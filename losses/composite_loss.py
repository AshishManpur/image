"""Weighted composite objective (Contract Part 6).

.. math::

    \\mathcal{L} = \\mathcal{L}_{\\text{Charb}}
                 + 0.15\\,\\mathcal{L}_{\\text{MS-SSIM}}
                 + 0.10\\,\\mathcal{L}_{\\text{wav}}
                 + 0.05\\,\\mathcal{L}_{\\text{FFT}}
                 + 0.05\\,\\mathcal{L}_{\\text{grad}}
                 + 0.02\\,\\mathcal{L}_{\\sigma}

Every weight is read from :class:`configs.sparc_config.LossConfig` — none is written as
a literal here, so the contract values live in exactly one place and are already
verified against the contract by ``tests/test_config.py``.

**Ordering constraint (Part 6):** all terms are computed *after de-normalisation and
clamping*, against unmodified ground truth. That is satisfied structurally:
:meth:`models.sparc_net.SPARCNet.forward_with_aux` de-normalises and clamps before
returning, so ``SparcOutput.image`` is already in the required state.

The module reports both weighted and unweighted values for every term. The unweighted
values are what tell you whether a term is *learning*; the weighted ones tell you what
it contributes to the gradient. Logging only one of the two makes loss-balance
questions unanswerable, and Part 6 requires every term to be logged separately.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from configs.sparc_config import LossConfig
from datasets.degradation import forward_operator
from losses.charbonnier import CharbonnierLoss
from losses.fft_loss import FFTLoss
from losses.gradient import GradientLoss
from losses.ms_ssim import MSSSIMLoss
from losses.noise_loss import NoiseAuxLoss
from losses.wavelet_loss import WaveletLoss
from utils.logging_utils import get_logger
from utils.numerics import fp32_island

_LOGGER = get_logger(__name__)

TERM_NAMES: tuple[str, ...] = (
    "charbonnier",
    "ms_ssim",
    "wavelet",
    "fft",
    "gradient",
    "noise",
    "clean_lr",
)
"""Contract Part 6 term order, plus the Phase 6 ``clean_lr`` term. ``total`` is appended
by the forward pass."""


class CompositeLoss(nn.Module):
    """The full SPARC-Base V1 objective.

    Args:
        config: Frozen loss configuration supplying every weight.
        enabled: Optional per-term switches, keyed by the names in :data:`TERM_NAMES`.
            A disabled term is neither constructed nor evaluated, which is what makes
            the Part 11 ablations cheap. Defaults to all enabled.
        channels: Channel count of the images, for the Sobel and SSIM kernels.

    Raises:
        ValueError: If ``enabled`` names a term that does not exist.
    """

    wants_aux: bool = True
    """Signals to :class:`trainer.trainer.Trainer` that this criterion consumes the
    model's auxiliary outputs and the input batch, not just ``(prediction, target)``."""

    def __init__(
        self,
        config: LossConfig | None = None,
        enabled: Mapping[str, bool] | None = None,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.config = config or LossConfig()
        cfg = self.config

        flags = {name: True for name in TERM_NAMES}
        if enabled is not None:
            unknown = set(enabled) - set(TERM_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown loss term(s) {sorted(unknown)}; known: {list(TERM_NAMES)}."
                )
            flags.update(enabled)
        self.enabled = flags

        self.weights: dict[str, float] = {
            "charbonnier": cfg.charbonnier,
            "ms_ssim": cfg.ms_ssim,
            "wavelet": cfg.wavelet,
            "fft": cfg.fft,
            "gradient": cfg.gradient,
            "noise": cfg.noise_aux,
            "clean_lr": cfg.clean_lr,
        }
        self.clean_lr_target_weight = cfg.clean_lr
        """Full-strength weight. ``weights["clean_lr"]`` is what is actually applied and
        is driven by :meth:`set_clean_lr_weight` during warmup."""

        self.charbonnier = (
            CharbonnierLoss(eps=cfg.charbonnier_eps) if flags["charbonnier"] else None
        )
        self.ms_ssim = (
            MSSSIMLoss(
                scales=cfg.ms_ssim_scales,
                window_size=cfg.ms_ssim_window,
                sigma=cfg.ms_ssim_sigma,
                channels=channels,
            )
            if flags["ms_ssim"]
            else None
        )
        self.wavelet = (
            WaveletLoss(levels=cfg.wavelet_levels, band_weights=cfg.wavelet_band_weights)
            if flags["wavelet"]
            else None
        )
        self.fft = FFTLoss() if flags["fft"] else None
        self.gradient = GradientLoss(channels=channels) if flags["gradient"] else None
        self.noise = NoiseAuxLoss() if flags["noise"] else None
        self.clean_lr = (
            CharbonnierLoss(eps=cfg.charbonnier_eps) if flags["clean_lr"] else None
        )

        active = [name for name, on in flags.items() if on]
        _LOGGER.info(
            "CompositeLoss: %d active terms %s", len(active), active
        )

    # ------------------------------------------------------------------ helpers
    def set_clean_lr_weight(self, weight: float) -> None:
        """Set the applied ``clean_lr`` weight, for the warmup ramp.

        The branch is zero-initialised, so on the first step of a warm-started run
        ``predicted_clean_LR`` is the noisy observation (measured 22.26 dB, Charbonnier
        ~0.07) against a converged ``charbonnier`` of ~0.031. Applying the full weight
        immediately would make the new term ~40 % of the objective and perturb a model
        that starts bit-identical to the 27.43 dB checkpoint.

        Args:
            weight: Non-negative weight to apply from now on.

        Raises:
            ValueError: If ``weight`` is negative.
        """
        if weight < 0.0:
            raise ValueError(f"clean_lr weight must be non-negative, got {weight}.")
        self.weights["clean_lr"] = float(weight)

    @staticmethod
    def _unpack(
        output: Any, batch: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Normalise the two accepted call signatures.

        Accepts either ``(SparcOutput, batch_dict)`` — the auxiliary path — or the
        plain ``(prediction_tensor, target_tensor)`` form, so the composite can still
        be unit-tested and used as a drop-in reconstruction criterion.

        Args:
            output: ``SparcOutput`` or a prediction tensor.
            batch: Batch mapping with ``gt``/``lr``, or a target tensor.

        Returns:
            ``(prediction, target, sigma_hat, noisy_lr, clean_lr)``.

        Raises:
            ValueError: If a batch mapping is supplied without a ``gt`` entry.
        """
        prediction = getattr(output, "image", output)
        sigma_hat = getattr(output, "sigma", None)
        clean_lr = getattr(output, "clean_lr", None)

        if isinstance(batch, torch.Tensor):
            return prediction, batch, sigma_hat, None, clean_lr
        if "gt" not in batch:
            raise ValueError("Batch mapping must contain a 'gt' entry.")
        return prediction, batch["gt"], sigma_hat, batch.get("lr"), clean_lr

    # ------------------------------------------------------------------ forward
    def forward(
        self, output: Any, batch: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate every enabled term and return the weighted total.

        Args:
            output: ``SparcOutput`` from ``forward_with_aux``, or a prediction tensor.
            batch: The input batch mapping, or a ground-truth tensor.

        Returns:
            ``(total, terms)``. ``terms`` holds the weighted contribution of each term
            under its name, the unweighted value under ``raw_<name>``, and ``total``.

        Raises:
            ValueError: If the prediction and target shapes differ.
        """
        prediction, target, sigma_hat, noisy_lr, clean_lr = self._unpack(output, batch)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {tuple(prediction.shape)} vs target "
                f"{tuple(target.shape)}."
            )
        with fp32_island(prediction.device.type):
            return self._evaluate(
                prediction.float(),
                target.float(),
                None if sigma_hat is None else sigma_hat.float(),
                None if noisy_lr is None else noisy_lr.float(),
                None if clean_lr is None else clean_lr.float(),
            )

    def _evaluate(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sigma_hat: torch.Tensor | None,
        noisy_lr: torch.Tensor | None,
        clean_lr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate every enabled term on float32 inputs, outside autocast.

        Split from :meth:`forward` so that the ``autocast(enabled=False)`` region is a
        single unbroken block covering every term. Terms whose internals use
        convolution — MS-SSIM and the gradient loss — were silently executing in fp16
        before Phase 4.10.1 despite their own ``.float()`` calls, because autocast
        re-casts convolution arguments regardless of what the caller passes.

        Args:
            prediction: Restored image, float32.
            target: Ground truth, float32.
            sigma_hat: Predicted sigma map, float32, or ``None``.
            noisy_lr: Observed LR image, float32, or ``None``.

        Returns:
            ``(total, terms)`` as documented on :meth:`forward`.
        """
        total = prediction.new_zeros(())
        terms: dict[str, float] = {}

        def _accumulate(name: str, value: torch.Tensor) -> None:
            nonlocal total
            weight = self.weights[name]
            total = total + weight * value
            terms[name] = float(weight * value.detach())
            terms[f"raw_{name}"] = float(value.detach())

        if self.charbonnier is not None:
            _accumulate("charbonnier", self.charbonnier(prediction, target))
        if self.ms_ssim is not None:
            _accumulate("ms_ssim", self.ms_ssim(prediction, target))
        if self.wavelet is not None:
            _accumulate("wavelet", self.wavelet(prediction, target))
        if self.fft is not None:
            _accumulate("fft", self.fft(prediction, target))
        if self.gradient is not None:
            _accumulate("gradient", self.gradient(prediction, target))

        # The noise term is skipped silently when the head is disabled or the batch
        # carries no LR image: an ablation that turns the head off must not crash the
        # objective, and validation loaders may not supply everything the term needs.
        if self.noise is not None and sigma_hat is not None and noisy_lr is not None:
            _accumulate("noise", self.noise(sigma_hat, noisy_lr, target))

        # Phase 6. Skipped silently when the branch is disabled, for the same reason the
        # noise term is: a model without the branch must not crash the objective.
        if self.clean_lr is not None and clean_lr is not None:
            clean_target = forward_operator(target, self.config.clean_lr_blur_sigma)
            if clean_lr.shape != clean_target.shape:
                raise ValueError(
                    f"Shape mismatch: clean_lr {tuple(clean_lr.shape)} vs target "
                    f"{tuple(clean_target.shape)}."
                )
            _accumulate("clean_lr", self.clean_lr(clean_lr, clean_target))

        terms["total"] = float(total.detach())
        return total, terms

    def extra_repr(self) -> str:
        active = ", ".join(
            f"{name}={self.weights[name]}" for name, on in self.enabled.items() if on
        )
        return f"weights({active})"
