"""Blind two-parameter noise estimator (Contract Part 2.9, Part 3 stage 1).

Phase 1 established that the degradation variance follows the two-parameter model

.. math::  \\operatorname{Var}(y \\mid I) = \\sigma_g^2 + \\sigma_s^2 I^2

where :math:`\\sigma_g` is the additive (Gaussian) term and :math:`\\sigma_s` the
multiplicative (speckle) term. The three-parameter form ``a + bI + cI^2`` was measured
to be ill-conditioned (``corr(b, c) = -0.90``), so only the identifiable pair is
estimated. This head predicts that pair blindly from the degraded LR image and expands
it into a per-pixel sigma map that conditions the restoration trunk.

Architecture (Contract Part 3 stage 1, ``4 x [Conv3x3 s2 + LN + SG] -> GAP -> MLP``)::

    (B,1,128,128)
      stage 0: Conv3x3 s2 (1  -> 32) -> SimpleGate -> 16 -> LayerNorm2d(16)   @ 64x64
      stage 1: Conv3x3 s2 (16 -> 48) -> SimpleGate -> 24 -> LayerNorm2d(24)   @ 32x32
      stage 2: Conv3x3 s2 (24 -> 64) -> SimpleGate -> 32 -> LayerNorm2d(32)   @ 16x16
      stage 3: Conv3x3 s2 (32 -> 64) -> SimpleGate -> 32 -> LayerNorm2d(32)   @  8x8
      GlobalAvgPool                                      -> (B,32)
      Linear(32 -> 64) -> SimpleGate -> (B,32) -> Linear(32 -> 2)
      softplus                                           -> (sigma_g, sigma_s)

**Parameters: exactly 42,050**, matching Part 3 stage 1.

.. note::
   **LayerNorm placement.** Part 3 writes the stage as ``Conv3x3 s2+LN+SG``, which
   reads as LayerNorm *before* SimpleGate — i.e. on the pre-gate width ``2w``, costing
   ``4w`` parameters per stage and giving a trunk of 40,080. No MLP with the specified
   hidden width of 64 completes that to 42,050 (the closest is 40,080 + 2,178 =
   42,258). Placing LayerNorm *after* SimpleGate, on the post-gate width ``w``, gives a
   trunk of 39,872, and 39,872 + 2,178 = **42,050 exactly**.

   Contract Part 9 states that the parameter count "matches Part 3 **exactly** (not
   within a tolerance)", so the authoritative Part 3 total disambiguates the ambiguous
   Part 3 prose. This ordering is also the one that makes ``NoiseHeadConfig
   .trunk_channels`` correct as documented — "post-SimpleGate widths of the four
   strided stages". Resolved as finding V-2 in ``reports/PHASE4_7_AUDIT.md``.

.. note::
   **MAC accounting.** Part 3 records 52.32 MMAC for this stage. That figure counts
   each strided convolution at its *input* resolution: 51.90 M that way, plus 0.41 M
   for the 5x5 sigma-map smoother, totals 52.31 M. Counted correctly at *output*
   resolution the convolutions cost 12.98 M, so the measured value is **13.39 MMAC**.
   This is a documentation erratum in the Part 3 table, not an implementation defect,
   and the module is deliberately **not** inflated to match it. Recorded as finding V-3
   in ``reports/PHASE4_7_AUDIT.md``; the top-level 2.449 GMAC budget stays within its
   +/-5 % tolerance regardless.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from configs.sparc_config import NoiseHeadConfig
from models.blocks.layer_norm import LayerNorm2d
from models.blocks.simple_gate import SimpleGate
from models.noise.noise_map import assemble_sigma_map, build_smoothing_kernel
from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)

SIGMA_GAUSS_BIAS = -3.718
"""Contract Part 2.9: ``softplus(-3.718) = 0.024``, the Phase 1 additive sigma."""

SIGMA_SPECKLE_BIAS = -1.718
"""Contract Part 2.9: ``softplus(-1.718) = 0.165``, the Phase 1 speckle sigma."""


class NoiseHeadOutput(NamedTuple):
    """Structured noise-head prediction.

    A ``NamedTuple`` rather than a dataclass so that ``NoiseHead.forward`` stays
    TorchScript-scriptable (Contract Part 9), matching
    :class:`~models.normalization.NormalizationStatistics`.

    Attributes:
        sigma_gauss: Per-image additive sigma, shape ``(B, 1)``, in image units.
        sigma_speckle: Per-image multiplicative sigma, shape ``(B, 1)``, in image units.
        sigma_map: Per-pixel sigma ``(B, 1, H, W)`` in **image units**, clamped to
            ``[sigma_min, sigma_max]``. This is what the auxiliary loss supervises
            against the analytic target derived from the ground truth.
        sigma_map_normalized: The same map divided by the per-image normalisation
            scale, ``(B, 1, H, W)``. This is what the trunk consumes, so that the noise
            channel lives in the same units as the normalised image channel.
    """

    sigma_gauss: torch.Tensor
    sigma_speckle: torch.Tensor
    sigma_map: torch.Tensor
    sigma_map_normalized: torch.Tensor


class NoiseStage(nn.Module):
    """One strided trunk stage: ``Conv3x3 s2 -> SimpleGate -> LayerNorm2d``.

    Args:
        in_channels: Incoming channel count.
        out_channels: Post-SimpleGate width ``w``. The convolution produces ``2w``.
        layer_norm_eps: Epsilon of the LayerNorm.

    Raises:
        ValueError: If either channel count is not positive.
    """

    def __init__(
        self, in_channels: int, out_channels: int, layer_norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"channels must be positive, got {in_channels} -> {out_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels, kernel_size=3, stride=2, padding=1
        )
        self.gate = SimpleGate()
        self.norm = LayerNorm2d(out_channels, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the resolution and project to ``out_channels``.

        Args:
            x: Tensor ``(B, in_channels, H, W)``.

        Returns:
            Tensor ``(B, out_channels, H // 2, W // 2)``.
        """
        return self.norm(self.gate(self.conv(x)))

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


class NoiseHead(nn.Module):
    """Blind estimator of ``(sigma_gauss, sigma_speckle)`` and the per-pixel sigma map.

    Args:
        config: Frozen noise-head configuration.

    Raises:
        ValueError: If ``trunk_channels`` is empty or the sigma bounds are invalid.
    """

    def __init__(self, config: NoiseHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or NoiseHeadConfig()
        cfg = self.config
        if not cfg.trunk_channels:
            raise ValueError("trunk_channels must not be empty.")
        if not 0.0 < cfg.sigma_min < cfg.sigma_max:
            raise ValueError(
                f"Require 0 < sigma_min < sigma_max, got {cfg.sigma_min} "
                f"and {cfg.sigma_max}."
            )

        # Plain float attributes, not reads through `self.config`: TorchScript cannot
        # hold an arbitrary Python object as a module attribute, so anything touched
        # inside forward has to be a scriptable type.
        self.sigma_min = float(cfg.sigma_min)
        self.sigma_max = float(cfg.sigma_max)

        widths = cfg.trunk_channels
        stages: list[nn.Module] = []
        in_channels = 1
        for width in widths:
            stages.append(NoiseStage(in_channels, width))
            in_channels = width
        self.stages = nn.Sequential(*stages)

        # GAP -> Linear(w -> 2w) -> SimpleGate -> Linear(w -> 2). The SimpleGate keeps
        # the head activation-free per Part 5 ("Activation: none, SimpleGate only");
        # a bare Linear->Linear stack would collapse to a single affine map and give
        # the estimator no nonlinearity at all.
        final_width = widths[-1]
        self.fc1 = nn.Linear(final_width, 2 * final_width)
        self.fc_gate = SimpleGate()
        self.fc2 = nn.Linear(final_width, 2)

        # Contract Part 2.9: weight = 0 and bias = (-3.718, -1.718) so that softplus
        # emits exactly (0.024, 0.165) at initialisation — the Phase 1 measured values.
        # `_custom_init` stops `utils.init.default_init` from overwriting this during
        # SPARCNet's `self.apply(default_init)` sweep.
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.copy_(
                torch.tensor([SIGMA_GAUSS_BIAS, SIGMA_SPECKLE_BIAS], dtype=torch.float32)
            )
        self.fc2._custom_init = True

        self.register_buffer(
            "smoothing_kernel", build_smoothing_kernel(cfg.smoother_kernel), persistent=False
        )

        _LOGGER.debug(
            "NoiseHead built: %d parameters, trunk widths %s",
            sum(p.numel() for p in self.parameters()),
            widths,
        )

    @staticmethod
    def parameter_count(config: NoiseHeadConfig | None = None) -> int:
        """Analytic parameter count, used by the Part 3 budget assertion.

        Args:
            config: Configuration to size. Defaults to the contract configuration.

        Returns:
            Number of parameters.
        """
        cfg = config or NoiseHeadConfig()
        total = 0
        in_channels = 1
        for width in cfg.trunk_channels:
            hidden = 2 * width
            total += in_channels * hidden * 9 + hidden  # Conv3x3 with bias
            total += 2 * width  # LayerNorm2d on the post-gate width
            in_channels = width
        final = cfg.trunk_channels[-1]
        total += final * (2 * final) + 2 * final  # fc1
        total += final * 2 + 2  # fc2
        return total

    def predict_parameters(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the two noise parameters from a degraded image.

        Args:
            y: Degraded LR image ``(B, 1, H, W)``, unclipped.

        Returns:
            ``(sigma_gauss, sigma_speckle)``, each ``(B, 1)`` and strictly positive.

        Raises:
            ValueError: If ``y`` is not a 4-D single-channel tensor.
        """
        if y.dim() != 4:
            raise ValueError(
                "NoiseHead expects a 4-D tensor, got " + str(y.dim()) + " dims."
            )
        if y.shape[1] != 1:
            raise ValueError(
                "NoiseHead expects 1 input channel, got " + str(y.shape[1]) + "."
            )

        features = self.stages(y)
        pooled = features.mean(dim=[2, 3])
        hidden = self.fc_gate(self.fc1(pooled))
        raw = self.fc2(hidden)

        # softplus in float32: under fp16 autocast the pre-activation can sit far
        # enough into the negative tail that the fp16 result quantises badly.
        sigma = F.softplus(raw.float())
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        return sigma[:, 0:1], sigma[:, 1:2]

    def forward(
        self, y: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> NoiseHeadOutput:
        """Predict the noise parameters and expand them into a sigma map.

        Args:
            y: Degraded LR image ``(B, 1, H, W)``, **unclipped** — the statistics must
                be read from the raw observation, per Contract Part 2.8.
            scale: Per-image normalisation scale ``(B, 1, 1, 1)`` from
                :class:`~models.normalization.RobustNormalizer`. When ``None`` the
                normalised map equals the physical map.

        Returns:
            A :class:`NoiseHeadOutput`.
        """
        sigma_gauss, sigma_speckle = self.predict_parameters(y)
        sigma_map = assemble_sigma_map(
            y,
            sigma_gauss,
            sigma_speckle,
            self.smoothing_kernel,
            self.sigma_min,
            self.sigma_max,
        )
        normalized = sigma_map
        if scale is not None:
            normalized = sigma_map / scale.to(sigma_map.dtype).clamp_min(1e-8)
        return NoiseHeadOutput(
            sigma_gauss=sigma_gauss,
            sigma_speckle=sigma_speckle,
            sigma_map=sigma_map,
            sigma_map_normalized=normalized.to(y.dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"trunk_channels={self.config.trunk_channels}, "
            f"sigma_range=({self.config.sigma_min}, {self.config.sigma_max})"
        )
