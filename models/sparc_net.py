"""SPARC-Net top-level assembly (Contract Part 1).

Data flow::

    y (B,1,128,128) unclipped
      -> RobustNormalizer                -> y_hat, (mean, std)
      -> NoiseHead                       -> sigma_hat (B,1,128,128)
      -> concat[y_hat, sigma_hat]        -> (B,2,128,128)
      -> Encoder (DWT stem + 3 levels)   -> bottleneck (B,160,16,16), 2 skips
      -> Decoder (2 stages, gated skips) -> (B,48,64,64)
      -> ReconstructionHead              -> (B,1,256,256)
      -> + bicubic_up2(y_hat)            global residual
      -> * std + mean                    de-normalise
      -> clamp(0, 1)                     GT is exactly [0,1] for all 3200 images
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing a deferred module
    from models.noise.noise_head import NoiseHeadOutput

from configs.sparc_config import SparcConfig, sparc_base
from models.decoder.decoder import Decoder
from models.decoder.clean_lr_branch import CleanLRBranch
from models.decoder.reconstruction_head import ReconstructionHead
from models.encoder import Encoder
from models.film import FiLMConditioner
from models.normalization import NormalizationStatistics, RobustNormalizer
from models.vst import VarianceStabiliser
from utils.init import default_init
from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class SparcOutput:
    """Model output plus the intermediates the loss needs.

    Attributes:
        image: Restored image ``(B, 1, 256, 256)`` in ``[0, 1]``.
        sigma: Predicted noise map ``(B, 1, 128, 128)`` in **image units**, or ``None``
            when the noise head is disabled. This is what the auxiliary noise loss
            supervises against the analytic target derived from the ground truth.
        stats: Normalisation statistics used for the inverse transform.
        noise: Full structured noise-head prediction — the two per-image sigmas plus
            both sigma maps — or ``None`` when the noise head is disabled. ``sigma`` is
            ``noise.sigma_map``; this field exposes the scalars the Phase 4.10
            auxiliary loss also needs.
    """

    image: torch.Tensor
    sigma: torch.Tensor | None
    stats: NormalizationStatistics
    noise: "NoiseHeadOutput | None" = None
    clean_lr: torch.Tensor | None = None
    """Phase 6 clean-LR estimate ``(B, 1, H, W)`` in **image units**, or ``None`` when
    ``use_clean_lr_branch`` is disabled. Already de-normalised and VST-inverted, so the
    composite loss can compare it directly against ``A(GT)``. This is an internal
    intermediate prediction, never the model's output."""


class SPARCNet(nn.Module):
    """Speckle-Prior Adaptive Restoration with Consistency, V1.0.

    Args:
        config: Frozen architecture configuration. Defaults to SPARC-Base.
    """

    def __init__(self, config: SparcConfig | None = None) -> None:
        super().__init__()
        self.config = config or sparc_base()
        cfg = self.config

        self.normalizer = RobustNormalizer(
            scale_floor=cfg.normalization_scale_floor, enabled=cfg.use_normalization
        )
        self.noise_head = self._build_noise_head(cfg)
        self.vst = VarianceStabiliser(
            enabled=cfg.use_vst, bias_correction=cfg.use_vst_bias_correction
        )
        self.film = (
            FiLMConditioner(cfg.film_sites, hidden=cfg.film_hidden)
            if cfg.use_film
            else None
        )
        stem_channels = cfg.in_channels + (1 if self.noise_head is not None else 0)

        self.encoder = Encoder(cfg, in_channels=stem_channels)
        self.decoder = Decoder(cfg)
        self.head = ReconstructionHead(cfg)
        self.clean_branch = CleanLRBranch(cfg) if cfg.use_clean_lr_branch else None

        self.apply(default_init)
        _LOGGER.info(
            "%s built: %.4f M parameters",
            cfg.name,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    @staticmethod
    def _build_noise_head(config: SparcConfig) -> nn.Module | None:
        """Construct the noise head, deferring the import until it exists."""
        if not config.use_noise_head:
            return None
        from models.noise.noise_head import NoiseHead

        return NoiseHead(config.noise)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Restore a batch of degraded images.

        Args:
            y: Degraded input ``(B, 1, H, W)``, **unclipped**.

        Returns:
            Restored image ``(B, 1, 2H, 2W)``.
        """
        return self.forward_with_aux(y).image

    def forward_with_aux(self, y: torch.Tensor) -> SparcOutput:
        """Restore a batch and return the auxiliary tensors the loss consumes.

        Args:
            y: Degraded input ``(B, 1, H, W)``, **unclipped**.

        Returns:
            A :class:`SparcOutput`.

        Raises:
            ValueError: If ``y`` is not 4-D or has the wrong channel count, or if its
                spatial size is not divisible by ``2 ** num_levels``.
        """
        cfg = self.config
        if y.dim() != 4:
            raise ValueError(
                "SPARCNet expects a 4-D tensor, got " + str(y.dim()) + " dims."
            )
        if y.shape[1] != cfg.in_channels:
            raise ValueError(
                "SPARCNet configured for " + str(cfg.in_channels)
                + " input channels, got " + str(y.shape[1]) + "."
            )
        divisor = 2**cfg.num_levels
        if y.shape[2] % divisor or y.shape[3] % divisor:
            raise ValueError(
                "Input spatial size must be divisible by " + str(divisor)
                + ", got " + str(y.shape[2]) + "x" + str(y.shape[3]) + "."
            )

        sigma: torch.Tensor | None = None
        noise: "NoiseHeadOutput | None" = None
        signal = y
        if self.noise_head is not None:
            # Statistics are read from the raw, unclipped observation (Part 2.8). The
            # scale is not known until the trunk input has been formed, so the map is
            # re-normalised below once `stats` exists.
            noise = self.noise_head(y)
            sigma = noise.sigma_map
            # Identity when `use_vst` is False, so the pre-Phase-5 path is unchanged.
            signal = self.vst(y, noise.sigma_gauss, noise.sigma_speckle)

        y_hat, stats = self.normalizer(signal)

        features = y_hat
        film_encoder: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        film_decoder: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if noise is not None:
            # The trunk consumes the map divided by the normalisation scale so that
            # both input channels share units.
            normalized = noise.sigma_map / stats.scale.to(
                noise.sigma_map.dtype
            ).clamp_min(1e-8)
            noise = noise._replace(sigma_map_normalized=normalized.to(y.dtype))
            features = torch.cat((y_hat, noise.sigma_map_normalized), dim=1)
            if self.film is not None:
                coefficients = self.film(noise.sigma_gauss, noise.sigma_speckle)
                film_encoder = coefficients[: cfg.num_levels]
                film_decoder = coefficients[cfg.num_levels :]

        bottleneck, skips = self.encoder(features, film_encoder)
        decoded = self.decoder(bottleneck, skips, film_decoder)
        out = self.head(decoded)

        # Phase 6. `clean_norm` is the residual base in normalised units; `clean_lr` is
        # the same tensor in image units, which is what `A(GT)` supervises. The branch
        # reads `decoded` — the tensor the head already consumes — so the encoder and
        # decoder are shared and nothing is recomputed.
        clean_norm = y_hat
        clean_lr: torch.Tensor | None = None
        if self.clean_branch is not None:
            clean_norm = y_hat + self.clean_branch(decoded)
            clean_lr = self.normalizer.denormalize(clean_norm, stats)
            if noise is not None:
                # Identity when `use_vst` is False.
                clean_lr = self.vst.inverse(
                    clean_lr, noise.sigma_gauss, noise.sigma_speckle
                )

        if cfg.use_global_residual:
            # `residual_source="noisy"` reproduces the V1 output path exactly, and is
            # still meaningful with the branch enabled: the clean-LR estimate is then
            # trained as a pure auxiliary task without altering the output.
            base = clean_norm if cfg.residual_source == "clean" else y_hat
            out = out + F.interpolate(
                base, scale_factor=float(cfg.scale), mode="bicubic", align_corners=False
            )

        out = self.normalizer.denormalize(out, stats)
        if noise is not None:
            # Identity when `use_vst` is False.
            out = self.vst.inverse(out, noise.sigma_gauss, noise.sigma_speckle)
        if cfg.clamp_output:
            out = out.clamp(cfg.output_min, cfg.output_max)
        return SparcOutput(
            image=out, sigma=sigma, stats=stats, noise=noise, clean_lr=clean_lr
        )


def build_model(variant: str = "sparc-base", **overrides: object) -> SPARCNet:
    """Build a SPARC-Net from a named variant.

    Args:
        variant: ``sparc-tiny`` or ``sparc-base``.
        **overrides: Configuration fields to replace.

    Returns:
        The constructed model.
    """
    from configs.sparc_config import build_sparc_config

    return SPARCNet(build_sparc_config(variant, **overrides))
