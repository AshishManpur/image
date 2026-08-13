"""Robust per-image normalisation (Contract Part 2.8, Stage 0).

Phase 1 measured per-image means spanning 0.016-0.959 and standard deviations spanning
0.016-0.451, so exposure and contrast are nuisance dimensions the network would
otherwise have to model. The transform is parameter-free and exactly invertible; the
statistics travel alongside the tensor so the output can be returned to original units.

Statistics are computed on the **raw, unclamped** input: Phase 1 measured 3.36 % of LR
pixels above 1.0 and 0.30 % below 0, and clamping them would discard real signal.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class NormalizationStatistics(NamedTuple):
    """Per-image location and scale, retained for exact inversion.

    Attributes:
        location: Per-image mean, shape ``(B, 1, 1, 1)``.
        scale: Per-image standard deviation, floored, shape ``(B, 1, 1, 1)``.
    """

    location: torch.Tensor
    scale: torch.Tensor


class RobustNormalizer(nn.Module):
    """Invertible per-image mean/std normalisation.

    Args:
        scale_floor: Lower clamp on the scale. Phase 1 found 7 near-featureless images
            with standard deviation below 0.03; the floor keeps the inverse finite.
        enabled: When ``False`` the module is an identity and returns unit statistics.

    Raises:
        ValueError: If ``scale_floor`` is not positive.
    """

    def __init__(self, scale_floor: float = 0.02, enabled: bool = True) -> None:
        super().__init__()
        if scale_floor <= 0.0:
            raise ValueError(f"scale_floor must be positive, got {scale_floor}.")
        self.scale_floor = scale_floor
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, NormalizationStatistics]:
        """Normalise ``x`` and return the statistics needed to invert it.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            ``(normalised, statistics)``.

        Raises:
            ValueError: If ``x`` is not 4-D.
        """
        if x.dim() != 4:
            raise ValueError(
                "RobustNormalizer expects a 4-D tensor, got " + str(x.dim()) + " dims."
            )
        if not self.enabled:
            ones = torch.ones(
                (x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device
            )
            return x, NormalizationStatistics(torch.zeros_like(ones), ones)

        location = x.mean(dim=[1, 2, 3], keepdim=True)
        scale = x.std(dim=[1, 2, 3], keepdim=True, unbiased=False).clamp_min(
            self.scale_floor
        )
        return (x - location) / scale, NormalizationStatistics(location, scale)

    @staticmethod
    def denormalize(
        x: torch.Tensor, stats: NormalizationStatistics
    ) -> torch.Tensor:
        """Invert :meth:`forward`.

        Args:
            x: Normalised tensor ``(B, C, H, W)``.
            stats: Statistics produced by :meth:`forward`.

        Returns:
            The tensor in original units.

        Raises:
            ValueError: If the batch dimensions do not match.
        """
        if stats.location.shape[0] != x.shape[0]:
            raise ValueError(
                "Statistics batch " + str(stats.location.shape[0])
                + " does not match input batch " + str(x.shape[0]) + "."
            )
        return x * stats.scale + stats.location

    def extra_repr(self) -> str:
        return f"scale_floor={self.scale_floor}, enabled={self.enabled}"
