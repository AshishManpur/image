"""Concatenation skip fusion — the ablation baseline (Contract Part 11, A3).

This is the fusion used by the step-6 model and by ablation A3's control arm. It is
deliberately the simplest thing that works: concatenate the encoder skip with the
decoder feature and project back to ``C`` with a 1x1 convolution.

Its limitation is exactly what ``GatedFuse`` is introduced to fix: the mixing weights
are fixed at training time, so the network cannot say "trust the skip more on this
image" even though Phase 1 measured the noise level varying 8.5x across images and,
because sigma scales with intensity, varying spatially within each image.
"""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Fuse an encoder skip with a decoder feature by concatenation and a 1x1 conv.

    Args:
        channels: Channel count ``C`` of both inputs and of the output.

    Raises:
        ValueError: If ``channels`` is not positive.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = channels
        self.project = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, skip: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        """Fuse two feature maps.

        Args:
            skip: Encoder feature ``(B, C, H, W)``.
            decoded: Decoder feature ``(B, C, H, W)``.

        Returns:
            Fused tensor ``(B, C, H, W)``.

        Raises:
            ValueError: If the two inputs disagree in shape or channel count.
        """
        if skip.shape != decoded.shape:
            raise ValueError(
                "ConcatFusion inputs must match: got "
                + str(list(skip.shape)) + " and " + str(list(decoded.shape)) + "."
            )
        if skip.shape[1] != self.channels:
            raise ValueError(
                "ConcatFusion configured for " + str(self.channels)
                + " channels, got " + str(skip.shape[1]) + "."
            )
        return self.project(torch.cat((skip, decoded), dim=1))

    def extra_repr(self) -> str:
        return f"channels={self.channels}"
