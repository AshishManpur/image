"""SimpleGate and LayerScale (Contract Parts 2.2 and 2.3).

SPARC-Net contains **no activation functions**. Following NAFNet, every nonlinearity is
a SimpleGate: split the channels in half and multiply them elementwise. This is one
elementwise multiply, it removes the optimisation pathologies that come with saturating
or piecewise activations, and it halves the channel count for free.

Every residual branch is scaled by a learnable per-channel vector initialised to 1e-2,
so the whole network starts near the identity and trains without warmup tricks.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleGate(nn.Module):
    """Gated linear unit without an activation: ``x1 * x2`` after a channel split.

    Input ``(B, 2C, H, W)`` becomes output ``(B, C, H, W)``. Parameter-free.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split ``x`` in half along the channel dimension and multiply.

        Args:
            x: Tensor of shape ``(B, 2C, H, W)`` with an even channel count.

        Returns:
            Tensor of shape ``(B, C, H, W)``.

        Raises:
            ValueError: If the channel count is odd.
        """
        if x.shape[1] % 2 != 0:
            raise ValueError(
                f"SimpleGate requires an even channel count, got {x.shape[1]}."
            )
        first, second = x.chunk(2, dim=1)
        return first * second


class LayerScale(nn.Module):
    """Learnable per-channel scaling of a residual branch.

    Args:
        num_channels: Number of channels ``C``.
        init_value: Initial value of every element; the contract fixes this at 1e-2.

    Raises:
        ValueError: If ``num_channels`` is not positive.
    """

    def __init__(self, num_channels: int, init_value: float = 1e-2) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        self.num_channels = num_channels
        self.init_value = init_value
        self.gamma = nn.Parameter(torch.full((1, num_channels, 1, 1), float(init_value)))
        self._custom_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` channel-wise.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of the same shape.

        Raises:
            ValueError: If the channel count does not match.
        """
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"LayerScale configured for {self.num_channels} channels, "
                f"got {x.shape[1]}."
            )
        return x * self.gamma

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, init_value={self.init_value}"
