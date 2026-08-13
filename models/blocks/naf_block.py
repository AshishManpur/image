"""NAF block — the local restoration operator (Contract Part 2.5).

This is the workhorse of the network. Phase 1 established that 84 % of the recoverable
performance is denoising and that the noise is spatially white, which makes restoration
primarily a local-statistics problem; Phase 2 selected the NAF block for it on three
measured grounds: best accuracy per FLOP, most stable training (no activation
nonlinearity, identity-initialised residual scales), and trivial export.

Structure, exactly as specified::

    t = LayerNorm2d(x)
    t = Conv1x1(C -> 2C) -> DWConv3x3(2C) -> SimpleGate -> SCA -> Conv1x1(C -> C)
    x = x + LayerScale_1 * t
    u = LayerNorm2d(x)
    u = Conv1x1(C -> 2C) -> SimpleGate -> Conv1x1(C -> C)
    x = x + LayerScale_2 * u

``SCA`` is Simplified Channel Attention: ``z * Conv1x1(GlobalAvgPool(z))`` — no sigmoid,
no hidden nonlinearity.

Parameter count is ``7C^2 + 33C``: 17,712 at C=48, 67,680 at C=96, 184,480 at C=160 and
8,224 at C=32, matching Contract Part 3 exactly.

.. note::
   Contract Part 2.5 prints the parameter formula as ``7C^2 + 8C``. That is an erratum
   in the prose; the Part 3 table (which is authoritative and was generated from the
   same arithmetic as this implementation) gives ``7C^2 + 33C``. The linear term is
   the two LayerNorms (4C), two LayerScales (2C), the depthwise 3x3 (20C including its
   bias) and the four remaining convolution biases (7C).
"""

from __future__ import annotations

import torch
from torch import nn

from models.blocks.layer_norm import LayerNorm2d
from models.blocks.simple_gate import LayerScale, SimpleGate


class NAFBlock(nn.Module):
    """Nonlinear-Activation-Free restoration block.

    Args:
        channels: Channel count ``C``; must be positive and even.
        expansion: Channel expansion inside both branches. The contract fixes this at 2.
        layer_scale_init: Initial value of both residual scales.
        layer_norm_eps: Epsilon of the two LayerNorms.

    Raises:
        ValueError: If ``channels`` is not a positive even number, or ``expansion``
            is not positive.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        layer_scale_init: float = 1e-2,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError(f"channels must be a positive even number, got {channels}.")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}.")

        self.channels = channels
        self.expansion = expansion
        hidden = channels * expansion

        # --- spatial branch
        self.norm1 = LayerNorm2d(channels, eps=layer_norm_eps)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.gate1 = SimpleGate()
        self.sca = nn.Conv2d(hidden // 2, hidden // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden // 2, channels, kernel_size=1)
        self.scale1 = LayerScale(channels, layer_scale_init)

        # --- channel branch
        self.norm2 = LayerNorm2d(channels, eps=layer_norm_eps)
        self.conv4 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(hidden // 2, channels, kernel_size=1)
        self.scale2 = LayerScale(channels, layer_scale_init)

    @staticmethod
    def parameter_count(channels: int, expansion: int = 2) -> int:
        """Analytic parameter count, used by the Part 3 budget assertions.

        Args:
            channels: Channel count ``C``.
            expansion: Channel expansion factor.

        Returns:
            Number of parameters.
        """
        c, hidden = channels, channels * expansion
        half = hidden // 2
        return (
            2 * c  # norm1
            + (c * hidden + hidden)  # conv1
            + (9 * hidden + hidden)  # dwconv
            + (half * half + half)  # sca
            + (half * c + c)  # conv3
            + c  # scale1
            + 2 * c  # norm2
            + (c * hidden + hidden)  # conv4
            + (half * c + c)  # conv5
            + c  # scale2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of the same shape.

        Raises:
            ValueError: If the channel count does not match.
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                "NAFBlock configured for " + str(self.channels)
                + " channels, got " + str(x.shape[1]) + "."
            )

        t = self.norm1(x)
        t = self.conv1(t)
        t = self.dwconv(t)
        t = self.gate1(t)
        t = t * self.sca(t.mean(dim=[2, 3], keepdim=True))
        t = self.conv3(t)
        x = x + self.scale1(t)

        u = self.norm2(x)
        u = self.conv4(u)
        u = self.gate2(u)
        u = self.conv5(u)
        return x + self.scale2(u)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, expansion={self.expansion}"
