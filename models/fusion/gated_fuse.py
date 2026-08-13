"""Gated skip fusion (Contract Part 2.7, Part 3 stages 15 and 20).

``ConcatFusion`` — the step-6 placeholder and the ablation A3 control arm — mixes the
encoder skip with the decoder feature using weights fixed at training time. Phase 1
measured the noise level varying **8.5x across images**, and because sigma scales with
intensity it varies spatially *within* each image too. A fixed mixing rule cannot say
"trust the skip more on this image": on a clean image the skip carries recoverable
high-frequency detail worth keeping, while on a noisy one it largely carries noise that
the decoder has already suppressed.

``GatedFuse`` makes that decision per channel, conditioned on the joint statistics of
both inputs::

    u = Conv1x1(2C -> C)(concat([skip, dec], dim=1))
    g = sigmoid( Conv1x1(C//4 -> C)( Conv1x1(C -> C//4)( GlobalAvgPool(u) ) ) )
    out = g * skip + (1 - g) * dec

Three properties follow from that form and are each pinned by a test:

* **The output is a convex combination.** ``g in (0, 1)`` and the two coefficients sum
  to exactly 1, so the fusion can never amplify: its output is bounded elementwise by
  the two inputs. This is what keeps the residual path numerically stable.
* **The gate is global per channel.** ``g`` has shape ``(B, C, 1, 1)``, so the decision
  is "how much do I trust this channel of the skip for this image", not a per-pixel
  mask. A per-pixel gate would be a different module and is not what the contract
  specifies.
* **The bottleneck is genuine.** ``C -> C//4 -> C`` is a rank-``C//4`` map, which forces
  the gate to key on a compressed summary rather than memorising channels.

Parameters ``2.5C^2 + 2C + C/4``: **23,256 at C=96** and **5,868 at C=48**, matching
Part 3 stages 15 and 20 exactly.

.. note::
   There is deliberately no activation between the two gate convolutions. Contract
   Part 5 fixes the activation set to "none (SimpleGate only)", and a SimpleGate here
   would halve the bottleneck width and break the parameter budget. The composition is
   therefore a low-rank linear map followed by the sigmoid, which supplies the
   nonlinearity. The Part 3 counts confirm exactly two convolutions in the gate.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedFuse(nn.Module):
    """Channel-gated convex fusion of an encoder skip and a decoder feature.

    Args:
        channels: Channel count ``C`` of both inputs and of the output.
        reduction: Bottleneck ratio of the gate MLP. The contract fixes this at 4.

    Raises:
        ValueError: If ``channels`` is not positive, or ``reduction`` is not positive,
            or the bottleneck ``channels // reduction`` would collapse to zero.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        hidden = channels // reduction
        if hidden <= 0:
            raise ValueError(
                f"channels // reduction must be at least 1, got {channels} // "
                f"{reduction} = {hidden}."
            )

        self.channels = channels
        self.reduction = reduction
        self.hidden = hidden

        self.project = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.squeeze = nn.Conv2d(channels, hidden, kernel_size=1)
        self.excite = nn.Conv2d(hidden, channels, kernel_size=1)

    @staticmethod
    def parameter_count(channels: int, reduction: int = 4) -> int:
        """Analytic parameter count, used by the Part 3 budget assertion.

        Args:
            channels: Channel count ``C``.
            reduction: Bottleneck ratio.

        Returns:
            Number of parameters.
        """
        hidden = channels // reduction
        return (
            (2 * channels * channels + channels)  # project
            + (channels * hidden + hidden)  # squeeze
            + (hidden * channels + channels)  # excite
        )

    def gate(self, skip: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        """Compute the per-channel mixing weight.

        Exposed separately so that tests and diagnostics can inspect the gate without
        re-deriving the fusion, and so a training run can log its distribution.

        Args:
            skip: Encoder feature ``(B, C, H, W)``.
            decoded: Decoder feature ``(B, C, H, W)``.

        Returns:
            Gate of shape ``(B, C, 1, 1)`` with every element in ``[0, 1]``.

        .. note::
           Strictly the sigmoid is open on ``(0, 1)``, but at feature magnitudes around
           ``1e3`` it saturates to exactly ``0.0`` or ``1.0`` in float32. The fusion
           stays well-defined there — a saturated gate simply selects one input
           outright — but the gate's own gradient vanishes, so the mix stops adapting.
           The trunk operates on normalised features, so this is far outside the
           operating range; it is documented because it is the one way this module can
           quietly stop learning.
        """
        joint = self.project(torch.cat((skip, decoded), dim=1))
        # mean over the spatial dims rather than AdaptiveAvgPool2d(1): numerically
        # identical, one op fewer, and it exports without a resize node.
        pooled = joint.mean(dim=[2, 3], keepdim=True)
        return torch.sigmoid(self.excite(self.squeeze(pooled)))

    def forward(self, skip: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        """Fuse an encoder skip with a decoder feature.

        Args:
            skip: Encoder feature ``(B, C, H, W)``.
            decoded: Decoder feature ``(B, C, H, W)``.

        Returns:
            Fused tensor ``(B, C, H, W)``, elementwise between the two inputs.

        Raises:
            ValueError: If the two inputs disagree in shape, or the channel count does
                not match the configuration.
        """
        # String concatenation rather than f-strings: this method must stay
        # TorchScript-scriptable (Contract Part 9), and TorchScript cannot size
        # `tuple(tensor.shape)`.
        if skip.shape != decoded.shape:
            raise ValueError(
                "GatedFuse inputs must match: got " + str(list(skip.shape))
                + " and " + str(list(decoded.shape)) + "."
            )
        if skip.shape[1] != self.channels:
            raise ValueError(
                "GatedFuse configured for " + str(self.channels)
                + " channels, got " + str(skip.shape[1]) + "."
            )

        g = self.gate(skip, decoded)
        return g * skip + (1.0 - g) * decoded

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, reduction={self.reduction}, "
            f"hidden={self.hidden}"
        )
