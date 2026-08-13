"""Sobel gradient loss (Contract Part 6, weight 0.05).

Edge fidelity is scored indirectly by SSIM but not directly by any per-pixel term.
This loss adds an explicit first-order constraint: the horizontal and vertical Sobel
responses of the prediction must match those of the ground truth in L1.

Contract Part 6 defines the term as **"L1 on Sobel-x and Sobel-y"** — the two
components are compared separately and averaged, *not* combined into a gradient
magnitude first. The distinction is not cosmetic: matching magnitudes alone would let
an edge point the wrong way at no cost, and the magnitude's ``sqrt`` has an unbounded
derivative at zero, which is exactly where flat regions sit. The component form is used
here; :meth:`GradientLoss.magnitude` is provided for diagnostics only and is not part
of the objective.

The kernels are registered **buffers**, never ``nn.Parameter`` and never ``nn.Conv2d``:
a loss module can end up inside a ``nn.Module`` tree, and a stray parameter there would
corrupt the Part 3 parameter budget and silently join the optimiser's decay group.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SOBEL_X: tuple[tuple[float, ...], ...] = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)
SOBEL_Y: tuple[tuple[float, ...], ...] = (
    (-1.0, -2.0, -1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 1.0),
)


class GradientLoss(nn.Module):
    """L1 between the Sobel-x and Sobel-y responses of two images.

    Args:
        channels: Channel count of the inputs; kernels are applied depthwise.
        normalize: Divide the kernels by 8 so the response approximates a true
            derivative rather than an 8x-scaled one. This only rescales the term, but
            it keeps the loss magnitude comparable to the other L1-style terms, which
            matters because the contract weights are fixed.

    Raises:
        ValueError: If ``channels`` is not positive.
    """

    def __init__(self, channels: int = 1, normalize: bool = True) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = channels
        self.normalize = normalize

        scale = 8.0 if normalize else 1.0
        kernel_x = torch.tensor(SOBEL_X, dtype=torch.float32) / scale
        kernel_y = torch.tensor(SOBEL_Y, dtype=torch.float32) / scale
        self.register_buffer(
            "kernel_x", kernel_x.expand(channels, 1, 3, 3).contiguous(), persistent=False
        )
        self.register_buffer(
            "kernel_y", kernel_y.expand(channels, 1, 3, 3).contiguous(), persistent=False
        )

    def gradients(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the horizontal and vertical Sobel responses.

        Replicate padding keeps the border response finite and unbiased; zero padding
        would manufacture a strong artificial edge around the whole image.

        Args:
            x: Tensor ``(B, C, H, W)``.

        Returns:
            ``(grad_x, grad_y)``, each of the same shape as ``x``.

        Raises:
            ValueError: If the channel count does not match.
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                "GradientLoss configured for " + str(self.channels)
                + " channels, got " + str(x.shape[1]) + "."
            )
        padded = F.pad(x, [1, 1, 1, 1], mode="replicate")
        grad_x = F.conv2d(padded, self.kernel_x.to(x.dtype), groups=self.channels)
        grad_y = F.conv2d(padded, self.kernel_y.to(x.dtype), groups=self.channels)
        return grad_x, grad_y

    def magnitude(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Gradient magnitude ``sqrt(gx^2 + gy^2)``. Diagnostics only.

        Args:
            x: Tensor ``(B, C, H, W)``.
            eps: Added under the square root to keep the gradient finite at zero.

        Returns:
            Magnitude of the same shape as ``x``.
        """
        grad_x, grad_y = self.gradients(x)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean L1 gradient discrepancy.

        Args:
            pred: Predicted images ``(B, C, H, W)``.
            target: Ground truth of the same shape.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If the shapes differ.
        """
        # String concatenation, not f-strings: this method is TorchScript-scriptable
        # (Part 9) and TorchScript cannot size `tuple(tensor.shape)`.
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: " + str(list(pred.shape))
                + " vs " + str(list(target.shape)) + "."
            )
        pred_x, pred_y = self.gradients(pred)
        target_x, target_y = self.gradients(target)
        return 0.5 * (
            F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
        )

    def extra_repr(self) -> str:
        return f"channels={self.channels}, normalize={self.normalize}"
