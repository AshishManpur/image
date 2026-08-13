"""Charbonnier loss (Contract Part 6).

The primary fidelity term. Charbonnier is a smooth approximation to L1,

.. math::  \\mathcal{L} = \\operatorname{mean}\\sqrt{(\\hat{x} - x)^2 + \\epsilon}

chosen over L2 because Phase 1 measured the speckle residual as heavy-tailed (excess
kurtosis +13.9 at low intensity). L2 would give those tails outsized gradients; the
square root bounds their influence while staying differentiable at zero, which plain
L1 is not.
"""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Smooth L1-like reconstruction loss.

    Args:
        eps: Squared smoothing constant added under the square root. The contract
            fixes this at 1e-6.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``eps`` is negative or ``reduction`` is unknown.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction '{reduction}'.")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred: Predicted tensor.
            target: Ground-truth tensor of the same shape.

        Returns:
            Scalar loss, or a per-element tensor when ``reduction="none"``.

        Raises:
            ValueError: If the shapes differ.
        """
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: " + str(list(pred.shape))
                + " vs " + str(list(target.shape)) + "."
            )
        loss = torch.sqrt((pred - target).pow(2) + self.eps)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction={self.reduction}"
