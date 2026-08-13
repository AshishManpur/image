"""Amplitude-spectrum FFT loss (Contract Part 6, weight 0.05).

Phase 1 measured the degradation as predominantly a loss of high-frequency energy.
Spatial-domain losses see that only indirectly: a uniformly blurred prediction and a
sharp one with small local errors can carry similar L1. Comparing amplitude spectra
makes the deficit explicit and global.

Contract Part 6::

    L_fft = mean( abs( |rfft2(x_hat)| - |rfft2(x)| ) )

**Amplitude only — phase is deliberately ignored.** Constraining phase would duplicate
what the spatial terms already enforce, and phase error is badly behaved near zero
amplitude where the angle is undefined.

``rfft2`` is used rather than ``fft2`` because the inputs are real: the spectrum is
conjugate-symmetric, so the half-spectrum carries all the information at half the cost.
"""

from __future__ import annotations

import torch
from torch import nn

_MAGNITUDE_EPS = 1e-12
"""Added inside the magnitude's square root so its gradient is finite at zero.

``torch.abs`` on a complex tensor is ``sqrt(re^2 + im^2)``, whose derivative is
undefined at the origin — and the origin is exactly where a well-fitted high-frequency
bin sits. Computing the magnitude explicitly with a floor keeps the backward pass
finite there.
"""


class FFTLoss(nn.Module):
    """L1 between the amplitude spectra of two images.

    Args:
        norm: Normalisation mode passed to ``torch.fft.rfft2``. ``"ortho"`` makes the
            loss scale-invariant with respect to image size, so the fixed contract
            weight stays meaningful if the patch size ever changes.
        eps: Floor inside the magnitude square root.

    Raises:
        ValueError: If ``norm`` is not a recognised mode.
    """

    def __init__(self, norm: str = "ortho", eps: float = _MAGNITUDE_EPS) -> None:
        super().__init__()
        if norm not in ("backward", "forward", "ortho"):
            raise ValueError(f"Unknown fft norm '{norm}'.")
        self.norm = norm
        self.eps = eps

    def amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return the half-spectrum amplitude of a real image.

        Args:
            x: Real tensor ``(B, C, H, W)``.

        Returns:
            Amplitude of shape ``(B, C, H, W // 2 + 1)``.
        """
        spectrum = torch.fft.rfft2(x, norm=self.norm)
        return torch.sqrt(spectrum.real.pow(2) + spectrum.imag.pow(2) + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean absolute amplitude-spectrum difference.

        Args:
            pred: Predicted images ``(B, C, H, W)``.
            target: Ground truth of the same shape.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If the shapes differ.
        """
        # String concatenation, not f-strings: TorchScript (Part 9) cannot size
        # `tuple(tensor.shape)`.
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: " + str(list(pred.shape))
                + " vs " + str(list(target.shape)) + "."
            )
        # float32 regardless of autocast: torch.fft has no half-precision CUDA kernel
        # for many sizes, and where it does the accumulated round-off across a 256-point
        # transform is large enough to bias the loss.
        return (self.amplitude(pred.float()) - self.amplitude(target.float())).abs().mean()

    def extra_repr(self) -> str:
        return f"norm={self.norm}"
