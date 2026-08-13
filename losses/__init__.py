"""Loss functions for SPARC-Base V1.0 (Contract Part 6).

Each term lives in its own module and is independently constructible and testable;
:class:`~losses.composite_loss.CompositeLoss` only weights and sums them.
"""

from losses.charbonnier import CharbonnierLoss
from losses.composite_loss import TERM_NAMES, CompositeLoss
from losses.fft_loss import FFTLoss
from losses.gradient import GradientLoss
from losses.ms_ssim import MSSSIMLoss
from losses.noise_loss import NoiseAuxLoss
from losses.wavelet_loss import WaveletLoss

__all__ = [
    "TERM_NAMES",
    "CharbonnierLoss",
    "CompositeLoss",
    "FFTLoss",
    "GradientLoss",
    "MSSSIMLoss",
    "NoiseAuxLoss",
    "WaveletLoss",
]
