"""Sigma-map assembly and the analytic auxiliary target (Contract Part 2.9).

The noise head predicts two scalars per image. The trunk, however, is conditioned on a
per-pixel map, because Phase 1 measured the speckle term as intensity-dependent: noise
scales with local brightness, so a single scalar would understate it in bright regions
and overstate it in dark ones. This module expands the pair into that map,

.. math::  \\hat{\\sigma}(x) = \\sqrt{\\sigma_g^2 + \\sigma_s^2 \\hat{I}(x)^2}

where :math:`\\hat{I}` is a locally smoothed estimate of the clean intensity. Smoothing
matters: using the raw observation would feed the noise realisation itself into the
intensity term, which biases the map upward exactly where it is already largest.

The **auxiliary training target** is the analytic sigma map computed from the ground
truth. Both halves of it — the closed-form per-image least-squares fit
``Var(r | I) = a + c I^2`` and the map assembly — already exist and are tested in
:mod:`datasets.degradation`; they are re-exported here so that Part 14's
``models/noise/noise_map.py`` is the single import site for noise-target code, without
duplicating the implementation.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from datasets.degradation import analytic_sigma_map, fit_noise_parameters

__all__ = [
    "analytic_sigma_map",
    "assemble_sigma_map",
    "build_smoothing_kernel",
    "fit_noise_parameters",
]

SIGMA_FLOOR_EPS: Final[float] = 1e-12
"""Added under the square root so the gradient stays finite when both sigmas vanish.

Consumed as the default value of ``assemble_sigma_map(..., floor=...)`` rather than
read inside the function body: TorchScript cannot close over a module-level float from
a free function (``Final`` only covers ``nn.Module`` class attributes), but it does bake
in default arguments at definition time.
"""


def build_smoothing_kernel(kernel_size: int) -> torch.Tensor:
    """Build the normalised box kernel used to estimate local intensity.

    A box filter is used rather than a Gaussian because it is exactly separable, has no
    free parameters to tune, and exports as a single convolution. The kernel is a
    non-trainable buffer, so it contributes **zero** parameters to the Part 3 budget.

    Args:
        kernel_size: Odd window size; the contract fixes this at 5.

    Returns:
        Tensor of shape ``(1, 1, kernel_size, kernel_size)`` summing to 1.

    Raises:
        ValueError: If ``kernel_size`` is not a positive odd integer.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    weight = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
    return weight / float(kernel_size * kernel_size)


def estimate_local_intensity(y: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Smooth an observation to approximate the clean local intensity.

    Padding is **replicate**, not zero. Zero padding would pull the estimate toward
    zero within half a window of every border, which in turn shrinks the speckle term
    exactly where there is no evidence that the image is dark — a systematic
    under-estimate of sigma in a 2-pixel frame around every image.

    Args:
        y: Observed image ``(B, 1, H, W)``.
        kernel: Kernel from :func:`build_smoothing_kernel`.

    Returns:
        Non-negative smoothed intensity of the same shape.
    """
    padding = kernel.shape[-1] // 2
    padded = F.pad(y, [padding, padding, padding, padding], mode="replicate")
    smoothed = F.conv2d(padded, kernel.to(y.dtype))
    # The observation is unclipped and may be negative; intensity cannot be.
    return smoothed.clamp_min(0.0)


def assemble_sigma_map(
    y: torch.Tensor,
    sigma_gauss: torch.Tensor,
    sigma_speckle: torch.Tensor,
    kernel: torch.Tensor,
    sigma_min: float = 1e-4,
    sigma_max: float = 2.0,
    floor: float = SIGMA_FLOOR_EPS,
) -> torch.Tensor:
    """Expand two per-image sigmas into a per-pixel sigma map.

    The arithmetic runs in float32 regardless of the surrounding autocast dtype: the
    squared terms are around ``5.8e-4`` for the contract's initial sigmas, which is
    close enough to the fp16 subnormal range that accumulating them in half precision
    loses meaningful accuracy.

    Args:
        y: Observed image ``(B, 1, H, W)``, unclipped.
        sigma_gauss: Additive sigma ``(B, 1)`` or ``(B,)``.
        sigma_speckle: Multiplicative sigma ``(B, 1)`` or ``(B,)``.
        kernel: Smoothing kernel from :func:`build_smoothing_kernel`.
        sigma_min: Lower clamp on the output.
        sigma_max: Upper clamp on the output.
        floor: Value added under the square root; see :data:`SIGMA_FLOOR_EPS`.

    Returns:
        Sigma map ``(B, 1, H, W)`` in image units, clamped to ``[sigma_min, sigma_max]``.

    Raises:
        ValueError: If ``y`` is not 4-D, or the batch dimensions disagree.
    """
    # String concatenation rather than f-strings: this function is reachable from
    # NoiseHead.forward and must stay TorchScript-scriptable, matching the convention
    # already used in models/blocks/naf_block.py and layer_norm.py.
    if y.dim() != 4:
        raise ValueError("Expected a 4-D observation, got " + str(y.dim()) + " dims.")
    batch = y.shape[0]
    if sigma_gauss.shape[0] != batch or sigma_speckle.shape[0] != batch:
        raise ValueError(
            "Batch mismatch: y has " + str(batch) + ", sigmas have "
            + str(sigma_gauss.shape[0]) + " and " + str(sigma_speckle.shape[0]) + "."
        )

    # The trailing `.float()` is not redundant with the one on the argument: autocast
    # re-casts convolution arguments to the reduced dtype whatever the caller passes,
    # so `estimate_local_intensity` returns fp16 under fp16 autocast however its inputs
    # were spelled. Re-casting the *result* is what actually puts the squares below in
    # float32 — and `intensity.pow(2)` lands around 1e-6, which is subnormal in fp16.
    # A plain cast is used rather than `autocast(enabled=False)` because this function
    # must stay TorchScript-scriptable (Part 9); the box filter itself is a convolution
    # of a [0, 1] image and is harmless in half precision.
    intensity = estimate_local_intensity(y.float(), kernel.float()).float()
    g = sigma_gauss.float().reshape(batch, 1, 1, 1)
    s = sigma_speckle.float().reshape(batch, 1, 1, 1)
    variance = g.pow(2) + s.pow(2) * intensity.pow(2)
    sigma = torch.sqrt(variance + floor)
    return sigma.clamp(sigma_min, sigma_max)
