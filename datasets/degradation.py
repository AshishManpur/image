"""Phase 1 forward degradation operator and LR re-synthesis (Contract Part 6).

The operator recovered in Phase 1 by joint RMSE and structure-leakage minimisation is

.. math::

    y = \\mathcal{D}^{\\downarrow 2,\\,\\text{noAA}}_{\\text{bicubic}}
        \\Big( (x * g_{\\sigma}) \\cdot \\frac{\\Gamma(L)}{L} + \\mathcal{N}(0, \\sigma_g^2) \\Big)

with :math:`\\sigma \\approx 0.4` px, multiplicative gamma speckle of
:math:`L \\approx 35` looks (:math:`\\sigma_s = L^{-1/2} \\approx 0.165`) and an additive
floor :math:`\\sigma_g \\approx 0.024`. Noise is injected **before** decimation, which
reproduces the measured lag-1 residual autocorrelation of -0.045. **No clipping is
applied at any stage** — Phase 1 measured 3.36 % of real LR pixels above 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from configs.sparc_config import DataConfig


@dataclass(frozen=True)
class DegradationParams:
    """Per-image degradation parameters."""

    blur_sigma: float
    looks: float
    gauss_sigma: float

    @property
    def speckle_sigma(self) -> float:
        """Multiplicative noise standard deviation, ``L**-0.5``."""
        return float(self.looks) ** -0.5


def gaussian_kernel1d(sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel.

    Args:
        sigma: Standard deviation in pixels; must be positive.
        kernel_size: Odd kernel length.

    Returns:
        Tensor of shape ``(kernel_size,)`` summing to 1.

    Raises:
        ValueError: If ``sigma`` is non-positive or ``kernel_size`` is even.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding.

    Args:
        x: Tensor of shape ``(B, C, H, W)``.
        sigma: Standard deviation in pixels.
        kernel_size: Odd kernel length.

    Returns:
        Blurred tensor of the same shape.
    """
    channels = x.shape[1]
    kernel = gaussian_kernel1d(sigma, kernel_size).to(x.device, x.dtype)
    pad = kernel_size // 2
    horizontal = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, horizontal, groups=channels)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x, vertical, groups=channels)


def bicubic_downsample2(x: torch.Tensor) -> torch.Tensor:
    """Bicubic 2x decimation **without** antialiasing, as recovered in Phase 1."""
    return F.interpolate(
        x, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=False
    )


def bicubic_upsample2(x: torch.Tensor) -> torch.Tensor:
    """Bicubic 2x upsampling used for the global residual and the baseline."""
    return F.interpolate(x, scale_factor=2.0, mode="bicubic", align_corners=False)


def forward_operator(x: torch.Tensor, blur_sigma: float = 0.4) -> torch.Tensor:
    """Noise-free forward operator ``A(x) = bicubic_down2(g_sigma * x)``.

    Args:
        x: High-resolution tensor ``(B, C, H, W)``.
        blur_sigma: Pre-blur standard deviation in HR pixels.

    Returns:
        Clean low-resolution tensor ``(B, C, H/2, W/2)``.
    """
    return bicubic_downsample2(gaussian_blur(x, blur_sigma))


def sample_degradation_params(
    config: DataConfig, generator: torch.Generator | None = None
) -> DegradationParams:
    """Draw per-image degradation parameters from the frozen ranges."""

    def _uniform(low: float, high: float) -> float:
        if high <= low:
            return float(low)
        u = torch.rand(1, generator=generator).item()
        return float(low + u * (high - low))

    return DegradationParams(
        blur_sigma=_uniform(*config.blur_sigma_range),
        looks=_uniform(*config.speckle_looks_range),
        gauss_sigma=_uniform(*config.gauss_sigma_range),
    )


def _apply_noise(
    x: torch.Tensor, params: DegradationParams, generator: torch.Generator | None
) -> torch.Tensor:
    """Apply multiplicative gamma speckle followed by additive Gaussian noise."""
    looks = torch.tensor(float(params.looks), dtype=x.dtype)
    gamma = torch._standard_gamma(looks.expand(x.shape), generator=generator) / looks
    x = x * gamma
    if params.gauss_sigma > 0.0:
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        x = x + noise * params.gauss_sigma
    return x


def synthesize_lr(
    gt: torch.Tensor,
    params: DegradationParams,
    generator: torch.Generator | None = None,
    noise_at_lr: bool = True,
) -> torch.Tensor:
    """Re-synthesise a degraded LR image from ground truth.

    Args:
        gt: Ground-truth tensor ``(B, C, H, W)`` in ``[0, 1]``.
        params: Degradation parameters for this sample.
        generator: Optional RNG for reproducibility.
        noise_at_lr: Amendment A-001. When ``True`` (default) noise is injected after
            decimation, which reproduces the measured LR-resolution noise level. When
            ``False`` the frozen-contract ordering is used, which delivers only 50 % of
            the measured speckle variance.

    Returns:
        Degraded tensor ``(B, C, H/2, W/2)``. **Not clipped** — Phase 1 measured 3.36 %
        of real LR pixels above 1.0 and 0.30 % below 0.

    Raises:
        ValueError: If ``gt`` is not 4-D.
    """
    if gt.dim() != 4:
        raise ValueError(f"synthesize_lr expects a 4-D tensor, got {tuple(gt.shape)}.")

    x = gaussian_blur(gt, params.blur_sigma)
    if noise_at_lr:
        return _apply_noise(bicubic_downsample2(x), params, generator)
    return bicubic_downsample2(_apply_noise(x, params, generator))


def fit_noise_parameters(
    noisy_lr: torch.Tensor, clean_lr: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-image least-squares fit of ``Var(r | I) = a + c I^2``.

    This produces the auxiliary supervision target for the noise head. Phase 1
    established that the three-parameter form ``a + bI + cI^2`` is ill-conditioned
    (``corr(b, c) = -0.90``), so the identifiable two-parameter model is used.

    Args:
        noisy_lr: Observed LR tensor ``(B, C, H, W)``.
        clean_lr: Noise-free LR tensor ``A(gt)`` of the same shape.
        eps: Numerical floor on the normal-equation determinant.

    Returns:
        ``(sigma_gauss, sigma_speckle)``, each of shape ``(B,)`` and non-negative.

    Raises:
        ValueError: If the shapes differ.
    """
    if noisy_lr.shape != clean_lr.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(noisy_lr.shape)} vs {tuple(clean_lr.shape)}."
        )
    batch = noisy_lr.shape[0]
    residual_sq = (noisy_lr - clean_lr).reshape(batch, -1).double().pow(2)
    intensity_sq = clean_lr.reshape(batch, -1).double().pow(2)

    n = residual_sq.shape[1]
    s_i2 = intensity_sq.sum(dim=1)
    s_i4 = intensity_sq.pow(2).sum(dim=1)
    s_r2 = residual_sq.sum(dim=1)
    s_i2r2 = (intensity_sq * residual_sq).sum(dim=1)

    det = n * s_i4 - s_i2 * s_i2
    det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
    a = (s_r2 * s_i4 - s_i2 * s_i2r2) / det
    c = (n * s_i2r2 - s_i2 * s_r2) / det

    sigma_gauss = a.clamp_min(0.0).sqrt().to(noisy_lr.dtype)
    sigma_speckle = c.clamp_min(0.0).sqrt().to(noisy_lr.dtype)
    return sigma_gauss, sigma_speckle


def analytic_sigma_map(
    clean_lr: torch.Tensor, sigma_gauss: torch.Tensor, sigma_speckle: torch.Tensor
) -> torch.Tensor:
    """Per-pixel noise standard deviation ``sqrt(sigma_g^2 + sigma_s^2 I^2)``.

    Args:
        clean_lr: Noise-free LR tensor ``(B, C, H, W)``.
        sigma_gauss: Per-image additive sigma, shape ``(B,)``.
        sigma_speckle: Per-image multiplicative sigma, shape ``(B,)``.

    Returns:
        Sigma map of shape ``(B, C, H, W)``.
    """
    g = sigma_gauss.reshape(-1, 1, 1, 1)
    s = sigma_speckle.reshape(-1, 1, 1, 1)
    return torch.sqrt(g.pow(2) + s.pow(2) * clean_lr.pow(2) + 1e-12)
