"""Evaluation metrics (Contract Part 6).

Metric definitions are pinned so that numbers are comparable across runs and against
the Phase 1 baselines:

* **PSNR** with ``data_range = 1.0``, computed against the unmodified ground truth.
* **SSIM** with an 11x11 Gaussian window, sigma 1.5, ``K = (0.01, 0.03)``, evaluated on
  the valid (unpadded) region — the standard formulation.
* **LPIPS** with the AlexNet backbone, grayscale replicated to three channels.
  Evaluation only; never used as a training loss in V1.

Two PSNR reductions are provided and they are **not** interchangeable:
``psnr_pooled`` averages the squared error over every pixel of every image before
taking the logarithm (this is what the Phase 1 baselines used), whereas ``psnr_per_image``
computes PSNR per image and then averages. Phase 1 also established that the 34
near-featureless images inflate the per-image mean, so median is reported alongside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size)
    return window.contiguous()


def psnr_pooled(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR over the pooled squared error of the whole batch.

    This is the reduction used for the Phase 1 baselines (bicubic 21.67 dB).

    Args:
        pred: Predicted tensor ``(B, C, H, W)``.
        target: Ground-truth tensor of the same shape.
        data_range: Dynamic range of the data.

    Returns:
        PSNR in dB.

    Raises:
        ValueError: If the shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    mse = torch.mean((pred.double() - target.double()) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item())


def psnr_per_image(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    """Per-image PSNR.

    Args:
        pred: Predicted tensor ``(B, C, H, W)``.
        target: Ground-truth tensor of the same shape.
        data_range: Dynamic range of the data.

    Returns:
        Tensor of shape ``(B,)`` with PSNR in dB.

    Raises:
        ValueError: If the shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    batch = pred.shape[0]
    mse = ((pred.double() - target.double()) ** 2).reshape(batch, -1).mean(dim=1)
    return 10.0 * torch.log10(data_range**2 / mse.clamp_min(1e-20))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = SSIM_WINDOW,
    sigma: float = SSIM_SIGMA,
) -> torch.Tensor:
    """Per-image SSIM on the valid (unpadded) region.

    Args:
        pred: Predicted tensor ``(B, C, H, W)``.
        target: Ground-truth tensor of the same shape.
        data_range: Dynamic range of the data.
        window_size: Gaussian window size.
        sigma: Gaussian window standard deviation.

    Returns:
        Tensor of shape ``(B,)``.

    Raises:
        ValueError: If shapes differ or the image is smaller than the window.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    if min(pred.shape[-2:]) < window_size:
        raise ValueError(
            f"Image {tuple(pred.shape[-2:])} is smaller than the SSIM window {window_size}."
        )

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    mu_p = F.conv2d(pred, window, groups=channels)
    mu_t = F.conv2d(target, window, groups=channels)
    mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p = F.conv2d(pred * pred, window, groups=channels) - mu_p_sq
    sigma_t = F.conv2d(target * target, window, groups=channels) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, window, groups=channels) - mu_pt

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    return (numerator / denominator).flatten(1).mean(dim=1)


class LpipsMetric:
    """Lazy LPIPS wrapper (AlexNet backbone, grayscale replicated to 3 channels).

    The ``lpips`` package is an optional dependency; construction is deferred until
    first use so that the rest of the pipeline runs without it.

    Args:
        net: Backbone name passed to ``lpips.LPIPS``.
        device: Device to run the metric on.
    """

    def __init__(self, net: str = "alex", device: torch.device | str = "cpu") -> None:
        self.net = net
        self.device = torch.device(device)
        self._model: Any = None

    @property
    def available(self) -> bool:
        """Whether the ``lpips`` package can be imported."""
        try:
            import lpips  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                import lpips
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "LPIPS requires the `lpips` package: pip install lpips"
                ) from exc
            self._model = lpips.LPIPS(net=self.net, verbose=False).to(self.device).eval()
        return self._model

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-image LPIPS.

        Args:
            pred: Predicted tensor ``(B, 1, H, W)`` in ``[0, 1]``.
            target: Ground-truth tensor of the same shape.

        Returns:
            Tensor of shape ``(B,)``; lower is better.
        """
        model = self._ensure_model()
        pred3 = pred.repeat(1, 3, 1, 1).to(self.device) * 2.0 - 1.0
        target3 = target.repeat(1, 3, 1, 1).to(self.device) * 2.0 - 1.0
        return model(pred3, target3).flatten()


@dataclass
class MetricAccumulator:
    """Accumulates per-image metrics and reports mean and median.

    Phase 1 measured 34 near-featureless training images whose PSNR is dominated by
    the noise floor; the median is therefore reported alongside the mean.
    """

    psnr: list[float] = None  # type: ignore[assignment]
    ssim: list[float] = None  # type: ignore[assignment]
    lpips: list[float] = None  # type: ignore[assignment]
    _sse: float = 0.0
    _count: int = 0

    def __post_init__(self) -> None:
        self.psnr = []
        self.ssim = []
        self.lpips = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lpips_values: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one batch.

        Args:
            pred: Predicted tensor ``(B, C, H, W)``.
            target: Ground-truth tensor of the same shape.
            lpips_values: Optional per-image LPIPS of shape ``(B,)``.
        """
        self.psnr.extend(psnr_per_image(pred, target).tolist())
        self.ssim.extend(ssim(pred, target).tolist())
        if lpips_values is not None:
            self.lpips.extend(lpips_values.tolist())
        diff = (pred.double() - target.double()) ** 2
        self._sse += diff.sum().item()
        self._count += diff.numel()

    def summary(self) -> dict[str, float]:
        """Return mean/median PSNR and SSIM, pooled PSNR, and mean LPIPS."""

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else float("nan")

        def _median(values: list[float]) -> float:
            if not values:
                return float("nan")
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return float(ordered[mid])
            return float(0.5 * (ordered[mid - 1] + ordered[mid]))

        pooled = (
            float("inf")
            if self._count == 0 or self._sse == 0.0
            else float(10.0 * torch.log10(torch.tensor(self._count / self._sse)).item())
        )
        result = {
            "psnr_mean": _mean(self.psnr),
            "psnr_median": _median(self.psnr),
            "psnr_pooled": pooled,
            "ssim_mean": _mean(self.ssim),
            "ssim_median": _median(self.ssim),
            "count": float(len(self.psnr)),
        }
        if self.lpips:
            result["lpips_mean"] = _mean(self.lpips)
            result["lpips_median"] = _median(self.lpips)
        return result


# --------------------------------------------------------------- band correlation
#: Radial frequency band edges as a fraction of Nyquist, matching the diagnostic that
#: motivated the Phase 6 clean-LR branch. ``corr(R, R*)`` measured on the sparc-base
#: checkpoint was LOW 0.599, MID 0.831, HIGH 0.815, NYQ 0.302 — the network being worst
#: in the band with the most signal energy is the failure this metric tracks.
BAND_EDGES: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.125),
    "mid": (0.125, 0.35),
    "high": (0.35, 0.7),
    "nyq": (0.7, 1.01),
}


class BandCorrelationAccumulator:
    """Streaming ``corr(R, R*)`` per radial frequency band.

    ``R = prediction - base`` is the residual the network actually produced and
    ``R* = target - base`` the ideal one, where ``base`` is the bicubic upsample of the
    observed LR. Correlation is accumulated from raw cross- and auto-products so the
    result over many batches is exact, not an average of per-batch correlations.
    """

    def __init__(self) -> None:
        self._rr: dict[str, float] = {k: 0.0 for k in BAND_EDGES}
        self._ss: dict[str, float] = {k: 0.0 for k in BAND_EDGES}
        self._rs: dict[str, float] = {k: 0.0 for k in BAND_EDGES}
        self._masks: dict[str, torch.Tensor] | None = None
        self._size: int | None = None

    def _build_masks(self, size: int, device: torch.device) -> dict[str, torch.Tensor]:
        coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
        radius = torch.sqrt(coords[:, None] ** 2 + coords[None, :] ** 2) / (size / 2)
        return {
            name: ((radius >= lo) & (radius < hi))
            for name, (lo, hi) in BAND_EDGES.items()
        }

    @torch.no_grad()
    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, base: torch.Tensor
    ) -> None:
        """Accumulate one batch.

        Args:
            prediction: Restored image ``(B, C, H, W)``.
            target: Ground truth of the same shape.
            base: Bicubic upsample of the observed LR, same shape.

        Raises:
            ValueError: If the shapes differ or the images are not square.
        """
        if not prediction.shape == target.shape == base.shape:
            raise ValueError(
                f"Shape mismatch: {tuple(prediction.shape)} / {tuple(target.shape)} / "
                f"{tuple(base.shape)}."
            )
        size = prediction.shape[-1]
        if prediction.shape[-2] != size:
            raise ValueError(f"Band correlation expects square images, got {size}.")
        if self._masks is None or self._size != size:
            self._masks = self._build_masks(size, prediction.device)
            self._size = size

        residual = torch.fft.fftshift(
            torch.fft.fft2((prediction - base).float()), dim=(-2, -1)
        )
        ideal = torch.fft.fftshift(
            torch.fft.fft2((target - base).float()), dim=(-2, -1)
        )
        for name, mask in self._masks.items():
            r = residual[..., mask]
            s = ideal[..., mask]
            # Real-valued inner products: the spectra are conjugate-symmetric, so the
            # imaginary parts cancel and Re(r * conj(s)) is the spatial-domain covariance
            # of the band-limited residuals by Parseval.
            self._rs[name] += float((r * s.conj()).real.sum())
            self._rr[name] += float((r * r.conj()).real.sum())
            self._ss[name] += float((s * s.conj()).real.sum())

    def summary(self) -> dict[str, float]:
        """Return ``rho_<band>`` for every band."""
        out: dict[str, float] = {}
        for name in BAND_EDGES:
            denominator = math.sqrt(self._rr[name] * self._ss[name])
            out[f"rho_{name}"] = (
                float(self._rs[name] / denominator) if denominator > 0.0 else float("nan")
            )
        return out
