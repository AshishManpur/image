"""Multi-scale SSIM loss (Contract Part 6, weight 0.15).

Charbonnier alone optimises a per-pixel criterion and is indifferent to whether the
error is structured. MS-SSIM adds a perceptual-structural term evaluated over a
Gaussian window at five scales, which is what keeps texture from being smoothed away.

Definition, following Wang et al. (2003)::

    MS-SSIM = l_M^alpha_M * prod_{j=1..M} cs_j^beta_j
    loss    = 1 - MS-SSIM

with ``M = 5`` scales, an 11x11 Gaussian window of sigma 1.5, ``data_range = 1.0``, and
the canonical scale weights. Only the finest-to-coarsest contrast-structure terms are
accumulated; the luminance term is taken at the coarsest scale, as in the original
formulation.

.. note::
   This is deliberately **not** shared with :func:`evaluation.metrics.ssim`. That
   function is the *scored* metric and must stay pinned to its own definition; coupling
   the trained objective to the reported metric would make a change to one silently
   change the other. The Gaussian window construction is the only thing in common and
   it is four lines.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MS_SSIM_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
"""Canonical five-scale weights from Wang et al. (2003); they sum to 1."""


def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a separable Gaussian window as a 4-D depthwise kernel.

    Args:
        window_size: Side length of the (square) window.
        sigma: Standard deviation in pixels.
        channels: Number of channels; the kernel is replicated per channel.

    Returns:
        Tensor of shape ``(channels, 1, window_size, window_size)`` summing to 1
        per channel.

    Raises:
        ValueError: If ``window_size`` is not a positive odd integer or ``sigma`` is
            not positive.
    """
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}."
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size)
    return window.contiguous()


class MSSSIMLoss(nn.Module):
    """``1 - MS-SSIM`` over a grayscale pair.

    Args:
        scales: Number of pyramid levels. The contract fixes this at 5.
        window_size: Gaussian window side length; contract value 11.
        sigma: Gaussian window standard deviation; contract value 1.5.
        data_range: Dynamic range of the inputs; contract value 1.0.
        channels: Channel count of the inputs.
        k1: SSIM stabiliser for the luminance term.
        k2: SSIM stabiliser for the contrast term.

    Raises:
        ValueError: If ``scales`` exceeds the number of available weights, or any
            hyperparameter is out of range.
    """

    def __init__(
        self,
        scales: int = 5,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channels: int = 1,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        if not 1 <= scales <= len(MS_SSIM_WEIGHTS):
            raise ValueError(
                f"scales must be in [1, {len(MS_SSIM_WEIGHTS)}], got {scales}."
            )
        if data_range <= 0.0:
            raise ValueError(f"data_range must be positive, got {data_range}.")

        self.scales = scales
        self.window_size = window_size
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

        # Buffers, not parameters: the window must move with `.to(device)` and must
        # never appear in the model's parameter count or the optimiser's groups.
        self.register_buffer(
            "window", gaussian_window(window_size, sigma, channels), persistent=False
        )
        weights = torch.tensor(MS_SSIM_WEIGHTS[:scales], dtype=torch.float32)
        self.register_buffer("weights", weights / weights.sum(), persistent=False)

    @property
    def minimum_size(self) -> int:
        """Smallest input side this configuration can process.

        Each of the ``scales - 1`` downsamplings halves the image, and the coarsest
        level must still be at least as large as the window.
        """
        # int() is load-bearing: TorchScript types `2 ** n` as float, and this property
        # is reached from the scriptable forward.
        return (self.window_size - 1) * int(2 ** (self.scales - 1)) + 1

    def _ssim_and_cs(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-image mean SSIM and mean contrast-structure at one scale."""
        channels = pred.shape[1]
        window = self.window.to(dtype=pred.dtype)

        mu_p = F.conv2d(pred, window, groups=channels)
        mu_t = F.conv2d(target, window, groups=channels)
        mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p = F.conv2d(pred * pred, window, groups=channels) - mu_p_sq
        sigma_t = F.conv2d(target * target, window, groups=channels) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, groups=channels) - mu_pt

        # The contrast-structure term is raised to a fractional power below, so it must
        # be strictly positive; variances can go slightly negative from cancellation.
        cs = (2 * sigma_pt + self.c2) / (sigma_p + sigma_t + self.c2)
        luminance = (2 * mu_pt + self.c1) / (mu_p_sq + mu_t_sq + self.c1)
        ssim = luminance * cs
        return ssim.flatten(1).mean(dim=1), cs.flatten(1).mean(dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ``1 - MS-SSIM``.

        Args:
            pred: Predicted images ``(B, C, H, W)``.
            target: Ground truth of the same shape.

        Returns:
            Scalar loss in ``[0, 1]`` for well-formed inputs.

        Raises:
            ValueError: If the shapes differ, or the images are too small for the
                configured number of scales.
        """
        # String concatenation, not f-strings: TorchScript (Part 9) cannot size
        # `tuple(tensor.shape)`.
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: " + str(list(pred.shape))
                + " vs " + str(list(target.shape)) + "."
            )
        smallest = min(pred.shape[-2], pred.shape[-1])
        if smallest < self.minimum_size:
            raise ValueError(
                "MS-SSIM with " + str(self.scales) + " scales and window "
                + str(self.window_size) + " needs images of at least "
                + str(self.minimum_size) + " px, got " + str(smallest) + "."
            )

        # float32 throughout: the products of five fractional powers underflow badly in
        # fp16, and torch.fft-style precision loss here shows up directly as a biased
        # loss rather than as noise.
        pred = pred.float()
        target = target.float()

        cs_values: list[torch.Tensor] = []
        ssim = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        for level in range(self.scales):
            ssim, cs = self._ssim_and_cs(pred, target)
            cs_values.append(cs.clamp_min(1e-6))
            if level < self.scales - 1:
                pred = F.avg_pool2d(pred, kernel_size=2)
                target = F.avg_pool2d(target, kernel_size=2)

        stacked = torch.stack(cs_values[:-1], dim=0)  # (scales-1, B)
        weights = self.weights.to(stacked.dtype)
        product = torch.prod(stacked ** weights[:-1, None], dim=0)
        ms_ssim = product * ssim.clamp_min(1e-6) ** weights[-1]
        return (1.0 - ms_ssim).mean()

    def extra_repr(self) -> str:
        return (
            f"scales={self.scales}, window_size={self.window_size}, "
            f"data_range={self.data_range}"
        )
