"""Orthonormal Haar wavelet transform (Contract Part 2.4).

The DWT is the resampling operator throughout SPARC-Net because it is **lossless**:
strided convolution and pooling both discard information before the network has
denoised it, and at the measured 7.1 dB median input SNR that loss is unrecoverable.

For the 2x2 pixel block ``[[a, b], [c, d]]`` the transform is

.. math::

    \\begin{pmatrix} LL \\\\ LH \\\\ HL \\\\ HH \\end{pmatrix}
    = \\frac{1}{2}
    \\begin{pmatrix}
      1 &  1 &  1 &  1 \\\\
      1 &  1 & -1 & -1 \\\\
      1 & -1 &  1 & -1 \\\\
      1 & -1 & -1 &  1
    \\end{pmatrix}
    \\begin{pmatrix} a \\\\ b \\\\ c \\\\ d \\end{pmatrix}

The matrix is :math:`\\tfrac{1}{2}H_4` with :math:`H_4` the order-4 Hadamard matrix.
Since :math:`H_4H_4^{\\top} = 4I` it is orthonormal, and since it is also symmetric the
inverse transform is the same operator applied again.

Packed channel order is ``[LL(C), LH(C), HL(C), HH(C)]``, so each sub-band is a
contiguous channel range. Only ``reshape``/``slice``/``add``/``mul`` are used, all of
which export cleanly to ONNX and TorchScript.

The inverse doubles resolution, which makes it the reconstruction head's upsampler:
``HaarIDWT`` is exactly ``PixelShuffle(2)`` composed with a fixed orthogonal 4x4 mix.
Checkerboard artefacts are impossible because the basis has no preferred sub-pixel
position, so no ICNR initialisation is required.
"""

from __future__ import annotations

import torch
from torch import nn



def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Single-level orthonormal Haar decomposition.

    Args:
        x: Tensor of shape ``(B, C, H, W)`` with ``H`` and ``W`` even.

    Returns:
        Tensor of shape ``(B, 4C, H // 2, W // 2)`` ordered ``[LL, LH, HL, HH]``.

    Raises:
        ValueError: If ``x`` is not 4-D or has odd spatial dimensions.
    """
    if x.dim() != 4:
        raise ValueError("haar_dwt expects a 4-D tensor, got " + str(x.dim()) + " dims.")
    batch = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "haar_dwt requires even spatial dimensions, got "
            + str(height) + "x" + str(width) + "."
        )

    blocks = x.reshape(batch, channels, height // 2, 2, width // 2, 2)
    top_left = blocks[:, :, :, 0, :, 0]
    top_right = blocks[:, :, :, 0, :, 1]
    bottom_left = blocks[:, :, :, 1, :, 0]
    bottom_right = blocks[:, :, :, 1, :, 1]

    low_low = (top_left + top_right + bottom_left + bottom_right) * 0.5
    low_high = (top_left + top_right - bottom_left - bottom_right) * 0.5
    high_low = (top_left - top_right + bottom_left - bottom_right) * 0.5
    high_high = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return torch.cat((low_low, low_high, high_low, high_high), dim=1)


def haar_idwt(x: torch.Tensor) -> torch.Tensor:
    """Single-level orthonormal Haar reconstruction, the inverse of :func:`haar_dwt`.

    Args:
        x: Tensor of shape ``(B, 4C, H, W)`` ordered ``[LL, LH, HL, HH]``.

    Returns:
        Tensor of shape ``(B, C, 2H, 2W)``.

    Raises:
        ValueError: If ``x`` is not 4-D or its channel count is not divisible by 4.
    """
    if x.dim() != 4:
        raise ValueError("haar_idwt expects a 4-D tensor, got " + str(x.dim()) + " dims.")
    batch = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    if channels % 4 != 0:
        raise ValueError(
            "haar_idwt requires a channel count divisible by 4, got "
            + str(channels) + "."
        )
    out_channels = channels // 4
    low_low = x[:, 0 * out_channels : 1 * out_channels]
    low_high = x[:, 1 * out_channels : 2 * out_channels]
    high_low = x[:, 2 * out_channels : 3 * out_channels]
    high_high = x[:, 3 * out_channels : 4 * out_channels]

    top_left = (low_low + low_high + high_low + high_high) * 0.5
    top_right = (low_low + low_high - high_low - high_high) * 0.5
    bottom_left = (low_low - low_high + high_low - high_high) * 0.5
    bottom_right = (low_low - low_high - high_low + high_high) * 0.5

    top = torch.stack((top_left, top_right), dim=-1).reshape(
        batch, out_channels, height, width * 2
    )
    bottom = torch.stack((bottom_left, bottom_right), dim=-1).reshape(
        batch, out_channels, height, width * 2
    )
    return torch.stack((top, bottom), dim=3).reshape(
        batch, out_channels, height * 2, width * 2
    )


class HaarDWT(nn.Module):
    """Module wrapper around :func:`haar_dwt`. Parameter-free."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose ``(B, C, H, W)`` into ``(B, 4C, H/2, W/2)``."""
        return haar_dwt(x)


class HaarIDWT(nn.Module):
    """Module wrapper around :func:`haar_idwt`. Parameter-free."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(B, 4C, H, W)`` into ``(B, C, 2H, 2W)``."""
        return haar_idwt(x)
