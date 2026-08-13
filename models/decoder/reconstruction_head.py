"""Sub-band reconstruction head (Contract Part 1, stage 5).

The head takes the decoder output at trunk resolution (64x64) to the final 256x256
image in two lossless Haar steps, with all convolution work done at 128x128 or below.

**No convolution ever runs at 256x256.** The output resolution is pure memory
bandwidth on the target card, and the head is already 23.6 % of MACs and 35.2 % of
activations without it.

The final ``Conv3x3(head_width -> 4)`` predicts the four Haar sub-bands
``[LL, LH, HL, HH]`` of the output. That gives the four predicted channels a defined
meaning, makes the band-wise wavelet loss directly applicable, and makes checkerboard
artefacts impossible: the orthogonal basis has no preferred sub-pixel position, so no
ICNR initialisation is required.
"""

from __future__ import annotations

import torch
from torch import nn

from configs.sparc_config import SparcConfig
from models.blocks.naf_block import NAFBlock
from models.wavelet.haar import HaarIDWT


class ReconstructionHead(nn.Module):
    """Trunk features to full-resolution image via two inverse Haar steps.

    Args:
        config: Model configuration.

    Raises:
        ValueError: If ``head_width`` is not a positive even number.
    """

    def __init__(self, config: SparcConfig) -> None:
        super().__init__()
        width = config.head_width
        if width <= 0 or width % 2 != 0:
            raise ValueError(f"head_width must be a positive even number, got {width}.")

        self.in_channels = config.widths[0]
        self.head_width = width
        self.out_channels = config.out_channels

        project_kernel = config.head_project_kernel_size
        self.project = nn.Conv2d(
            config.widths[0], 4 * width, kernel_size=project_kernel,
            padding=project_kernel // 2,
        )
        self.idwt_mid = HaarIDWT()
        self.blocks = nn.Sequential(
            *[
                NAFBlock(
                    width,
                    expansion=config.naf_expansion,
                    layer_scale_init=config.layer_scale_init,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.head_naf_blocks)
            ]
        )
        self.to_subbands = nn.Conv2d(
            width, 4 * config.out_channels, kernel_size=3, padding=1
        )
        self.idwt_out = HaarIDWT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full-resolution image.

        Args:
            x: Decoder output ``(B, widths[0], T, T)`` at trunk resolution.

        Returns:
            Tensor of shape ``(B, out_channels, 4T, 4T)``.

        Raises:
            ValueError: If the channel count does not match.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "ReconstructionHead configured for " + str(self.in_channels)
                + " channels, got " + str(x.shape[1]) + "."
            )
        x = self.idwt_mid(self.project(x))
        x = self.blocks(x)
        return self.idwt_out(self.to_subbands(x))

    def predict_subbands(self, x: torch.Tensor) -> torch.Tensor:
        """Return the packed sub-bands before the final inverse transform.

        Exposed so that the wavelet loss can supervise ``[LL, LH, HL, HH]`` directly.

        Args:
            x: Decoder output ``(B, widths[0], T, T)``.

        Returns:
            Tensor of shape ``(B, 4 * out_channels, 2T, 2T)``.
        """
        return self.to_subbands(self.blocks(self.idwt_mid(self.project(x))))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, head_width={self.head_width}, "
            f"out_channels={self.out_channels}"
        )
