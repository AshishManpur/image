"""Supervised clean-LR branch (Phase 6).

The V1 output path is ``out = head(decoded) + bicubic(y_hat)``, where ``y_hat`` is the
*noisy* observation. That formulation hands the network the observation noise and
requires it to synthesise an exact cancelling signal. Measured on the 320-image
validation split with ``checkpoints/sparc_base_50/best_psnr.pt``:

===========================  ==========  ==========  =============
band                         model err   oracle err  model/oracle
===========================  ==========  ==========  =============
LOW   (0-.125 Nyquist)          1125.9         7.6         148.4x
MID   (.125-.35)                 594.0        52.7          11.3x
HIGH  (.35-.7)                   276.2       170.0           1.62x
NYQ   (.7-1.0)                    97.6        97.2           1.00x
===========================  ==========  ==========  =============

The network reduces the low-frequency error energy of ``bicubic(y_hat)`` by only 34 %,
against 66-69 % in the mid and high bands, and ``corr(R, R*)`` — the correlation between
the learned and ideal residuals — is 0.599 at low frequency against 0.831/0.815 at
mid/high. Low frequencies carry the most signal energy and average down fastest under
noise, so they should be the *easiest* band; the network being worst there is the
architectural signal this branch addresses.

This module reads the decoder trunk — the same tensor the reconstruction head consumes,
so the encoder and decoder are shared at zero duplicated cost — and predicts ``Delta`` at
LR resolution. ``clean_norm = y_hat + Delta`` is supervised against ``A(GT)`` and becomes
the residual base when ``residual_source="clean"``.

The structure deliberately mirrors :class:`~models.decoder.reconstruction_head.Reconstruction\
Head`'s ``project -> HaarIDWT`` stage rather than introducing a new upsampling scheme:
the orthogonal Haar basis has no preferred sub-pixel position, so no ICNR initialisation
is required and checkerboard artefacts are impossible.

``to_clean`` is zero-initialised, which makes ``Delta = 0`` and the whole network
**bit-identical** to the V1 path at step 0 — verified ``max|new - old| = 0.000e+00``.
It carries ``_custom_init`` so :func:`utils.init.default_init` does not overwrite it.
"""

from __future__ import annotations

import torch
from torch import nn

from configs.sparc_config import SparcConfig
from models.blocks.naf_block import NAFBlock
from models.wavelet.haar import HaarIDWT


class CleanLRBranch(nn.Module):
    """Trunk features to a clean-LR correction at LR resolution.

    Args:
        config: Model configuration.

    Raises:
        ValueError: If ``clean_branch_width`` is not a positive even number.
    """

    def __init__(self, config: SparcConfig) -> None:
        super().__init__()
        width = config.clean_branch_width
        if width <= 0 or width % 2 != 0:
            raise ValueError(
                f"clean_branch_width must be a positive even number, got {width}."
            )

        self.in_channels = config.widths[0]
        self.width = width
        self.out_channels = config.out_channels

        self.project = nn.Conv2d(config.widths[0], 4 * width, kernel_size=3, padding=1)
        self.idwt = HaarIDWT()
        self.blocks = nn.Sequential(
            *[
                NAFBlock(
                    width,
                    expansion=config.naf_expansion,
                    layer_scale_init=config.layer_scale_init,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.clean_branch_naf_blocks)
            ]
        )
        self.to_clean = nn.Conv2d(width, config.out_channels, kernel_size=3, padding=1)
        if config.clean_branch_zero_init:
            nn.init.zeros_(self.to_clean.weight)
            nn.init.zeros_(self.to_clean.bias)
            # Stops `utils.init.default_init` from overwriting the zero-init during the
            # model-wide `apply`, exactly as `NoiseHead.fc2` does.
            self.to_clean._custom_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the clean-LR correction.

        Args:
            x: Decoder output ``(B, widths[0], T, T)`` at trunk resolution.

        Returns:
            Correction ``Delta`` of shape ``(B, out_channels, 2T, 2T)``, i.e. LR
            resolution. Zero at initialisation.

        Raises:
            ValueError: If the channel count does not match.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "CleanLRBranch configured for " + str(self.in_channels)
                + " channels, got " + str(x.shape[1]) + "."
            )
        return self.to_clean(self.blocks(self.idwt(self.project(x))))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, width={self.width}, "
            f"out_channels={self.out_channels}"
        )
