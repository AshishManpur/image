"""SPARC-Net decoder (Contract Part 1, stage 4).

Mirrors the encoder: each stage projects to ``4 * C_finer`` channels, applies the
inverse Haar transform to double the resolution, fuses the encoder skip, then runs the
level's attention and NAF blocks.

Attention appears only in the coarse decoder stage (D1, 32x32). Contract Part 2.6
forbids it at 64x64 and above: a GSA block at 64x64 would cost 16x the 32x32 matmul.
"""

from __future__ import annotations

import torch
from torch import nn

from configs.sparc_config import SparcConfig
from models.blocks.naf_block import NAFBlock
from models.encoder import build_gsa_blocks
from models.film import apply_film
from models.fusion import build_fusion
from models.wavelet.haar import HaarIDWT


class Upsample(nn.Module):
    """Lossless 2x upsample: ``Conv1x1(Cin -> 4*Cout)`` then ``HaarIDWT``.

    Args:
        in_channels: Channels of the coarser feature map.
        out_channels: Channels of the finer level.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)
        self.idwt = HaarIDWT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double the resolution without introducing checkerboard artefacts."""
        return self.idwt(self.proj(x))


class DecoderLevel(nn.Module):
    """One decoder stage: upsample, fuse the skip, then GSA and NAF blocks.

    Args:
        config: Model configuration.
        level: Trunk level index of the **output** of this stage.
        stage: Decoder stage index, ordered coarse-to-fine.
    """

    def __init__(self, config: SparcConfig, level: int, stage: int) -> None:
        super().__init__()
        channels = config.widths[level]
        self.level = level
        self.channels = channels
        self.upsample = Upsample(config.widths[level + 1], channels)
        self.fusion = build_fusion(config, channels)
        self.gsa = nn.Sequential(
            *build_gsa_blocks(config, level, config.dec_gsa_blocks[stage])
        )
        self.naf = nn.Sequential(
            *[
                NAFBlock(
                    channels,
                    expansion=config.naf_expansion,
                    layer_scale_init=config.layer_scale_init,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.dec_naf_blocks[stage])
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Upsample ``x``, fuse ``skip``, refine, then apply the optional FiLM.

        Args:
            x: Coarser decoder feature ``(B, C_coarse, H, W)``.
            skip: Encoder feature ``(B, C, 2H, 2W)``.
            film: ``(gamma, beta)`` for this stage, or ``None`` when unconditioned.

        Returns:
            Tensor of shape ``(B, C, 2H, 2W)``.
        """
        x = self.upsample(x)
        x = self.fusion(skip, x)
        return apply_film(self.naf(self.gsa(x)), film)

    def extra_repr(self) -> str:
        return f"level={self.level}, channels={self.channels}"


class Decoder(nn.Module):
    """Full decoder, coarse to fine.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: SparcConfig) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            DecoderLevel(config, level=config.num_levels - 2 - stage, stage=stage)
            for stage in range(config.num_levels - 1)
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
        film: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Decode to trunk level 0.

        Args:
            bottleneck: Coarsest encoder output.
            skips: Encoder skips ordered fine-to-coarse.
            film: One ``(gamma, beta)`` pair per decoder stage, or ``None``.

        Returns:
            Tensor of shape ``(B, widths[0], trunk, trunk)``.

        Raises:
            ValueError: If the number of skips or of FiLM entries does not match the
                number of stages.
        """
        if len(skips) != len(self.stages):
            raise ValueError(
                f"Decoder expects {len(self.stages)} skips, got {len(skips)}."
            )
        if film is not None and len(film) != len(self.stages):
            raise ValueError(
                "Decoder expects " + str(len(self.stages))
                + " FiLM entries, got " + str(len(film)) + "."
            )
        x = bottleneck
        for stage, module in enumerate(self.stages):
            x = module(
                x,
                skips[len(skips) - 1 - stage],
                None if film is None else film[stage],
            )
        return x
