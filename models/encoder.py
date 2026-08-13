"""SPARC-Net encoder (Contract Part 1, stages 2-3).

The trunk begins **one Haar level below the input**. The DWT stem is what makes the
model fit a 4 GB card: it cuts compute and activation memory 4x at the most expensive
resolution, and because the DWT is lossless nothing is given up to get that.

Resolution schedule for a 128x128 input::

    (B, 2, 128, 128)  concat[y_hat, sigma_hat]
      -> HaarDWT   -> (B,   8, 64, 64)
      -> Conv3x3   -> (B,  48, 64, 64)   L0   4 x NAF                    -> skip_0
      -> HaarDWT   -> (B, 192, 32, 32)
      -> Conv1x1   -> (B,  96, 32, 32)   L1   4 x NAF + 2 x GSA          -> skip_1
      -> HaarDWT   -> (B, 384, 16, 16)
      -> Conv1x1   -> (B, 160, 16, 16)   L2   4 x NAF + 3 x GSA   bottleneck
"""

from __future__ import annotations

import torch
from torch import nn

from configs.sparc_config import SparcConfig
from models.blocks.naf_block import NAFBlock
from models.film import apply_film
from models.wavelet.haar import HaarDWT


def build_gsa_blocks(config: SparcConfig, level: int, count: int) -> list[nn.Module]:
    """Construct the global self-attention blocks for one level.

    Args:
        config: Model configuration.
        level: Trunk level index.
        count: Number of blocks requested by the configuration.

    Returns:
        A list of ``GSABlock`` instances, empty when attention is disabled or the
        requested count is zero. The import is deferred so that the step-6 model can
        be built before ``models/attention`` exists (Contract Part 8).
    """
    if count <= 0 or not config.use_attention:
        return []
    from models.attention.gsa_block import GSABlock

    return [
        GSABlock(
            channels=config.widths[level],
            heads=config.num_heads[level],
            spatial_size=config.level_size(level),
            attn_dim_ratio=config.attn_dim_ratio,
            ffn_expansion=config.ffn_expansion,
            layer_scale_init=config.layer_scale_init,
            layer_norm_eps=config.layer_norm_eps,
            use_sdpa=config.use_sdpa,
            use_relative_position_bias=config.use_relative_position_bias,
            use_attention_branch=config.use_attention_branch,
        )
        for _ in range(count)
    ]


class Stem(nn.Module):
    """Lossless DWT stem: ``(B, Cin, H, W) -> (B, Cout, H/2, W/2)``.

    Args:
        in_channels: Input channels (2 with the noise map, 1 without).
        out_channels: Trunk level-0 width.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = HaarDWT()
        self.proj = nn.Conv2d(4 * in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose and project to trunk width."""
        return self.proj(self.dwt(x))

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


class Downsample(nn.Module):
    """Lossless 2x downsample: ``HaarDWT`` then ``Conv1x1(4*Cin -> Cout)``.

    Args:
        in_channels: Channels of the incoming feature map.
        out_channels: Channels of the next level.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dwt = HaarDWT()
        self.proj = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the resolution without losing information."""
        return self.proj(self.dwt(x))


class EncoderLevel(nn.Module):
    """One encoder level: ``n`` NAF blocks followed by ``m`` GSA blocks.

    Args:
        config: Model configuration.
        level: Trunk level index.
    """

    def __init__(self, config: SparcConfig, level: int) -> None:
        super().__init__()
        channels = config.widths[level]
        self.level = level
        self.channels = channels
        self.naf = nn.Sequential(
            *[
                NAFBlock(
                    channels,
                    expansion=config.naf_expansion,
                    layer_scale_init=config.layer_scale_init,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.enc_naf_blocks[level])
            ]
        )
        self.gsa = nn.Sequential(
            *build_gsa_blocks(config, level, config.enc_gsa_blocks[level])
        )

    def forward(
        self,
        x: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the level's blocks in order, then the optional FiLM modulation.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.
            film: ``(gamma, beta)`` for this level, or ``None`` when unconditioned.

        Returns:
            Tensor of the same shape.
        """
        return apply_film(self.gsa(self.naf(x)), film)

    def extra_repr(self) -> str:
        return f"level={self.level}, channels={self.channels}"


class Encoder(nn.Module):
    """DWT stem plus the trunk levels, returning the bottleneck and the skips.

    Args:
        config: Model configuration.
        in_channels: Channels entering the stem.
    """

    def __init__(self, config: SparcConfig, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.stem = Stem(in_channels, config.widths[0])
        self.levels = nn.ModuleList(
            EncoderLevel(config, level) for level in range(config.num_levels)
        )
        self.downsamples = nn.ModuleList(
            Downsample(config.widths[level], config.widths[level + 1])
            for level in range(config.num_levels - 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        film: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode an input tensor.

        Args:
            x: Tensor of shape ``(B, in_channels, H, W)``.
            film: One ``(gamma, beta)`` pair per trunk level, or ``None``.

        Returns:
            ``(bottleneck, skips)`` where ``skips`` is ordered fine-to-coarse and
            excludes the bottleneck.

        Raises:
            ValueError: If ``film`` is given with the wrong number of entries.
        """
        if film is not None and len(film) != len(self.levels):
            raise ValueError(
                "Encoder expects " + str(len(self.levels))
                + " FiLM entries, got " + str(len(film)) + "."
            )
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for index, level in enumerate(self.levels):
            x = level(x, None if film is None else film[index])
            if index < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[index](x)
        return x, skips
