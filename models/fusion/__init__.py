"""Skip-fusion modules (Contract Part 2.7)."""

from __future__ import annotations

from torch import nn

from configs.sparc_config import SparcConfig
from models.fusion.concat_fuse import ConcatFusion
from models.fusion.gated_fuse import GatedFuse

__all__ = ["ConcatFusion", "GatedFuse", "build_fusion"]


def build_fusion(config: SparcConfig, channels: int) -> nn.Module:
    """Construct the skip-fusion module selected by the configuration.

    Args:
        config: Model configuration.
        channels: Channel count of the fusion.

    Returns:
        ``GatedFuse`` when ``config.use_gated_fusion`` is set, otherwise
        ``ConcatFusion`` (the ablation A3 control arm).
    """
    if config.use_gated_fusion:
        return GatedFuse(channels, reduction=config.fusion_reduction)
    return ConcatFusion(channels)
