"""Frozen configuration objects for SPARC-Base V1.0."""

from configs.sparc_config import (
    ATTENTION_HEAD_DIM,
    PROJECT_ROOT,
    DataConfig,
    LossConfig,
    NoiseHeadConfig,
    SparcConfig,
    TrainingConfig,
    build_sparc_config,
    sparc_base,
    sparc_tiny,
)

__all__ = [
    "ATTENTION_HEAD_DIM",
    "DataConfig",
    "LossConfig",
    "NoiseHeadConfig",
    "PROJECT_ROOT",
    "SparcConfig",
    "TrainingConfig",
    "build_sparc_config",
    "sparc_base",
    "sparc_tiny",
]
