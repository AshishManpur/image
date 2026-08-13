"""Training framework for SPARC-Base V1.0 (Contract Parts 5 and 6)."""

from trainer.ema import ModelEma
from trainer.trainer import Trainer, TrainState, build_param_groups, warmup_cosine_lambda

__all__ = [
    "ModelEma",
    "TrainState",
    "Trainer",
    "build_param_groups",
    "warmup_cosine_lambda",
]
