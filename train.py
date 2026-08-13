"""Train SPARC-Net (Contract Part 8, step 7).

Usage::

    # The full V1 model (Phase 4.12): normaliser + noise head + gated fusion +
    # 3 GSA groups + composite loss. 2,345,650 parameters. This is the scored model.
    python train.py --variant sparc-base --epochs 400 --amp-dtype bf16

    # GPU shakedown: measures VRAM, epoch time and AMP stability in a few minutes.
    python train.py --variant sparc-base --epochs 3 --warmup-epochs 1 --amp-dtype bf16

    python train.py --resume checkpoints/sparc-base/last.pt

Ablation arms only — **not** the scored model::

    python train.py --variant sparc-base --no-attention   # ablation A5 control
    python train.py --stage6 --variant sparc-tiny --epochs 5

.. note::
   ``--no-attention`` drops the three GSA groups (608,088 parameters) and produces a
   1,737,562-parameter network. It exists for ablation A5 and for the pre-4.12
   baseline; a checkpoint from it will not load into the full V1 model. The startup
   banner prints the module state and the parameter count so the two cannot be
   confused after the fact.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.sparc_config import DataConfig, TrainingConfig, build_sparc_config
from datasets.packed_dataset import build_datasets
from datasets.splits import group_aware_split, verify_no_group_overlap
from losses import CompositeLoss
from losses.charbonnier import CharbonnierLoss
from models.sparc_net import SPARCNet
from trainer.trainer import DivergenceError, Trainer
from utils.complexity import count_parameters, measure_complexity
from utils.logging_utils import configure_logging, get_logger
from utils.seed import set_seed

_LOGGER = get_logger(__name__)

STAGE6_OVERRIDES = {
    "use_noise_head": False,
    "use_gated_fusion": False,
    "use_attention": False,
}


FULL_V1_PARAMETERS = 2_345_650
"""Contract Part 3, TOTAL row — the full Phase 4.12 model."""


def log_module_state(model: SPARCNet, config) -> None:
    """Print which optional modules are live, so a run cannot be misattributed later.

    A checkpoint records weights, not intent. Three of the four switches below change
    the parameter count by hundreds of thousands and every one of them still trains to
    a plausible-looking loss curve, so the only cheap defence against training the
    wrong network for eleven hours is to state the configuration at startup and check
    the total against Part 3.

    Args:
        model: The constructed model.
        config: Its :class:`~configs.sparc_config.SparcConfig`.
    """
    from models.attention.gsa_block import GSABlock

    blocks = [m for m in model.modules() if isinstance(m, GSABlock)]
    gsa_params = sum(sum(p.numel() for p in b.parameters()) for b in blocks)
    total = sum(p.numel() for p in model.parameters())

    _LOGGER.info(
        "Module state | attention=%s (%d GSA blocks, %d params) | attention_branch=%s | "
        "gated_fusion=%s | noise_head=%s | normalisation=%s | global_residual=%s",
        "ENABLED" if config.use_attention else "DISABLED",
        len(blocks),
        gsa_params,
        "ENABLED" if getattr(config, "use_attention_branch", True) else "DISABLED (GDFN-only arm)",
        "ENABLED" if config.use_gated_fusion else "DISABLED",
        "ENABLED" if config.use_noise_head else "DISABLED",
        "ENABLED" if config.use_normalization else "DISABLED",
        "ENABLED" if config.use_global_residual else "DISABLED",
    )
    if total == FULL_V1_PARAMETERS:
        _LOGGER.info(
            "Parameter count %d matches Contract Part 3 exactly: this is the FULL "
            "Phase 4.12 V1 model.", total
        )
    else:
        _LOGGER.warning(
            "Parameter count %d differs from the Contract Part 3 total %d (delta %+d). "
            "This is not the scored V1 model — confirm that is intended.",
            total, FULL_V1_PARAMETERS, total - FULL_V1_PARAMETERS,
        )


def build_loaders(
    data_config: DataConfig, train_config: TrainingConfig
) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation loaders with a leak-free split.

    Args:
        data_config: Dataset configuration.
        train_config: Training configuration.

    Returns:
        ``(train_loader, val_loader)``.
    """
    split = group_aware_split(
        data_config.expected_train,
        block_size=train_config.val_block_size,
        every_n=train_config.val_every_n_blocks,
    )
    verify_no_group_overlap(
        split, train_config.val_block_size, data_config.expected_train
    )
    train_set, val_set = build_datasets(
        data_config, split.train, split.val, seed=train_config.seed
    )

    common = {
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory,
    }
    if train_config.num_workers > 0:
        common["persistent_workers"] = train_config.persistent_workers
        common["prefetch_factor"] = train_config.prefetch_factor

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=train_config.drop_last,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Train SPARC-Net.")
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--stage6",
        action="store_true",
        help="Disable the modules deferred past Contract step 6.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help=(
            "ABLATION ARM (A5), not the scored model. Drops the three GSA groups "
            "(608,088 params), leaving the 1,737,562-parameter pre-attention network. "
            "Omit this for the full Phase 4.12 V1 model."
        ),
    )
    parser.add_argument(
        "--no-attention-branch",
        action="store_true",
        help=(
            "ABLATION ARM (GDFN-only), not the scored model. Keeps the six GSA blocks "
            "and their convolutional GDFN branch but drops the self-attention "
            "sub-branch (274,776 params), leaving 2,070,874. --no-attention drops the "
            "whole GSA group instead, which removes attention AND 333,312 params of "
            "GDFN; this flag isolates attention as the single independent variable."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("composite", "charbonnier"),
        default="composite",
        help=(
            "Training objective. 'composite' is the Contract Part 6 V1 objective and "
            "the default; 'charbonnier' is the reconstruction-only ablation arm."
        ),
    )

    numerics = parser.add_argument_group(
        "numerical stability (Phase 4.10.1)",
        "Operational settings. None of these changes a contract hyperparameter, the "
        "architecture, or the parameter count.",
    )
    numerics.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default=None,
        help=(
            "Autocast dtype. Default fp16 (contract). bf16 has fp32's exponent range "
            "and so removes the 65504 activation ceiling that caused the Phase 4.10 "
            "divergence; it needs Ampere or newer and runs without a GradScaler."
        ),
    )
    numerics.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=None,
        help=(
            "Abort after this many consecutive non-finite batches (default 25). "
            "Skipping does not touch the weights, so a diverged model fails every "
            "batch forever; Phase 4.10 skipped 300+ and still reported success."
        ),
    )
    numerics.add_argument(
        "--detect-anomaly",
        action="store_true",
        help=(
            "Run each step inside torch.autograd.detect_anomaly to name the backward "
            "op that produces the first NaN. Debugging only: roughly halves speed."
        ),
    )
    numerics.add_argument(
        "--debug-numerics",
        action="store_true",
        help=(
            "Write a per-step tensor-range trace to <log_dir>/<run>/numerics.jsonl. "
            "Forces a device sync per step; pair with --debug-numerics-from-step."
        ),
    )
    numerics.add_argument(
        "--debug-numerics-from-step",
        type=int,
        default=None,
        help="First global step at which --debug-numerics emits (default 0).",
    )
    args = parser.parse_args()

    configure_logging()

    train_config = TrainingConfig()
    overrides = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("learning_rate", args.lr),
            ("num_workers", args.num_workers),
            ("warmup_epochs", args.warmup_epochs),
            ("seed", args.seed),
            ("amp_dtype", args.amp_dtype),
            ("max_consecutive_nonfinite", args.max_consecutive_nonfinite),
            ("debug_numerics_from_step", args.debug_numerics_from_step),
            # store_true flags: only override when set, so the config default stands.
            ("detect_anomaly", args.detect_anomaly or None),
            ("debug_numerics", args.debug_numerics or None),
        )
        if value is not None
    }
    if overrides:
        # TrainingConfig is a slots=True dataclass, so instances have no __dict__.
        # dataclasses.replace() is the only correct way to derive a modified copy.
        train_config = dataclasses.replace(train_config, **overrides)
    train_config.validate()

    set_seed(train_config.seed)
    device = torch.device(args.device)

    model_overrides = dict(STAGE6_OVERRIDES) if args.stage6 else {}
    if args.no_attention:
        model_overrides["use_attention"] = False
    if args.no_attention_branch:
        model_overrides["use_attention_branch"] = False
    model_config = build_sparc_config(args.variant, **model_overrides)
    model = SPARCNet(model_config)
    total, trainable = count_parameters(model)
    report = measure_complexity(model, torch.randn(1, 1, 128, 128))
    _LOGGER.info(
        "%s: %.4f M params (%.4f M trainable) | %.4f GMAC | %.3f GFLOP",
        model_config.name,
        total / 1e6,
        trainable / 1e6,
        report.gmacs,
        report.gflops,
    )

    log_module_state(model, model_config)
    if not model_config.use_attention:
        _LOGGER.warning(
            "Attention is DISABLED: this is the pre-Phase-4.12 baseline, missing the "
            "three GSA groups (608,088 params). Checkpoints from this run will NOT "
            "load into the full V1 model."
        )

    criterion = CompositeLoss() if args.loss == "composite" else CharbonnierLoss()
    if args.loss == "charbonnier":
        _LOGGER.warning(
            "Training on Charbonnier only. The Contract Part 6 objective is the "
            "composite loss; use --loss composite for a scored run."
        )

    train_loader, val_loader = build_loaders(DataConfig(), train_config)
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    total_optimizer_steps = train_steps * train_config.epochs
    _LOGGER.info(
        "Schedule: %d train steps/epoch from %d training samples | %d val steps/epoch "
        "from %d validation samples | batch size %d | total optimizer steps %d",
        train_steps,
        train_samples,
        val_steps,
        val_samples,
        train_config.batch_size,
        total_optimizer_steps,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        run_name=args.run_name or model_config.name,
    )
    if args.resume is not None:
        trainer.load(args.resume, resume=True)

    try:
        state = trainer.fit()
    except DivergenceError as exc:
        # A non-zero exit code so a sweep or scheduler treats this as the failure it
        # is. Phase 4.10 exited 0 after learning nothing for two of its three epochs.
        _LOGGER.error("Training aborted: %s", exc)
        if exc.report_path is not None:
            _LOGGER.error("Diagnostic report: %s", exc.report_path)
        return 2

    _LOGGER.info(
        "Done. Best val PSNR %.3f dB, best EMA PSNR %.3f dB after %d epochs. "
        "%d batches skipped, %d GradScaler overflows.",
        state.best_psnr,
        state.best_ema_psnr,
        state.epoch + 1,
        trainer.skipped_batches,
        trainer.overflow_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
