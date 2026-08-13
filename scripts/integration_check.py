"""Phase 4.10 integration checkpoint — short end-to-end run with the composite loss.

This is **not** a training run and does not attempt to maximise PSNR. It answers six
questions and nothing else:

1. Does a real epoch complete with every loss term active, without NaN?
2. Does each term decrease over a handful of epochs?
3. Does the auxiliary noise prediction actually train (does sigma move toward truth)?
4. Does reconstruction quality improve rather than regress?
5. Are all terms logged independently to TensorBoard?
6. Do checkpointing and resume still work with the new criterion?

Usage::

    python scripts/integration_check.py --epochs 5 --subset 128
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import DataConfig, TrainingConfig, build_sparc_config  # noqa: E402
from datasets.packed_dataset import build_datasets  # noqa: E402
from datasets.splits import group_aware_split  # noqa: E402
from losses import TERM_NAMES, CompositeLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from trainer.trainer import Trainer  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)


def build_short_loaders(
    subset: int, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Build small train/val loaders from the real packed dataset.

    Args:
        subset: Number of training images to use.
        batch_size: Batch size.
        seed: Dataset seed.

    Returns:
        ``(train_loader, val_loader)``.
    """
    data_config = DataConfig()
    split = group_aware_split(data_config.expected_train, block_size=32, every_n=10)
    train_set, val_set = build_datasets(data_config, split.train, split.val, seed=seed)

    train_set = Subset(train_set, range(min(subset, len(train_set))))
    val_set = Subset(val_set, range(min(subset // 2, len(val_set))))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def noise_head_error(model: SPARCNet, loader: DataLoader, criterion: CompositeLoss) -> float:
    """Mean absolute log-sigma error of the noise head on a loader.

    This is the direct measure of whether the auxiliary branch is learning, independent
    of whether the reconstruction improved.

    Args:
        model: Model with an active noise head.
        loader: Data loader.
        criterion: Composite loss supplying the analytic target.

    Returns:
        Mean absolute error in log space, or ``nan`` when unavailable.
    """
    if criterion.noise is None or model.noise_head is None:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        output = model.forward_with_aux(batch["lr"])
        value = criterion.noise(output.sigma, batch["lr"], batch["gt"])
        total += float(value) * batch["lr"].shape[0]
        count += batch["lr"].shape[0]
    return total / max(count, 1)


def main() -> int:
    """Entry point.

    Returns:
        0 when every integration assertion holds, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Phase 4.10 integration check.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=Path("outputs/integration_4_10.json"))
    parser.add_argument(
        "--concat-fusion",
        action="store_true",
        help="Use ConcatFusion instead of GatedFuse (the ablation A3 control arm).",
    )
    parser.add_argument("--run-name", default="integration")
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help=(
            "Drop the three GSA groups (ablation A5 control arm). Attention is ON by "
            "default since Phase 4.12: this script previously hardcoded it off, which "
            "would now silently check the 1.74 M pre-attention model instead of the "
            "2.35 M scored one."
        ),
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the check on."
    )
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    model_config = build_sparc_config(
        args.variant,
        use_attention=not args.no_attention,
        use_gated_fusion=not args.concat_fusion,
    )
    model = SPARCNet(model_config)
    _LOGGER.info(
        "Integration check: attention=%s, fusion=%s, %d parameters",
        "ON" if model_config.use_attention else "OFF",
        "concat" if args.concat_fusion else "gated",
        sum(p.numel() for p in model.parameters()),
    )
    criterion = CompositeLoss()
    train_loader, val_loader = build_short_loaders(
        args.subset, args.batch_size, args.seed
    )

    config = dataclasses.replace(
        TrainingConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=1,
        num_workers=0,
        checkpoint_dir=Path("outputs/integration_ckpt"),
        log_dir=Path("outputs/logs") / args.run_name,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        run_name=args.run_name,
    )

    sigma_before = noise_head_error(model, val_loader, criterion)
    _LOGGER.info("Noise-head log-sigma MAE before training: %.4f", sigma_before)

    state = trainer.fit()
    sigma_after = noise_head_error(model, val_loader, criterion)
    _LOGGER.info("Noise-head log-sigma MAE after training:  %.4f", sigma_after)

    history = state.history
    first, last = history[0], history[-1]
    report = {
        "fusion": "concat" if args.concat_fusion else "gated",
        "attention": model_config.use_attention,
        "device": args.device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": len(history),
        "terms": {},
        "psnr_first": first["val_psnr"],
        "psnr_last": last["val_psnr"],
        "psnr_improved": last["val_psnr"] > first["val_psnr"],
        "noise_mae_before": sigma_before,
        "noise_mae_after": sigma_after,
        "noise_head_improved": sigma_after < sigma_before,
        "any_nan": False,
    }

    for name in (*TERM_NAMES, "total"):
        key = f"train_raw_{name}" if f"train_raw_{name}" in first else f"train_{name}"
        if key not in first:
            continue
        start, end = first[key], last[key]
        report["terms"][name] = {
            "first": start, "last": end, "decreased": end < start
        }

    for record in history:
        for value in record.values():
            if isinstance(value, float) and value != value:  # NaN check
                report["any_nan"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Phase 4.10 integration check ---")
    print(f"  epochs run          : {report['epochs']}")
    print(f"  NaN encountered     : {report['any_nan']}")
    print(f"  val PSNR            : {report['psnr_first']:.3f} -> {report['psnr_last']:.3f} dB")
    print(f"  noise log-sigma MAE : {sigma_before:.4f} -> {sigma_after:.4f}")
    print("  loss terms:")
    for name, values in report["terms"].items():
        mark = "down" if values["decreased"] else "UP  "
        print(f"    {name:14s} {values['first']:10.6f} -> {values['last']:10.6f}  {mark}")

    ok = (
        not report["any_nan"]
        and report["psnr_improved"]
        and report["terms"].get("total", {}).get("decreased", False)
    )
    print(f"\n  RESULT: {'PASS' if ok else 'REVIEW REQUIRED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
