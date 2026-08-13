"""CPU smoke run over real packed data (Phase 4.10.1).

Exercises the full :meth:`trainer.trainer.Trainer.fit` path — composite loss, EMA,
validation, checkpointing, divergence guard and the numerics trace — on a small slice
of the real packed training set.

This is **not** the Task 9 shakedown and does not substitute for it: AMP is CUDA-only,
so a CPU run says nothing about fp16 stability. What it does establish is that the
Phase 4.10.1 changes leave the training loop working end to end on real data, which is
the part that can be checked without a GPU.

Usage::

    python -m scripts.cpu_smoke --images 64 --epochs 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.sparc_config import DataConfig, TrainingConfig, build_sparc_config
from losses.composite_loss import CompositeLoss
from models.sparc_net import SPARCNet
from trainer.trainer import DivergenceError, Trainer
from utils.logging_utils import configure_logging, get_logger

_LOGGER = get_logger(__name__)


class PackedSlice(Dataset):
    """The first ``count`` LR/GT pairs of the packed training set.

    Args:
        count: Number of image pairs to load.
        packed_root: Directory holding ``train_lr.npy`` and ``train_gt.npy``.

    Raises:
        FileNotFoundError: If the packed arrays are absent.
    """

    def __init__(self, count: int, packed_root: Path | None = None) -> None:
        root = packed_root or DataConfig().packed_root
        lr_path, gt_path = root / "train_lr.npy", root / "train_gt.npy"
        for path in (lr_path, gt_path):
            if not path.exists():
                raise FileNotFoundError(f"Packed array not found: {path}")
        # np.asarray materialises the mmap slice; the arrays are read-only otherwise
        # and torch.from_numpy refuses to share a non-writable buffer.
        self.lr = np.asarray(np.load(lr_path, mmap_mode="r")[:count])
        self.gt = np.asarray(np.load(gt_path, mmap_mode="r")[:count])

    def __len__(self) -> int:
        return int(self.lr.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.from_numpy(self.lr[index]).float().unsqueeze(0),
            "gt": torch.from_numpy(self.gt[index]).float().unsqueeze(0),
            "index": torch.tensor(index),
        }


def main() -> int:
    """Run the smoke test.

    Returns:
        Process exit code: 2 if training diverged, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default="smoke")
    args = parser.parse_args()

    configure_logging()
    dataset = PackedSlice(args.images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=8,  # contract-valid; the loader batch is set independently above
        warmup_epochs=1,
        num_workers=0,
        checkpoint_dir=Path("outputs/smoke/ckpt"),
        log_dir=Path("outputs/smoke/logs"),
        output_dir=Path("outputs/smoke"),
        debug_numerics=True,
    )
    trainer = Trainer(
        model=SPARCNet(build_sparc_config("sparc-base", use_attention=False)),
        criterion=CompositeLoss(),
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=args.device,
        run_name=args.run_name,
    )

    start = time.perf_counter()
    try:
        state = trainer.fit()
    except DivergenceError as exc:
        _LOGGER.error("Smoke run diverged: %s", exc)
        return 2

    _LOGGER.info(
        "COMPLETED in %.0f s | skipped=%d overflow=%d consecutive_nonfinite=%d",
        time.perf_counter() - start,
        trainer.skipped_batches,
        trainer.overflow_steps,
        trainer.consecutive_nonfinite,
    )
    for record in state.history:
        _LOGGER.info(
            "epoch %d: loss=%.5f val_psnr=%.3f val_ssim=%.4f ema_psnr=%.3f",
            record["epoch"],
            record["train_loss"],
            record["val_psnr"],
            record["val_ssim"],
            record["ema_psnr"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
