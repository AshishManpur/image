"""Reproduce the Phase 1 baselines (Contract Part 8, step 3).

Acceptance gate: bicubic x2 must reproduce **21.67 dB** and nearest x2 **20.38 dB** on
the Phase 1 evaluation sample. Those figures were computed with a *pooled* PSNR — the
squared error is averaged over every pixel of every image before the logarithm — on
300 training pairs drawn with ``numpy.random.default_rng(0)``.

Nothing downstream is trustworthy until this script reproduces those numbers: it
exercises the packing, the LR/GT pairing, the value ranges, and the metric definition
all at once.

Usage::

    python scripts/baselines.py [--samples 300] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import DataConfig  # noqa: E402
from datasets.degradation import (  # noqa: E402
    bicubic_downsample2,
    bicubic_upsample2,
    forward_operator,
)
from evaluation.metrics import psnr_per_image, psnr_pooled, ssim  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

PHASE1_BICUBIC_DB = 21.67
PHASE1_NEAREST_DB = 20.38
PHASE1_ORACLE_DB = 27.36
TOLERANCE_DB = 0.05


def load_sample(
    config: DataConfig, num_samples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the Phase 1 evaluation sample from the packed arrays.

    Args:
        config: Dataset configuration.
        num_samples: Number of pairs to draw.
        seed: Seed for ``numpy.random.default_rng``, matching Phase 1.

    Returns:
        ``(lr, gt)`` tensors of shape ``(N, 1, 128, 128)`` and ``(N, 1, 256, 256)``.

    Raises:
        FileNotFoundError: If the packed arrays are missing.
    """
    root = Path(config.packed_root)
    lr_path, gt_path = root / "train_lr.npy", root / "train_gt.npy"
    for path in (lr_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `python scripts/pack_dataset.py` first."
            )
    lr_all = np.load(lr_path, mmap_mode="r")
    gt_all = np.load(gt_path, mmap_mode="r")
    index = np.sort(
        np.random.default_rng(seed).choice(lr_all.shape[0], num_samples, replace=False)
    )
    lr = torch.from_numpy(np.asarray(lr_all[index], dtype=np.float32)).unsqueeze(1)
    gt = torch.from_numpy(np.asarray(gt_all[index], dtype=np.float32)).unsqueeze(1)
    return lr, gt


def compute_baselines(lr: torch.Tensor, gt: torch.Tensor) -> dict[str, dict[str, float]]:
    """Compute every Phase 1 baseline predictor.

    Args:
        lr: Low-resolution tensor ``(N, 1, 128, 128)``.
        gt: Ground-truth tensor ``(N, 1, 256, 256)``.

    Returns:
        Mapping from predictor name to its metric dictionary.
    """
    predictions = {
        "bicubic": bicubic_upsample2(lr).clamp(0.0, 1.0),
        "nearest": torch.nn.functional.interpolate(lr, scale_factor=2, mode="nearest")
        .clamp(0.0, 1.0),
        "mean_image": gt.mean(dim=(1, 2, 3), keepdim=True).expand_as(gt).clone(),
        # Phase 1 computed the denoising oracle as decimate-then-upsample with no
        # pre-blur; that is the 27.36 dB figure. The blurred variant is reported
        # separately because the sigma=0.4 pre-blur suppresses the aliasing that
        # bicubic decimation introduces, which *raises* the reconstruction score.
        "oracle_clean_lr_bicubic": bicubic_upsample2(bicubic_downsample2(gt)).clamp(
            0.0, 1.0
        ),
        "oracle_blurred_lr_bicubic": bicubic_upsample2(forward_operator(gt, 0.4)).clamp(
            0.0, 1.0
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for name, pred in predictions.items():
        per_image = psnr_per_image(pred, gt)
        results[name] = {
            "psnr_pooled": psnr_pooled(pred, gt),
            "psnr_mean": float(per_image.mean().item()),
            "psnr_median": float(per_image.median().item()),
            "ssim_mean": float(ssim(pred, gt).mean().item()),
        }
    return results


def main() -> int:
    """Entry point.

    Returns:
        0 if the Phase 1 baselines are reproduced within tolerance, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Reproduce the Phase 1 baselines.")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    config = DataConfig()
    lr, gt = load_sample(config, args.samples, args.seed)
    _LOGGER.info("Loaded %d pairs: LR %s, GT %s", lr.shape[0], tuple(lr.shape), tuple(gt.shape))

    results = compute_baselines(lr, gt)
    _LOGGER.info("%-26s %10s %10s %10s %8s", "predictor", "pooled dB", "mean dB", "median dB", "SSIM")
    for name, metrics in results.items():
        _LOGGER.info(
            "%-26s %10.3f %10.3f %10.3f %8.4f",
            name,
            metrics["psnr_pooled"],
            metrics["psnr_mean"],
            metrics["psnr_median"],
            metrics["ssim_mean"],
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = True
    for name, expected in (
        ("bicubic", PHASE1_BICUBIC_DB),
        ("nearest", PHASE1_NEAREST_DB),
        ("oracle_clean_lr_bicubic", PHASE1_ORACLE_DB),
    ):
        actual = results[name]["psnr_pooled"]
        delta = abs(actual - expected)
        status = "OK " if delta <= TOLERANCE_DB else "FAIL"
        _LOGGER.info(
            "[%s] %-26s %.3f dB vs Phase 1 %.2f dB (delta %.3f)",
            status,
            name,
            actual,
            expected,
            delta,
        )
        ok &= delta <= TOLERANCE_DB

    if not ok:
        _LOGGER.error("Baseline reproduction FAILED. Do not proceed to model code.")
        return 1
    _LOGGER.info("All Phase 1 baselines reproduced within %.2f dB.", TOLERANCE_DB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
