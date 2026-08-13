"""Capacity sanity check: overfit a handful of images (Contract Part 8, step 6).

Acceptance gate: **SPARC-Tiny must exceed 45 dB PSNR on 8 images within 2000 steps.**

This is the single most informative early test. Passing it proves, together, that the
data path is correct, that gradients reach every parameter, that the reconstruction
head can represent full-resolution detail, and that the optimiser configuration is
sane. Failing it localises the bug before any real training run is attempted.

Usage::

    python scripts/overfit.py [--variant sparc-tiny] [--images 8] [--steps 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import DataConfig, TrainingConfig, build_sparc_config  # noqa: E402
from evaluation.metrics import psnr_pooled, ssim  # noqa: E402
from losses.charbonnier import CharbonnierLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.complexity import count_parameters  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

TARGET_PSNR_DB = 45.0


def load_fixed_batch(
    config: DataConfig, count: int, device: torch.device, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a small fixed batch of paired samples.

    Args:
        config: Dataset configuration.
        count: Number of images.
        device: Device to place the tensors on.
        offset: Index of the first image, so that a disjoint held-out batch can be
            drawn by passing ``offset=images``.

    Returns:
        ``(lr, gt)`` tensors.

    Raises:
        FileNotFoundError: If the packed arrays are missing.
    """
    root = Path(config.packed_root)
    lr_path, gt_path = root / "train_lr.npy", root / "train_gt.npy"
    if not lr_path.exists():
        raise FileNotFoundError(
            f"{lr_path} missing. Run `python scripts/pack_dataset.py` first."
        )
    lr_all = np.load(lr_path, mmap_mode="r")
    gt_all = np.load(gt_path, mmap_mode="r")
    index = np.arange(offset, offset + count)
    lr = torch.from_numpy(np.asarray(lr_all[index], dtype=np.float32)).unsqueeze(1)
    gt = torch.from_numpy(np.asarray(gt_all[index], dtype=np.float32)).unsqueeze(1)
    return lr.to(device), gt.to(device)


@torch.no_grad()
def evaluate(
    model: SPARCNet,
    criterion: CharbonnierLoss,
    lr_batch: torch.Tensor,
    gt_batch: torch.Tensor,
) -> dict[str, float]:
    """Score a batch in inference mode.

    Args:
        model: Model to evaluate. Restored to training mode by the caller.
        criterion: Fidelity criterion.
        lr_batch: Degraded inputs.
        gt_batch: Ground truth.

    Returns:
        Mapping with ``loss``, ``psnr`` (pooled, dB) and ``ssim`` (mean).
    """
    was_training = model.training
    model.eval()
    prediction = model(lr_batch)
    result = {
        "loss": float(criterion(prediction, gt_batch).item()),
        "psnr": psnr_pooled(prediction, gt_batch),
        "ssim": float(ssim(prediction, gt_batch).mean().item()),
    }
    if was_training:
        model.train()
    return result


def overfit(
    variant: str,
    images: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    log_every: int,
    overrides: dict[str, object] | None = None,
    val_images: int = 8,
    eval_every: int = 50,
    trace_path: Path | None = None,
) -> dict[str, object]:
    """Run the overfit loop and return the full metric report.

    The training set is the first ``images`` packed pairs. A disjoint held-out batch
    of ``val_images`` pairs is scored alongside it: on a memorisation test the train
    curve must fall away from the validation curve, which is how a genuine overfit is
    distinguished from a model that is merely learning the bicubic prior.

    Args:
        variant: SPARC variant name.
        images: Number of images to overfit.
        steps: Maximum optimisation steps.
        learning_rate: AdamW learning rate.
        device: Device to train on.
        log_every: Logging interval in steps.
        overrides: Configuration overrides applied to the variant.
        val_images: Size of the disjoint held-out batch. ``0`` disables it.
        eval_every: Interval in steps between inference-mode evaluations.
        trace_path: Optional JSONL file receiving one record per evaluation.

    Returns:
        A report mapping with the final and best train/validation metrics, the step
        at which the target was reached, and timing.
    """
    config = build_sparc_config(variant, **(overrides or {}))
    model = SPARCNet(config).to(device).train()
    total, _ = count_parameters(model)
    _LOGGER.info("%s: %.4f M parameters", config.name, total / 1e6)

    data_config = DataConfig()
    lr_batch, gt_batch = load_fixed_batch(data_config, images, device)
    val_batch = (
        load_fixed_batch(data_config, val_images, device, offset=images)
        if val_images > 0
        else None
    )
    _LOGGER.info(
        "train %d images, held-out %d images, %.2f parameters per output pixel",
        images,
        val_images,
        total / (images * gt_batch.shape[-1] * gt_batch.shape[-2]),
    )

    criterion = CharbonnierLoss(eps=1e-6)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")

    best = -float("inf")
    best_step = 0
    train_eval: dict[str, float] = {}
    val_eval: dict[str, float] = {}
    train_loss = float("nan")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)
        prediction = model(lr_batch)
        loss = criterion(prediction, gt_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        train_loss = float(loss.item())

        with torch.no_grad():
            current = psnr_pooled(prediction.detach(), gt_batch)
        if current > best:
            best, best_step = current, step

        if step % eval_every == 0 or step == 1 or step == steps:
            train_eval = evaluate(model, criterion, lr_batch, gt_batch)
            val_eval = (
                evaluate(model, criterion, *val_batch) if val_batch is not None else {}
            )
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "train_loss_step": train_loss,
                "train": train_eval,
                "val": val_eval,
                "elapsed_s": time.perf_counter() - start,
            }
            if trace_path is not None:
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if step % log_every == 0 or step == 1 or step == steps:
                _LOGGER.info(
                    "step %5d | train loss %.6f PSNR %6.2f dB SSIM %.4f "
                    "| val loss %.6f PSNR %6.2f dB SSIM %.4f | best %6.2f dB | %.0f s",
                    step,
                    train_eval["loss"],
                    train_eval["psnr"],
                    train_eval["ssim"],
                    val_eval.get("loss", float("nan")),
                    val_eval.get("psnr", float("nan")),
                    val_eval.get("ssim", float("nan")),
                    best,
                    time.perf_counter() - start,
                )
        if best >= TARGET_PSNR_DB:
            _LOGGER.info("Target reached at step %d (%.2f dB).", step, best)
            break

    if not train_eval:  # steps == 0
        train_eval = evaluate(model, criterion, lr_batch, gt_batch)
    elapsed = time.perf_counter() - start
    report: dict[str, object] = {
        "variant": config.name,
        "parameters": total,
        "images": images,
        "val_images": val_images,
        "steps_run": step,
        "steps_budget": steps,
        "learning_rate": learning_rate,
        "device": str(device),
        "train_loss_final": train_loss,
        "train": train_eval,
        "val": val_eval,
        "best_psnr_db": best,
        "best_psnr_step": best_step,
        "target_psnr_db": TARGET_PSNR_DB,
        "passed": bool(best >= TARGET_PSNR_DB),
        "elapsed_s": elapsed,
        "seconds_per_step": elapsed / max(step, 1),
    }
    _LOGGER.info(
        "Finished in %.1f s. Best PSNR %.2f dB (target %.1f dB).",
        elapsed,
        best,
        TARGET_PSNR_DB,
    )
    return report


def main() -> int:
    """Entry point.

    Returns:
        0 if the target PSNR was reached, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Overfit a few images.")
    parser.add_argument("--variant", default="sparc-tiny")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="gate", help="Suffix for the report filenames.")
    parser.add_argument(
        "--stage6",
        action="store_true",
        help="Disable the modules not yet implemented at Contract step 6.",
    )
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)
    overrides = (
        {"use_noise_head": False, "use_gated_fusion": False, "use_attention": False}
        if args.stage6
        else {}
    )
    output_dir = Path(TrainingConfig().output_dir)
    report = overfit(
        args.variant,
        args.images,
        args.steps,
        args.lr,
        torch.device(args.device),
        args.log_every,
        overrides,
        val_images=args.val_images,
        eval_every=args.eval_every,
        trace_path=output_dir / f"overfit_{args.tag}_trace.jsonl",
    )
    report_path = output_dir / f"overfit_{args.tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _LOGGER.info("Report written to %s", report_path)

    if not report["passed"]:
        _LOGGER.error(
            "Overfit gate FAILED: %.2f dB < %.1f dB.",
            report["best_psnr_db"],
            TARGET_PSNR_DB,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
