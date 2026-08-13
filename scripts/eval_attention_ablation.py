"""Paired attention vs pre-attention evaluation (read-only ablation harness).

This script answers one question and nothing else: **on the same images, under the same
preprocessing, does the GSA attention variant restore better than the pre-attention
baseline?** It is deliberately a separate entry point rather than an edit to
``scripts/infer.py`` so that nothing in the trained pipeline moves while an ablation is
being judged.

Design decisions that make the comparison paired rather than merely parallel:

* **Both checkpoints see byte-identical inputs.** Images come from the packed arrays
  (``Data/packed/*.npy``) that training itself reads, not from the raw ``.npy`` tree, so
  there is no second decode path that could differ. The same index array is used for
  both models in the same order.
* **Augmentation and LR re-synthesis are off.** ``PackedRestorationDataset`` applies
  geometric ops and ``synthesize_lr`` only when ``training=True``; this harness reads the
  memmaps directly, so every model sees the real recorded NoisyLR.
* **The split is recomputed, not remembered.** ``group_aware_split`` is deterministic
  (no RNG), so ``--split val`` reproduces exactly the 320 images the trainer validated
  on, and ``--split train`` the 2880 it fitted. Train and val are reported separately
  because train-set PSNR is not evidence of generalisation.
* **No dataset-level normalisation.** ``RobustNormalizer`` lives inside the model and
  de-normalises before returning, so both models are fed and scored in the same units.

Metrics are the project's own (``evaluation.metrics``) so numbers are comparable with
the trainer's validation logs and the Phase 1 baselines.

Usage::

    python scripts/eval_attention_ablation.py \\
        --pre  <pre_attention_ckpt.pt> --attn <attention_ckpt.pt> \\
        --split val --limit 0 --tag my_run
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import DataConfig, TrainingConfig  # noqa: E402
from datasets.splits import group_aware_split  # noqa: E402
from evaluation.metrics import psnr_per_image, ssim  # noqa: E402
from scripts.infer import AMP_DTYPES, load_model, resolve_device, restore  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

REPORT_ROOT = PROJECT_ROOT / "reports" / "attention_vs_pre_attention"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "attention_vs_pre_attention"


# --------------------------------------------------------------------------- data
def resolve_packed_root(explicit: Path | None) -> Path:
    """Locate the packed dataset directory.

    ``DataConfig.packed_root`` spells the folder ``data/packed`` while the repository
    ships ``Data/packed``; those are the same path on NTFS but not on a case-sensitive
    filesystem, so both spellings are probed.

    Args:
        explicit: User-supplied path, or ``None`` to auto-detect.

    Returns:
        Directory containing ``train_lr.npy`` / ``train_gt.npy``.

    Raises:
        FileNotFoundError: If no candidate contains the packed arrays.
    """
    candidates = [explicit] if explicit else [
        Path(DataConfig().packed_root), PROJECT_ROOT / "Data" / "packed"
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "train_lr.npy").exists():
            return candidate
    raise FileNotFoundError(
        f"No packed dataset found. Looked in: {[str(c) for c in candidates]}. "
        "Run `python scripts/pack_dataset.py` first."
    )


def select_indices(split: str, n_samples: int, config: TrainingConfig) -> np.ndarray:
    """Reproduce the trainer's partition for the requested split.

    Args:
        split: ``train``, ``val`` or ``all``.
        n_samples: Total packed sample count.
        config: Training configuration carrying the block-split constants.

    Returns:
        Sorted index array.

    Raises:
        ValueError: If ``split`` is unknown.
    """
    if split == "all":
        return np.arange(n_samples, dtype=np.int64)
    indices = group_aware_split(
        n_samples, block_size=config.val_block_size, every_n=config.val_every_n_blocks
    )
    if split == "train":
        return np.sort(indices.train)
    if split == "val":
        return np.sort(indices.val)
    raise ValueError(f"Unknown split '{split}'.")


# ---------------------------------------------------------------------- statistics
@dataclass
class PairedStats:
    """Summary of a paired per-image difference (attention minus pre-attention)."""

    n: int
    mean: float
    median: float
    std: float
    ci95_low: float
    ci95_high: float
    t_statistic: float | None
    t_pvalue: float | None
    wilcoxon_statistic: float | None
    wilcoxon_pvalue: float | None
    cohens_dz: float | None
    n_improved: int
    n_degraded: int
    n_tied: int
    pct_improved: float
    pct_degraded: float
    largest_improvement: float
    largest_degradation: float


def paired_stats(delta: np.ndarray, tie_epsilon: float = 1e-4) -> PairedStats:
    """Summarise paired differences with both a parametric and a rank-based test.

    Both tests are reported because PSNR deltas on this dataset are not expected to be
    normal: Phase 1 identified 34 near-featureless images whose PSNR sits far from the
    bulk, and those are exactly the points that move a t-test's mean while leaving a
    signed-rank test alone. Agreement between the two is the signal; disagreement means
    the outliers are driving the result.

    Args:
        delta: Per-image differences, shape ``(N,)``.
        tie_epsilon: Absolute difference below which a pair counts as tied rather than
            improved or degraded. Guards against calling a 1e-7 dB float wobble a win.

    Returns:
        The populated :class:`PairedStats`.

    Raises:
        ValueError: If ``delta`` is empty.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        raise ValueError("Cannot summarise an empty delta array.")

    n = int(delta.size)
    mean = float(delta.mean())
    # ddof=1: this is a sample of images drawn from a larger population, not the
    # population itself.
    std = float(delta.std(ddof=1)) if n > 1 else 0.0

    t_stat = t_p = w_stat = w_p = None
    ci_low = ci_high = mean
    dz = None
    if n > 1 and std > 0.0:
        from scipy import stats as scipy_stats

        stderr = std / np.sqrt(n)
        critical = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_low, ci_high = mean - critical * stderr, mean + critical * stderr
        result = scipy_stats.ttest_rel(delta, np.zeros_like(delta))
        t_stat, t_p = float(result.statistic), float(result.pvalue)
        dz = mean / std
        if np.any(delta != 0.0):
            try:
                wilcoxon = scipy_stats.wilcoxon(delta, zero_method="wilcox")
                w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
            except ValueError:  # all-zero after dropping ties
                pass

    improved = int(np.sum(delta > tie_epsilon))
    degraded = int(np.sum(delta < -tie_epsilon))
    return PairedStats(
        n=n,
        mean=mean,
        median=float(np.median(delta)),
        std=std,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        t_statistic=t_stat,
        t_pvalue=t_p,
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_p,
        cohens_dz=dz,
        n_improved=improved,
        n_degraded=degraded,
        n_tied=n - improved - degraded,
        pct_improved=100.0 * improved / n,
        pct_degraded=100.0 * degraded / n,
        largest_improvement=float(delta.max()),
        largest_degradation=float(delta.min()),
    )


def describe(values: np.ndarray) -> dict[str, float]:
    """Mean/median/std/min/max of a metric column."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


# ------------------------------------------------------------------------ visuals
def _to_uint8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _upscale_nearest(array: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def compose_quad(panels: list[tuple[str, np.ndarray]], caption: str) -> Any:
    """Lay out INPUT LR | PRE-ATTENTION | ATTENTION | GROUND TRUTH.

    The LR panel is nearest-neighbour upscaled, never bicubic: bicubic would make the
    input look better than the 21.67 dB baseline it actually is.

    Args:
        panels: ``(label, image (H, W) float in [0, 1])`` pairs, all the same size.
        caption: Footer line, typically the per-image metrics.

    Returns:
        A PIL ``Image`` in mode ``L``.
    """
    from PIL import Image, ImageDraw

    label_height, gap, footer = 22, 8, 20
    height, width = panels[0][1].shape
    canvas = Image.new(
        "L",
        (width * len(panels) + gap * (len(panels) - 1), height + label_height + footer),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * (width + gap)
        canvas.paste(Image.fromarray(_to_uint8(image)), (x, label_height))
        draw.text((x + 4, 5), label, fill=0)
    draw.text((4, height + label_height + 4), caption, fill=0)
    return canvas


def select_representatives(rows: list[dict[str, Any]], n_random: int, seed: int) -> dict[str, int]:
    """Pick the images worth looking at by eye.

    Args:
        rows: Per-image result records.
        n_random: How many additional random cases to include.
        seed: RNG seed for the random picks, for reproducibility.

    Returns:
        Mapping from case label to row position.
    """
    deltas = np.array([r["psnr_delta"] for r in rows])
    order = np.argsort(deltas)
    chosen = {
        "largest_improvement": int(order[-1]),
        "largest_degradation": int(order[0]),
        "median_case": int(order[len(order) // 2]),
        "worst_absolute_psnr": int(np.argmin([r["psnr_attn"] for r in rows])),
    }
    rng = np.random.default_rng(seed)
    pool = [i for i in range(len(rows)) if i not in set(chosen.values())]
    for count, position in enumerate(rng.choice(pool, size=min(n_random, len(pool)), replace=False)):
        chosen[f"random_{count}"] = int(position)
    return chosen


# ---------------------------------------------------------------------- evaluation
@torch.inference_mode()
def evaluate_pair(
    pre: Any,
    attn: Any,
    lr_array: np.ndarray,
    gt_array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> list[dict[str, Any]]:
    """Run both models over the same indices and score every image.

    Args:
        pre: Loaded pre-attention model wrapper.
        attn: Loaded attention model wrapper.
        lr_array: Memmapped LR array ``(N, 128, 128)``.
        gt_array: Memmapped GT array ``(N, 256, 256)``.
        indices: Sample indices to evaluate.
        device: Compute device.
        amp_dtype: Autocast dtype or ``None`` for fp32.

    Returns:
        One record per image with both models' PSNR/SSIM, the deltas, and latencies.
    """
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        lr = np.asarray(lr_array[index], dtype=np.float32)
        gt = np.asarray(gt_array[index], dtype=np.float32)

        pre_out, pre_ms = restore(pre.model, lr, device, amp_dtype)
        attn_out, attn_ms = restore(attn.model, lr, device, amp_dtype)

        gt_t = torch.from_numpy(gt)[None, None]
        pre_t = torch.from_numpy(pre_out)[None, None]
        attn_t = torch.from_numpy(attn_out)[None, None]

        record = {
            "position": position,
            "index": int(index),
            "stem": f"{int(index):06d}",
            "psnr_pre": float(psnr_per_image(pre_t, gt_t).item()),
            "psnr_attn": float(psnr_per_image(attn_t, gt_t).item()),
            "ssim_pre": float(ssim(pre_t, gt_t).item()),
            "ssim_attn": float(ssim(attn_t, gt_t).item()),
            "latency_pre_ms": pre_ms * 1e3,
            "latency_attn_ms": attn_ms * 1e3,
            "gt_std": float(gt.std()),
        }
        record["psnr_delta"] = record["psnr_attn"] - record["psnr_pre"]
        record["ssim_delta"] = record["ssim_attn"] - record["ssim_pre"]
        rows.append(record)

        if (position + 1) % 25 == 0 or position + 1 == len(indices):
            _LOGGER.info("Evaluated %d/%d images", position + 1, len(indices))
    return rows


def measure_latency(model: Any, device: torch.device, amp_dtype: torch.dtype | None,
                    repeats: int, size: int) -> dict[str, float]:
    """Time single-image forward passes after a warm-up.

    The first call pays lazy context creation and kernel autotune, so it is discarded;
    reporting it would overstate the cost of whichever model ran first.

    Args:
        model: Loaded model wrapper.
        device: Compute device.
        amp_dtype: Autocast dtype or ``None``.
        repeats: Timed iterations.
        size: Input edge length.

    Returns:
        Mean/median/std latency in milliseconds.
    """
    probe = np.zeros((size, size), dtype=np.float32)
    for _ in range(3):
        restore(model.model, probe, device, amp_dtype)
    samples = np.array([restore(model.model, probe, device, amp_dtype)[1] * 1e3
                        for _ in range(repeats)])
    return {
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "std_ms": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired attention vs pre-attention ablation on identical images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pre", type=Path, required=True,
                        help="Pre-attention checkpoint (.pt).")
    parser.add_argument("--attn", type=Path, required=True,
                        help="Attention checkpoint (.pt).")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="Which partition to score. 'val' is the only "
                             "generalisation evidence.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N images of the split; 0 = all.")
    parser.add_argument("--tag", default="run",
                        help="Sub-directory name under reports/ and outputs/.")
    parser.add_argument("--packed-root", type=Path, default=None,
                        help="Override the packed dataset directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-dtype", default="fp32", choices=sorted(AMP_DTYPES))
    parser.add_argument("--no-ema", action="store_true",
                        help="Use live weights for BOTH models instead of the EMA "
                             "shadow. The flag is deliberately not per-model: "
                             "comparing one model's EMA against the other's live "
                             "weights is not an architecture comparison.")
    parser.add_argument("--n-random-visuals", type=int, default=3,
                        help="Random cases to render in addition to the extremes.")
    parser.add_argument("--latency-repeats", type=int, default=20,
                        help="Timed iterations for the standalone latency benchmark.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for the random visual picks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    device = resolve_device(args.device)
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    packed_root = resolve_packed_root(args.packed_root)

    lr_array = np.load(packed_root / "train_lr.npy", mmap_mode="r")
    gt_array = np.load(packed_root / "train_gt.npy", mmap_mode="r")
    indices = select_indices(args.split, lr_array.shape[0], TrainingConfig())
    if args.limit > 0:
        indices = indices[: args.limit]

    pre = load_model(args.pre, device, prefer_ema=not args.no_ema)
    attn = load_model(args.attn, device, prefer_ema=not args.no_ema)

    # An ablation is only an ablation if exactly one thing differs. Refuse the run
    # rather than emit a report whose headline number conflates two changes.
    if pre.config.use_attention:
        raise SystemExit(
            f"--pre ({args.pre}) has attention ENABLED; it is not a pre-attention "
            "baseline. Check the checkpoint paths."
        )
    if not attn.config.use_attention:
        raise SystemExit(
            f"--attn ({args.attn}) has attention DISABLED; there is nothing to ablate."
        )
    if pre.source != attn.source:
        raise SystemExit(
            f"Weight sources differ: pre='{pre.source}' vs attn='{attn.source}'. "
            "Comparing EMA against live weights measures the averaging, not the "
            "architecture."
        )

    report_dir = REPORT_ROOT / args.tag
    output_dir = OUTPUT_ROOT / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Split '%s': %d images from %s", args.split, len(indices), packed_root)
    _LOGGER.info("pre  = %s (%d params, epoch=%s)", pre.variant, pre.parameters, pre.epoch)
    _LOGGER.info("attn = %s (%d params, epoch=%s)", attn.variant, attn.parameters, attn.epoch)

    started = time.perf_counter()
    rows = evaluate_pair(pre, attn, lr_array, gt_array, indices, device, amp_dtype)
    elapsed = time.perf_counter() - started

    psnr_delta = np.array([r["psnr_delta"] for r in rows])
    ssim_delta = np.array([r["ssim_delta"] for r in rows])
    stats = {
        "psnr_delta": asdict(paired_stats(psnr_delta)),
        "ssim_delta": asdict(paired_stats(ssim_delta, tie_epsilon=1e-5)),
    }

    summary = {
        "psnr_pre": describe(np.array([r["psnr_pre"] for r in rows])),
        "psnr_attn": describe(np.array([r["psnr_attn"] for r in rows])),
        "ssim_pre": describe(np.array([r["ssim_pre"] for r in rows])),
        "ssim_attn": describe(np.array([r["ssim_attn"] for r in rows])),
    }

    latency = {
        "pre": measure_latency(pre, device, amp_dtype, args.latency_repeats,
                               pre.config.input_size),
        "attn": measure_latency(attn, device, amp_dtype, args.latency_repeats,
                                attn.config.input_size),
    }
    latency["overhead_pct"] = (
        100.0 * (latency["attn"]["median_ms"] - latency["pre"]["median_ms"])
        / latency["pre"]["median_ms"]
    )
    param_overhead_pct = 100.0 * (attn.parameters - pre.parameters) / pre.parameters

    # --------------------------------------------------------------- visual cases
    cases = select_representatives(rows, args.n_random_visuals, args.seed)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, str] = {}
    for label, position in cases.items():
        row = rows[position]
        index = row["index"]
        lr = np.asarray(lr_array[index], dtype=np.float32)
        gt = np.asarray(gt_array[index], dtype=np.float32)
        pre_out, _ = restore(pre.model, lr, device, amp_dtype)
        attn_out, _ = restore(attn.model, lr, device, amp_dtype)
        caption = (
            f"{row['stem']}  PRE {row['psnr_pre']:.2f}dB/{row['ssim_pre']:.4f}  "
            f"ATTN {row['psnr_attn']:.2f}dB/{row['ssim_attn']:.4f}  "
            f"dPSNR {row['psnr_delta']:+.3f}dB  dSSIM {row['ssim_delta']:+.5f}"
        )
        image = compose_quad(
            [
                ("INPUT LR (nearest x2)", _upscale_nearest(lr, 2)),
                ("PRE-ATTENTION", pre_out),
                ("ATTENTION", attn_out),
                ("GROUND TRUTH", gt),
            ],
            caption,
        )
        destination = visual_dir / f"{label}_{row['stem']}.png"
        image.save(destination)
        rendered[label] = str(destination.relative_to(PROJECT_ROOT))

    # -------------------------------------------------------------------- reports
    csv_path = report_dir / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "attention_vs_pre_attention/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "amp_dtype": args.amp_dtype,
            "cuda_available": torch.cuda.is_available(),
        },
        "configuration": {
            "split": args.split,
            "n_images": len(indices),
            "limit": args.limit,
            "packed_root": str(packed_root),
            "weight_source": pre.source,
            "seed": args.seed,
            "index_range": [int(indices.min()), int(indices.max())],
        },
        "models": {
            "pre": {
                "checkpoint": str(args.pre),
                "parameters": pre.parameters,
                "epoch": pre.epoch,
                "use_attention": pre.config.use_attention,
                "variant": pre.variant,
            },
            "attn": {
                "checkpoint": str(args.attn),
                "parameters": attn.parameters,
                "epoch": attn.epoch,
                "use_attention": attn.config.use_attention,
                "variant": attn.variant,
            },
            "parameter_overhead_pct": param_overhead_pct,
        },
        "summary": summary,
        "paired_statistics": stats,
        "latency": latency,
        "visuals": rendered,
        "evaluation_seconds": elapsed,
    }
    json_path = report_dir / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------------- stdout
    delta_stats = stats["psnr_delta"]
    print("=" * 78)
    print(f"  split            : {args.split}  ({len(indices)} images, weights={pre.source})")
    print(f"  pre-attention    : {pre.parameters:,} params, epoch {pre.epoch}")
    print(f"  attention        : {attn.parameters:,} params, epoch {attn.epoch}"
          f"  ({param_overhead_pct:+.1f}% params)")
    print("-" * 78)
    print(f"  PSNR  pre        : mean {summary['psnr_pre']['mean']:.3f} dB   "
          f"median {summary['psnr_pre']['median']:.3f} dB   "
          f"std {summary['psnr_pre']['std']:.3f}")
    print(f"  PSNR  attn       : mean {summary['psnr_attn']['mean']:.3f} dB   "
          f"median {summary['psnr_attn']['median']:.3f} dB   "
          f"std {summary['psnr_attn']['std']:.3f}")
    print(f"  SSIM  pre        : mean {summary['ssim_pre']['mean']:.4f}   "
          f"median {summary['ssim_pre']['median']:.4f}")
    print(f"  SSIM  attn       : mean {summary['ssim_attn']['mean']:.4f}   "
          f"median {summary['ssim_attn']['median']:.4f}")
    print("-" * 78)
    print(f"  dPSNR mean       : {delta_stats['mean']:+.4f} dB")
    print(f"  dPSNR median     : {delta_stats['median']:+.4f} dB")
    print(f"  dPSNR std        : {delta_stats['std']:.4f} dB")
    print(f"  dPSNR 95% CI     : [{delta_stats['ci95_low']:+.4f}, "
          f"{delta_stats['ci95_high']:+.4f}] dB")
    if delta_stats["t_pvalue"] is not None:
        print(f"  paired t-test    : t={delta_stats['t_statistic']:+.3f}  "
              f"p={delta_stats['t_pvalue']:.4g}  dz={delta_stats['cohens_dz']:+.3f}")
    if delta_stats["wilcoxon_pvalue"] is not None:
        print(f"  Wilcoxon         : W={delta_stats['wilcoxon_statistic']:.1f}  "
              f"p={delta_stats['wilcoxon_pvalue']:.4g}")
    print(f"  improved         : {delta_stats['n_improved']} "
          f"({delta_stats['pct_improved']:.1f}%)")
    print(f"  degraded         : {delta_stats['n_degraded']} "
          f"({delta_stats['pct_degraded']:.1f}%)")
    print(f"  tied             : {delta_stats['n_tied']}")
    print(f"  largest gain     : {delta_stats['largest_improvement']:+.3f} dB")
    print(f"  largest loss     : {delta_stats['largest_degradation']:+.3f} dB")
    print("-" * 78)
    print(f"  dSSIM mean       : {stats['ssim_delta']['mean']:+.6f}")
    print(f"  dSSIM median     : {stats['ssim_delta']['median']:+.6f}")
    print("-" * 78)
    print(f"  latency pre      : {latency['pre']['median_ms']:.1f} ms (median)")
    print(f"  latency attn     : {latency['attn']['median_ms']:.1f} ms (median)  "
          f"({latency['overhead_pct']:+.1f}%)")
    print("-" * 78)
    print(f"  per-image CSV    : {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  summary JSON     : {json_path.relative_to(PROJECT_ROOT)}")
    print(f"  visuals          : {visual_dir.relative_to(PROJECT_ROOT)} "
          f"({len(rendered)} figures)")
    print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
