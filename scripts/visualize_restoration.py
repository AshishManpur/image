"""Side-by-side restoration comparison for visual inspection.

PSNR is a scalar summary of a 65,536-pixel decision and it hides the failure modes that
matter here: over-smoothed speckle, ringing on strong edges, and hallucinated texture
all move it by a few tenths of a dB. This script puts the panels next to each other so
those are visible, and prints the metrics alongside rather than instead of them.

Layout::

    +-------------+-------------+-------------+
    | Input / LR  |  Restored   | Ground truth|      when GT is available
    +-------------+-------------+-------------+

    +-------------+-------------+
    | Input / LR  |  Restored   |             otherwise
    +-------------+-------------+

The LR panel is nearest-neighbour upscaled to 256x256 so the three panels align. It is
labelled as such: bicubic would make the input look better than the 21.67 dB baseline it
actually is, which would misrepresent the comparison.

Usage::

    python scripts/visualize_restoration.py --weights <ckpt.pt> \\
        --input Data/train/train/NoisyLR/000000.npy \\
        --gt    Data/train/train/GT/000000.npy \\
        --output outputs/compare_000000.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import LpipsMetric, psnr_per_image, ssim  # noqa: E402
from scripts.infer import (  # noqa: E402
    AMP_DTYPES,
    check_input_size,
    load_model,
    read_image,
    resolve_device,
    restore,
)
from utils.logging_utils import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

LABEL_HEIGHT = 22
GAP = 8


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _upscale_nearest(array: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def compose(panels: list[tuple[str, np.ndarray]]) -> "object":
    """Lay the panels out horizontally with a caption strip above each.

    Args:
        panels: ``(label, image (H, W) float in [0, 1])`` pairs, all the same size.

    Returns:
        A PIL ``Image`` in mode ``L``.
    """
    from PIL import Image, ImageDraw

    height, width = panels[0][1].shape
    canvas = Image.new(
        "L",
        (width * len(panels) + GAP * (len(panels) - 1), height + LABEL_HEIGHT),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * (width + GAP)
        canvas.paste(Image.fromarray(_to_uint8(image)), (x, LABEL_HEIGHT))
        draw.text((x + 4, 5), label, fill=0)
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an input / restored / ground-truth comparison strip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint (.pt).")
    parser.add_argument("--input", type=Path, required=True, help="LR input image.")
    parser.add_argument("--gt", type=Path, help="Optional ground-truth 256x256 image.")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/comparison.png"),
        help="Destination PNG.",
    )
    parser.add_argument(
        "--save-restored", type=Path, help="Also write the restored image on its own."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-dtype", default="fp32", choices=sorted(AMP_DTYPES))
    parser.add_argument(
        "--no-ema", action="store_true", help="Use live weights instead of the EMA."
    )
    parser.add_argument(
        "--lpips", action="store_true",
        help="Also compute LPIPS (requires the optional `lpips` package).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    device = resolve_device(args.device)
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    loaded = load_model(args.weights, device, prefer_ema=not args.no_ema)

    lr, _ = read_image(args.input)
    check_input_size(lr, loaded.config)

    # one untimed warm-up: the first call pays lazy CUDA context and kernel autotune
    restore(loaded.model, lr, device, amp_dtype)
    start = time.perf_counter()
    restored, elapsed = restore(loaded.model, lr, device, amp_dtype)
    wall = time.perf_counter() - start

    panels = [
        (f"Input LR {lr.shape[0]}x{lr.shape[1]} (nearest x2)", _upscale_nearest(lr, 2)),
        (f"Restored {restored.shape[0]}x{restored.shape[1]}", restored),
    ]

    metrics: dict[str, float] = {}
    if args.gt is not None:
        gt, _ = read_image(args.gt)
        if gt.shape != restored.shape:
            raise SystemExit(
                f"Ground truth is {gt.shape} but the model produced {restored.shape}."
            )
        panels.append((f"Ground truth {gt.shape[0]}x{gt.shape[1]}", gt))

        pred_t = torch.from_numpy(restored)[None, None]
        gt_t = torch.from_numpy(gt)[None, None]
        metrics["psnr_db"] = float(psnr_per_image(pred_t, gt_t).item())
        metrics["ssim"] = float(ssim(pred_t, gt_t).item())
        if args.lpips:
            metric = LpipsMetric(device="cpu")
            if metric.available:
                metrics["lpips"] = float(metric(pred_t, gt_t).item())
            else:
                _LOGGER.warning("LPIPS requested but the `lpips` package is not installed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compose(panels).save(args.output)
    if args.save_restored is not None:
        from scripts.infer import write_image

        write_image(args.save_restored, restored)

    print("=" * 66)
    print(f"  model variant   : {loaded.variant}")
    print(f"  parameters      : {loaded.parameters:,}")
    print(f"  checkpoint      : {args.weights}")
    if loaded.epoch is not None:
        print(f"  epoch           : {loaded.epoch}")
    print(f"  device          : {device}  ({args.amp_dtype})")
    print(f"  input           : {args.input.name}  {lr.shape[0]}x{lr.shape[1]}")
    print(f"  output          : {restored.shape[0]}x{restored.shape[1]}")
    print(f"  inference time  : {elapsed * 1e3:.1f} ms (wall {wall * 1e3:.1f} ms)")
    if metrics:
        print(f"  PSNR            : {metrics['psnr_db']:.2f} dB")
        print(f"  SSIM            : {metrics['ssim']:.4f}")
        if "lpips" in metrics:
            print(f"  LPIPS           : {metrics['lpips']:.4f}")
    else:
        print("  PSNR/SSIM/LPIPS : n/a (no --gt supplied)")
    print(f"  comparison      : {args.output}")
    print("=" * 66)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
