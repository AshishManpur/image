"""Convergence plots for the Phase 4.7 overfit gate.

Renders the four diagnostic views that the root-cause analysis rests on:

1. PSNR vs step (train and held-out) against the 45 dB gate line.
2. Loss vs step, log scale — the memorisation signature.
3. SSIM vs step.
4. PSNR vs log-steps with a fitted trend, used to extrapolate how many steps a
   45 dB result would actually require.

Usage::

    python reports/plot_overfit.py [--trace outputs/overfit_gate_trace.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET_DB = 45.0


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Read the JSONL trace into arrays.

    Args:
        path: Trace file written by ``scripts/overfit.py``.

    Returns:
        Mapping of column name to array.

    Raises:
        FileNotFoundError: If the trace is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"No trace at {path}.")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "step": np.array([r["step"] for r in records], dtype=float),
        "lr": np.array([r["lr"] for r in records], dtype=float),
        "train_psnr": np.array([r["train"]["psnr"] for r in records]),
        "train_loss": np.array([r["train"]["loss"] for r in records]),
        "train_ssim": np.array([r["train"]["ssim"] for r in records]),
        "val_psnr": np.array([r["val"]["psnr"] for r in records]),
        "val_loss": np.array([r["val"]["loss"] for r in records]),
        "val_ssim": np.array([r["val"]["ssim"] for r in records]),
    }


def render(data: dict[str, np.ndarray], out_path: Path, title: str) -> Path:
    """Render the four-panel figure.

    Args:
        data: Trace arrays.
        out_path: Destination PNG.
        title: Figure title.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    step = data["step"]

    ax = axes[0, 0]
    ax.plot(step, data["train_psnr"], label="train (8 memorised)", color="#2563eb", lw=2)
    ax.plot(step, data["val_psnr"], label="held-out (8 unseen)", color="#dc2626", lw=2)
    ax.axhline(TARGET_DB, ls="--", color="#16a34a", lw=1.5, label=f"gate {TARGET_DB:.0f} dB")
    ax.axhline(21.67, ls=":", color="#6b7280", lw=1.2, label="bicubic 21.67 dB")
    ax.set_xlabel("step"), ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR — gate never approached")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(step, data["train_loss"], label="train", color="#2563eb", lw=2)
    ax.semilogy(step, data["val_loss"], label="held-out", color="#dc2626", lw=2)
    ax.set_xlabel("step"), ax.set_ylabel("Charbonnier loss (log)")
    ax.set_title("Loss — clean memorisation signature")
    ax.legend(fontsize=8), ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.plot(step, data["train_ssim"], label="train", color="#2563eb", lw=2)
    ax.plot(step, data["val_ssim"], label="held-out", color="#dc2626", lw=2)
    ax.set_xlabel("step"), ax.set_ylabel("SSIM")
    ax.set_title("SSIM — train 0.93, held-out collapsing")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    ax = axes[1, 1]
    mask = step >= 100
    logs, psnr = np.log10(step[mask]), data["train_psnr"][mask]
    slope, intercept = np.polyfit(logs, psnr, 1)
    span = np.linspace(np.log10(100), np.log10(1e8), 200)
    ax.plot(logs, psnr, color="#2563eb", lw=2, label="measured train PSNR")
    ax.plot(span, slope * span + intercept, ls="--", color="#f59e0b", lw=1.5,
            label=f"fit: {slope:.2f} dB / decade")
    ax.axhline(TARGET_DB, ls="--", color="#16a34a", lw=1.5, label="gate 45 dB")
    needed = 10 ** ((TARGET_DB - intercept) / slope)
    ax.set_xlabel("log10(step)"), ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Extrapolation: 45 dB needs ~{needed:.1e} steps")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"slope = {slope:.3f} dB/decade | steps for 45 dB = {needed:.3e}")
    return out_path


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Plot the overfit gate.")
    parser.add_argument("--trace", type=Path, default=Path("outputs/overfit_gate_trace.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/figures/overfit_gate.png"))
    parser.add_argument("--title", default="SPARC-Tiny overfit gate — 8 images, 2000 steps (FAILED)")
    args = parser.parse_args()
    path = render(load_trace(args.trace), args.out, args.title)
    print(f"Figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
