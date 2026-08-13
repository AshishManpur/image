"""Per-step VRAM trace (Phase 4.12, post-A-003 reconciliation).

``scripts/vram_reconcile.py`` isolated the ~56 MB gap between the single-step gate
measurement (1.9964 GB) and the 20-step shakedown (2.0526 GB) to the number of
iterations, not to gradient clipping, ``GradScaler``, or peak-reset timing — those
three together only account for +10.45 MB (1.9964 -> 2.0069 GB); going from 1 step to
20 steps in an otherwise-identical loop accounts for the other +45.8 MB
(2.0069 -> 2.0526 GB) on its own.

"Repeated iterations cost more" has two very different explanations that look
identical from the outside (a single peak-VRAM number) but are opposite in severity:

* **Allocator plateau (benign).** The caching allocator does not defragment
  proactively. Early steps may need to request a handful of block sizes it has not
  seen before (first bf16 activation of a given shape, first workspace for a given
  cuDNN call); once every size class has been requested once, later steps reuse the
  same blocks and the peak stops moving. This is real but *bounded* — the peak
  converges within the first few steps and does not grow further no matter how many
  more steps run.
* **Reference-cycle accumulation (a real, if subtle, leak).** Autograd graphs contain
  reference cycles (a `Node` and the tensors it saved for backward can each hold a
  reference back to the other). CPython's refcounting frees acyclic garbage
  immediately, but cyclic garbage needs the generational collector, which only runs
  automatically after ~700 new container allocations (`gc.get_threshold()`). Over a
  short 20-step loop that threshold may not trigger even once, so graph objects from
  steps 1..19 can still be *reachable* — not used, but not yet collected — when step
  20 hits its peak. This looks identical to a "leak" in a single peak-VRAM reading but
  is not: forcing `gc.collect()` reclaims it immediately, and unlike a real leak it
  does not grow without bound over a long run.

This script tells the two apart by recording, every step for >= 50 steps, both the
persistent live-tensor count (`memory_allocated`, read after each step settles — this
is what a genuine leak would grow) and the transient step-local peak
(`max_memory_allocated`, reset before each step so it measures *that step's* peak, not
a running one). It then repeats the run with `gc.collect()` forced every step to see
whether the plateau drops and stays down (allocator + gc-cycle explanation) or is
unaffected (something else is genuinely retaining tensors).

Run on the A400::

    python scripts/vram_step_trace.py --steps 50 --json reports/phase4_12_step_trace.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import LossConfig, sparc_base  # noqa: E402
from losses.composite_loss import CompositeLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.seed import set_seed  # noqa: E402

MB = 1e6


def trace(steps: int, batch: int, force_gc: bool) -> list[dict[str, float]]:
    """Run ``steps`` real training iterations, recording memory every step.

    Mirrors ``trainer.py``'s per-step operations exactly: zero_grad(set_to_none=True),
    autocast bf16 forward, GradScaler-wrapped backward, unscale, clip_grad_norm_(1.0),
    scaler step/update. Everything up to but not including the dataloader, EMA and
    logging, which the reconciliation already showed are not where the gap is.
    """
    set_seed(1337)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = SPARCNet(sparc_base()).to(device).to(memory_format=torch.channels_last)
    model.train()
    criterion = CompositeLoss(LossConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.9), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    rows: list[dict[str, float]] = []
    cumulative_peak = 0.0

    for step in range(steps):
        torch.cuda.reset_peak_memory_stats(device)

        x = torch.rand(batch, 1, 128, 128, device=device).to(memory_format=torch.channels_last)
        gt = torch.rand(batch, 1, 256, 256, device=device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_with_aux(x)
            loss, _ = criterion(output, {"gt": gt, "lr": x})

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Drop the loop's only references to this step's graph and inputs before the
        # measurement, exactly as `trainer.py` does by reassigning `x`/`gt`/`loss` next
        # iteration — the point is to see what's reachable *without* help, then what
        # forcing collection changes.
        del x, gt, loss, output
        if force_gc:
            gc.collect()

        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated(device) / MB
        reserved_mb = torch.cuda.memory_reserved(device) / MB
        step_peak_mb = torch.cuda.max_memory_allocated(device) / MB
        cumulative_peak = max(cumulative_peak, step_peak_mb)

        rows.append({
            "step": step,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "step_peak_mb": step_peak_mb,
            "cumulative_peak_mb": cumulative_peak,
        })

    del model, criterion, optimizer
    torch.cuda.empty_cache()
    return rows


def _linear_slope(values: list[float]) -> float:
    """MB/step trend via simple least squares."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs) or 1.0
    return num / den


def classify(rows: list[dict[str, float]]) -> str:
    """A: rises then plateaus. B: grows continuously. C: oscillates with a ceiling."""
    peaks = [r["step_peak_mb"] for r in rows]
    n = len(peaks)
    back_half = peaks[n // 2:]
    slope_per_step = _linear_slope(back_half)

    if slope_per_step > 0.5:  # still climbing > 0.5 MB/step deep into the run
        return "B (increases continuously)"

    first_quarter_max = max(peaks[: max(n // 4, 1)])
    overall_max = max(peaks)
    if overall_max - first_quarter_max > 5.0:
        # The ceiling wasn't reached until later than the first quarter, but the back
        # half is flat: it climbed once, then stopped.
        return "A (rises then plateaus)"
    return "C (oscillates, stable ceiling)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-step VRAM trace.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_12_step_trace.json"
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("No CUDA device. Run this on the RTX A400.", file=sys.stderr)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n")

    print(f"== pass 1: no forced gc (matches trainer.py exactly) ==")
    rows_no_gc = trace(args.steps, args.batch_size, force_gc=False)
    print(f"{'step':>4s} {'allocated_MB':>13s} {'reserved_MB':>12s} "
          f"{'step_peak_MB':>13s} {'cumulative_peak_MB':>19s}")
    for r in rows_no_gc:
        print(f"{r['step']:>4d} {r['allocated_mb']:13.2f} {r['reserved_mb']:12.2f} "
              f"{r['step_peak_mb']:13.2f} {r['cumulative_peak_mb']:19.2f}")

    verdict_no_gc = classify(rows_no_gc)
    print(f"\npattern (no forced gc): {verdict_no_gc}")
    print(f"final cumulative peak: {rows_no_gc[-1]['cumulative_peak_mb'] / 1000:.4f} GB")

    print(f"\n== pass 2: gc.collect() forced every step (diagnosis only, not applied "
          f"to trainer.py) ==")
    rows_gc = trace(args.steps, args.batch_size, force_gc=True)
    verdict_gc = classify(rows_gc)
    print(f"pattern (forced gc): {verdict_gc}")
    print(f"final cumulative peak: {rows_gc[-1]['cumulative_peak_mb'] / 1000:.4f} GB")

    delta_gb = (
        rows_no_gc[-1]["cumulative_peak_mb"] - rows_gc[-1]["cumulative_peak_mb"]
    ) / 1000.0
    print(f"\nforced-gc peak reduction: {delta_gb * 1000:+.2f} MB")
    if abs(delta_gb) * 1000 > 20:
        print(
            "  -> forcing gc.collect() materially lowers the peak: consistent with "
            "reference-cycle accumulation in the autograd graph, not a genuine leak. "
            "The plateau itself (not the gc-collected floor) is still the number that "
            "matters for the real, gc-untouched trainer.py loop."
        )
    else:
        print(
            "  -> forcing gc.collect() does not materially change the peak: the "
            "growth is not explained by uncollected reference cycles. Needs deeper "
            "tracing (retained tensors via gc.get_objects(), or a real leak in "
            "trainer.py-adjacent state)."
        )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "no_forced_gc": {"rows": rows_no_gc, "pattern": verdict_no_gc},
        "forced_gc": {"rows": rows_gc, "pattern": verdict_gc},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
