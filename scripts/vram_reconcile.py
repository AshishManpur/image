"""Reconcile the two batch-8 training-VRAM measurements (Phase 4.12, post-A-003).

Two scripts measure the same quantity and disagree:

    scripts/vram_profile.py            (== the Part 10 gate / pytest test methodology)
        peak = 1.9978 GB

    scripts/cuda_shakedown.py          (== trainer.py's real per-step loop, x20)
        peak = 2.051 GB

The gap is ~53 MB. Before touching anything, this script isolates *which*
methodological difference between the two accounts for it. There are four candidates,
each toggled independently against a common baseline that otherwise matches
``cuda_shakedown.measure_training`` exactly:

* **grad clipping**   — ``cuda_shakedown``/``trainer.py`` call
  ``torch.nn.utils.clip_grad_norm_`` every step; ``vram_profile``/the pytest gate do
  not. Real training always clips (``Config.grad_clip_norm``), so if this is the cause
  it is not an artifact — the gate is under-measuring the real loop.
* **GradScaler wrapper** — ``cuda_shakedown``/``trainer.py`` route the step through
  ``torch.amp.GradScaler`` (a no-op under bf16, but not a zero-cost no-op: ``.scale``,
  ``.unscale_``, ``.step``, ``.update`` are still real Python/dispatcher calls that may
  each transiently touch the allocator).
* **multi-step (x20) vs single-step, one reset** — ``cuda_shakedown`` resets peak
  stats once and runs 20 steps in that window; ``vram_profile``/the gate measure one
  step. If the true per-step peak creeps across iterations (fragmentation, lazy
  workspace growth) rather than staying flat, only the multi-step run would see it.
* **reset timing** — ``cuda_shakedown`` resets peak stats *before* building the
  criterion and optimizer object; the gate builds them first, then resets. Negligible
  in isolation (both are ~0 MB before ``.step()`` is ever called) but included for
  completeness.

Run on the A400::

    python scripts/vram_reconcile.py --json reports/phase4_12_reconcile.json
"""

from __future__ import annotations

import argparse
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
BUDGET_GB = 2.06
"""Contract Part 10, as amended by A-004. Unchanged by this diagnostic — this script's
2.0526 GB shakedown-methodology reading is one of the three measurements A-004 cites."""


def _peak_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / MB


def _reserved_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_reserved(device) / MB


def run(
    *,
    steps: int,
    clip_grad: bool,
    use_scaler: bool,
    reset_before_optimizer: bool,
    batch: int = 8,
) -> dict[str, float]:
    """One configuration of the training loop, peak measured across ``steps`` steps.

    Mirrors ``cuda_shakedown.measure_training`` exactly except for the four toggles
    under test, so a run with all four set to match ``cuda_shakedown``'s defaults
    (``steps=20, clip_grad=True, use_scaler=True, reset_before_optimizer=True``)
    should reproduce its 2.051 GB, and one set to match the gate's defaults
    (``steps=1, clip_grad=False, use_scaler=False, reset_before_optimizer=False``)
    should reproduce ``vram_profile``'s 1.9978 GB.
    """
    set_seed(1337)
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    model = SPARCNet(sparc_base()).to(device).to(memory_format=torch.channels_last)
    model.train()

    if reset_before_optimizer:
        torch.cuda.reset_peak_memory_stats(device)
        criterion = CompositeLoss(LossConfig()).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.9), eps=1e-8
        )
    else:
        criterion = CompositeLoss(LossConfig()).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.9), eps=1e-8
        )
        torch.cuda.reset_peak_memory_stats(device)

    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16: always a no-op arm

    for _ in range(steps):
        x = torch.rand(batch, 1, 128, 128, device=device).to(
            memory_format=torch.channels_last
        )
        gt = torch.rand(batch, 1, 256, 256, device=device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = criterion(model.forward_with_aux(x), {"gt": gt, "lr": x})

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    torch.cuda.synchronize()
    peak_mb = _peak_mb(device)
    reserved_mb = _reserved_mb(device)

    del model, criterion, optimizer, x, gt, loss
    torch.cuda.empty_cache()

    return {
        "steps": steps,
        "clip_grad": clip_grad,
        "use_scaler": use_scaler,
        "reset_before_optimizer": reset_before_optimizer,
        "peak_mb": peak_mb,
        "peak_gb": peak_mb / 1000.0,
        "reserved_mb": reserved_mb,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile the batch-8 VRAM gap.")
    parser.add_argument(
        "--json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_12_reconcile.json"
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("No CUDA device. Run this on the RTX A400.", file=sys.stderr)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n")

    gate_defaults = dict(steps=1, clip_grad=False, use_scaler=False, reset_before_optimizer=False)
    shakedown_defaults = dict(steps=20, clip_grad=True, use_scaler=True, reset_before_optimizer=True)

    arms: dict[str, dict[str, float]] = {}
    arms["gate_methodology"] = run(**gate_defaults)
    arms["shakedown_methodology"] = run(**shakedown_defaults)

    # Single-toggle arms, each starting from the gate baseline and flipping one knob
    # toward the shakedown's setting, isolating that knob's individual contribution.
    arms["gate+clip_grad"] = run(**{**gate_defaults, "clip_grad": True})
    arms["gate+use_scaler"] = run(**{**gate_defaults, "use_scaler": True})
    arms["gate+reset_before_optimizer"] = run(
        **{**gate_defaults, "reset_before_optimizer": True}
    )
    arms["gate+steps20"] = run(**{**gate_defaults, "steps": 20})

    # And the reverse: start from shakedown, remove one knob at a time.
    arms["shakedown-clip_grad"] = run(**{**shakedown_defaults, "clip_grad": False})
    arms["shakedown-use_scaler"] = run(**{**shakedown_defaults, "use_scaler": False})
    arms["shakedown-reset_before_optimizer"] = run(
        **{**shakedown_defaults, "reset_before_optimizer": False}
    )
    arms["shakedown-steps1"] = run(**{**shakedown_defaults, "steps": 1})

    baseline_gb = arms["gate_methodology"]["peak_gb"]
    print(f"{'arm':32s} {'peak GB':>9s}  {'Δ vs gate':>10s}  {'reserved GB':>12s}")
    for name, arm in arms.items():
        delta_mb = arm["peak_mb"] - arms["gate_methodology"]["peak_mb"]
        print(
            f"{name:32s} {arm['peak_gb']:9.4f}  {delta_mb:+9.2f} MB "
            f" {arm['reserved_mb'] / 1000.0:11.4f}"
        )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "budget_gb": BUDGET_GB,
        "arms": arms,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {args.json}")

    print(
        "\nReconciliation: compare gate_methodology (should read ~1.998 GB, matching "
        "vram_profile.py) against shakedown_methodology (should read ~2.051 GB, "
        "matching cuda_shakedown.py). The single-toggle arms show which knob(s) carry "
        "the ~53 MB gap between them: whichever gate+X arm moves furthest off "
        "gate_methodology's baseline (and whichever shakedown-X arm moves furthest "
        "off shakedown_methodology's) is the dominant cause. If gate+steps20 alone "
        "accounts for most of the gap, it's a real multi-step effect (creep across "
        "iterations) rather than a one-off single-step measurement being wrong. If "
        "gate+clip_grad accounts for most of it, the gate itself is under-measuring "
        "what trainer.py actually does every step, since trainer.py always clips."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
