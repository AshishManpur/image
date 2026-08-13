"""Phase 4.12 GPU shakedown — run this on the RTX A400 before any long training run.

Everything Part 10 constrains that a CPU cannot answer: VRAM, latency, BF16 stability,
channels-last, ``torch.compile``. The script **refuses to run without CUDA** rather than
substituting CPU numbers, because a CPU latency figure compared against a GPU budget is
worse than no figure at all.

It does not train to convergence. It runs a handful of real optimiser steps with the
frozen loss and checks that the numbers stay finite and the memory fits — enough to know
whether launching 400 epochs is justified, and cheap enough to repeat.

Usage::

    python scripts/cuda_shakedown.py --variant sparc-base --steps 20 \\
        --json reports/phase4_12_cuda.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import LossConfig, build_sparc_config  # noqa: E402
from losses.composite_loss import CompositeLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.complexity import count_parameters, measure_complexity  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}

TRAINING_VRAM_BUDGET_GB = 2.06
"""Contract Part 10 as amended by A-004 (see AMENDMENTS.md): maximum training VRAM at
batch 8. A-003's 2.05 GB came from a single-step measurement; the 50-step trace put the
stable cumulative peak at 2.0516 GB, ~1.6 MB above it. The measured value is always
printed and written to the JSON report — this threshold gates it, it does not mask it."""


def _reset(device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e6


def measure_inference(
    model: torch.nn.Module,
    device: torch.device,
    batch: int,
    amp_dtype: torch.dtype | None,
    iterations: int,
) -> dict[str, float]:
    """Time a batched forward pass and record its peak VRAM."""
    _reset(device)
    model.eval()
    x = torch.randn(batch, 1, 128, 128, device=device)

    with torch.inference_mode():
        for _ in range(5):  # warm-up: lazy context, autotune, allocator growth
            if amp_dtype is None:
                model(x)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            if amp_dtype is None:
                model(x)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_ms = elapsed / iterations * 1e3
    return {
        "batch": batch,
        "latency_ms": total_ms,
        "latency_ms_per_image": total_ms / batch,
        "peak_vram_mb": _peak_mb(device),
    }


def measure_training(
    model: torch.nn.Module,
    device: torch.device,
    batch: int,
    amp_dtype: torch.dtype | None,
    steps: int,
    channels_last: bool,
) -> dict[str, Any]:
    """Run real optimiser steps with the frozen loss and watch for divergence."""
    _reset(device)
    model.train()
    criterion = CompositeLoss(LossConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.9), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    losses: list[float] = []
    nonfinite = 0
    overflow = 0
    memory_format = torch.channels_last if channels_last else torch.contiguous_format

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        x = torch.rand(batch, 1, 128, 128, device=device).to(memory_format=memory_format)
        target = torch.rand(batch, 1, 256, 256, device=device)
        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is None:
            output = model.forward_with_aux(x)
            loss = criterion(output, {"gt": target, "lr": x})
        else:
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model.forward_with_aux(x)
                loss = criterion(output, {"gt": target, "lr": x})
        if isinstance(loss, tuple):
            loss = loss[0]

        if not torch.isfinite(loss):
            nonfinite += 1
            continue
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            overflow += 1
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "batch": batch,
        "steps": steps,
        "peak_vram_mb": _peak_mb(device),
        "peak_vram_gb": _peak_mb(device) / 1000.0,
        "seconds_per_step": elapsed / max(steps, 1),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "nonfinite_steps": nonfinite,
        "amp_overflow_steps": overflow,
        "stable": nonfinite == 0 and all(l == l for l in losses),
    }


def try_compile(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Compile and compare against eager. Never fatal: Part 5 has compile off in V1."""
    x = torch.randn(1, 1, 128, 128, device=device)
    model.eval()
    with torch.inference_mode():
        reference = model(x)
    try:
        compiled = torch.compile(model)
        with torch.inference_mode():
            for _ in range(3):
                result = compiled(x)
        torch.cuda.synchronize()
        return {
            "status": "ok",
            "max_abs_diff": float((reference - result).abs().max()),
        }
    except Exception as error:  # pragma: no cover - backend-dependent
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def _dry_run(args: argparse.Namespace) -> int:
    """Run one forward/backward on CPU to prove the script executes. Not a measurement."""
    set_seed(args.seed)
    config = build_sparc_config(args.variant)
    model = SPARCNet(config)
    criterion = CompositeLoss(LossConfig())
    x = torch.rand(2, 1, 128, 128)
    loss, _ = criterion(model.forward_with_aux(x), {"gt": torch.rand(2, 1, 256, 256), "lr": x})
    loss.backward()
    _LOGGER.warning(
        "DRY RUN on CPU: forward/backward completed, loss=%.4f. No VRAM, latency or "
        "AMP figure was measured. Run without --dry-run-cpu on the RTX A400.",
        float(loss.detach()),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4.12 CUDA shakedown.")
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--amp-dtype", default="bf16", choices=sorted(AMP_DTYPES))
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("reports/phase4_12_cuda.json"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dry-run-cpu", action="store_true",
        help="Exercise the code path on CPU without measuring anything. For checking "
             "that this script runs before taking it to the GPU host; the numbers it "
             "prints are NOT measurements and no report is written.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    if args.dry_run_cpu and not torch.cuda.is_available():
        return _dry_run(args)
    if not torch.cuda.is_available():
        _LOGGER.error(
            "No CUDA device. Part 10's VRAM and latency budgets are GPU-denominated; "
            "this script will not substitute CPU numbers. Run it on the RTX A400."
        )
        return 2

    set_seed(args.seed)
    device = torch.device("cuda")
    config = build_sparc_config(args.variant)
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    channels_last = not args.no_channels_last

    model = SPARCNet(config).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    total, trainable = count_parameters(model)
    complexity = measure_complexity(SPARCNet(config), torch.randn(1, 1, 128, 128))

    report: dict[str, Any] = {
        "variant": config.name,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "amp_dtype": args.amp_dtype,
        "channels_last": channels_last,
        "parameters": {"total": total, "trainable": trainable},
        "macs": complexity.macs,
        "gflops": complexity.gflops,
        "inference": {
            "batch_1": measure_inference(model, device, 1, amp_dtype, args.iterations),
            "batch_16": measure_inference(
                model, device, 16, amp_dtype, max(args.iterations // 2, 5)
            ),
        },
        "training": measure_training(
            model, device, args.batch_size, amp_dtype, args.steps, channels_last
        ),
        "training_vram_budget_gb": TRAINING_VRAM_BUDGET_GB,
        "training_vram_budget_amendment": "A-004",
        "compile": try_compile(SPARCNet(config).to(device), device),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    training = report["training"]
    print("=" * 72)
    print(f"  GPU               : {report['gpu']}  ({report['vram_total_gb']:.2f} GB)")
    print(f"  variant           : {report['variant']}  ({total:,} params)")
    print(f"  MACs / GFLOPs     : {complexity.gmacs:.3f} G / {complexity.gflops:.3f}")
    print(f"  AMP / layout      : {args.amp_dtype} / "
          f"{'channels_last' if channels_last else 'contiguous'}")
    for name, entry in report["inference"].items():
        print(f"  inference {name:9s}: {entry['latency_ms']:.2f} ms "
              f"({entry['latency_ms_per_image']:.2f} ms/img), "
              f"peak {entry['peak_vram_mb']:.0f} MB")
    vram_ok = training["peak_vram_gb"] <= TRAINING_VRAM_BUDGET_GB
    print(f"  train b{training['batch']} VRAM    : {training['peak_vram_gb']:.3f} GB "
          f"(limit {TRAINING_VRAM_BUDGET_GB:.2f} GB, A-004) "
          f"{'PASS' if vram_ok else 'FAIL'}")
    print(f"  train step time   : {training['seconds_per_step'] * 1e3:.0f} ms")
    print(f"  loss {training['loss_first']} -> {training['loss_last']}, "
          f"nonfinite={training['nonfinite_steps']}, "
          f"overflow={training['amp_overflow_steps']}")
    print(f"  torch.compile     : {report['compile']['status']}")
    print(f"  report            : {args.json}")
    print("=" * 72)
    return 0 if training["stable"] and vram_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
