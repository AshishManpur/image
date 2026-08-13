"""Benchmark SPARC-Net against every Contract Part 10 budget (Phase 4.14).

Prepared during Phase 4.7; **runnable once Phase 4.13 lands** (the full Base model
cannot be constructed until the noise head, fusion and attention modules exist).

The script is deliberately honest about what the host can and cannot measure. Contract
Part 10 states several budgets in GPU terms — training VRAM at batch 8, inference VRAM
at batch 16, and latency limits benchmarked on the target card. On a CPU-only host
those are **not** measurable, and reporting CPU numbers against a GPU budget would be
worse than reporting nothing. Every such row is emitted as ``UNMEASURED (no CUDA)``
rather than silently substituted.

Target deployment card: **NVIDIA RTX A400, 4 GB VRAM**.

Usage::

    python scripts/benchmark.py                       # full report, current device
    python scripts/benchmark.py --device cuda          # target-card run
    python scripts/benchmark.py --variant sparc-tiny  # step-6 model
    python scripts/benchmark.py --json reports/benchmark.json
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import build_sparc_config  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.complexity import count_parameters, measure_complexity, parameter_table  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.profiling import benchmark_latency  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

UNMEASURED = "UNMEASURED (no CUDA)"

# ----------------------------------------------------------------- Part 10 budgets
CONTRACT_PARAMS = 2_345_650
CONTRACT_MACS = 2_449_370_000

BUDGETS: dict[str, tuple[float, str]] = {
    "params": (2.60e6, "Maximum parameters"),
    "macs": (2.80e9, "Maximum MACs (128^2 -> 256^2)"),
    "gflops": (5.60, "Maximum GFLOPs"),
    "train_vram_gb": (2.06, "Maximum training VRAM @ batch 8"),
    "infer_vram_gb": (1.50, "Maximum inference VRAM @ batch 16"),
    "latency_b1_ms": (35.0, "Maximum inference latency (batch 1)"),
    "latency_b16_ms_per_image": (10.0, "Maximum inference latency (batch 16)"),
    "disk_mb": (12.0, "Maximum model size on disk (fp32)"),
}

"""Part 10 limits. ``train_vram_gb`` is 2.06 GB per **A-004** (see AMENDMENTS.md); this
table was never updated for A-003 and still read the original 2.00 GB until Phase 4.12's
GPU validation, which is erratum M-2 below."""

TOLERANCES = {"params": 0.02, "macs": 0.05}
"""Contract Part 8 acceptance: params +/-2 %, MACs +/-5 % of the Part 3 table."""

VRAM_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


@dataclass
class BudgetRow:
    """One budget line: measured value, limit, and verdict."""

    name: str
    description: str
    measured: float | str
    limit: float
    unit: str
    passed: bool | None

    def render(self) -> str:
        if isinstance(self.measured, str):
            return f"  {self.description:44s} {self.measured:>22s}  (limit {self.limit:g} {self.unit})"
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[self.passed]
        return (
            f"  {self.description:44s} {self.measured:14.3f} {self.unit:<7s} "
            f"(limit {self.limit:g}) {verdict}"
        )


def measure_disk_size_mb(model: torch.nn.Module) -> float:
    """Serialised fp32 size in MB — the actual bytes ``torch.save`` writes.

    Measured by serialising rather than by summing tensor bytes. Since Phase 4.12 the
    two differ substantially: each ``GSABlock`` holds an ``(N, N)`` int64
    relative-position **index** buffer (8 MB at N=1024), registered non-persistent
    because it is a derived constant rebuilt at construction. Summing
    ``model.buffers()`` counts 26.7 MB that never reach the file and would report the
    checkpoint at 36 MB against a 12 MB budget it in fact meets at 9.6 MB. The resident
    cost of those buffers is real and is reported separately as ``buffer_mb``.

    Args:
        model: Model to size.

    Returns:
        Size in megabytes.
    """
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.tell() / 1e6


def measure_resident_buffer_mb(model: torch.nn.Module) -> float:
    """Bytes held by non-parameter buffers at run time, in MB.

    Not a Part 10 budget line, but it occupies VRAM, so it is reported.
    """
    return sum(b.numel() * b.element_size() for b in model.buffers()) / 1e6


def measure_training_vram_gb(
    model: torch.nn.Module, batch_size: int, device: torch.device, amp_dtype: str
) -> float | str:
    """Peak allocated VRAM for one training step, in GB.

    Runs a genuine forward/backward/optimiser step so that retained activations are
    included — the dominant term in Contract Part 4's 140.7 MB/image figure.

    The probe reproduces the **frozen Part 5 training configuration**: bf16 autocast,
    ``channels_last``, the real ``CompositeLoss`` over ``forward_with_aux``, and an
    AdamW step. Anything else measures a configuration nobody trains in — see erratum
    M-2. This is deliberately the same methodology as
    ``tests/test_full_model_training.py::test_training_vram_at_batch_eight_is_within_budget``
    and ``scripts/cuda_shakedown.py``, so the three agree instead of contradicting.

    Args:
        model: Model to profile.
        batch_size: Training batch size.
        device: Device; must be CUDA for a meaningful measurement.
        amp_dtype: One of ``bf16`` (frozen default), ``fp16`` or ``fp32``.

    Returns:
        Peak GB, or the ``UNMEASURED`` sentinel on a non-CUDA device.
    """
    if device.type != "cuda":
        return UNMEASURED

    from losses.composite_loss import CompositeLoss  # local: CPU hosts never reach here

    dtype = VRAM_AMP_DTYPES[amp_dtype]
    model = model.to(device).to(memory_format=torch.channels_last).train()
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sample = torch.rand(batch_size, 1, 128, 128, device=device).to(
        memory_format=torch.channels_last
    )
    target = torch.rand(batch_size, 1, 256, 256, device=device)

    with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
        loss, _ = criterion(
            model.forward_with_aux(sample), {"gt": target, "lr": sample}
        )
    loss.backward()
    optimizer.step()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return peak


def build_report(
    variant: str, device: torch.device, amp_dtype: str, iterations: int
) -> dict[str, Any]:
    """Measure everything Part 10 constrains.

    Args:
        variant: SPARC variant name.
        device: Device to benchmark on.
        amp_dtype: Autocast dtype for the training-VRAM probe (``bf16`` frozen default).
        iterations: Timed iterations per latency measurement.

    Returns:
        A JSON-serialisable report.
    """
    config = build_sparc_config(variant)
    model = SPARCNet(config)
    total, trainable = count_parameters(model)

    complexity = measure_complexity(model, torch.randn(1, 1, 128, 128))
    disk_mb = measure_disk_size_mb(model)
    buffer_mb = measure_resident_buffer_mb(model)

    latency_b1 = benchmark_latency(
        model, (1, 128, 128), batch_size=1, iterations=iterations, device=device
    )
    latency_b16 = benchmark_latency(
        model, (1, 128, 128), batch_size=16, iterations=max(iterations // 4, 5), device=device
    )

    # `benchmark_latency` moved `model` onto the device in place and it is still
    # resident. `max_memory_allocated` is a device-global counter, so leaving it there
    # would charge the training probe for a model it is not training.
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_vram = measure_training_vram_gb(SPARCNet(config), 8, device, amp_dtype)
    infer_vram = (
        latency_b16.peak_memory_mb / 1000.0 if device.type == "cuda" else UNMEASURED
    )

    rows = [
        BudgetRow("params", BUDGETS["params"][1], total, BUDGETS["params"][0], "",
                  total <= BUDGETS["params"][0]),
        BudgetRow("macs", BUDGETS["macs"][1], complexity.macs, BUDGETS["macs"][0], "",
                  complexity.macs <= BUDGETS["macs"][0]),
        BudgetRow("gflops", BUDGETS["gflops"][1], complexity.gflops, BUDGETS["gflops"][0],
                  "GFLOP", complexity.gflops <= BUDGETS["gflops"][0]),
        BudgetRow("disk_mb", BUDGETS["disk_mb"][1], disk_mb, BUDGETS["disk_mb"][0], "MB",
                  disk_mb <= BUDGETS["disk_mb"][0]),
        BudgetRow(
            "train_vram_gb", BUDGETS["train_vram_gb"][1], train_vram,
            BUDGETS["train_vram_gb"][0], "GB",
            None if isinstance(train_vram, str) else train_vram <= BUDGETS["train_vram_gb"][0],
        ),
        BudgetRow(
            "infer_vram_gb", BUDGETS["infer_vram_gb"][1], infer_vram,
            BUDGETS["infer_vram_gb"][0], "GB",
            None if isinstance(infer_vram, str) else infer_vram <= BUDGETS["infer_vram_gb"][0],
        ),
        BudgetRow(
            "latency_b1_ms", BUDGETS["latency_b1_ms"][1],
            latency_b1.mean_ms if device.type == "cuda" else UNMEASURED,
            BUDGETS["latency_b1_ms"][0], "ms",
            latency_b1.mean_ms <= BUDGETS["latency_b1_ms"][0] if device.type == "cuda" else None,
        ),
        BudgetRow(
            "latency_b16_ms_per_image", BUDGETS["latency_b16_ms_per_image"][1],
            latency_b16.mean_ms / 16.0 if device.type == "cuda" else UNMEASURED,
            BUDGETS["latency_b16_ms_per_image"][0], "ms",
            (latency_b16.mean_ms / 16.0) <= BUDGETS["latency_b16_ms_per_image"][0]
            if device.type == "cuda" else None,
        ),
    ]

    param_delta = (total - CONTRACT_PARAMS) / CONTRACT_PARAMS
    mac_delta = (complexity.macs - CONTRACT_MACS) / CONTRACT_MACS

    return {
        "variant": config.name,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "parameters": {
            "total": total,
            "trainable": trainable,
            "contract": CONTRACT_PARAMS,
            "delta_fraction": param_delta,
            "within_tolerance": abs(param_delta) <= TOLERANCES["params"],
            "per_module": parameter_table(model, depth=1),
        },
        "resident_buffer_mb": buffer_mb,
        "macs": {
            "measured": complexity.macs,
            "contract": CONTRACT_MACS,
            "delta_fraction": mac_delta,
            "within_tolerance": abs(mac_delta) <= TOLERANCES["macs"],
        },
        "latency": {
            "batch_1": asdict(latency_b1),
            "batch_16": asdict(latency_b16),
            "dtype": "fp32",
            "memory_format": "contiguous",
            "note": None if device.type == "cuda" else "CPU timings; not comparable to Part 10.",
        },
        "train_vram_probe": {
            "amp_dtype": amp_dtype,
            "memory_format": "channels_last",
            "loss": "CompositeLoss",
            "optimizer_step": True,
            "budget_gb": BUDGETS["train_vram_gb"][0],
            "budget_amendment": "A-004",
        },
        "budgets": [asdict(row) for row in rows],
        "all_measured_budgets_pass": all(r.passed for r in rows if r.passed is not None),
        "_rows": rows,
    }


def main() -> int:
    """Entry point.

    Returns:
        0 if every *measured* budget passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Benchmark SPARC-Net against Part 10.")
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--vram-amp-dtype", choices=sorted(VRAM_AMP_DTYPES), default="bf16",
        help="Autocast dtype for the training-VRAM probe. Default bf16 = the frozen "
             "Part 5 training configuration, which is what the budget denominates.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Deprecated no-op: the VRAM probe now always runs under --vram-amp-dtype "
             "(bf16 by default). Kept so existing invocations do not break.",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device)

    if device.type != "cuda":
        _LOGGER.warning(
            "Running on %s. Part 10's VRAM and latency budgets are GPU-denominated "
            "and will be reported as UNMEASURED.", device
        )

    if args.amp:
        _LOGGER.warning(
            "--amp is a deprecated no-op; the VRAM probe runs under --vram-amp-dtype "
            "(%s).", args.vram_amp_dtype
        )
    report = build_report(args.variant, device, args.vram_amp_dtype, args.iterations)
    rows: list[BudgetRow] = report.pop("_rows")

    print(f"\nSPARC benchmark — {report['variant']} on {report['device']}")
    print(f"  GPU: {report['gpu_name'] or 'none'}   torch {report['torch']}\n")
    print("Contract Part 8 acceptance:")
    params, macs = report["parameters"], report["macs"]
    print(f"  parameters {params['total']:>10,d}  vs contract {params['contract']:>10,d}  "
          f"({params['delta_fraction']:+.2%})  {'PASS' if params['within_tolerance'] else 'FAIL'} (+/-2 %)")
    print(f"  MACs       {macs['measured']:>10,d}  vs contract {macs['contract']:>10,d}  "
          f"({macs['delta_fraction']:+.2%})  {'PASS' if macs['within_tolerance'] else 'FAIL'} (+/-5 %)")
    print("\nContract Part 10 budgets:")
    for row in rows:
        print(row.render())

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _LOGGER.info("Report written to %s", args.json)

    return 0 if report["all_measured_budgets_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
