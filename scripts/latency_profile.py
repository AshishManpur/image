"""Locate the source of the Part 10 inference-latency miss (Phase 4.12, open item §6.1c).

On the RTX A400 — the contract's own target card (Part 1) — both latency budgets are
missed, in both the deployment path and the benchmark's fp32 path::

    latency @ b1    35 ms limit    49.23 ms bf16 / 43.36 ms fp32
    latency @ b16   10 ms/img      17.96 ms/img bf16 / 23.89 ms/img fp32

Picking a different dtype does not rescue it, so before any threshold is proposed this
script separates *why*. Four hypotheses, each isolated as its own arm:

* **cuDNN determinism.** Every script that has ever reported a latency number calls
  ``utils.seed.set_seed``, which pins ``cudnn.deterministic = True`` and
  ``cudnn.benchmark = False``. For a conv-heavy network at a fixed input shape that is
  the slow configuration twice over: autotuning is disabled, so cuDNN cannot pick the
  best algorithm for these shapes, and the deterministic constraint excludes the fastest
  algorithms outright. **No deployment would run this way.** This is the leading
  hypothesis and it is a measurement-configuration defect, not an architecture one.
* **Eager execution.** ``torch.compile`` *fails* on the GPU host (no ``triton``, no
  ``cl``), so the model runs fully eager. Part 10's estimate may have assumed a compiled
  deployment path. This script reports exactly why compilation fails rather than
  recording "failed".
* **Launch-bound at batch 1.** bf16 is *slower* than fp32 at b1 (49.23 vs 43.36 ms) and
  faster at b16 (17.96 vs 23.89 ms/img). That inversion is the signature of a
  launch-latency-bound regime where autocast's casts are not amortised. Timing host wall
  time against CUDA-event device time separates dispatch overhead from real compute.
* **A genuinely slow stage.** If one stage dominates, the fix is local. The per-stage
  breakdown measures each contract stage (Part 1) on its own.

Nothing here changes the model, and nothing here changes a threshold. It produces the
evidence a latency decision needs — optimise, amend with numbers, or accept the miss.

Run on the A400::

    python scripts/latency_profile.py --json reports/phase4_12_latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import sparc_base  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

BUDGET_B1_MS = 35.0
BUDGET_B16_MS_PER_IMAGE = 10.0
"""Contract Part 10. **Not amended by this script** — it measures, it does not gate."""

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


# --------------------------------------------------------------- environment probes
def compile_environment() -> dict[str, Any]:
    """Report why ``torch.compile`` can or cannot work on this host.

    ``cuda_shakedown.py`` reports a bare ``failed``, which is not actionable. Inductor
    needs a Triton backend for GPU kernels and a host C++ compiler (``cl.exe`` on
    Windows) for the wrapper; either one missing produces the same unhelpful status.
    """
    info: dict[str, Any] = {
        "cl_on_path": shutil.which("cl") is not None,
        "cxx_on_path": shutil.which("g++") or shutil.which("clang++"),
    }
    try:
        import triton  # noqa: F401

        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - host-dependent
        info["triton"] = None
        info["triton_error"] = f"{type(exc).__name__}: {exc}"
    return info


def try_compile(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    """Compile and run once, capturing the real failure reason if it does not work."""
    try:
        compiled = torch.compile(model)
        with torch.inference_mode():
            compiled(sample)
        return {"status": "ok", "model": compiled}
    except Exception as exc:  # pragma: no cover - host-dependent
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "model": None,
        }


@contextmanager
def cudnn_settings(deterministic: bool, benchmark: bool) -> Iterator[None]:
    """Temporarily set the cuDNN algorithm-selection policy, restoring it after."""
    previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous


# ------------------------------------------------------------------------- timing
def time_forward(
    call: Callable[[], Any],
    device: torch.device,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Time a callable in both host wall time and CUDA device time.

    Returning both is the point: ``host_ms`` is what a caller experiences, ``device_ms``
    is what the GPU actually spent. A large ``host_ms - device_ms`` gap means the run is
    bound by Python and kernel-launch overhead, which compilation removes and a wider
    architecture does not.

    Args:
        call: Zero-argument callable performing one forward pass.
        device: Device being timed.
        iterations: Timed iterations.
        warmup: Untimed warm-up iterations. Must be enough for ``cudnn.benchmark``
            autotuning to settle, or the first timed call carries the search cost.

    Returns:
        Mean/median/p90 host ms, and mean device ms.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)

    host: list[float] = []
    device_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start = time.perf_counter()
        start_event.record()
        call()
        end_event.record()
        torch.cuda.synchronize(device)
        host.append((time.perf_counter() - start) * 1e3)
        device_ms.append(start_event.elapsed_time(end_event))

    host.sort()
    return {
        "host_ms": sum(host) / len(host),
        "host_median_ms": host[len(host) // 2],
        "host_p90_ms": host[int(0.9 * (len(host) - 1))],
        "device_ms": sum(device_ms) / len(device_ms),
        "launch_overhead_ms": sum(host) / len(host) - sum(device_ms) / len(device_ms),
    }


def measure_arm(
    model: SPARCNet,
    device: torch.device,
    batch: int,
    amp_dtype: str,
    channels_last: bool,
    deterministic: bool,
    benchmark: bool,
    iterations: int,
    warmup: int,
    compiled: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Measure one configuration at one batch size."""
    dtype = AMP_DTYPES[amp_dtype]
    target = compiled if compiled is not None else model
    sample = torch.rand(batch, 1, 128, 128, device=device)
    if channels_last:
        sample = sample.to(memory_format=torch.channels_last)
        target = target.to(memory_format=torch.channels_last)
    else:
        target = target.to(memory_format=torch.contiguous_format)

    def call() -> None:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=dtype or torch.float32, enabled=dtype is not None
        ):
            target(sample)

    with cudnn_settings(deterministic, benchmark):
        timing = time_forward(call, device, iterations, warmup)

    per_image = timing["host_ms"] / batch
    limit = BUDGET_B1_MS if batch == 1 else BUDGET_B16_MS_PER_IMAGE
    measured = timing["host_ms"] if batch == 1 else per_image
    return {
        "batch": batch,
        "amp_dtype": amp_dtype,
        "channels_last": channels_last,
        "cudnn_deterministic": deterministic,
        "cudnn_benchmark": benchmark,
        "compiled": compiled is not None,
        **timing,
        "ms_per_image": per_image,
        "budget_ms": limit,
        "within_budget": measured <= limit,
        "over_by_fraction": measured / limit - 1.0,
    }


def stage_breakdown(
    model: SPARCNet,
    device: torch.device,
    batch: int,
    amp_dtype: str,
    iterations: int,
) -> list[dict[str, Any]]:
    """Time each contract stage (Part 1) separately, in the deployment configuration.

    Mirrors ``SPARCNet.forward_with_aux`` in the same order rather than hooking, so a
    stage's cost is attributed to the name it has in the contract. Stage times are
    measured independently and will not sum exactly to the end-to-end figure — the
    question they answer is *which stage dominates*, not accounting to the microsecond.
    """
    dtype = AMP_DTYPES[amp_dtype]
    sample = torch.rand(batch, 1, 128, 128, device=device).to(
        memory_format=torch.channels_last
    )

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=dtype or torch.float32, enabled=dtype is not None
    ):
        y_hat, stats = model.normalizer(sample)
        features = y_hat
        if model.noise_head is not None:
            noise = model.noise_head(sample, stats.scale)
            features = torch.cat((y_hat, noise.sigma_map_normalized), dim=1)
        bottleneck, skips = model.encoder(features)
        decoded = model.decoder(bottleneck, skips)

    def timed(name: str, call: Callable[[], Any]) -> dict[str, Any]:
        def wrapped() -> None:
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                call()

        return {"stage": name, **time_forward(wrapped, device, iterations, warmup=5)}

    stages = [
        timed("01_normalizer", lambda: model.normalizer(sample)),
        timed("03_encoder", lambda: model.encoder(features)),
        timed("04_decoder", lambda: model.decoder(bottleneck, skips)),
        timed("05_head", lambda: model.head(decoded)),
    ]
    if model.noise_head is not None:
        stages.insert(
            1, timed("02_noise_head", lambda: model.noise_head(sample, stats.scale))
        )
    return stages


def main() -> int:
    """Entry point. Always returns 0: this is a diagnostic, not a gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Must cover cudnn.benchmark autotuning; 20 is not tight.")
    parser.add_argument("--json", type=Path, default=Path("reports/phase4_12_latency.json"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging()
    if not torch.cuda.is_available():
        print(
            "No CUDA device. Part 10's latency budgets are denominated on the RTX A400 "
            "and this script will not substitute CPU timings. Run it on the A400.",
            file=sys.stderr,
        )
        return 2

    set_seed(args.seed)  # deliberately: this is the configuration under investigation
    device = torch.device("cuda")
    model = SPARCNet(sparc_base()).to(device).eval()

    compiled = try_compile(
        SPARCNet(sparc_base()).to(device).eval(),
        torch.rand(1, 1, 128, 128, device=device),
    )
    compiled_model = compiled.pop("model")

    # Arms are ordered so each isolates one variable against the row above it.
    arms: list[dict[str, Any]] = []
    for batch in (1, 16):
        common = dict(iterations=args.iterations, warmup=args.warmup)
        arms.append({"arm": "baseline_benchmark_py", **measure_arm(
            model, device, batch, "fp32", False, True, False, **common)})
        arms.append({"arm": "deployment_path_shakedown", **measure_arm(
            model, device, batch, "bf16", True, True, False, **common)})
        arms.append({"arm": "cudnn_autotuned", **measure_arm(
            model, device, batch, "bf16", True, False, True, **common)})
        arms.append({"arm": "cudnn_autotuned_fp32", **measure_arm(
            model, device, batch, "fp32", False, False, True, **common)})
        arms.append({"arm": "fp16_autotuned", **measure_arm(
            model, device, batch, "fp16", True, False, True, **common)})
        if compiled_model is not None:
            arms.append({"arm": "compiled_autotuned", **measure_arm(
                model, device, batch, "bf16", True, False, True,
                compiled=compiled_model, **common)})

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "budgets": {
            "latency_b1_ms": BUDGET_B1_MS,
            "latency_b16_ms_per_image": BUDGET_B16_MS_PER_IMAGE,
            "note": "Part 10, unamended. This script measures; it does not gate.",
        },
        "compile_environment": compile_environment(),
        "compile": compiled,
        "arms": arms,
        "stage_breakdown_b1": stage_breakdown(model, device, 1, "bf16", args.iterations),
        "stage_breakdown_b16": stage_breakdown(model, device, 16, "bf16", max(args.iterations // 4, 5)),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"  GPU: {report['gpu']}  torch {report['torch']}")
    print(f"  torch.compile: {compiled['status']}"
          + (f" ({compiled.get('error_type')})" if compiled["status"] != "ok" else ""))
    env = report["compile_environment"]
    print(f"  triton={env['triton']}  cl_on_path={env['cl_on_path']}")
    print("-" * 78)
    print(f"  {'arm':<28}{'b':>3}{'host ms':>10}{'device ms':>11}"
          f"{'launch':>9}{'/img':>9}  verdict")
    for row in report["arms"]:
        measured = row["host_ms"] if row["batch"] == 1 else row["ms_per_image"]
        print(f"  {row['arm']:<28}{row['batch']:>3}{row['host_ms']:>10.2f}"
              f"{row['device_ms']:>11.2f}{row['launch_overhead_ms']:>9.2f}"
              f"{row['ms_per_image']:>9.2f}  "
              f"{'PASS' if row['within_budget'] else 'OVER'} "
              f"({measured / row['budget_ms'] - 1:+.0%} vs {row['budget_ms']:.0f} ms)")
    print("-" * 78)
    for name, key in (("b1", "stage_breakdown_b1"), ("b16", "stage_breakdown_b16")):
        parts = ", ".join(
            f"{s['stage']}={s['host_ms']:.2f}" for s in report[key]
        )
        print(f"  stages {name}: {parts}")
    print(f"  report: {args.json}")
    print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
