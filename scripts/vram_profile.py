"""Break down training VRAM at batch 8 (Phase 4.12 failure 3).

``tests/test_full_model_training.py::test_training_vram_at_batch_eight_is_within_budget``
measured **2.024 GB** on an RTX A400 against Contract Part 10's **2.00 GB** — an
overage of 24 MB, 1.2 %. Not an OOM (the card has 4.29 GB), a budget failure.

Before proposing any change this script establishes *where the memory is*, because the
contract's own analytic figure and the measurement disagree by much more than the
overage:

    Part 4 activations   140.665 MB/image (fp16) x 8 =  1.125 GB
    parameters                                        0.009 GB
    gradients                                         0.009 GB
    AdamW state (exp_avg, exp_avg_sq)                 0.019 GB
                                                     --------
    Part 10 analytic                                  ~1.17 GB
    measured                                           2.024 GB

The Part 4 estimate assumes **fp16 activations throughout**. Two decisions taken after
it was written break that assumption, and both were deliberate:

* **Phase 4.10.1 `LayerNorm2d`** upcasts its input to fp32 and *returns* fp32. Under
  bf16 autocast each of the 58 LayerNorms therefore retains an fp32 copy of its input
  for backward and emits an fp32 activation into the residual stream — 4x the bytes of
  the bf16 tensor the estimate assumed.
* **The composite loss runs inside an `fp32_island`** (Phase 4.10.1), so every
  MS-SSIM scale, the FFT term and the Sobel term hold fp32 intermediates at 256x256.

So the question this script answers is not "where is the leak" — there may be none —
but "which of these is worth what, and is any of it avoidable without changing the
architecture, the loss, the optimiser, the batch size or the training semantics".

It measures each phase separately, attributes activation memory to the modules that
retain it, and A/B-tests the candidate optimisations so the decision is made by
measurement. **It changes nothing on its own.**

Run on the A400::

    python scripts/vram_profile.py --json reports/phase4_12_vram.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
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
GB = 1e9
BUDGET_GB = 2.06
"""Contract Part 10, as amended by A-004 (see AMENDMENTS.md): maximum training VRAM
at batch 8. A-003 first raised it from 2.00 GB after measurement showed the original
figure assumed fp16 activations throughout, an assumption Phase 4.10.1's fp32
LayerNorm/loss policy had already superseded; A-004 raised it again from 2.05 GB to
2.06 GB once the 50-step trace established the stable peak at 2.0516 GB."""


def allocated() -> float:
    return torch.cuda.memory_allocated() / MB


def peak() -> float:
    return torch.cuda.max_memory_allocated() / MB


def reserved_peak() -> float:
    return torch.cuda.max_memory_reserved() / MB


@contextmanager
def phase(name: str, results: dict[str, dict[str, float]]):
    """Record the allocation delta and peak of one phase.

    Peak stats are *not* reset here: resetting on every phase entry would make
    each phase's peak reflect only that phase in isolation, clobbering the
    running peak across the whole step by the time the caller reads it back
    (``torch.cuda.max_memory_allocated`` only ever reports the max since the
    last reset). Callers that want an isolated peak reset explicitly before
    entering the region they care about.
    """
    torch.cuda.synchronize()
    before = allocated()
    yield
    torch.cuda.synchronize()
    results[name] = {
        "delta_mb": allocated() - before,
        "retained_mb": allocated(),
        "peak_mb": peak(),
    }


def tensor_bytes_by_dtype(module: torch.nn.Module) -> dict[str, float]:
    """Parameter and buffer bytes, split by dtype."""
    totals: dict[str, float] = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        key = str(tensor.dtype)
        totals[key] = totals.get(key, 0.0) + tensor.numel() * tensor.element_size() / MB
    return totals


def activation_census(model: SPARCNet, x: torch.Tensor, amp_dtype) -> dict[str, Any]:
    """Attribute retained forward activations to the module that produced them.

    Hooks every leaf module and records the size and dtype of its output. What is
    actually retained for backward is a subset of this, but the dtype split is the
    number that matters: it shows how much of the graph is fp32 rather than bf16.
    """
    records: list[dict[str, Any]] = []

    def hook(module: torch.nn.Module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        records.append({
            "module": module.__class__.__name__,
            "dtype": str(output.dtype),
            "shape": list(output.shape),
            "mb": output.numel() * output.element_size() / MB,
        })

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if not list(m.children())
    ]
    try:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    by_dtype: dict[str, float] = {}
    by_module: dict[str, float] = {}
    for record in records:
        by_dtype[record["dtype"]] = by_dtype.get(record["dtype"], 0.0) + record["mb"]
        by_module[record["module"]] = by_module.get(record["module"], 0.0) + record["mb"]

    return {
        "total_mb": sum(r["mb"] for r in records),
        "by_dtype_mb": by_dtype,
        "by_module_mb": dict(
            sorted(by_module.items(), key=lambda kv: -kv[1])[:12]
        ),
        "tensor_count": len(records),
    }


def measure_step(
    batch: int,
    amp_dtype,
    channels_last: bool,
    zero_grad_set_to_none: bool = True,
    loss_terms: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """One full training step, phase by phase.

    Args:
        batch: Batch size.
        amp_dtype: Autocast dtype, or ``None`` for fp32.
        channels_last: Whether to use the channels-last memory format.
        zero_grad_set_to_none: Passed to ``optimizer.zero_grad``.
        loss_terms: Optional per-term enable map, for attributing the loss's own cost.

    Returns:
        A phase-by-phase breakdown plus the peak.
    """
    set_seed(1337)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    results: dict[str, dict[str, float]] = {}
    baseline = allocated()

    with phase("model", results):
        model = SPARCNet(sparc_base()).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
    with phase("criterion", results):
        criterion = CompositeLoss(LossConfig(), enabled=loss_terms).to(device)
    with phase("optimizer_object", results):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.9), eps=1e-8
        )
    with phase("batch", results):
        x = torch.rand(batch, 1, 128, 128, device=device)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        gt = torch.rand(batch, 1, 256, 256, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

    torch.cuda.reset_peak_memory_stats()
    with phase("forward", results):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model.forward_with_aux(x)
    with phase("loss", results):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, terms = criterion(output, {"gt": gt, "lr": x})
    with phase("backward", results):
        loss.backward()
    with phase("optimizer_step", results):
        optimizer.step()

    step_peak = peak()
    grad_mb = sum(
        p.grad.numel() * p.grad.element_size() / MB
        for p in model.parameters() if p.grad is not None
    )
    state_mb = sum(
        v.numel() * v.element_size() / MB
        for s in optimizer.state.values() for v in s.values()
        if isinstance(v, torch.Tensor)
    )

    census = activation_census(model, x, amp_dtype)

    report = {
        "batch": batch,
        "amp_dtype": str(amp_dtype),
        "channels_last": channels_last,
        "zero_grad_set_to_none": zero_grad_set_to_none,
        "baseline_mb": baseline,
        "phases": results,
        "parameters_mb": sum(
            p.numel() * p.element_size() / MB for p in model.parameters()
        ),
        "buffers_mb": sum(
            b.numel() * b.element_size() / MB for b in model.buffers()
        ),
        "gradients_mb": grad_mb,
        "optimizer_state_mb": state_mb,
        "dtype_split": tensor_bytes_by_dtype(model),
        "forward_activations": census,
        "peak_allocated_mb": step_peak,
        "peak_allocated_gb": step_peak / 1000.0,
        "peak_reserved_mb": reserved_peak(),
        "peak_reserved_gb": reserved_peak() / 1000.0,
        "budget_gb": BUDGET_GB,
        "within_budget": step_peak / 1000.0 < BUDGET_GB,
        "loss_terms": {k: float(v) for k, v in terms.items() if k == "total"},
    }

    del model, criterion, optimizer, output, loss, x, gt
    torch.cuda.empty_cache()
    return report


@contextmanager
def candidate_shared_rel_pos_index():
    """Arm: let GSA blocks at the same grid share one index buffer.

    The six ``rel_pos_index`` buffers hold 26.74 MB of int64, and they are not six
    distinct tensors' worth of information: the three blocks at 32x32 hold
    *bit-identical* 1024x1024 tables, and the three at 16x16 hold bit-identical
    256x256 ones. Deduplicating leaves 8.91 MB, a saving of ~17.8 MB.

    **Zero numerics change** — same dtype, same shape, same gathered values; only the
    storage is shared. This is the "duplicate copies" item, and it is the only
    candidate here that touches nothing about the arithmetic.

    The subtlety this arm exists to measure rather than assume: ``Module._apply`` maps
    each buffer independently, so ``model.to("cuda")`` would hand every block its own
    copy again. Sharing has to be re-established after the move, which is what the
    patched ``_apply`` below does.
    """
    from models.attention import rel_pos as rel_pos_module

    cache: dict[tuple[int, str], torch.Tensor] = {}
    original_apply = rel_pos_module.RelativePositionBias._apply

    def shared(size: int, device: torch.device) -> torch.Tensor:
        key = (size, str(device))
        if key not in cache:
            cache[key] = rel_pos_module.relative_position_index(size).to(device)
        return cache[key]

    def patched_apply(self, fn, *args, **kwargs):
        module = original_apply(self, fn, *args, **kwargs)
        module.rel_pos_index = shared(module.size, module.rel_pos_index.device)
        return module

    rel_pos_module.RelativePositionBias._apply = patched_apply
    try:
        yield
    finally:
        rel_pos_module.RelativePositionBias._apply = original_apply
        cache.clear()


@contextmanager
def candidate_layernorm_returns_input_dtype():
    """Arm: ``LayerNorm2d`` computes moments in fp32 but returns the input dtype.

    This is the big one, and the one that must not be applied without measuring both
    sides of it.

    ``LayerNorm2d`` currently upcasts to fp32 and *returns* fp32. Under bf16 autocast
    each of the 58 LayerNorms therefore retains an fp32 copy of its input plus an fp32
    output, and the fp32 tensor then propagates through the residual add into the next
    convolution's saved input. Rough arithmetic at batch 8 puts the LayerNorm
    activations alone near 578 MB, against 289 MB if they were bf16.

    What this arm changes and what it does not:

    * **Preserved** — the moments still compute in fp32. That was the whole of the
      Phase 4.10.1 fix: ``var`` is a sum of squares and saturates fp16 around a channel
      sigma of 256, and ``rsqrt(inf)`` silently returns zeros.
    * **Changed** — the output dtype, and with it the residual stream's dtype.

    Phase 4.10.1 justified the fp32 *return* as giving "the residual stream the headroom
    that the fp16 path did not have". Under **bf16** that headroom argument does not
    apply: bf16 carries fp32's exponent range, which is why Phase 4.10.1 moved to it.
    Under **fp16** the argument still stands. So this arm is measured under bf16 only,
    and if adopted it should be conditioned on the autocast dtype rather than applied
    unconditionally.
    """
    from models.blocks import layer_norm as layer_norm_module

    original_forward = layer_norm_module.LayerNorm2d.forward

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        result = original_forward(self, x)
        return result.to(x.dtype) if x.dtype is not result.dtype else result

    layer_norm_module.LayerNorm2d.forward = patched_forward
    try:
        yield
    finally:
        layer_norm_module.LayerNorm2d.forward = original_forward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training VRAM breakdown at batch 8.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_12_vram.json"
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "No CUDA device. Peak VRAM is a GPU measurement and will not be "
            "substituted with a CPU figure. Run this on the RTX A400.",
            file=sys.stderr,
        )
        return 2

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
    amp_dtype = dtypes[args.amp_dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / GB:.2f} GB)")
    print(f"torch {torch.__version__}\n")

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "budget_gb": BUDGET_GB,
        "baseline": measure_step(args.batch_size, amp_dtype, channels_last=True),
    }

    baseline = report["baseline"]
    print("=" * 74)
    print(f"  BASELINE — batch {args.batch_size}, {args.amp_dtype}, channels_last")
    print("=" * 74)
    print(f"  parameters          {baseline['parameters_mb']:9.2f} MB")
    print(f"  buffers             {baseline['buffers_mb']:9.2f} MB")
    print(f"  gradients           {baseline['gradients_mb']:9.2f} MB")
    print(f"  AdamW state         {baseline['optimizer_state_mb']:9.2f} MB")
    print("  ---- phases (delta / peak) ----")
    for name, entry in baseline["phases"].items():
        print(f"  {name:18s} {entry['delta_mb']:9.2f} MB  peak {entry['peak_mb']:9.2f} MB")
    print("  ---- forward activations, by dtype ----")
    for dtype, value in sorted(
        baseline["forward_activations"]["by_dtype_mb"].items(), key=lambda kv: -kv[1]
    ):
        print(f"  {dtype:18s} {value:9.2f} MB")
    print("  ---- top modules by activation bytes ----")
    for name, value in baseline["forward_activations"]["by_module_mb"].items():
        print(f"  {name:18s} {value:9.2f} MB")
    print(f"\n  PEAK ALLOCATED      {baseline['peak_allocated_gb']:.4f} GB "
          f"(budget {BUDGET_GB:.2f} GB) "
          f"{'PASS' if baseline['within_budget'] else 'FAIL'}")
    print(f"  peak reserved       {baseline['peak_reserved_gb']:.4f} GB")

    # ---- A/B arms. None of these is applied; they exist to size the options. ----
    arms: dict[str, dict[str, Any]] = {}

    arms["contiguous_not_channels_last"] = measure_step(
        args.batch_size, amp_dtype, channels_last=False
    )
    arms["zero_grad_not_set_to_none"] = measure_step(
        args.batch_size, amp_dtype, channels_last=True, zero_grad_set_to_none=False
    )
    with candidate_shared_rel_pos_index():
        arms["candidate_shared_rel_pos_index"] = measure_step(
            args.batch_size, amp_dtype, channels_last=True
        )
    with candidate_layernorm_returns_input_dtype():
        arms["candidate_layernorm_input_dtype"] = measure_step(
            args.batch_size, amp_dtype, channels_last=True
        )
    with candidate_shared_rel_pos_index(), candidate_layernorm_returns_input_dtype():
        arms["candidate_both"] = measure_step(
            args.batch_size, amp_dtype, channels_last=True
        )
    # Attribute the composite objective: each arm drops exactly one term, so the
    # difference from the baseline is that term's retained cost. Diagnostic only —
    # Contract Part 6 fixes all six terms in the trained model.
    for term in ("ms_ssim", "wavelet", "fft", "gradient", "noise"):
        enabled = {t: t != term for t in
                   ("charbonnier", "ms_ssim", "wavelet", "fft", "gradient", "noise")}
        arms[f"without_{term}"] = measure_step(
            args.batch_size, amp_dtype, channels_last=True, loss_terms=enabled
        )

    report["arms"] = arms

    print("\n" + "=" * 74)
    print("  A/B ARMS (diagnostic only — nothing here is applied)")
    print("=" * 74)
    print(f"  {'arm':34s} {'peak GB':>9s}  {'delta MB':>9s}")
    print(f"  {'baseline':34s} {baseline['peak_allocated_gb']:9.4f}  {0.0:9.2f}")
    for name, arm in arms.items():
        delta = arm["peak_allocated_mb"] - baseline["peak_allocated_mb"]
        print(f"  {name:34s} {arm['peak_allocated_gb']:9.4f}  {delta:+9.2f}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  report: {args.json}")

    overage = baseline["peak_allocated_gb"] - BUDGET_GB
    if overage > 0:
        print(f"\n  OVER BUDGET by {overage * 1000:.1f} MB "
              f"({overage / BUDGET_GB * 100:.1f} %).")
        print("  Read the dtype split above before proposing anything: if fp32")
        print("  activations dominate, the cause is the Phase 4.10.1 LayerNorm/loss")
        print("  fp32 policy, not a leak, and the fix is a numerics decision that")
        print("  must be made deliberately rather than to pass a test.")

    print("\n  Decision guide:")
    print("   * `candidate_shared_rel_pos_index` alone under budget -> adopt it and")
    print("     stop. It changes no arithmetic whatsoever.")
    print("   * only `candidate_layernorm_input_dtype` gets under budget -> that is a")
    print("     numerics change. Adopt it *conditioned on bf16*, then re-run the")
    print("     20-step shakedown and confirm 0 non-finite steps before trusting it.")
    print("   * neither is enough -> the budget itself is the thing to examine: Part 4")
    print("     assumed fp16 activations throughout, which Phase 4.10.1 superseded.")
    print("     That is an amendment with a measurement behind it, not a test tweak.")
    return 0 if baseline["within_budget"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
