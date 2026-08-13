"""Phase 4.11 verification measurements for ``GatedFuse`` (Contract Part 2.7).

Produces every number quoted in the Phase 4.11 verification report by measurement
rather than derivation: tensor shapes at both fusion sites, exact parameter counts,
MACs/FLOPs at module and model level, retained-activation bytes, gradient reach,
numerical stability, and the GPU/export readiness matrix (AMP, channels-last,
``torch.compile``, TorchScript, ONNX, CUDA).

Every figure is compared against the contract's Part 3 stages 15 and 20 and against
``ConcatFusion``, which is both the Phase 4.10 baseline and the ablation A3 control arm.

Usage::

    python reports/inspect_fusion.py [--json reports/report_fusion.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch.onnx's exporter prints a U+2705 progress marker, which raises
# UnicodeEncodeError on this host's cp1252 console. Nothing to do with the module
# under test, but it aborts the run before ONNX parity is reported.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.sparc_config import build_sparc_config  # noqa: E402
from models.fusion import ConcatFusion, GatedFuse, build_fusion  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.complexity import measure_complexity  # noqa: E402

CONTRACT_PARAMETERS = {96: 23_256, 48: 5_868}
"""Contract Part 3, stages 15 and 20. Part 9 requires an exact match, not a tolerance."""

FUSION_SITES = (
    ("Dec D1 (Part 3 stage 15)", 96, 32),
    ("Dec D0 (Part 3 stage 20)", 48, 64),
)
"""``(label, channels, spatial)`` for the two — and only two — long skips."""


def _inputs(channels: int, size: int, batch: int = 1, seed: int = 1337):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, channels, size, size, generator=generator),
        torch.rand(batch, channels, size, size, generator=generator),
    )


# ------------------------------------------------------------------ measurements
def measure_shapes() -> list[dict[str, Any]]:
    """Record input/output/gate shapes at both fusion sites for B in {1, 2, 8}."""
    rows = []
    for label, channels, size in FUSION_SITES:
        module = GatedFuse(channels)
        for batch in (1, 2, 8):
            skip, decoded = _inputs(channels, size, batch)
            out = module(skip, decoded)
            gate = module.gate(skip, decoded)
            rows.append(
                {
                    "site": label,
                    "batch": batch,
                    "skip": list(skip.shape),
                    "decoded": list(decoded.shape),
                    "gate": list(gate.shape),
                    "output": list(out.shape),
                    "shape_preserved": out.shape == skip.shape,
                }
            )
    return rows


def measure_parameters() -> list[dict[str, Any]]:
    """Compare measured, analytic and contract parameter counts against concat."""
    rows = []
    for label, channels, _ in FUSION_SITES:
        gated = GatedFuse(channels)
        concat = ConcatFusion(channels)
        measured = sum(p.numel() for p in gated.parameters())
        rows.append(
            {
                "site": label,
                "channels": channels,
                "measured": measured,
                "analytic": GatedFuse.parameter_count(channels),
                "contract": CONTRACT_PARAMETERS[channels],
                "exact_match": measured == CONTRACT_PARAMETERS[channels],
                "concat": sum(p.numel() for p in concat.parameters()),
                "breakdown": {
                    name: sum(p.numel() for p in child.parameters())
                    for name, child in gated.named_children()
                },
            }
        )
    return rows


def measure_module_flops() -> list[dict[str, Any]]:
    """Module-level FLOPs/MACs at both sites, gated vs concat, at B=1."""
    rows = []
    for label, channels, size in FUSION_SITES:
        skip, decoded = _inputs(channels, size)
        gated = measure_complexity(GatedFuse(channels), (skip, decoded))
        concat = measure_complexity(ConcatFusion(channels), (skip, decoded))
        rows.append(
            {
                "site": label,
                "gated_macs": gated.macs,
                "gated_flops": gated.flops,
                "concat_macs": concat.macs,
                "concat_flops": concat.flops,
                "delta_macs": gated.macs - concat.macs,
                # Part 2.7: 2C^2*H*W for the projection, + C^2/2 for the pooled gate.
                "contract_macs": 2 * channels**2 * size**2 + channels**2 // 2,
            }
        )
    return rows


def measure_model_complexity() -> dict[str, Any]:
    """Whole-model parameters and MACs with each fusion, on the contract input."""
    sample = torch.rand(1, 1, 128, 128)
    out: dict[str, Any] = {}
    for key, gated in (("gated", True), ("concat", False)):
        model = SPARCNet(
            build_sparc_config(
                "sparc-base", use_attention=False, use_gated_fusion=gated
            )
        )
        report = measure_complexity(model, sample)
        out[key] = {
            "parameters": report.params_total,
            "macs": report.macs,
            "flops": report.flops,
            "disk_mb_fp32": report.params_total * 4 / 1024**2,
        }
    out["delta_parameters"] = out["gated"]["parameters"] - out["concat"]["parameters"]
    out["delta_macs"] = out["gated"]["macs"] - out["concat"]["macs"]
    out["mac_overhead_fraction"] = out["delta_macs"] / out["concat"]["macs"]
    return out


CONTRACT_ACT_MB = {96: 0.590, 48: 1.180}
"""Contract Part 3 stages 15 and 20, fp16 decimal megabytes (``3*C*H*W`` elements)."""


def measure_activation_memory() -> list[dict[str, Any]]:
    """Activation elements the module allocates, measured with ``saved_tensors_hooks``.

    Part 2.7 budgets ``3*C*H*W``: the concatenated tensor (``2*C*H*W``) plus the output
    (``C*H*W``). Those are the tensors this module *allocates*. ``skip`` and ``decoded``
    are counted by the stages that produced them, and the ``(B,C,1,1)`` gate tensors are
    negligible — so the measurement retains autograd's saved tensors minus the module's
    own inputs and parameters, then adds the output.
    """
    rows = []
    for label, channels, size in FUSION_SITES:
        for name, module in (
            ("gated", GatedFuse(channels)),
            ("concat", ConcatFusion(channels)),
        ):
            skip, decoded = _inputs(channels, size)
            skip.requires_grad_(True)
            decoded.requires_grad_(True)
            own = {id(skip), id(decoded), *(id(p) for p in module.parameters())}
            saved: dict[int, int] = {}

            def pack(tensor: torch.Tensor, _saved=saved, _own=own) -> torch.Tensor:
                if id(tensor) not in _own:
                    _saved[id(tensor)] = tensor.numel()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                out = module(skip, decoded)
                out.sum().backward()

            elements = sum(saved.values()) + out.numel()
            row = {
                "site": label,
                "fusion": name,
                "elements": elements,
                "elements_per_cHW": elements / (channels * size * size),
                "fp16_mb": elements * 2 / 1e6,
            }
            if name == "gated":
                contract = CONTRACT_ACT_MB[channels]
                row["contract_mb"] = contract
                row["deviation"] = (row["fp16_mb"] - contract) / contract
            rows.append(row)
    return rows


def measure_gradient_reach() -> dict[str, Any]:
    """Every parameter and both inputs must receive a finite, non-zero gradient."""
    module = GatedFuse(96)
    skip, decoded = _inputs(96, 32, batch=2)
    skip.requires_grad_(True)
    decoded.requires_grad_(True)
    module(skip, decoded).pow(2).mean().backward()

    params = {
        name: {
            "finite": bool(torch.isfinite(p.grad).all()),
            "nonzero": int(torch.count_nonzero(p.grad)),
            "numel": p.numel(),
        }
        for name, p in module.named_parameters()
    }
    return {
        "parameters": params,
        "all_reached": all(v["nonzero"] > 0 and v["finite"] for v in params.values()),
        "skip_grad_nonzero": int(torch.count_nonzero(skip.grad)),
        "decoded_grad_nonzero": int(torch.count_nonzero(decoded.grad)),
    }


def measure_stability() -> dict[str, Any]:
    """Part 9: 100 random batches including x1e-3 and x1e3 magnitudes."""
    module = GatedFuse(96)
    generator = torch.Generator().manual_seed(7)
    gate_min, gate_max, worst_violation = 1.0, 0.0, 0.0
    for index in range(100):
        magnitude = {0: 1e-3, 1: 1e3}.get(index % 3, 1.0)
        skip = torch.randn(2, 96, 16, 16, generator=generator) * magnitude
        decoded = torch.randn(2, 96, 16, 16, generator=generator) * magnitude
        with torch.no_grad():
            out = module(skip, decoded)
            gate = module.gate(skip, decoded)
        if not torch.isfinite(out).all():
            return {"finite": False, "failed_at": index}
        gate_min = min(gate_min, float(gate.min()))
        gate_max = max(gate_max, float(gate.max()))
        # convexity: output must lie between the two inputs elementwise
        low = torch.minimum(skip, decoded)
        high = torch.maximum(skip, decoded)
        violation = float(
            torch.maximum(low - out, out - high).clamp_min(0).max()
        )
        worst_violation = max(worst_violation, violation)
    return {
        "finite": True,
        "batches": 100,
        "gate_min": gate_min,
        "gate_max": gate_max,
        "gate_in_unit_interval": gate_min >= 0.0 and gate_max <= 1.0,
        "worst_convexity_violation": worst_violation,
    }


def measure_readiness() -> dict[str, Any]:
    """AMP, channels-last, torch.compile, TorchScript, ONNX, CUDA."""
    module = GatedFuse(96).eval()
    skip, decoded = _inputs(96, 32, batch=2)
    with torch.no_grad():
        reference = module(skip, decoded)
    results: dict[str, Any] = {}

    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
        amp_out = module(skip, decoded)
    results["autocast_bf16"] = {
        "finite": bool(torch.isfinite(amp_out.float()).all()),
        "max_abs_diff": float((amp_out.float() - reference).abs().max()),
    }

    with torch.no_grad():
        cl_out = module(
            skip.to(memory_format=torch.channels_last),
            decoded.to(memory_format=torch.channels_last),
        )
    results["channels_last"] = {
        "max_abs_diff": float((cl_out - reference).abs().max())
    }

    with torch.no_grad():
        scripted = torch.jit.script(module)
        results["torchscript"] = {
            "max_abs_diff": float((scripted(skip, decoded) - reference).abs().max())
        }

    with torch.no_grad():
        compiled = torch.compile(module, backend="eager", fullgraph=True)
        results["torch_compile"] = {
            "backend": "eager",
            "fullgraph": True,
            "max_abs_diff": float((compiled(skip, decoded) - reference).abs().max()),
        }

    results["onnx"] = _measure_onnx(module, reference)

    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this host
        cuda_out = module.cuda()(skip.cuda(), decoded.cuda()).cpu()
        results["cuda"] = {"max_abs_diff": float((cuda_out - reference).abs().max())}
        module.cpu()
    else:
        results["cuda"] = "SKIPPED (no CUDA device)"
    return results


def _measure_onnx(module: nn.Module, reference: torch.Tensor) -> Any:
    """Export and compare against eager, reporting the opset actually produced.

    Contract Part 9 asks for opset 17. This torch build's exporter implements opset 18
    and up, and its automatic down-conversion to 17 fails on this graph, so the request
    is recorded alongside the opset the file really carries.
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        return "SKIPPED (onnx / onnxruntime not installed)"

    import tempfile

    skip, decoded = _inputs(96, 32, batch=2)
    results = {}
    for requested in (17, 18):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gated_fuse.onnx"
            try:
                torch.onnx.export(
                    module,
                    (skip, decoded),
                    str(path),
                    opset_version=requested,
                    input_names=["skip", "decoded"],
                    output_names=["fused"],
                    dynamic_axes={
                        "skip": {0: "batch"},
                        "decoded": {0: "batch"},
                        "fused": {0: "batch"},
                    },
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                results[f"requested_{requested}"] = f"EXPORT FAILED: {type(exc).__name__}"
                continue

            produced_opset = max(
                imp.version for imp in onnx.load(str(path)).opset_import
            )
            session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            produced = session.run(
                None, {"skip": skip.numpy(), "decoded": decoded.numpy()}
            )[0]
            results[f"requested_{requested}"] = {
                "produced_opset": produced_opset,
                "max_abs_diff": float(abs(produced - reference.numpy()).max()),
            }
    return results


def measure_latency(repeats: int = 40) -> dict[str, Any]:
    """Whole-model CPU inference latency, gated vs concat, batch 1.

    CPU latency is not a contract budget — Part 10's limits are GPU-denominated — so
    this only establishes that the fusion adds no material inference cost.

    **The two arms are interleaved, and the order is alternated.** Timing one model to
    completion and then the other attributes any drift in CPU frequency or scheduler
    state to whichever ran second: doing exactly that reported a spurious +11.8 % for
    the gated arm, which interleaving resolves to roughly zero. Per-iteration spread on
    this host is wide (p10 ~115 ms, p90 ~275 ms), so the medians are only meaningful
    because both arms sample the same noise.
    """
    sample = torch.rand(1, 1, 128, 128)
    models = {
        key: SPARCNet(
            build_sparc_config(
                "sparc-base", use_attention=False, use_gated_fusion=gated
            )
        ).eval()
        for key, gated in (("gated", True), ("concat", False))
    }
    timings: dict[str, list[float]] = {"gated": [], "concat": []}

    with torch.no_grad():
        for model in models.values():
            for _ in range(5):
                model(sample)
        for index in range(repeats):
            order = ("gated", "concat") if index % 2 == 0 else ("concat", "gated")
            for key in order:
                start = time.perf_counter()
                models[key](sample)
                timings[key].append((time.perf_counter() - start) * 1e3)

    out: dict[str, Any] = {"repeats": repeats, "interleaved": True}
    for key, values in timings.items():
        values.sort()
        out[key] = {
            "median_ms": values[len(values) // 2],
            "min_ms": values[0],
            "p10_ms": values[len(values) // 10],
            "p90_ms": values[-1 - len(values) // 10],
        }
    out["median_overhead_fraction"] = (
        out["gated"]["median_ms"] - out["concat"]["median_ms"]
    ) / out["concat"]["median_ms"]
    out["min_overhead_fraction"] = (
        out["gated"]["min_ms"] - out["concat"]["min_ms"]
    ) / out["concat"]["min_ms"]
    return out


def measure_factory() -> dict[str, Any]:
    """The public abstraction must be unchanged: only the implementation swaps."""
    gated_config = build_sparc_config("sparc-base", use_gated_fusion=True)
    concat_config = build_sparc_config("sparc-base", use_gated_fusion=False)
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    fusions = [stage.fusion for stage in model.decoder.stages]
    return {
        "build_fusion_gated": type(build_fusion(gated_config, 96)).__name__,
        "build_fusion_concat": type(build_fusion(concat_config, 96)).__name__,
        "reduction": build_fusion(gated_config, 96).reduction,
        "decoder_fusion_types": [type(f).__name__ for f in fusions],
        "decoder_fusion_channels": [f.channels for f in fusions],
        "decoder_fusion_parameters": [
            sum(p.numel() for p in f.parameters()) for f in fusions
        ],
    }


# ------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.11 fusion verification.")
    parser.add_argument("--json", type=Path, default=Path("reports/report_fusion.json"))
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1337)
    report: dict[str, Any] = {
        "shapes": measure_shapes(),
        "parameters": measure_parameters(),
        "module_flops": measure_module_flops(),
        "model_complexity": measure_model_complexity(),
        "activation_memory": measure_activation_memory(),
        "gradients": measure_gradient_reach(),
        "stability": measure_stability(),
        "readiness": measure_readiness(),
        "factory": measure_factory(),
    }
    if not args.skip_latency:
        report["latency"] = measure_latency()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Phase 4.11 — GatedFuse verification ===\n")

    print("-- Parameters (Contract Part 3 stages 15 / 20) --")
    for row in report["parameters"]:
        mark = "EXACT" if row["exact_match"] else "MISMATCH"
        print(
            f"  C={row['channels']:3d}  measured {row['measured']:6d}  "
            f"contract {row['contract']:6d}  [{mark}]  concat {row['concat']:6d}"
        )

    print("\n-- Shapes --")
    for row in report["shapes"]:
        print(
            f"  {row['site']:26s} B={row['batch']}  "
            f"skip {row['skip']} + dec {row['decoded']} "
            f"-> gate {row['gate']} -> out {row['output']}"
        )

    print("\n-- Module MACs --")
    for row in report["module_flops"]:
        print(
            f"  {row['site']:26s} gated {row['gated_macs']:>12,}  "
            f"concat {row['concat_macs']:>12,}  delta {row['delta_macs']:>+9,}  "
            f"contract {row['contract_macs']:>12,}"
        )

    complexity = report["model_complexity"]
    print("\n-- Whole model (sparc-base, attention off) --")
    for key in ("gated", "concat"):
        entry = complexity[key]
        print(
            f"  {key:6s} params {entry['parameters']:>9,}  "
            f"MACs {entry['macs']:>13,}  disk {entry['disk_mb_fp32']:.2f} MB"
        )
    print(
        f"  delta  params {complexity['delta_parameters']:>+9,}  "
        f"MACs {complexity['delta_macs']:>+13,}  "
        f"({complexity['mac_overhead_fraction']:.4%})"
    )

    print("\n-- Activations allocated (B=1, fp16 decimal MB as in Part 3) --")
    for row in report["activation_memory"]:
        contract = (
            f"  contract {row['contract_mb']:.3f} ({row['deviation']:+.2%})"
            if "contract_mb" in row
            else ""
        )
        print(
            f"  {row['site']:26s} {row['fusion']:6s} "
            f"{row['elements']:>9,} elements "
            f"({row['elements_per_cHW']:.2f} x C*H*W)  "
            f"{row['fp16_mb']:.3f} MB{contract}"
        )

    print("\n-- Gradients / stability --")
    print(f"  every parameter reached : {report['gradients']['all_reached']}")
    stability = report["stability"]
    print(f"  100 batches finite      : {stability['finite']}")
    print(
        f"  gate range              : "
        f"[{stability['gate_min']:.6f}, {stability['gate_max']:.6f}]"
    )
    print(
        f"  worst convexity breach  : {stability['worst_convexity_violation']:.3e}"
    )

    print("\n-- GPU / export readiness --")
    for name, value in report["readiness"].items():
        print(f"  {name:16s} {value}")

    if "latency" in report:
        latency = report["latency"]
        print(
            "\n-- CPU inference latency (batch 1, interleaved, not a contract budget) --"
        )
        for key in ("gated", "concat"):
            entry = latency[key]
            print(
                f"  {key:6s} median {entry['median_ms']:6.1f} ms  "
                f"p10 {entry['p10_ms']:6.1f}  p90 {entry['p90_ms']:6.1f}  "
                f"min {entry['min_ms']:6.1f}"
            )
        print(
            f"  overhead: median {latency['median_overhead_fraction']:+.2%}, "
            f"min {latency['min_overhead_fraction']:+.2%}"
        )

    print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
