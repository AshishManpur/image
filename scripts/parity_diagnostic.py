"""Locate the source of a CPU-vs-CUDA output difference (Phase 4.12 failure 2).

``tests/test_inference.py::test_cuda_inference_matches_cpu`` compares a CPU fp32
forward against a CUDA fp32 forward and requires ``max|Δ| < 1e-4``. On the RTX A400 it
measured **2.415e-04**. This script answers *why*, rather than assuming.

The leading hypothesis is **TF32**. The A400 is Ampere (compute 8.6), and PyTorch
leaves ``torch.backends.cudnn.allow_tf32 = True`` by default, so every convolution in a
nominally-fp32 CUDA forward runs with a **10-bit mantissa** — a relative precision of
about 4.9e-04. ``utils/seed.set_seed`` pins ``cudnn.deterministic`` and
``cudnn.benchmark`` but says nothing about TF32, so nothing in this project has ever
turned it off. That would make "CUDA fp32" a reduced-precision path the caller never
asked for, which is a defect in the inference path, not in the tolerance.

The competing hypotheses, each of which this script separates:

* **bicubic interpolation** — CPU and CUDA use different kernels for
  ``F.interpolate(mode="bicubic")``, and it feeds the global residual.
* **SDPA** — the fused attention kernel on CUDA is a different algorithm from the CPU
  math path, with a different reduction order.
* **plain fp32 reduction-order noise** — real, but should land near 1e-6 for a network
  this deep, not 2.4e-4.

Run on the A400::

    python scripts/parity_diagnostic.py --json reports/phase4_12_parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import sparc_base  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def tf32_state() -> dict[str, bool]:
    """Report every TF32 switch that affects a nominally-fp32 CUDA forward."""
    return {
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
    }


def set_tf32(enabled: bool) -> None:
    """Turn TF32 on or off for both convolutions and matmuls."""
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def stage_outputs(model: SPARCNet, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture the output of every major stage of one forward pass.

    Mirrors ``SPARCNet.forward_with_aux`` rather than hooking, so each tensor is named
    for the contract stage it belongs to (Part 1).

    Args:
        model: Model in eval mode.
        x: Input ``(1, 1, 128, 128)``.

    Returns:
        Stage name -> tensor, on CPU in float64 for lossless comparison.
    """
    out: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        y_hat, stats = model.normalizer(x)
        out["01_normalized"] = y_hat

        features = y_hat
        if model.noise_head is not None:
            noise = model.noise_head(x, stats.scale)
            out["02_noise_sigma"] = noise.sigma_map
            features = torch.cat((y_hat, noise.sigma_map_normalized), dim=1)
        out["03_stem_input"] = features

        stem = model.encoder.stem(features)
        out["04_stem"] = stem

        h = stem
        for index, level in enumerate(model.encoder.levels):
            h_naf = level.naf(h)
            out[f"05_enc{index}_naf"] = h_naf
            h = level.gsa(h_naf)
            out[f"06_enc{index}_gsa"] = h
            if index < len(model.encoder.downsamples):
                h = model.encoder.downsamples[index](h)
                out[f"07_down{index}"] = h

        bottleneck, skips = model.encoder(features)
        out["08_bottleneck"] = bottleneck

        decoded = model.decoder(bottleneck, skips)
        out["09_decoder"] = decoded

        head = model.head(decoded)
        out["10_head"] = head

        residual = F.interpolate(
            y_hat, scale_factor=2.0, mode="bicubic", align_corners=False
        )
        out["11_bicubic_residual"] = residual
        out["12_final"] = model(x)

    return {k: v.detach().to("cpu", torch.float64) for k, v in out.items()}


def isolated_ops(device: torch.device) -> dict[str, torch.Tensor]:
    """Run the individual operations under suspicion, in isolation."""
    generator = torch.Generator(device="cpu").manual_seed(1337)
    image = torch.randn(1, 1, 128, 128, generator=generator)
    features = torch.randn(1, 96, 32, 32, generator=generator)
    weight = torch.randn(96, 96, 3, 3, generator=generator)
    q = torch.randn(1, 3, 1024, 16, generator=generator)
    k = torch.randn(1, 3, 1024, 16, generator=generator)
    v = torch.randn(1, 3, 1024, 16, generator=generator)

    image, features = image.to(device), features.to(device)
    weight = weight.to(device)
    q, k, v = q.to(device), k.to(device), v.to(device)

    from models.wavelet.haar import HaarDWT, HaarIDWT

    with torch.inference_mode():
        results = {
            "conv3x3": F.conv2d(features, weight, padding=1),
            "bicubic_up2": F.interpolate(
                image, scale_factor=2.0, mode="bicubic", align_corners=False
            ),
            "bicubic_down2": F.interpolate(
                image, scale_factor=0.5, mode="bicubic",
                align_corners=False, antialias=False,
            ),
            "haar_roundtrip": HaarIDWT().to(device)(HaarDWT().to(device)(features)),
            "sdpa": F.scaled_dot_product_attention(q, k, v, scale=16**-0.5),
            "explicit_attention": torch.matmul(
                (torch.matmul(q, k.transpose(-2, -1)) * 16**-0.5).softmax(-1), v
            ),
            "layernorm_moments": features.float().var(dim=1, keepdim=True),
        }
    return {name: t.detach().to("cpu", torch.float64) for name, t in results.items()}


def compare(
    label: str,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Max abs difference per stage, in the order the stages run."""
    return {
        name: max_abs(reference[name], candidate[name])
        for name in reference
        if name in candidate
    }


def run_config(
    build: Callable[[], SPARCNet], x: torch.Tensor, device: torch.device, tf32: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build a fresh model on ``device`` and capture stage + isolated-op outputs."""
    if device.type == "cuda":
        set_tf32(tf32)
    model = build().to(device).eval()
    return stage_outputs(model, x.to(device)), isolated_ops(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU vs CUDA parity diagnostic.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_12_parity.json"
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "No CUDA device. This diagnostic compares CPU against CUDA and cannot "
            "run on a CPU-only host.",
            file=sys.stderr,
        )
        return 2

    set_seed(args.seed)
    x = torch.rand(1, 1, 128, 128)

    def build() -> SPARCNet:
        set_seed(args.seed)  # identical weights in every configuration
        return SPARCNet(sparc_base())

    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"TF32 defaults as this project leaves them: {tf32_state()}\n")

    cpu_stages, cpu_ops = run_config(build, x, torch.device("cpu"), tf32=False)
    cuda = torch.device("cuda")

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "tf32_defaults": tf32_state(),
        "tolerance": args.tolerance,
        "configurations": {},
    }

    for label, tf32 in (("cuda_fp32_tf32_on", True), ("cuda_fp32_tf32_off", False)):
        stages, ops = run_config(build, x, cuda, tf32=tf32)
        stage_diff = compare(label, cpu_stages, stages)
        op_diff = compare(label, cpu_ops, ops)
        report["configurations"][label] = {
            "tf32": tf32_state(),
            "final_max_abs_diff": stage_diff["12_final"],
            "passes_tolerance": stage_diff["12_final"] < args.tolerance,
            "per_stage": stage_diff,
            "per_op": op_diff,
        }

        print(f"--- {label} ---")
        print(f"  final max|Δ| = {stage_diff['12_final']:.3e}  "
              f"({'PASS' if stage_diff['12_final'] < args.tolerance else 'FAIL'} "
              f"vs {args.tolerance:g})")
        print("  per stage (first stage to exceed the tolerance is the culprit):")
        for name, value in stage_diff.items():
            flag = "  <-- exceeds" if value >= args.tolerance else ""
            print(f"    {name:24s} {value:.3e}{flag}")
        print("  isolated operations:")
        for name, value in op_diff.items():
            print(f"    {name:24s} {value:.3e}")
        print()

    # Reduced-precision arms, for context only: these are expected to differ far more,
    # and are here so the fp32 numbers can be read against a known scale.
    for label, dtype in (("cuda_bf16", torch.bfloat16), ("cuda_fp16", torch.float16)):
        set_tf32(False)
        model = build().to(cuda).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = model(x.to(cuda))
        diff = max_abs(cpu_stages["12_final"], out.to("cpu", torch.float64))
        report["configurations"][label] = {"final_max_abs_diff": diff}
        print(f"--- {label} ---  final max|Δ| = {diff:.3e}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    on = report["configurations"]["cuda_fp32_tf32_on"]["final_max_abs_diff"]
    off = report["configurations"]["cuda_fp32_tf32_off"]["final_max_abs_diff"]
    print("\n" + "=" * 72)
    print(f"  TF32 on : {on:.3e}")
    print(f"  TF32 off: {off:.3e}")
    if off < args.tolerance <= on:
        print("  VERDICT: TF32 is the cause. Disabling it restores fp32 parity, so the")
        print("           inference path must request true fp32 — not the tolerance.")
    elif off >= args.tolerance:
        print("  VERDICT: NOT TF32 alone. Read the per-stage table above: the first")
        print("           stage exceeding the tolerance names the real cause.")
    else:
        print("  VERDICT: both configurations pass; the failure did not reproduce.")
    print(f"  report: {args.json}")
    print("=" * 72)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
