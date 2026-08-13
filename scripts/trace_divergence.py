"""Localise the first non-finite tensor in a SPARC-Net forward pass (Task 1).

The Phase 4.10 shakedown reported only that the total loss was non-finite. That is the
last link in the chain, not the first: by the time the scalar is NaN, the tensor that
produced it is many modules back. This script walks the forward pass with
:class:`utils.numerics.ModuleTracer` and names the first module whose output goes bad,
together with the activation magnitudes on the way there.

Two modes:

``--checkpoint``
    Load real weights — including ``divergence.pt``, which the hardened trainer writes
    on abort — and trace one batch. This is the mode to use on the actual failure.

``--weight-scale``
    Multiply every weight matrix by a constant to walk a freshly initialised model
    toward the failure, and report the scale at which it first breaks. This reproduces
    the failure *mechanism* without needing the diverged checkpoint, and is how the
    Phase 4.10.1 root cause was established.

Both modes report the same thing: which module, in which dtype, at what magnitude.

Usage::

    python -m scripts.trace_divergence --device cuda --dtype fp16
    python -m scripts.trace_divergence --checkpoint checkpoints/shakedown/divergence.pt
    python -m scripts.trace_divergence --dtype bf16 --weight-scale 2.5 3.0 4.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from configs.sparc_config import DataConfig, build_sparc_config
from models.sparc_net import SPARCNet
from utils.checkpoint import load_checkpoint
from utils.logging_utils import configure_logging, get_logger
from utils.numerics import FP16_MAX, ModuleTracer

_LOGGER = get_logger(__name__)

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_batch(
    device: torch.device, batch_size: int, use_real_data: bool
) -> torch.Tensor:
    """Load a batch of degraded LR images.

    Real packed data is preferred: a divergence that only appears on real inputs would
    be missed by ``torch.rand``, which has none of the intensity structure the speckle
    term keys off.

    Args:
        device: Device to place the batch on.
        batch_size: Number of images.
        use_real_data: Whether to read the packed training array.

    Returns:
        LR batch ``(B, 1, 128, 128)``.
    """
    if use_real_data:
        path = DataConfig().packed_root / "train_lr.npy"
        if path.exists():
            import numpy as np

            # The packed array is (N, H, W); the model needs an explicit channel axis.
            array = np.asarray(np.load(path, mmap_mode="r")[:batch_size])
            return torch.from_numpy(array).float().unsqueeze(1).to(device)
        _LOGGER.warning("Packed data not found at %s; falling back to random.", path)
    torch.manual_seed(0)
    return (torch.rand(batch_size, 1, 128, 128, device=device) * 0.6 + 0.05)


def trace_once(
    model: SPARCNet, x: torch.Tensor, device: torch.device, dtype: torch.dtype | None
) -> dict[str, Any]:
    """Trace one forward pass and summarise its numerical health.

    Args:
        model: Model to run.
        x: Input batch.
        device: Device, for the autocast device type.
        dtype: Autocast dtype, or ``None`` for plain fp32.

    Returns:
        A record naming the first non-finite module and the largest activations.
    """
    tracer = ModuleTracer(model)
    with tracer, torch.no_grad():
        if dtype is None:
            output = model.forward_with_aux(x)
        else:
            with torch.amp.autocast(device.type, dtype=dtype):
                output = model.forward_with_aux(x)

    image = output.image.float()
    first_bad = tracer.first_bad()
    absmax = max(
        (r["absmax"] for r in tracer.records if r["absmax"] == r["absmax"]), default=0.0
    )
    return {
        "output_finite": bool(torch.isfinite(image).all()),
        "output_absmax": float(image.abs().max()),
        "peak_activation": absmax,
        # Below 1.0 means some activation is already unrepresentable in fp16, which is
        # the condition that turns into NaN at the next convolution.
        "fp16_headroom": FP16_MAX / absmax if absmax > 0 else float("inf"),
        "nonfinite_module_count": len(tracer.bad()),
        "first_nonfinite": (
            None
            if first_bad is None
            else {
                "name": first_bad["name"],
                "class": first_bad["class"],
                "input_dtype": first_bad["input_dtype"],
                "output_dtype": first_bad["dtype"],
                "has_nan": first_bad["has_nan"],
                "has_inf": first_bad["has_inf"],
            }
        ),
        "largest_activations": [
            {"name": r["name"], "class": r["class"], "dtype": r["dtype"],
             "absmax": r["absmax"]}
            for r in tracer.worst_headroom(10)
        ],
        # The full record list is what supports a before/after activation comparison.
        "records": [
            {"name": r["name"], "class": r["class"], "dtype": r["dtype"],
             "absmax": r["absmax"], "mean": r["mean"], "std": r["std"],
             "finite": not (r["has_nan"] or r["has_inf"])}
            for r in tracer.records
        ],
    }


def main() -> int:
    """Run the trace and report.

    Returns:
        Process exit code: 1 if any traced configuration produced a non-finite output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument(
        "--weight-scale",
        type=float,
        nargs="*",
        default=[1.0],
        help="Multiply every weight matrix by each of these before tracing.",
    )
    parser.add_argument("--random-data", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    config = build_sparc_config("sparc-base", use_attention=args.attention)
    model = SPARCNet(config).to(device).eval()
    if args.checkpoint is not None:
        payload = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        _LOGGER.info("Loaded weights from %s", args.checkpoint)

    x = load_batch(device, args.batch_size, use_real_data=not args.random_data)
    baseline = {name: param.detach().clone() for name, param in model.named_parameters()}

    results: list[dict[str, Any]] = []
    for scale in args.weight_scale:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(baseline[name])
                if param.ndim > 1:
                    param.mul_(scale)

        record = trace_once(model, x, device, dtype)
        record["weight_scale"] = scale
        results.append(record)

        first = record["first_nonfinite"]
        _LOGGER.info(
            "scale=%-5g output_finite=%-5s peak_activation=%-11.4g fp16_headroom=%-9.3g "
            "first_bad=%s",
            scale,
            record["output_finite"],
            record["peak_activation"],
            record["fp16_headroom"],
            f"{first['name']} ({first['class']}, {first['output_dtype']})"
            if first else "none",
        )

    print(f"\n{'scale':>8} {'finite':>7} {'peak act':>12} {'fp16 room':>11}  first non-finite module")
    print("-" * 86)
    for record in results:
        first = record["first_nonfinite"]
        print(
            f"{record['weight_scale']:>8g} {str(record['output_finite']):>7} "
            f"{record['peak_activation']:>12.4g} {record['fp16_headroom']:>11.3g}  "
            f"{first['name'] + ' (' + first['class'] + ')' if first else '-'}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"device": str(device), "dtype": args.dtype, "traces": results},
                       indent=2),
            encoding="utf-8",
        )
        _LOGGER.info("Wrote %s", args.json)

    return 0 if all(r["output_finite"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
