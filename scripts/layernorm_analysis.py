"""LayerNorm2d numerical analysis (Task 2).

:class:`models.blocks.layer_norm.LayerNorm2d` is hand-rolled from ``mean``, ``var`` and
``rsqrt`` rather than calling ``F.layer_norm``, to avoid two permutes per call. That
choice has a cost the docstring did not record: ``F.layer_norm`` is on CUDA autocast's
fp32 promotion list, whereas ``mean`` and ``var`` are on no list at all and simply
follow their input dtype. Under fp16 autocast the moments are therefore computed in
fp16, and a variance is a sum of squares.

fp16 saturates at 65504, so a channel variance overflows once activations reach about
256 in magnitude. What follows is worse than an obvious crash: ``rsqrt(inf)`` is ``0``,
so the layer returns all zeros (or its bias) and silently deletes the signal instead of
raising. If the input itself has already overflowed to ``inf``, ``inf - inf`` yields
NaN directly.

This script measures three things:

1. The exact activation magnitude at which the fp16 variance overflows.
2. What the layer returns on either side of that threshold, fp16 versus fp32 moments.
3. The agreement between the fp32-moment implementation and the fp16 one in the safe
   regime, which is what establishes that the fix changes stability and not maths.

Usage::

    python -m scripts.layernorm_analysis --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from models.blocks.layer_norm import LayerNorm2d
from utils.logging_utils import configure_logging, get_logger
from utils.numerics import FP16_MAX

_LOGGER = get_logger(__name__)


def reference_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Channel LayerNorm with moments forced to float64 — the ground truth.

    float64 rather than float32 so that the fp32 implementation is being compared
    against something strictly better than itself, not against its own arithmetic.

    Args:
        x: Tensor ``(B, C, H, W)``.
        eps: Variance epsilon.

    Returns:
        The normalised tensor in float64.
    """
    x64 = x.double()
    mean = x64.mean(dim=1, keepdim=True)
    var = x64.var(dim=1, keepdim=True, unbiased=False)
    return (x64 - mean) * torch.rsqrt(var + eps)


def overflow_sweep(
    device: torch.device, channels: int = 48, eps: float = 1e-6
) -> list[dict[str, Any]]:
    """Sweep activation magnitude and record where the fp16 variance overflows.

    Args:
        device: Device to run on.
        channels: Channel count; the variance is a mean of ``channels`` squares.
        eps: Variance epsilon.

    Returns:
        One record per magnitude.
    """
    torch.manual_seed(0)
    records: list[dict[str, Any]] = []
    magnitudes = [1, 4, 16, 64, 128, 181, 200, 256, 300, 362, 512, 1024, 4096]

    for magnitude in magnitudes:
        base = torch.randn(4, channels, 8, 8, device=device) * magnitude
        x16 = base.half()

        var16 = x16.var(dim=1, keepdim=True, unbiased=False)
        var32 = base.float().var(dim=1, keepdim=True, unbiased=False)
        out16 = (x16 - x16.mean(dim=1, keepdim=True)) * torch.rsqrt(var16 + eps)
        out32 = (
            base.float() - base.float().mean(dim=1, keepdim=True)
        ) * torch.rsqrt(var32 + eps)

        records.append(
            {
                "magnitude": magnitude,
                "input_absmax": float(base.abs().max()),
                # The overflow threshold is set by the channel standard deviation, not
                # by absmax: the variance is a *mean* of squares, so it reaches 65504
                # when sigma reaches ~256, which for Gaussian data is an absmax of
                # roughly 3.5x that.
                "input_channel_std": float(
                    base.float().std(dim=1, unbiased=False).max()
                ),
                "var_fp16": float(var16.max()),
                "var_fp32": float(var32.max()),
                "var_fp16_overflowed": bool(torch.isinf(var16).any()),
                "input_fp16_overflowed": bool(torch.isinf(x16).any()),
                "out_fp16_has_nan": bool(torch.isnan(out16).any()),
                "out_fp16_absmax": float(out16.float().abs().max()),
                "out_fp32_absmax": float(out32.abs().max()),
                # Once the variance overflows, rsqrt(inf) is 0 and the layer returns
                # zeros. This is the silent failure: no NaN, no error, no signal.
                "out_fp16_all_zero": bool((out16 == 0).all()),
            }
        )
    return records


def theoretical_threshold(channels: int) -> dict[str, float]:
    """Closed-form magnitude at which an fp16 channel variance overflows.

    The variance is ``mean((x - mu)^2)`` over ``C`` channels. The intermediate squares
    are what overflow, not the mean, so the threshold is set by a single element:
    ``x^2 > 65504``.

    Args:
        channels: Channel count, recorded for context.

    Returns:
        The threshold and the fp16 ceiling.
    """
    return {
        "channels": float(channels),
        "fp16_max": FP16_MAX,
        "elementwise_square_overflow_at": float(FP16_MAX**0.5),
    }


def agreement_check(device: torch.device, eps: float = 1e-6) -> dict[str, Any]:
    """Compare fp32-moment and fp16-moment LayerNorm against a float64 reference.

    Establishes that computing the moments in fp32 is a strict improvement in the safe
    regime rather than a change of definition.

    Args:
        device: Device to run on.
        eps: Variance epsilon.

    Returns:
        Max absolute deviation from the float64 reference for each implementation.
    """
    torch.manual_seed(0)
    layer = LayerNorm2d(48, eps=eps, affine=False).to(device)
    results: dict[str, Any] = {}

    for magnitude in (1.0, 10.0, 100.0):
        x = torch.randn(4, 48, 16, 16, device=device) * magnitude
        reference = reference_normalize(x, eps)

        fp32_moments = layer(x.float())
        x16 = x.half()
        fp16_moments = (
            x16 - x16.mean(dim=1, keepdim=True)
        ) * torch.rsqrt(x16.var(dim=1, keepdim=True, unbiased=False) + eps)

        results[f"magnitude_{magnitude:g}"] = {
            "fp32_moments_max_abs_err": float(
                (fp32_moments.double() - reference).abs().max()
            ),
            "fp16_moments_max_abs_err": float(
                (fp16_moments.double() - reference).abs().max()
            ),
        }
    return results


def main() -> int:
    """Run the analysis and print or persist the results.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    device = torch.device(args.device)

    payload = {
        "device": str(device),
        "threshold": theoretical_threshold(args.channels),
        "sweep": overflow_sweep(device, channels=args.channels),
        "agreement": agreement_check(device),
    }

    threshold = payload["threshold"]
    print(f"\nfp16 max = {threshold['fp16_max']:.0f};  x^2 overflows at |x| > "
          f"{threshold['elementwise_square_overflow_at']:.1f}\n")
    header = (
        f"{'|x|max':>10} {'var fp16':>12} {'var fp32':>12} "
        f"{'var ovf':>8} {'out NaN':>8} {'out all-0':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["sweep"]:
        print(
            f"{row['input_absmax']:>10.4g} {row['var_fp16']:>12.4g} "
            f"{row['var_fp32']:>12.4g} {str(row['var_fp16_overflowed']):>8} "
            f"{str(row['out_fp16_has_nan']):>8} {str(row['out_fp16_all_zero']):>10}"
        )

    print("\nAgreement against a float64 reference (max abs error):")
    for name, row in payload["agreement"].items():
        print(
            f"  {name:<16} fp32 moments {row['fp32_moments_max_abs_err']:.3e}   "
            f"fp16 moments {row['fp16_moments_max_abs_err']:.3e}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOGGER.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
