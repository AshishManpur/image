"""AMP audit: what actually runs in reduced precision under autocast (Task 3).

Autocast does not respect explicit casts. A tensor passed as ``x.float()`` into an
operation on autocast's reduced-precision list is cast straight back down before the
kernel runs, so ``.float()`` in a loss body is a comment, not a guarantee. This script
measures the truth rather than reading the source:

1. **Operation probe** — runs each operation this codebase uses under autocast, with
   float32 inputs, and records the output dtype. This is the ground truth for which
   operations are promoted to fp32 and which are demoted.
2. **Module probe** — runs the real SPARCNet under autocast with
   :class:`utils.numerics.ModuleTracer` and tabulates every module by output dtype,
   input dtype and fp16 headroom.
3. **Loss probe** — runs every term of the composite objective under autocast and
   reports the dtype its internal convolutions actually executed in.

Run it on the training device; the autocast dispatch lists are per-device-type, so a
CPU result does not by itself establish CUDA behaviour for every operation.

Usage::

    python -m scripts.amp_audit --device cuda --dtype fp16
    python -m scripts.amp_audit --device cuda --dtype bf16 --json reports/amp_bf16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from configs.sparc_config import build_sparc_config
from losses.composite_loss import CompositeLoss
from models.sparc_net import SPARCNet
from utils.logging_utils import configure_logging, get_logger
from utils.numerics import ModuleTracer, tensor_stats

_LOGGER = get_logger(__name__)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

CUDA_FP32_POLICY: frozenset[str] = frozenset(
    {
        "acos", "asin", "binary_cross_entropy_with_logits", "cdist", "cosh",
        "cosine_embedding_loss", "cosine_similarity", "cumprod", "cumsum", "dist",
        "erfinv", "exp", "expm1", "group_norm", "hinge_embedding_loss", "kl_div",
        "l1_loss", "layer_norm", "log", "log10", "log1p", "log2", "log_softmax",
        "logsumexp", "margin_ranking_loss", "mse_loss", "multi_margin_loss",
        "multilabel_margin_loss", "nll_loss", "nll_loss2d", "norm", "pdist",
        "poisson_nll_loss", "pow", "prod", "reciprocal", "renorm", "rsqrt", "sinh",
        "smooth_l1_loss", "soft_margin_loss", "softmax", "softplus", "sum", "tan",
        "triplet_margin_loss",
    }
)
"""Operations CUDA autocast promotes to fp32, from ``torch.testing._internal``.

Recorded here because **the CPU and CUDA autocast policies are not the same**, so a
CPU run of this script does not establish CUDA behaviour. The difference that matters
for Phase 4.10.1: on CUDA ``layer_norm`` is promoted to fp32, but ``mean``, ``var`` and
elementwise ``mul`` are in no list at all and simply follow their input dtype. A
hand-rolled channel LayerNorm built from ``mean``/``var``/``rsqrt`` therefore computes
its moments in fp16 on CUDA, where ``F.layer_norm`` would have computed them in fp32.
``rsqrt`` *is* promoted — but it is promoted after the variance has already overflowed,
so the promotion cannot help.
"""

CUDA_REDUCED_POLICY: frozenset[str] = frozenset(
    {
        "_convolution", "addbmm", "addmm", "addmv", "addr", "baddbmm", "bmm",
        "chain_matmul", "conv1d", "conv2d", "conv3d", "conv_tbc", "conv_transpose1d",
        "conv_transpose2d", "conv_transpose3d", "convolution", "cudnn_convolution",
        "cudnn_convolution_transpose", "einsum", "linear", "matmul", "mm", "mv",
        "prelu",
    }
)
"""Operations CUDA autocast demotes to the reduced dtype, whatever their input dtype.

``conv2d`` and ``linear`` are the entries that make an explicit ``.float()`` in a loss
body ineffective: autocast re-casts the arguments before the kernel runs.
"""


# --------------------------------------------------------------------- op probe
def _op_probe_cases(
    device: torch.device, dtype: torch.dtype
) -> dict[str, Callable[[], torch.Tensor]]:
    """Build the operation probes, keyed by a human-readable operation name.

    Each probe is run twice by :func:`probe_operations`, once on float32 inputs and
    once on ``dtype`` inputs, because neither alone classifies an operation. An
    operation that returns fp32 from fp32 input might be promoting or merely passing
    through; only feeding it reduced-precision input distinguishes the two.

    Args:
        device: Device to allocate probe tensors on.
        dtype: Reduced precision to build the low-precision probe inputs in.

    Returns:
        Mapping from operation label to a zero-argument callable returning a tensor.
    """
    return _build_cases(device, dtype)


def _build_cases(
    device: torch.device, dtype: torch.dtype
) -> dict[str, Callable[[], torch.Tensor]]:
    """Construct probe callables bound to tensors of a given dtype.

    Args:
        device: Device to allocate on.
        dtype: Input dtype for every probe tensor.

    Returns:
        Mapping from operation label to callable.
    """
    x = torch.randn(2, 8, 16, 16, device=device, dtype=dtype)
    w = torch.randn(8, 8, 3, 3, device=device, dtype=dtype)
    dw = torch.randn(8, 1, 3, 3, device=device, dtype=dtype)
    vec = torch.randn(2, 8, device=device, dtype=dtype)
    mat = torch.randn(8, 8, device=device, dtype=dtype)

    return {
        # Reduced-precision list: these are why .float() does not survive autocast.
        "conv2d": lambda: F.conv2d(x, w, padding=1),
        "conv2d(.float() inputs)": lambda: F.conv2d(x.float(), w.float(), padding=1),
        "conv2d depthwise": lambda: F.conv2d(x, dw, padding=1, groups=8),
        "conv_transpose2d": lambda: F.conv_transpose2d(x, w, padding=1),
        "linear": lambda: F.linear(vec, mat),
        "matmul": lambda: mat @ mat,
        "einsum": lambda: torch.einsum("bc,cd->bd", vec, mat),
        "scaled_dot_product_attention": lambda: F.scaled_dot_product_attention(
            x.flatten(2).transpose(1, 2),
            x.flatten(2).transpose(1, 2),
            x.flatten(2).transpose(1, 2),
        ),
        # fp32 list: autocast promotes these regardless of input dtype.
        "layer_norm (F.layer_norm)": lambda: F.layer_norm(x, (16,)),
        "softmax": lambda: F.softmax(x, dim=1),
        "log_softmax": lambda: F.log_softmax(x, dim=1),
        "softplus": lambda: F.softplus(x),
        "fft.rfft2": lambda: torch.fft.rfft2(x),
        "l1_loss": lambda: F.l1_loss(x, x * 0.5),
        "mse_loss": lambda: F.mse_loss(x, x * 0.5),
        # No autocast policy at all: dtype follows the input. These are exactly what a
        # hand-rolled LayerNorm is built from, which is why it forfeits the fp32
        # guarantee that F.layer_norm receives.
        "mean(dim=1)": lambda: x.mean(dim=1),
        "var(dim=1)": lambda: x.var(dim=1, unbiased=False),
        "rsqrt": lambda: torch.rsqrt(x.abs() + 1e-6),
        "sqrt": lambda: torch.sqrt(x.abs() + 1e-6),
        "log": lambda: torch.log(x.abs() + 1e-6),
        "pow(2)": lambda: x.pow(2),
        "elementwise mul": lambda: x * x,
        "avg_pool2d": lambda: F.avg_pool2d(x, 2),
        "interpolate bicubic": lambda: F.interpolate(
            x, scale_factor=2.0, mode="bicubic", align_corners=False
        ),
        "pad replicate": lambda: F.pad(x, [1, 1, 1, 1], mode="replicate"),
    }


def probe_operations(device: torch.device, dtype: torch.dtype) -> list[dict[str, Any]]:
    """Classify each probed operation's autocast policy.

    Every operation is run three ways: without autocast on fp32 inputs (the baseline),
    under autocast on fp32 inputs, and under autocast on reduced-precision inputs. The
    last is what separates "promoted to fp32" from "no policy, dtype follows input" —
    a distinction that decides whether an explicit ``.float()`` in the source survives.

    Args:
        device: Device to run on.
        dtype: Autocast dtype.

    Returns:
        One record per operation with all three observed dtypes and a ``policy``
        classification.
    """
    records: list[dict[str, Any]] = []
    fp32_cases = _op_probe_cases(device, torch.float32)
    low_cases = _op_probe_cases(device, dtype)
    low = str(dtype)

    for name in fp32_cases:
        try:
            with torch.no_grad():
                baseline = str(fp32_cases[name]().dtype)
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
                from_fp32 = str(fp32_cases[name]().dtype)
                from_low = str(low_cases[name]().dtype)
        except (RuntimeError, NotImplementedError) as exc:
            records.append(
                {"op": name, "baseline": "error", "autocast_from_fp32": "error",
                 "autocast_from_reduced": "error", "policy": "unsupported",
                 "note": str(exc)[:140]}
            )
            continue

        if from_fp32 == low:
            # Demotes fp32 down: an explicit .float() on the input does NOT survive.
            policy = "DEMOTES to reduced (.float() does not survive)"
        elif from_low == "torch.float32":
            # Upcasts reduced input: the fp32 guarantee holds whatever you pass.
            policy = "promotes to fp32 (safe)"
        elif from_fp32 == from_low:
            policy = "no policy (dtype follows input)"
        else:
            policy = "dtype follows input"
        records.append(
            {
                "op": name,
                "baseline": baseline,
                "autocast_from_fp32": from_fp32,
                "autocast_from_reduced": from_low,
                "policy": policy,
            }
        )
    return records


# ----------------------------------------------------------------- module probe
def probe_modules(
    device: torch.device, dtype: torch.dtype, use_attention: bool
) -> dict[str, Any]:
    """Trace every module of the real model under autocast.

    Args:
        device: Device to run on.
        dtype: Autocast dtype.
        use_attention: Whether to build the model with GSA groups.

    Returns:
        A mapping with the full record list and the summary tables.
    """
    config = build_sparc_config("sparc-base", use_attention=use_attention)
    model = SPARCNet(config).to(device).eval()
    x = (torch.rand(2, 1, 128, 128, device=device) * 0.6).float()

    tracer = ModuleTracer(model)
    with tracer, torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
        model.forward_with_aux(x)

    low = str(dtype)
    by_class: dict[str, dict[str, int]] = {}
    for record in tracer.records:
        bucket = by_class.setdefault(record["class"], {})
        bucket[record["dtype"]] = bucket.get(record["dtype"], 0) + 1

    # The normalisation layers are the interesting case: a fp32 affine parameter
    # promotes the *output* even when the moments were computed in reduced precision,
    # so the output dtype alone would report them as safe.
    norm_sites = [
        {
            "name": r["name"],
            "input_dtype": r["input_dtype"],
            "output_dtype": r["dtype"],
            "absmax": r["absmax"],
        }
        for r in tracer.records
        if r["class"] == "LayerNorm2d"
    ]

    return {
        "records": tracer.records,
        "by_class": by_class,
        "reduced_precision_outputs": len(tracer.by_dtype(low)),
        "fp32_outputs": len(tracer.by_dtype("torch.float32")),
        "layer_norm_sites": norm_sites,
        "layer_norm_reduced_inputs": [
            s["name"] for s in norm_sites if s["input_dtype"] == low
        ],
        "worst_headroom": [
            {"name": r["name"], "class": r["class"], "dtype": r["dtype"],
             "absmax": r["absmax"]}
            for r in tracer.worst_headroom(12)
        ],
    }


# ------------------------------------------------------------------- loss probe
def probe_losses(device: torch.device, dtype: torch.dtype) -> list[dict[str, Any]]:
    """Record the execution dtype inside each loss term under autocast.

    Each term is run on float32 inputs inside an autocast region — exactly how the
    trainer calls it. The probe records the dtype of the term's own internal
    convolution (where it has one) as well as the returned scalar's dtype, because a
    term can return fp32 while having done its real work in fp16.

    Args:
        device: Device to run on.
        dtype: Autocast dtype.

    Returns:
        One record per loss term.
    """
    criterion = CompositeLoss().to(device)
    pred = torch.rand(2, 1, 256, 256, device=device)
    target = torch.rand(2, 1, 256, 256, device=device)

    terms: dict[str, nn.Module | None] = {
        "charbonnier": criterion.charbonnier,
        "ms_ssim": criterion.ms_ssim,
        "wavelet": criterion.wavelet,
        "fft": criterion.fft,
        "gradient": criterion.gradient,
    }

    records: list[dict[str, Any]] = []
    for name, module in terms.items():
        if module is None:
            continue
        internal: list[str] = []
        original = F.conv2d

        def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = original(*args, **kwargs)
            internal.append(str(out.dtype))
            return out

        F.conv2d = spy  # type: ignore[assignment]
        try:
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
                value = module(pred, target)
        finally:
            F.conv2d = original  # type: ignore[assignment]

        records.append(
            {
                "term": name,
                "returned_dtype": str(value.dtype),
                "internal_conv_dtypes": sorted(set(internal)) or ["n/a (no conv2d)"],
                "value": float(value),
                "finite": bool(torch.isfinite(value)),
            }
        )
    return records


# ------------------------------------------------------------------------ report
def render_markdown(payload: dict[str, Any]) -> str:
    """Render the audit payload as a Markdown report.

    Args:
        payload: The mapping produced by :func:`main`.

    Returns:
        A Markdown document.
    """
    device, dtype = payload["device"], payload["dtype"]
    lines = [
        f"# AMP audit — device `{device}`, autocast dtype `{dtype}`",
        "",
        f"torch {payload['torch']}",
        "",
        "> **Autocast policy is per device type.** A CPU run does not establish CUDA",
        "> behaviour: `layer_norm` is promoted to fp32 on CUDA but not on CPU. The",
        "> reference CUDA lists are in `scripts/amp_audit.py`. Run this on the training",
        "> device before drawing conclusions about the training device.",
        "",
        "## 1. Operation policy",
        "",
        "| Operation | No autocast | autocast(fp32 in) | autocast(reduced in) | Policy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in payload["operations"]:
        lines.append(
            f"| `{r['op']}` | {r['baseline']} | **{r['autocast_from_fp32']}** | "
            f"{r['autocast_from_reduced']} | {r['policy']} |"
        )

    modules = payload["modules"]
    lines += [
        "",
        "## 2. Model modules",
        "",
        f"- Reduced-precision outputs: **{modules['reduced_precision_outputs']}**",
        f"- fp32 outputs: **{modules['fp32_outputs']}**",
        "",
        "### LayerNorm2d sites receiving reduced-precision input",
        "",
        "These compute mean, variance and rsqrt in reduced precision. The output dtype",
        "is fp32 only because the affine parameters are fp32 — the moments are not.",
        "",
    ]
    reduced = modules["layer_norm_reduced_inputs"]
    if reduced:
        for name in reduced:
            lines.append(f"- `{name}`")
    else:
        lines.append("- _none_")

    lines += ["", "### Largest activations (fp16 ceiling is 65504)", "",
              "| Module | Class | dtype | absmax |", "| --- | --- | --- | --- |"]
    for r in modules["worst_headroom"]:
        lines.append(
            f"| `{r['name']}` | {r['class']} | {r['dtype']} | {r['absmax']:.4g} |"
        )

    lines += [
        "",
        "## 3. Loss terms",
        "",
        "| Term | Returned dtype | Internal conv2d dtype | Finite |",
        "| --- | --- | --- | --- |",
    ]
    for r in payload["losses"]:
        lines.append(
            f"| {r['term']} | {r['returned_dtype']} | "
            f"{', '.join(r['internal_conv_dtypes'])} | {r['finite']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the audit and write the report.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    parser.add_argument("--attention", action="store_true", help="Build with GSA groups.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    _LOGGER.info("AMP audit on %s with autocast dtype %s", device, dtype)
    payload = {
        "device": str(device),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "operations": probe_operations(device, dtype),
        "modules": probe_modules(device, dtype, use_attention=args.attention),
        "losses": probe_losses(device, dtype),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOGGER.info("Wrote %s", args.json)
    markdown = render_markdown(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
        _LOGGER.info("Wrote %s", args.markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
