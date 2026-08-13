"""Loss stability matrix (Task 4).

Every term of the Contract Part 6 objective is exercised in isolation across the axes
that matter for a mixed-precision training run:

* **fp32** and **autocast** (fp16 and bf16) execution,
* **forward** finiteness and **backward** gradient finiteness,
* **TorchScript** scriptability, required by Contract Part 9,
* **torch.compile**, where the toolchain supports it,
* **determinism** — the same inputs must give bit-identical outputs twice.

Each term is additionally run on adversarial inputs, not just well-behaved random ones:
saturated images, near-identical pairs (where SSIM's contrast term and the FFT
magnitude's square root both approach their singularities), and constant images (zero
variance everywhere). A term that is finite on ``torch.rand`` but not on a flat patch
will still take down a real run.

Usage::

    python -m scripts.loss_stability --device cuda
    python -m scripts.loss_stability --device cuda --compile --json reports/loss.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from configs.sparc_config import LossConfig
from losses.charbonnier import CharbonnierLoss
from losses.fft_loss import FFTLoss
from losses.gradient import GradientLoss
from losses.ms_ssim import MSSSIMLoss
from losses.wavelet_loss import WaveletLoss
from utils.logging_utils import configure_logging, get_logger

_LOGGER = get_logger(__name__)

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_terms(config: LossConfig | None = None) -> dict[str, nn.Module]:
    """Construct every loss term that takes a ``(pred, target)`` pair.

    The noise term is excluded: its signature is ``(sigma_hat, noisy_lr, gt)`` and its
    target comes from the degradation model, so it is not comparable on this matrix and
    is covered by ``tests/test_losses.py`` instead.

    Args:
        config: Loss configuration. Defaults to the contract values.

    Returns:
        Mapping from term name to module.
    """
    cfg = config or LossConfig()
    return {
        "charbonnier": CharbonnierLoss(eps=cfg.charbonnier_eps),
        "ms_ssim": MSSSIMLoss(
            scales=cfg.ms_ssim_scales,
            window_size=cfg.ms_ssim_window,
            sigma=cfg.ms_ssim_sigma,
        ),
        "wavelet": WaveletLoss(
            levels=cfg.wavelet_levels, band_weights=cfg.wavelet_band_weights
        ),
        "fft": FFTLoss(),
        "gradient": GradientLoss(),
    }


def build_inputs(
    device: torch.device, size: int = 256
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build the adversarial input cases.

    Args:
        device: Device to allocate on.
        size: Spatial size; must admit 5 MS-SSIM scales, so at least 161.

    Returns:
        Mapping from case name to ``(prediction, target)``.
    """
    torch.manual_seed(0)
    shape = (2, 1, size, size)
    rand = torch.rand(shape, device=device)

    return {
        # Ordinary case.
        "random": (torch.rand(shape, device=device), torch.rand(shape, device=device)),
        # Identical pair: MS-SSIM's contrast term hits 1.0 and its clamp_min(1e-6)
        # guard, and the FFT magnitude difference is exactly zero where sqrt' is
        # unbounded. Both are the classic "loss is 0 but the gradient is NaN" traps.
        "identical": (rand, rand.clone()),
        # Constant images: zero variance in every SSIM window, zero Sobel response,
        # and an all-zero spectrum away from DC.
        "constant": (
            torch.full(shape, 0.5, device=device),
            torch.full(shape, 0.5, device=device),
        ),
        # Saturated extremes: the model clamps to [0, 1], so this is reachable.
        "saturated": (
            torch.zeros(shape, device=device),
            torch.ones(shape, device=device),
        ),
        # Tiny differences, where a fp16 subtraction loses every significant digit.
        "near_identical": (rand, rand + 1e-4),
    }


def _run_case(
    module: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Run one term on one input case, forward and backward.

    Args:
        module: Loss module.
        pred: Prediction tensor.
        target: Target tensor.
        device: Device to run on.
        dtype: Autocast dtype, or ``None`` for plain fp32.

    Returns:
        A record with the loss value and the forward/backward finiteness verdicts.
    """
    pred = pred.clone().detach().requires_grad_(True)
    try:
        if dtype is None:
            value = module(pred, target)
        else:
            with torch.amp.autocast(device.type, dtype=dtype):
                value = module(pred, target)

        forward_finite = bool(torch.isfinite(value).all())
        value.float().backward()
        grad = pred.grad
        backward_finite = grad is not None and bool(torch.isfinite(grad).all())
        return {
            "value": float(value.detach()),
            "value_dtype": str(value.dtype),
            "forward_finite": forward_finite,
            "backward_finite": backward_finite,
            "grad_absmax": float(grad.abs().max()) if grad is not None else None,
            "ok": forward_finite and backward_finite,
        }
    except (RuntimeError, NotImplementedError) as exc:
        return {"error": str(exc)[:200], "ok": False}


def _check_deterministic(
    module: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> bool:
    """Whether two identical fp32 calls return bit-identical results.

    Args:
        module: Loss module.
        pred: Prediction tensor.
        target: Target tensor.

    Returns:
        ``True`` if the two results are bitwise equal.
    """
    with torch.no_grad():
        return bool(torch.equal(module(pred, target), module(pred, target)))


def _check_scriptable(
    module: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    """Whether the term scripts and matches eager to 1e-5 (Contract Part 9).

    Args:
        module: Loss module.
        pred: Prediction tensor.
        target: Target tensor.

    Returns:
        A record with the verdict and, on failure, the error.
    """
    try:
        scripted = torch.jit.script(module)
        with torch.no_grad():
            matches = bool(
                torch.allclose(scripted(pred, target), module(pred, target), atol=1e-5)
            )
        return {"scriptable": True, "matches_eager": matches}
    except Exception as exc:  # noqa: BLE001 - torch.jit raises many types
        return {"scriptable": False, "error": str(exc)[:200]}


def _check_compilable(
    module: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    """Whether the term survives ``torch.compile`` and matches eager.

    Compilation is opt-in because it needs a working Triton toolchain, which the
    Phase 4.10 shakedown log shows is absent on at least one of the machines involved
    ("triton not found"). A missing compiler is reported, not raised.

    Args:
        module: Loss module.
        pred: Prediction tensor.
        target: Target tensor.

    Returns:
        A record with the verdict.
    """
    try:
        compiled = torch.compile(module, fullgraph=False)
        with torch.no_grad():
            matches = bool(
                torch.allclose(compiled(pred, target), module(pred, target), atol=1e-4)
            )
        return {"compilable": True, "matches_eager": matches}
    except Exception as exc:  # noqa: BLE001 - inductor raises many types
        return {"compilable": False, "error": str(exc)[:200]}


def run_matrix(
    device: torch.device, size: int, try_compile: bool
) -> dict[str, Any]:
    """Run the full stability matrix.

    Args:
        device: Device to run on.
        size: Spatial size of the probe images.
        try_compile: Whether to attempt ``torch.compile``.

    Returns:
        The results payload.
    """
    terms = {name: module.to(device) for name, module in build_terms().items()}
    cases = build_inputs(device, size=size)
    dtypes = {
        name: dtype
        for name, dtype in DTYPES.items()
        # Autocast dtype support is per device type; skip what the device cannot do.
        if dtype is None
        or device.type == "cpu"
        or (dtype is not torch.bfloat16 or torch.cuda.is_bf16_supported())
    }

    results: dict[str, Any] = {}
    for term_name, module in terms.items():
        pred, target = cases["random"]
        entry: dict[str, Any] = {
            "deterministic": _check_deterministic(module, pred, target),
            "torchscript": _check_scriptable(module, pred, target),
            "cases": {},
        }
        if try_compile:
            entry["torch_compile"] = _check_compilable(module, pred, target)

        for case_name, (case_pred, case_target) in cases.items():
            entry["cases"][case_name] = {
                dtype_name: _run_case(module, case_pred, case_target, device, dtype)
                for dtype_name, dtype in dtypes.items()
            }
        results[term_name] = entry

    return {"device": str(device), "size": size, "terms": results}


def render(payload: dict[str, Any]) -> str:
    """Render the matrix as a text table.

    Args:
        payload: Result payload from :func:`run_matrix`.

    Returns:
        A printable report.
    """
    lines = [f"Loss stability matrix — device {payload['device']}, "
             f"{payload['size']}x{payload['size']}", ""]
    for term_name, entry in payload["terms"].items():
        script = entry["torchscript"]
        script_text = (
            f"script={script['scriptable']}/match={script.get('matches_eager')}"
            if script["scriptable"] else f"script=FAILED ({script.get('error', '')[:60]})"
        )
        compile_text = ""
        if "torch_compile" in entry:
            comp = entry["torch_compile"]
            compile_text = (
                f"  compile={comp['compilable']}/match={comp.get('matches_eager')}"
                if comp["compilable"] else "  compile=unavailable"
            )
        lines.append(
            f"### {term_name}   deterministic={entry['deterministic']}  "
            f"{script_text}{compile_text}"
        )
        header = f"  {'case':<16}" + "".join(f"{d:>26}" for d in DTYPES)
        lines.append(header)
        for case_name, by_dtype in entry["cases"].items():
            cells = []
            for dtype_name in DTYPES:
                record = by_dtype.get(dtype_name)
                if record is None:
                    cells.append(f"{'skipped':>26}")
                elif "error" in record:
                    cells.append(f"{'ERROR':>26}")
                else:
                    mark = "ok " if record["ok"] else "BAD"
                    cells.append(
                        f"{mark} f={str(record['forward_finite'])[0]} "
                        f"b={str(record['backward_finite'])[0]} "
                        f"v={record['value']:.4g}".rjust(26)
                    )
            lines.append(f"  {case_name:<16}" + "".join(cells))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run the matrix and report.

    Returns:
        Process exit code: 1 if any configuration produced a non-finite result.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--compile", action="store_true", help="Also try torch.compile.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    device = torch.device(args.device)
    payload = run_matrix(device, args.size, args.compile)
    print(render(payload))

    failures = [
        f"{term}/{case}/{dtype}"
        for term, entry in payload["terms"].items()
        for case, by_dtype in entry["cases"].items()
        for dtype, record in by_dtype.items()
        if not record.get("ok", False)
    ]
    if failures:
        _LOGGER.error("%d non-finite configurations: %s", len(failures), failures)
    else:
        _LOGGER.info("All configurations finite in forward and backward.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOGGER.info("Wrote %s", args.json)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
