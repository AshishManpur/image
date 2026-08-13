"""Export SPARC-Net to TorchScript and ONNX, and verify numerical parity (Phase 4.15).

Prepared during Phase 4.7; **runnable once Phase 4.13 lands**.

Contract Part 9 acceptance:

* TorchScript — ``torch.jit.script`` succeeds and matches eager to **1e-5**.
* ONNX — export at **opset 17** succeeds; ``onnxruntime`` matches eager to **1e-3**.

Two export-path subtleties are handled here rather than discovered later:

1. **GSA must export through its explicit path.** Contract Part 2.6 mandates SDPA in
   training (it saves 20.8 MB/image) but requires the explicit
   ``matmul -> add bias -> softmax -> matmul`` formulation for export, matching SDPA to
   1e-4. This script flips ``use_sdpa`` off for the exported copy.
2. **Export runs in eval mode on a deep copy.** Exporting the live training module
   risks baking in training-mode behaviour and mutating the model being trained.

Usage::

    python scripts/export_onnx.py --checkpoint checkpoints/sparc-base/best_ema_psnr.pt
    python scripts/export_onnx.py --variant sparc-tiny --no-onnx
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import build_sparc_config  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

OPSET = 17
TORCHSCRIPT_TOLERANCE = 1e-5
ONNX_TOLERANCE = 1e-3


def load_model(variant: str, checkpoint: Path | None, export_safe: bool) -> SPARCNet:
    """Build a model for export, optionally restoring trained weights.

    Args:
        variant: SPARC variant name.
        checkpoint: Optional checkpoint. ``ema`` weights are preferred when present,
            since those are the weights the contract evaluates.
        export_safe: When ``True``, disable SDPA so attention takes the explicit path.

    Returns:
        The model in ``eval`` mode.

    Raises:
        FileNotFoundError: If ``checkpoint`` is given but missing.
    """
    overrides: dict[str, Any] = {"use_sdpa": False} if export_safe else {}
    model = SPARCNet(build_sparc_config(variant, **overrides))

    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint}.")
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if "ema" in payload and "module" in payload["ema"]:
            model.load_state_dict(payload["ema"]["module"])
            _LOGGER.info("Loaded EMA weights from %s", checkpoint)
        else:
            model.load_state_dict(payload["model"])
            _LOGGER.info("Loaded live weights from %s", checkpoint)

    return model.eval()


def export_torchscript(model: SPARCNet, sample: torch.Tensor, path: Path) -> dict[str, Any]:
    """Script the model and check parity against eager.

    Args:
        model: Model in eval mode.
        sample: Representative input.
        path: Destination ``.pt`` file.

    Returns:
        A result mapping with ``max_abs_diff`` and ``passed``.
    """
    scripted = torch.jit.script(copy.deepcopy(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))

    with torch.no_grad():
        reference = model(sample)
        produced = scripted(sample)
    diff = (reference - produced).abs().max().item()

    _LOGGER.info("TorchScript max|diff| = %.3e (tolerance %.0e)", diff, TORCHSCRIPT_TOLERANCE)
    return {"path": str(path), "max_abs_diff": diff, "passed": diff <= TORCHSCRIPT_TOLERANCE}


def export_onnx(
    model: SPARCNet, sample: torch.Tensor, path: Path, dynamic_batch: bool
) -> dict[str, Any]:
    """Export to ONNX and check parity with onnxruntime.

    Args:
        model: Model in eval mode.
        sample: Representative input.
        path: Destination ``.onnx`` file.
        dynamic_batch: Whether to mark the batch dimension dynamic.

    Returns:
        A result mapping with ``max_abs_diff`` and ``passed``. When ``onnxruntime`` is
        unavailable the export still runs and ``passed`` is ``None``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        copy.deepcopy(model),
        sample,
        str(path),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    _LOGGER.info("ONNX written to %s (opset %d)", path, OPSET)

    try:
        import onnxruntime
    except ImportError:
        _LOGGER.warning("onnxruntime not installed; parity check skipped.")
        return {"path": str(path), "max_abs_diff": None, "passed": None}

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"input": sample.numpy()})[0]
    with torch.no_grad():
        reference = model(sample)
    diff = (reference - torch.from_numpy(produced)).abs().max().item()

    _LOGGER.info("ONNX max|diff| = %.3e (tolerance %.0e)", diff, ONNX_TOLERANCE)
    return {"path": str(path), "max_abs_diff": diff, "passed": diff <= ONNX_TOLERANCE}


def main() -> int:
    """Entry point.

    Returns:
        0 if every attempted parity check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Export SPARC-Net.")
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/export"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--no-torchscript", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    model = load_model(args.variant, args.checkpoint, export_safe=True)
    sample = torch.randn(args.batch_size, 1, 128, 128)

    results: dict[str, Any] = {"variant": args.variant, "opset": OPSET}
    if not args.no_torchscript:
        results["torchscript"] = export_torchscript(
            model, sample, args.output_dir / f"{args.variant}.torchscript.pt"
        )
    if not args.no_onnx:
        results["onnx"] = export_onnx(
            model, sample, args.output_dir / f"{args.variant}.onnx", args.dynamic_batch
        )

    report_path = args.output_dir / f"{args.variant}_export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    checks = [v["passed"] for v in results.values() if isinstance(v, dict)]
    failed = [c for c in checks if c is False]
    if failed:
        _LOGGER.error("Export parity FAILED.")
        return 1
    _LOGGER.info("Export parity OK. Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
