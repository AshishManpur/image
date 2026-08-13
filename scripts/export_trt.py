"""Export SPARC-Net to TensorRT and verify parity (Phase 4.15, checklist item 23).

Prepared during Phase 4.7. **This script cannot run on the current development host**
— TensorRT requires CUDA, and the development machine is CPU-only. It is written to be
executed unchanged on the target card.

Target: **NVIDIA RTX A400, 4 GB VRAM**.
Contract acceptance: TensorRT output matches eager to **1e-2**.

Pipeline: eager -> ONNX (opset 17, explicit attention path) -> TensorRT engine.
The ONNX stage is delegated to ``scripts/export_onnx.py`` so there is exactly one
definition of how SPARC-Net becomes ONNX.

Prerequisites on the target machine::

    pip install tensorrt onnxruntime-gpu

Usage::

    python scripts/export_trt.py --checkpoint checkpoints/sparc-base/best_ema_psnr.pt
    python scripts/export_trt.py --precision fp16 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_onnx import export_onnx, load_model  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

_LOGGER = get_logger(__name__)

TRT_TOLERANCE = 1e-2
WORKSPACE_MB = 1024
"""Builder workspace. Kept modest: the RTX A400 has only 4 GB total."""


def require_cuda() -> None:
    """Fail early and clearly when no CUDA device is present.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires a CUDA device. The development host is CPU-only; "
            "run this script on the target machine (RTX A400)."
        )


def build_engine(
    onnx_path: Path, engine_path: Path, precision: str, batch_size: int
) -> Path:
    """Build a TensorRT engine from an ONNX file.

    Args:
        onnx_path: Source ONNX model.
        engine_path: Destination ``.engine`` file.
        precision: ``fp32`` or ``fp16``.
        batch_size: Fixed batch size for the optimisation profile.

    Returns:
        The engine path.

    Raises:
        ImportError: If the ``tensorrt`` package is unavailable.
        RuntimeError: If the build fails or the ONNX file cannot be parsed.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - target-machine dependency
        raise ImportError(
            "TensorRT export requires the `tensorrt` package: pip install tensorrt"
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB << 20)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            _LOGGER.warning("Platform reports no fast fp16; building anyway.")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    shape = (batch_size, 1, 128, 128)
    profile.set_shape("input", min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    _LOGGER.info("Building TensorRT engine (%s, batch %d)...", precision, batch_size)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    _LOGGER.info("Engine written to %s", engine_path)
    return engine_path


def verify_engine(
    engine_path: Path, model: torch.nn.Module, sample: torch.Tensor
) -> dict[str, Any]:
    """Run the engine and compare against eager PyTorch.

    Args:
        engine_path: Serialised engine.
        model: Eager reference model (CUDA, eval mode).
        sample: Input tensor on CUDA.

    Returns:
        A mapping with ``max_abs_diff`` and ``passed``.

    Raises:
        ImportError: If ``tensorrt`` or ``pycuda`` is unavailable.
        RuntimeError: If inference fails.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialise {engine_path}.")
    context = engine.create_execution_context()

    with torch.no_grad():
        reference = model(sample)
    produced = torch.empty_like(reference)

    context.set_tensor_address(engine.get_tensor_name(0), sample.contiguous().data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), produced.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("TensorRT inference failed.")
    torch.cuda.synchronize()

    diff = (reference - produced).abs().max().item()
    _LOGGER.info("TensorRT max|diff| = %.3e (tolerance %.0e)", diff, TRT_TOLERANCE)
    return {"max_abs_diff": diff, "passed": diff <= TRT_TOLERANCE}


def main() -> int:
    """Entry point.

    Returns:
        0 on parity success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Export SPARC-Net to TensorRT.")
    parser.add_argument("--variant", default="sparc-base")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/export"))
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)
    require_cuda()

    model = load_model(args.variant, args.checkpoint, export_safe=True)
    sample_cpu = torch.randn(args.batch_size, 1, 128, 128)

    onnx_path = args.output_dir / f"{args.variant}.onnx"
    export_onnx(model, sample_cpu, onnx_path, dynamic_batch=False)

    engine_path = build_engine(
        onnx_path,
        args.output_dir / f"{args.variant}.{args.precision}.engine",
        args.precision,
        args.batch_size,
    )

    result = verify_engine(
        engine_path, model.cuda().eval(), sample_cpu.cuda()
    )
    result.update(
        {"variant": args.variant, "precision": args.precision, "engine": str(engine_path)}
    )

    report_path = args.output_dir / f"{args.variant}_trt_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not result["passed"]:
        _LOGGER.error("TensorRT parity FAILED.")
        return 1
    _LOGGER.info("TensorRT parity OK. Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
