"""One-step CUDA memory probe for SPARC training variants.

Runs a realistic training step and reports both the live and peak CUDA allocator
counters. On a CPU-only host it exits honestly after reporting that CUDA is unavailable.

Usage::

    python scripts/cuda_memory_probe.py --variant sparc-xl-moderate --batch-size 8 --amp-dtype bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import build_sparc_config  # noqa: E402
from losses.composite_loss import CompositeLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from utils.complexity import count_parameters  # noqa: E402

AMP_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,
}


def _mb(value: int) -> float:
    return value / (1024.0 * 1024.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="sparc-xl-moderate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--amp-dtype", choices=sorted(AMP_DTYPES), default="bf16")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "variant": args.variant,
        "torch.cuda.is_available": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "input_resolution": [args.input_size, args.input_size],
        "amp_dtype": args.amp_dtype,
    }
    if not torch.cuda.is_available():
        report["status"] = "CUDA unavailable; probe not executed."
        return report

    device = torch.device("cuda")
    report["gpu_name"] = torch.cuda.get_device_name(device)

    config = build_sparc_config(args.variant)
    model = SPARCNet(config).to(device).to(memory_format=torch.channels_last).train()
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameters, _ = count_parameters(model)
    report["parameter_count"] = parameters

    dtype = AMP_DTYPES[args.amp_dtype]
    sample = torch.rand(args.batch_size, 1, args.input_size, args.input_size, device=device)
    sample = sample.to(memory_format=torch.channels_last)
    target = torch.rand(
        args.batch_size, 1, args.input_size * config.scale, args.input_size * config.scale,
        device=device,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
        loss, _ = criterion(model.forward_with_aux(sample), {"gt": target, "lr": sample})
    loss.backward()

    report["torch.cuda.memory_allocated"] = torch.cuda.memory_allocated(device)
    report["torch.cuda.memory_reserved"] = torch.cuda.memory_reserved(device)
    report["torch.cuda.max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
    report["torch.cuda.max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
    report["peak_allocated_mb"] = _mb(report["torch.cuda.max_memory_allocated"])
    report["peak_reserved_mb"] = _mb(report["torch.cuda.max_memory_reserved"])

    optimizer.zero_grad(set_to_none=True)
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_probe(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
