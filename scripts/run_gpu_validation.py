"""Phase 4.12 GPU validation — the whole sequence, one command, on the RTX A400.

Runs the six checks in dependency order and stops at the first failure, because a
later stage's numbers are meaningless if an earlier one broke:

1. **CUDA tests** — the GPU-marked tests that skip on a CPU host.
2. **20-step shakedown** — full model, CompositeLoss, backward, optimiser, BF16.
3. **3-epoch training** — real data, real loop, EMA, checkpoints, TensorBoard.
4. **Checkpoint resume** — reload `last.pt` and continue for one more epoch.
5. **Benchmark** — parameters, MACs, latency, VRAM against Part 10.
6. **Inference + visual comparison** — using the checkpoint step 3 produced.

Everything writes into `reports/` and `outputs/`; nothing touches the pre-attention
baseline checkpoints.

Usage::

    python scripts/run_gpu_validation.py
    python scripts/run_gpu_validation.py --skip benchmark inference   # partial re-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_NAME = "phase4_12_gpu_shakedown"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / RUN_NAME / "last.pt"
BEST = PROJECT_ROOT / "checkpoints" / RUN_NAME / "best_ema_psnr.pt"

STAGES: dict[str, list[str]] = {
    "cuda-tests": [
        sys.executable, "-m", "pytest",
        "tests/test_attention.py", "tests/test_inference.py",
        "tests/test_full_model_training.py", "-q", "-p", "no:warnings",
    ],
    "shakedown": [
        sys.executable, "scripts/cuda_shakedown.py",
        "--variant", "sparc-base", "--batch-size", "8", "--steps", "20",
        "--amp-dtype", "bf16", "--json", "reports/phase4_12_cuda.json",
    ],
    "train": [
        sys.executable, "train.py",
        "--variant", "sparc-base", "--epochs", "3", "--warmup-epochs", "1",
        "--batch-size", "8", "--amp-dtype", "bf16", "--run-name", RUN_NAME,
    ],
    "resume": [
        sys.executable, "train.py",
        "--variant", "sparc-base", "--epochs", "4", "--warmup-epochs", "1",
        "--batch-size", "8", "--amp-dtype", "bf16", "--run-name", RUN_NAME,
        "--resume", str(CHECKPOINT),
    ],
    "benchmark": [
        sys.executable, "scripts/benchmark.py",
        "--variant", "sparc-base", "--device", "cuda", "--iterations", "50",
        "--json", "reports/phase4_12_benchmark_cuda.json",
    ],
}

DIAGNOSTICS: dict[str, list[str]] = {
    "parity": [
        sys.executable, "scripts/parity_diagnostic.py",
        "--json", "reports/phase4_12_parity.json",
    ],
    "vram": [
        sys.executable, "scripts/vram_profile.py",
        "--batch-size", "8", "--amp-dtype", "bf16",
        "--json", "reports/phase4_12_vram.json",
    ],
}
"""Run with --diagnostics. These explain a failure; they are not pass/fail gates."""


def preflight() -> list[str]:
    """Check that this working copy actually contains the fixes it is about to test.

    The Phase 4.12 GPU run reported
    ``test_validation_autocast_follows_the_configured_dtype`` observing fp16 while the
    development machine's identical test passed. The cause was not code: the GPU host
    was running a **stale ``trainer.py``**. When files are synced selectively — the
    repo root carries a ``phase4_10_1_changes.zip``, so they have been — a new test can
    arrive without the fix it was written for, and the failure looks like a bug in
    code that is already correct.

    Each check below is a source-level invariant of a fix, so a partial sync is caught
    in seconds instead of after a multi-hour run.

    Returns:
        A list of human-readable problems; empty when the working copy is current.
    """
    problems: list[str] = []

    trainer = (PROJECT_ROOT / "trainer" / "trainer.py").read_text(encoding="utf-8")
    if "dtype=torch.float16" in trainer:
        problems.append(
            "trainer/trainer.py still contains a hardcoded `dtype=torch.float16` "
            "autocast. This file is stale — validation would be scored in fp16 even "
            "under --amp-dtype bf16."
        )
    if "def autocast(self)" not in trainer:
        problems.append(
            "trainer/trainer.py has no `Trainer.autocast` helper. This file predates "
            "the single-source-of-truth autocast fix."
        )

    infer = (PROJECT_ROOT / "scripts" / "infer.py").read_text(encoding="utf-8")
    if "true_float32" not in infer:
        problems.append(
            "scripts/infer.py has no `true_float32` guard. This file predates the TF32 "
            "fix, so CPU/CUDA fp32 parity will fail at ~2.4e-04."
        )

    rel_pos = (PROJECT_ROOT / "models" / "attention" / "rel_pos.py").read_text(
        encoding="utf-8"
    )
    if "inference_mode(False)" not in rel_pos:
        problems.append(
            "models/attention/rel_pos.py does not build its shared index under "
            "`torch.inference_mode(False)`. This file predates the inference-tensor "
            "fix, so any GSA block that reaches CUDA after an inference-mode test will "
            "fail with 'Inference tensors cannot be saved for backward'."
        )

    for path in (
        PROJECT_ROOT / "models" / "attention" / "gsa_block.py",
        PROJECT_ROOT / "models" / "attention" / "rel_pos.py",
        PROJECT_ROOT / "scripts" / "vram_profile.py",
        PROJECT_ROOT / "scripts" / "parity_diagnostic.py",
    ):
        if not path.exists():
            problems.append(f"missing: {path.relative_to(PROJECT_ROOT)}")

    from configs.sparc_config import sparc_base
    from models.sparc_net import SPARCNet

    total = sum(p.numel() for p in SPARCNet(sparc_base()).parameters())
    if total != 2_345_650:
        problems.append(
            f"sparc-base builds {total} parameters, expected 2,345,650. The model "
            "source is stale or altered."
        )
    return problems


def run(name: str, command: list[str]) -> dict[str, object]:
    """Run one stage, streaming its output, and record the outcome."""
    print("\n" + "=" * 78)
    print(f"  [{name}]  " + " ".join(command[1:]))
    print("=" * 78, flush=True)
    start = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  [{name}] {status} in {elapsed:.1f} s (exit {result.returncode})", flush=True)
    return {
        "stage": name,
        "command": " ".join(command),
        "returncode": result.returncode,
        "status": status,
        "seconds": elapsed,
    }


def inference_stage(lr_image: Path, gt_image: Path | None) -> list[dict[str, object]]:
    """Restore one real image with the 3-epoch checkpoint and render a comparison."""
    weights = BEST if BEST.exists() else CHECKPOINT
    outcomes = [
        run("inference", [
            sys.executable, "scripts/infer.py",
            "--weights", str(weights),
            "--input", str(lr_image),
            "--output", "outputs/phase4_12_gpu/restored.png",
            "--device", "cuda", "--amp-dtype", "bf16",
        ])
    ]
    command = [
        sys.executable, "scripts/visualize_restoration.py",
        "--weights", str(weights),
        "--input", str(lr_image),
        "--output", "outputs/phase4_12_gpu/comparison.png",
        "--device", "cuda", "--amp-dtype", "bf16",
    ]
    if gt_image is not None and gt_image.exists():
        command += ["--gt", str(gt_image)]
    outcomes.append(run("visualize", command))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4.12 GPU validation sequence.")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=[*STAGES, "inference"],
        help="Stages to skip (for partial re-runs).",
    )
    parser.add_argument(
        "--lr-image", type=Path,
        default=PROJECT_ROOT / "Data" / "train" / "train" / "NoisyLR" / "000000.npy",
        help="LR image for the inference stage.",
    )
    parser.add_argument(
        "--gt-image", type=Path,
        default=PROJECT_ROOT / "Data" / "train" / "train" / "GT" / "000000.npy",
        help="Matching ground truth, for PSNR/SSIM in the comparison.",
    )
    parser.add_argument(
        "--json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_12_gpu_validation.json"
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help=(
            "Run the parity and VRAM diagnostics instead of the validation gates. "
            "Use this when a gate fails and you need to know why."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print(
            "No CUDA device visible. This sequence measures GPU-denominated budgets "
            "and will not substitute CPU numbers. Run it on the RTX A400.",
            file=sys.stderr,
        )
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB), "
          f"torch {torch.__version__}")

    problems = preflight()
    if problems:
        print("\nPREFLIGHT FAILED — this working copy is not current:\n", file=sys.stderr)
        for problem in problems:
            print(f"  * {problem}", file=sys.stderr)
        print(
            "\nSync the full repository to this machine and re-run. Testing a stale "
            "copy produces failures in code that is already fixed.",
            file=sys.stderr,
        )
        return 3
    print("Preflight: working copy is current.")

    if args.diagnostics:
        outcomes = [run(name, command) for name, command in DIAGNOSTICS.items()]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print("\nDiagnostics complete. Read reports/phase4_12_parity.json and "
              "reports/phase4_12_vram.json.")
        return 0

    outcomes: list[dict[str, object]] = []
    for name, command in STAGES.items():
        if name in args.skip:
            print(f"  [{name}] SKIPPED by request")
            continue
        outcome = run(name, command)
        outcomes.append(outcome)
        if outcome["returncode"] != 0:
            print(f"\nStopped at '{name}'. Later stages depend on it; fix this first.")
            break
    else:
        if "inference" not in args.skip:
            if not (BEST.exists() or CHECKPOINT.exists()):
                print("  [inference] SKIPPED: no checkpoint from the training stage.")
            else:
                outcomes += inference_stage(args.lr_image, args.gt_image)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    for outcome in outcomes:
        print(f"  {outcome['stage']:<12s} {outcome['status']:<5s} "
              f"{outcome['seconds']:7.1f} s")
    print(f"  report: {args.json}")
    print("=" * 78)

    failed = [o for o in outcomes if o["returncode"] != 0]
    if failed:
        print(f"\nNOT READY FOR LONG TRAINING — {len(failed)} stage(s) failed.")
        return 1
    print("\nAll stages passed. Review reports/phase4_12_cuda.json and "
          "reports/phase4_12_benchmark_cuda.json against Part 10 before launching.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
