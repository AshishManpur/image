"""Pack the raw dataset into memory-mapped arrays (Contract Part 8, step 2).

Read-only with respect to ``Data/``. Writes ``data/packed/{train_lr,train_gt,
test_lr}.npy`` plus ``manifest.json``.

Verification performed by this script, all of which must pass:
  * exact file counts (3200 / 3200 / 400) after filtering ``__MACOSX`` decoys;
  * every LR stem has exactly one GT stem;
  * every array loads, is 2-D float32, and has the expected shape;
  * float16 round-trip PSNR >= 70 dB, otherwise packing aborts.

Usage::

    python scripts/pack_dataset.py [--dtype float16] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sparc_config import DataConfig  # noqa: E402
from utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

MIN_ROUNDTRIP_PSNR_DB = 70.0


def discover(directory: Path) -> list[Path]:
    """List real ``.npy`` files, excluding macOS AppleDouble decoys.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of genuine ``.npy`` paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    return sorted(
        p
        for p in directory.rglob("*.npy")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def roundtrip_psnr(reference: np.ndarray, dtype: str) -> float:
    """PSNR of a cast-and-restore round trip, in dB.

    Args:
        reference: Float32 array in ``[0, 1]``-ish units.
        dtype: Storage dtype name.

    Returns:
        PSNR in dB, or ``inf`` when the round trip is exact.
    """
    restored = reference.astype(dtype).astype(np.float32)
    mse = float(np.mean((reference - restored) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def pack_split(
    files: list[Path], out_path: Path, size: int, dtype: str, label: str
) -> dict[str, object]:
    """Pack one split into a single array and verify it.

    Args:
        files: Ordered list of source ``.npy`` files.
        out_path: Destination ``.npy`` path.
        size: Expected spatial size.
        dtype: Storage dtype.
        label: Human-readable split name for logging.

    Returns:
        Manifest entry for this split.

    Raises:
        ValueError: If any array has an unexpected shape/dtype, or the float16
            round-trip PSNR falls below the acceptance threshold.
    """
    count = len(files)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.dtype(dtype), shape=(count, size, size)
    )

    worst_psnr = float("inf")
    stems: list[str] = []
    for position, path in enumerate(files):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or array.shape != (size, size):
            raise ValueError(
                f"{path.name}: expected shape ({size}, {size}), got {array.shape}."
            )
        if array.dtype != np.float32:
            raise ValueError(f"{path.name}: expected float32, got {array.dtype}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{path.name}: contains NaN or Inf.")
        if dtype != "float32" and position % 200 == 0:
            worst_psnr = min(worst_psnr, roundtrip_psnr(array, dtype))
        packed[position] = array.astype(dtype)
        stems.append(path.stem)

    packed.flush()
    del packed

    if dtype != "float32" and worst_psnr < MIN_ROUNDTRIP_PSNR_DB:
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"{label}: float16 round-trip PSNR {worst_psnr:.1f} dB is below the "
            f"{MIN_ROUNDTRIP_PSNR_DB} dB acceptance threshold. Re-run with "
            f"--dtype float32."
        )

    _LOGGER.info(
        "Packed %-8s %4d x %d^2 -> %s (%.1f MB, worst round-trip %.1f dB)",
        label,
        count,
        size,
        out_path.name,
        out_path.stat().st_size / 1e6,
        worst_psnr,
    )
    return {
        "count": count,
        "size": size,
        "dtype": dtype,
        "path": out_path.name,
        "first_stem": stems[0],
        "last_stem": stems[-1],
        "roundtrip_psnr_db": None if worst_psnr == float("inf") else round(worst_psnr, 2),
    }


def main() -> int:
    """Entry point.

    Returns:
        Process exit code: 0 on success, 1 on verification failure.
    """
    parser = argparse.ArgumentParser(description="Pack the SPARC dataset.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing pack.")
    args = parser.parse_args()

    configure_logging()
    config = DataConfig()
    out_root = Path(config.packed_root)

    if (out_root / "manifest.json").exists() and not args.force:
        _LOGGER.error("Pack already exists at %s. Use --force to overwrite.", out_root)
        return 1

    raw = Path(config.raw_root)
    train_lr = discover(raw / config.train_lr_subpath)
    train_gt = discover(raw / config.train_gt_subpath)
    test_lr = discover(raw / config.test_lr_subpath)

    if len(train_lr) != config.expected_train:
        _LOGGER.error("Train LR: expected %d files, found %d.", config.expected_train, len(train_lr))
        return 1
    if len(train_gt) != config.expected_train:
        _LOGGER.error("Train GT: expected %d files, found %d.", config.expected_train, len(train_gt))
        return 1
    if len(test_lr) != config.expected_test:
        _LOGGER.error("Test LR: expected %d files, found %d.", config.expected_test, len(test_lr))
        return 1

    lr_stems = [p.stem for p in train_lr]
    gt_stems = [p.stem for p in train_gt]
    if lr_stems != gt_stems:
        missing = set(lr_stems) ^ set(gt_stems)
        _LOGGER.error("LR/GT stems do not match; %d unpaired stems.", len(missing))
        return 1

    _LOGGER.info("Verified %d paired train samples and %d test samples.", len(train_lr), len(test_lr))

    try:
        manifest = {
            "dtype": args.dtype,
            "lr_size": config.lr_size,
            "gt_size": config.gt_size,
            "train_lr": pack_split(
                train_lr, out_root / "train_lr.npy", config.lr_size, args.dtype, "train_lr"
            ),
            "train_gt": pack_split(
                train_gt, out_root / "train_gt.npy", config.gt_size, args.dtype, "train_gt"
            ),
            "test_lr": pack_split(
                test_lr, out_root / "test_lr.npy", config.lr_size, args.dtype, "test_lr"
            ),
        }
    except ValueError as exc:
        _LOGGER.error("Packing failed: %s", exc)
        return 1

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _LOGGER.info("Manifest written to %s", out_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
