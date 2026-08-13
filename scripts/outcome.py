from pathlib import Path
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================

# You are running this script from:
# Image restoration/scripts/

LR_DIR = Path("../Data/train/train/NoisyLR")

# IMPORTANT:
# Change this to the folder containing your training GT images.
GT_DIR = Path("../Data/train/train/GT")

PRE_WEIGHTS = Path(
    "../checkpoints/pre_attention_baseline_50ep/best_ema_psnr.pt"
)

ATTN_WEIGHTS = Path(
    "../checkpoints/sparc_base_50/best_ema_psnr.pt"
)

TEMP_LR_DIR = Path("../Data/_comparison_10_lr")

PRE_OUTPUT = Path("../outputs/comparison_pre_attention_10")
ATTN_OUTPUT = Path("../outputs/comparison_attention_10")

FINAL_OUTPUT = Path("../outputs/train_10_comparison")

NUM_IMAGES = 10


# ============================================================
# HELPERS
# ============================================================

def load_array(path):
    """Load npy/png/etc and return float32 image."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))

    arr = np.asarray(arr).astype(np.float32)
    arr = np.nan_to_num(arr)

    # Convert HxWx1 -> HxW
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Handle common 8-bit / 16-bit data.
    max_value = arr.max() if arr.size else 1

    if max_value > 255:
        arr = arr / 65535.0
    elif max_value > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def find_gt(stem):
    """
    Find GT image with the same filename stem.

    Searches recursively under GT_DIR.
    Supports npy/png/jpg/tif/tiff.
    """
    extensions = [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    for ext in extensions:
        matches = list(GT_DIR.rglob(stem + ext))
        if matches:
            return matches[0]

    return None


def to_display(arr, size=(256, 256)):
    """Convert image array to displayable PIL image."""
    arr = np.clip(arr, 0, 1)

    img = Image.fromarray(
        (arr * 255).astype(np.uint8),
        mode="L"
    )

    return img.resize(size, Image.Resampling.BICUBIC)


def psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)


def ssim_score(pred, gt):
    try:
        from skimage.metrics import structural_similarity

        return structural_similarity(
            pred,
            gt,
            data_range=1.0
        )
    except ImportError:
        return None


# ============================================================
# CREATE TEMPORARY 10-IMAGE DATASET
# ============================================================

TEMP_LR_DIR.mkdir(parents=True, exist_ok=True)

for old_file in TEMP_LR_DIR.iterdir():
    if old_file.is_file():
        old_file.unlink()

lr_files = sorted(LR_DIR.glob("*.npy"))[:NUM_IMAGES]

if len(lr_files) < NUM_IMAGES:
    raise RuntimeError(
        f"Only found {len(lr_files)} LR images in {LR_DIR}"
    )

print("\nSelected training images:")

for p in lr_files:
    print(" ", p.name)
    shutil.copy2(p, TEMP_LR_DIR / p.name)


# ============================================================
# RUN PRE-ATTENTION INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("PRE-ATTENTION 50-EPOCH MODEL")
print("=" * 70)

PRE_OUTPUT.mkdir(parents=True, exist_ok=True)

cmd_pre = [
    "python",
    "infer.py",
    "--weights",
    str(PRE_WEIGHTS),
    "--input-dir",
    str(TEMP_LR_DIR),
    "--output-dir",
    str(PRE_OUTPUT),
    "--device",
    "cuda",
    "--amp-dtype",
    "bf16",
]

subprocess.run(cmd_pre, check=True)


# ============================================================
# RUN ATTENTION INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("ATTENTION 50-EPOCH MODEL")
print("=" * 70)

ATTN_OUTPUT.mkdir(parents=True, exist_ok=True)

cmd_attn = [
    "python",
    "infer.py",
    "--weights",
    str(ATTN_WEIGHTS),
    "--input-dir",
    str(TEMP_LR_DIR),
    "--output-dir",
    str(ATTN_OUTPUT),
    "--device",
    "cuda",
    "--amp-dtype",
    "bf16",
]

subprocess.run(cmd_attn, check=True)


# ============================================================
# CREATE COMPARISONS
# ============================================================

FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

results = []

print("\n" + "=" * 70)
print("CREATING COMPARISONS")
print("=" * 70)

for lr_path in lr_files:

    stem = lr_path.stem

    gt_path = find_gt(stem)

    if gt_path is None:
        print(f"\nWARNING: GT not found for {stem}")
        continue

    pre_path = PRE_OUTPUT / f"{stem}.png"
    attn_path = ATTN_OUTPUT / f"{stem}.png"

    if not pre_path.exists():
        print(f"WARNING: pre-attention output missing: {pre_path}")
        continue

    if not attn_path.exists():
        print(f"WARNING: attention output missing: {attn_path}")
        continue

    # --------------------------------------------------------
    # Load images for metrics
    # --------------------------------------------------------

    gt = load_array(gt_path)
    pre = load_array(pre_path)
    attn = load_array(attn_path)

    # Make sure dimensions match
    target_shape = gt.shape

    if pre.shape != target_shape:
        pre = np.array(
            Image.fromarray((pre * 255).astype(np.uint8))
            .resize(
                (target_shape[1], target_shape[0]),
                Image.Resampling.BICUBIC
            )
        ).astype(np.float32) / 255.0

    if attn.shape != target_shape:
        attn = np.array(
            Image.fromarray((attn * 255).astype(np.uint8))
            .resize(
                (target_shape[1], target_shape[0]),
                Image.Resampling.BICUBIC
            )
        ).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    pre_psnr = psnr(pre, gt)
    attn_psnr = psnr(attn, gt)

    pre_ssim = ssim_score(pre, gt)
    attn_ssim = ssim_score(attn, gt)

    results.append({
        "image": stem,
        "pre_psnr": pre_psnr,
        "attention_psnr": attn_psnr,
        "psnr_delta": attn_psnr - pre_psnr,
        "pre_ssim": pre_ssim,
        "attention_ssim": attn_ssim,
        "ssim_delta": (
            attn_ssim - pre_ssim
            if pre_ssim is not None and attn_ssim is not None
            else None
        )
    })

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    lr_arr = load_array(lr_path)
    gt_arr = gt

    lr_img = to_display(lr_arr)
    pre_img = to_display(pre)
    attn_img = to_display(attn)
    gt_img = to_display(gt_arr)

    W = 256
    H = 256
    HEADER = 55

    canvas = Image.new(
        "RGB",
        (W * 4, H + HEADER),
        "white"
    )

    canvas.paste(lr_img.convert("RGB"), (0, HEADER))
    canvas.paste(pre_img.convert("RGB"), (W, HEADER))
    canvas.paste(attn_img.convert("RGB"), (W * 2, HEADER))
    canvas.paste(gt_img.convert("RGB"), (W * 3, HEADER))

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (70, 10),
        "NOISY LR",
        fill="black"
    )

    draw.text(
        (315, 10),
        f"PRE-ATTENTION\nPSNR {pre_psnr:.2f}",
        fill="black"
    )

    draw.text(
        (580, 10),
        f"ATTENTION\nPSNR {attn_psnr:.2f}",
        fill="black"
    )

    draw.text(
        (850, 10),
        "GROUND TRUTH",
        fill="black"
    )

    canvas.save(
        FINAL_OUTPUT / f"{stem}_comparison.png"
    )

    print(
        f"{stem}: "
        f"Pre={pre_psnr:.2f} dB | "
        f"Attention={attn_psnr:.2f} dB | "
        f"Delta={attn_psnr-pre_psnr:+.2f} dB"
    )


# ============================================================
# SUMMARY
# ============================================================

if not results:
    raise RuntimeError(
        "\nNo comparisons were created.\n"
        "Most likely GT_DIR is incorrect:\n"
        f"  {GT_DIR}\n"
    )

pre_avg = np.mean([r["pre_psnr"] for r in results])
attn_avg = np.mean([r["attention_psnr"] for r in results])

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Images evaluated       : {len(results)}")
print(f"Pre-attention PSNR     : {pre_avg:.3f} dB")
print(f"Attention PSNR         : {attn_avg:.3f} dB")
print(f"Improvement            : {attn_avg - pre_avg:+.3f} dB")

if all(r["pre_ssim"] is not None for r in results):
    pre_ssim_avg = np.mean([r["pre_ssim"] for r in results])
    attn_ssim_avg = np.mean([r["attention_ssim"] for r in results])

    print(f"Pre-attention SSIM     : {pre_ssim_avg:.4f}")
    print(f"Attention SSIM         : {attn_ssim_avg:.4f}")
    print(f"SSIM improvement       : {attn_ssim_avg-pre_ssim_avg:+.4f}")

print("\nComparison images:")
print(FINAL_OUTPUT.resolve())