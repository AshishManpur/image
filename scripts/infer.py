"""Single-image and folder inference for SPARC-Net.

The model is a **128x128 -> 256x256 restorer** and nothing else: the input it expects is
exactly the degraded LR representation the training loader produced, and Contract Part 1
fixes the pre/post-processing to a short, non-negotiable list:

* the input is a **single-channel float array**, nominally in ``[0, 1]`` but **not
  clipped** — Phase 1 measured 3.36 % of real LR pixels above 1.0 and 0.30 % below 0,
  and the packed training arrays reproduce that (range ``[-0.20, 1.70]``);
* **there is no dataset-level normalisation.** ``RobustNormalizer`` lives inside the
  model (Part 2.8), and de-normalisation plus ``clamp(0, 1)`` are the model's last two
  ops. Subtracting a mean here would be applying it twice;
* the output is ``(B, 1, 256, 256)`` already in ``[0, 1]``. Nothing is left to undo.

``.npy`` input is therefore the faithful path — it is bit-for-bit what the loader feeds
the model. PNG/TIFF input is supported for convenience and is divided by the integer
dtype's maximum, but 8-bit PNG has already clipped and quantised the out-of-range tail,
so its result is an approximation of the ``.npy`` result rather than equal to it. See
``docs/INFERENCE.md``.

Usage::

    python scripts/infer.py --weights <ckpt.pt> --input <img.npy> --output <out.png>
    python scripts/infer.py --weights <ckpt.pt> --input-dir <in/> --output-dir <out/>
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import (  # noqa: E402
    SPARC_VARIANTS,
    SparcConfig,
    sparc_base,
)
from models.sparc_net import SPARCNet  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

IMAGE_SUFFIXES = (".npy", ".png", ".tif", ".tiff")
AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- config
def infer_config_from_state_dict(state: dict[str, torch.Tensor]) -> SparcConfig:
    """Reconstruct the architecture configuration a checkpoint was written from.

    SPARC checkpoints store weights only (``trainer.Trainer.save``), so the variant has
    to be read back off the tensor names and shapes. This is deliberately structural
    rather than a name lookup: a checkpoint from any staged build — pre-attention,
    concat-fusion, SPARC-Tiny — reconstructs correctly, and a mismatch surfaces as a
    strict ``load_state_dict`` failure rather than as silently wrong pixels.

    Args:
        state: Model ``state_dict``.

    Returns:
        The reconstructed :class:`SparcConfig`.

    Raises:
        KeyError: If the payload is not a SPARC-Net state dict.
    """
    if "encoder.stem.proj.weight" not in state:
        raise KeyError(
            "This does not look like a SPARC-Net checkpoint: no "
            "'encoder.stem.proj.weight'. Keys begin with: "
            f"{sorted(state)[:5]}"
        )

    widths = (
        int(state["encoder.stem.proj.weight"].shape[0]),
        int(state["encoder.downsamples.0.proj.weight"].shape[0]),
        int(state["encoder.downsamples.1.proj.weight"].shape[0]),
    )

    def _depth(prefix: str) -> int:
        indices = {
            int(key[len(prefix):].split(".", 1)[0])
            for key in state
            if key.startswith(prefix)
        }
        return max(indices) + 1 if indices else 0

    enc_naf = tuple(_depth(f"encoder.levels.{i}.naf.") for i in range(3))
    enc_gsa = tuple(_depth(f"encoder.levels.{i}.gsa.") for i in range(3))
    dec_naf = tuple(_depth(f"decoder.stages.{s}.naf.") for s in range(2))
    dec_gsa = tuple(_depth(f"decoder.stages.{s}.gsa.") for s in range(2))

    heads = [0, 0, 0]
    for level in range(3):
        key = f"encoder.levels.{level}.gsa.0.rel_pos.rel_pos_table"
        if key in state:
            heads[level] = int(state[key].shape[0])
    if heads[1] == 0 and "decoder.stages.0.gsa.0.rel_pos.rel_pos_table" in state:
        heads[1] = int(state["decoder.stages.0.gsa.0.rel_pos.rel_pos_table"].shape[0])

    use_attention = sum(enc_gsa) + sum(dec_gsa) > 0
    use_noise_head = any(key.startswith("noise_head.") for key in state)
    use_gated_fusion = any("fusion.squeeze" in key for key in state)
    use_relative_position_bias = any("rel_pos_table" in key for key in state)
    # GDFN-only arm: the GSA blocks are present (they carry `ffn_*`) but their
    # attention sub-branch is not, so there are no `qkv` weights to size heads from.
    # With no GSA blocks at all the flag is inert, so report the default rather than
    # labelling the pre-attention baseline as a GDFN-only arm.
    use_attention_branch = not use_attention or any(
        ".gsa." in key and ".qkv." in key for key in state
    )

    # Phase 5 mechanisms. The stabiliser is weightless apart from its bias-correction
    # scalar, so `use_vst` is inferred from that parameter; a checkpoint trained with
    # `use_vst_bias_correction=False` is indistinguishable from one without the VST and
    # would be rebuilt without it, which is why the correction defaults to on.
    use_vst = any(key.startswith("vst.") for key in state)
    use_film = any(key.startswith("film.") for key in state)
    # Phase 6 clean-LR branch. `residual_source` is not recoverable from the weights —
    # it selects between two tensors, it does not create parameters — so a checkpoint
    # carrying the branch is rebuilt with `residual_source="clean"`, which is the only
    # configuration the branch was introduced for. A run that deliberately trained the
    # branch as a pure auxiliary (`residual_source="noisy"`) must be rebuilt explicitly.
    use_clean_lr_branch = any(key.startswith("clean_branch.") for key in state)
    clean_branch_width = (
        int(state["clean_branch.project.weight"].shape[0]) // 4
        if "clean_branch.project.weight" in state
        else sparc_base().clean_branch_width
    )
    film_hidden = (
        int(state["film.fc1.weight"].shape[0]) if "film.fc1.weight" in state else 64
    )

    defaults = sparc_base()
    if use_attention and not use_attention_branch:
        # `heads` is unused when the branch is absent, but SparcConfig.validate still
        # requires a head count that divides the attention dim into head_dim 16 at any
        # level that instantiates a GSA block. Restore the variant's own values.
        heads = list(defaults.num_heads)
    # Name the checkpoint after the registered variant whose *shape* it matches.
    # `use_attention` is deliberately not compared: it is an ablation axis applied on
    # top of a variant, so the pre-attention and GDFN-only arms are still sparc-base.
    head_width = int(state["head.project.weight"].shape[0]) // 4
    name = "sparc-custom"
    for variant, factory in SPARC_VARIANTS.items():
        candidate = factory()
        if (
            candidate.widths == widths
            and candidate.enc_naf_blocks == enc_naf
            and candidate.dec_naf_blocks == dec_naf
            and candidate.head_width == head_width
        ):
            name = variant
            break

    return SparcConfig(
        name=name,
        widths=widths,
        enc_naf_blocks=enc_naf,
        enc_gsa_blocks=enc_gsa,
        dec_naf_blocks=dec_naf,
        dec_gsa_blocks=dec_gsa,
        num_heads=tuple(heads),
        head_width=head_width,
        head_naf_blocks=_depth("head.blocks."),
        use_noise_head=use_noise_head,
        use_gated_fusion=use_gated_fusion,
        use_attention=use_attention,
        use_attention_branch=use_attention_branch,
        use_relative_position_bias=use_relative_position_bias,
        use_vst=use_vst,
        use_film=use_film,
        film_hidden=film_hidden,
        use_clean_lr_branch=use_clean_lr_branch,
        clean_branch_width=clean_branch_width,
        clean_branch_naf_blocks=_depth("clean_branch.blocks."),
        residual_source="clean" if use_clean_lr_branch else "noisy",
    )


def extract_state_dict(payload: Any, prefer_ema: bool) -> tuple[dict[str, torch.Tensor], str]:
    """Pull the model weights out of a checkpoint payload.

    Args:
        payload: Whatever ``torch.load`` returned.
        prefer_ema: Use the EMA shadow weights when the checkpoint carries them.
            Contract Part 5 evaluates EMA separately and it is normally the better
            of the two; ``--no-ema`` selects the live weights.

    Returns:
        ``(state_dict, source_label)``.

    Raises:
        KeyError: If no recognisable weights are present.
    """
    if not isinstance(payload, dict):
        raise KeyError(f"Unsupported checkpoint payload of type {type(payload).__name__}.")
    if prefer_ema and isinstance(payload.get("ema"), dict) and "module" in payload["ema"]:
        return payload["ema"]["module"], "ema"
    if isinstance(payload.get("model"), dict):
        return payload["model"], "model"
    if isinstance(payload.get("ema"), dict) and "module" in payload["ema"]:
        return payload["ema"]["module"], "ema"
    if all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload, "raw"
    raise KeyError(
        "Checkpoint has neither a 'model' nor an 'ema' entry. Top-level keys: "
        f"{sorted(payload)}"
    )


@dataclass
class LoadedModel:
    """A ready-to-run model plus the provenance needed for reporting."""

    model: SPARCNet
    config: SparcConfig
    source: str
    epoch: int | None
    parameters: int

    @property
    def variant(self) -> str:
        if not self.config.use_attention:
            attention = "pre-attention"
        elif not self.config.use_attention_branch:
            attention = "gdfn-only"
        else:
            attention = "attention"
        mechanisms = [attention]
        if self.config.use_vst:
            mechanisms.append("vst")
        if self.config.use_film:
            mechanisms.append("film")
        return f"{self.config.name} ({'+'.join(mechanisms)}, weights={self.source})"


def load_model(weights: Path, device: torch.device, prefer_ema: bool = True) -> LoadedModel:
    """Load a checkpoint into the exact architecture it was trained with.

    Args:
        weights: Checkpoint path.
        device: Device to place the model on.
        prefer_ema: Prefer EMA weights when present.

    Returns:
        A :class:`LoadedModel` in ``eval`` mode with gradients disabled.

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
        RuntimeError: If the reconstructed architecture does not match the weights.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"No checkpoint at {weights}.")

    payload = load_checkpoint(weights, map_location="cpu")
    state, source = extract_state_dict(payload, prefer_ema)
    config = infer_config_from_state_dict(state)

    model = SPARCNet(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # rel_pos_index is a non-persistent buffer: rebuilt at construction, never stored.
    missing = [k for k in missing if not k.endswith("rel_pos_index")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the reconstructed architecture.\n"
            f"  missing:    {missing[:10]}\n  unexpected: {unexpected[:10]}"
        )

    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    epoch = None
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        epoch = payload["state"].get("epoch")
    return LoadedModel(
        model=model,
        config=config,
        source=source,
        epoch=epoch,
        parameters=sum(p.numel() for p in model.parameters()),
    )


# ------------------------------------------------------------------------------ io
def read_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one LR image into a float32 array in the model's value convention.

    Args:
        path: ``.npy``, ``.png``, ``.tif`` or ``.tiff`` file.

    Returns:
        ``(array (H, W) float32, metadata)``. ``.npy`` is passed through untouched;
        integer image formats are divided by their dtype maximum.

    Raises:
        ValueError: If the file is not 2-D single-channel, or the format is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        meta = {"source": "npy", "scale": 1.0, "clipped_by_format": False}
    elif suffix in (".png", ".tif", ".tiff"):
        from PIL import Image

        with Image.open(path) as handle:
            array = np.array(handle)
        if array.dtype == np.uint8:
            divisor = 255.0
        elif array.dtype == np.uint16:
            divisor = 65535.0
        elif np.issubdtype(array.dtype, np.floating):
            divisor = 1.0
        else:
            raise ValueError(f"Unsupported image dtype {array.dtype} in {path}.")
        array = array.astype(np.float32) / divisor
        meta = {"source": suffix[1:], "scale": divisor, "clipped_by_format": divisor > 1.0}
    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'. Expected one of {IMAGE_SUFFIXES}."
        )

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"{path} has shape {array.shape}; SPARC-Net is a grayscale model and "
            "expects a single-channel 2-D array."
        )
    meta["shape"] = array.shape
    return array, meta


def write_image(path: Path, array: np.ndarray, bit_depth: int = 16) -> None:
    """Write a restored image.

    The network output is already in ``[0, 1]`` (Contract Part 1, step 28 clamps it), so
    the only conversion is a scale to the integer range. 16-bit is the default because
    8-bit quantisation costs ~0.1 dB of the PSNR the model was scored on.

    Args:
        path: Destination. ``.npy`` stores the float array losslessly.
        array: Restored image ``(H, W)`` float32 in ``[0, 1]``.
        bit_depth: 8 or 16, for PNG/TIFF output.

    Raises:
        ValueError: If ``bit_depth`` is not 8 or 16, or the suffix is unsupported.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, array.astype(np.float32))
        return
    if suffix not in (".png", ".tif", ".tiff"):
        raise ValueError(f"Unsupported output format '{suffix}'.")
    if bit_depth == 8:
        data = np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
    elif bit_depth == 16:
        data = np.rint(array * 65535.0).clip(0, 65535).astype(np.uint16)
    else:
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}.")

    from PIL import Image

    Image.fromarray(data).save(path)


# ---------------------------------------------------------------------- inference
def resolve_device(name: str) -> torch.device:
    """Resolve ``--device``; ``auto`` prefers CUDA."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible.")
    return torch.device(name)


def check_input_size(array: np.ndarray, config: SparcConfig) -> None:
    """Reject inputs the model cannot process correctly, rather than resizing them.

    Two independent constraints:

    * the trunk applies ``num_levels`` Haar decimations, so both dimensions must be
      divisible by 8;
    * **with attention enabled the size is fixed**. Each ``GSABlock`` builds its
      relative-position index for one grid at construction time, so a model trained at
      128x128 accepts 128x128 only. Silently resizing would change the noise statistics
      the noise head is conditioned on, which is exactly the failure this check exists
      to prevent.

    Args:
        array: Input array ``(H, W)``.
        config: The loaded model's configuration.

    Raises:
        ValueError: If the size is not processable.
    """
    height, width = array.shape
    divisor = 2**config.num_levels
    if height % divisor or width % divisor:
        raise ValueError(
            f"Input is {height}x{width}; both dimensions must be divisible by "
            f"{divisor} (stem DWT plus {config.num_levels - 1} encoder stages)."
        )
    if config.use_attention and (height != config.input_size or width != config.input_size):
        raise ValueError(
            f"Input is {height}x{width} but this model has attention enabled, which "
            f"fixes the input to {config.input_size}x{config.input_size}: the "
            "relative-position tables are built for one grid. Tile the image or use a "
            "pre-attention checkpoint."
        )


@contextmanager
def true_float32(enabled: bool):
    """Make fp32 mean fp32 on Ampere and later.

    PyTorch leaves ``torch.backends.cudnn.allow_tf32 = True`` by default, so a
    nominally-fp32 CUDA convolution actually runs in **TF32**: a 10-bit mantissa, about
    4.9e-04 of relative precision. Nothing in this project ever turned it off —
    ``utils.seed.set_seed`` pins determinism and benchmarking but is silent on TF32 —
    which made "CUDA fp32 inference" a reduced-precision path the caller never asked
    for. It is measurable: `tests/test_inference.py::test_cuda_inference_matches_cpu`
    saw 2.415e-04 between CPU and CUDA, against a 1e-4 requirement.

    ``--amp-dtype bf16`` is the fast path and is unaffected. ``fp32`` is the reference
    path, and a reference that is quietly 10-bit is not a reference.

    Args:
        enabled: Whether to force true fp32. Pass ``False`` to leave the global state
            alone (e.g. when the caller asked for autocast anyway).

    Yields:
        ``None``. The previous TF32 settings are restored on exit.
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    previous = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous[0]
        torch.backends.cuda.matmul.allow_tf32 = previous[1]


@torch.inference_mode()
def restore(
    model: SPARCNet,
    array: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    allow_tf32: bool = False,
) -> tuple[np.ndarray, float]:
    """Restore a single image.

    Args:
        model: A loaded SPARC-Net in ``eval`` mode.
        array: LR input ``(H, W)`` float32, unclipped.
        device: Device to run on.
        amp_dtype: Autocast dtype, or ``None`` for full fp32.
        allow_tf32: Permit TF32 convolutions in the fp32 path. Default ``False`` so
            that fp32 is genuinely fp32; see :func:`true_float32`.

    Returns:
        ``(restored (2H, 2W) float32 in [0, 1], elapsed_seconds)``.
    """
    x = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with true_float32(amp_dtype is None and not allow_tf32):
        if amp_dtype is None:
            out = model(x)
        else:
            with torch.autocast(device.type, dtype=amp_dtype):
                out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # The model's own last op is clamp(0, 1) (Part 1, step 28); no further clamping is
    # applied here. Autocast can return bf16/fp16, so cast for the numpy round-trip.
    return out.squeeze(0).squeeze(0).float().cpu().numpy(), elapsed


def iter_inputs(directory: Path) -> Iterable[Path]:
    """Yield every supported image in ``directory``, sorted."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore 128x128 LR images to 256x256 with SPARC-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint (.pt).")
    parser.add_argument("--input", type=Path, help="Single input image (.npy/.png/.tif).")
    parser.add_argument("--output", type=Path, help="Single output image path.")
    parser.add_argument("--input-dir", type=Path, help="Folder of input images.")
    parser.add_argument("--output-dir", type=Path, help="Folder for restored images.")
    parser.add_argument(
        "--output-suffix", default=".png", help="Extension used for folder inference."
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Compute device; 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--amp-dtype", default="fp32", choices=sorted(AMP_DTYPES),
        help="Autocast dtype. bf16 is CUDA-only in practice; fp32 is the reference.",
    )
    parser.add_argument(
        "--bit-depth", type=int, default=16, choices=[8, 16],
        help="Integer depth of PNG/TIFF output.",
    )
    parser.add_argument(
        "--no-ema", action="store_true",
        help="Use the live weights instead of the EMA shadow.",
    )
    parser.add_argument(
        "--channels-last", action="store_true",
        help="Run in channels-last memory format (CUDA only; Contract Part 5).",
    )
    parser.add_argument(
        "--allow-tf32", action="store_true",
        help=(
            "Permit TF32 convolutions in the fp32 path. Faster on Ampere+, but only "
            "10 mantissa bits, so fp32 output stops matching a CPU reference to 1e-4. "
            "Off by default; --amp-dtype bf16 is the intended fast path."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    single = args.input is not None
    folder = args.input_dir is not None
    if single == folder:
        raise SystemExit("Provide exactly one of --input or --input-dir.")
    if single and args.output is None:
        raise SystemExit("--input requires --output.")
    if folder and args.output_dir is None:
        raise SystemExit("--input-dir requires --output-dir.")

    device = resolve_device(args.device)
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    if amp_dtype is not None and device.type == "cpu" and amp_dtype is torch.float16:
        raise SystemExit("fp16 autocast is not supported on CPU; use bf16 or fp32.")

    loaded = load_model(args.weights, device, prefer_ema=not args.no_ema)
    if args.channels_last:
        loaded.model.to(memory_format=torch.channels_last)

    _LOGGER.info(
        "Loaded %s: %d parameters, device=%s, amp=%s",
        loaded.variant, loaded.parameters, device, args.amp_dtype,
    )

    if single:
        paths = [args.input]
        outputs = [args.output]
    else:
        paths = list(iter_inputs(args.input_dir))
        if not paths:
            raise SystemExit(f"No {IMAGE_SUFFIXES} files found in {args.input_dir}.")
        outputs = [
            args.output_dir / (path.stem + args.output_suffix) for path in paths
        ]

    total = 0.0
    for path, destination in zip(paths, outputs):
        array, meta = read_image(path)
        check_input_size(array, loaded.config)
        if meta["clipped_by_format"]:
            _LOGGER.warning(
                "%s is an integer image: its out-of-range tail was already clipped "
                "before this script saw it. Use .npy for the exact training-time input.",
                path.name,
            )
        restored, elapsed = restore(
            loaded.model, array, device, amp_dtype, allow_tf32=args.allow_tf32
        )
        total += elapsed
        write_image(destination, restored, args.bit_depth)
        _LOGGER.info(
            "%s %s -> %s %s  (%.1f ms)",
            path.name, array.shape, destination.name, restored.shape, elapsed * 1e3,
        )

    _LOGGER.info(
        "Restored %d image(s) in %.2f s (%.1f ms/image).",
        len(paths), total, total / max(len(paths), 1) * 1e3,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
