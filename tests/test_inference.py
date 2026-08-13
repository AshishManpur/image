"""Inference-pipeline tests (``scripts/infer.py``, ``docs/INFERENCE.md``).

The properties worth pinning here are the ones that fail *silently*: a checkpoint loaded
into the wrong architecture still produces an image, and so does an input that has been
normalised twice or resized behind your back. Every one of those would score badly and
look plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configs.sparc_config import build_sparc_config, sparc_base
from models.sparc_net import SPARCNet
from scripts.infer import (
    check_input_size,
    extract_state_dict,
    infer_config_from_state_dict,
    load_model,
    read_image,
    resolve_device,
    restore,
    write_image,
)

VARIANTS = {
    "sparc-tiny": build_sparc_config("sparc-tiny"),
    "sparc-xl-moderate": build_sparc_config("sparc-xl-moderate"),
    "pre-attention": build_sparc_config("sparc-base", use_attention=False),
    "concat-fusion": build_sparc_config(
        "sparc-base", use_attention=False, use_gated_fusion=False
    ),
    "full-base": sparc_base(),
}


def _checkpoint(tmp_path, config, name: str = "ckpt.pt", with_ema: bool = True):
    model = SPARCNet(config)
    payload = {"model": model.state_dict(), "state": {"epoch": 7}}
    if with_ema:
        payload["ema"] = {"module": model.state_dict(), "steps": 100}
    path = tmp_path / name
    torch.save(payload, path)
    return path, model


# ------------------------------------------------------------- config round-trip
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_config_is_reconstructed_from_the_state_dict(name: str) -> None:
    config = VARIANTS[name]
    state = SPARCNet(config).state_dict()
    recovered = infer_config_from_state_dict(state)
    for field in (
        "widths", "enc_naf_blocks", "dec_naf_blocks", "head_width",
        "head_naf_blocks", "use_noise_head", "use_gated_fusion", "use_attention",
    ):
        assert getattr(recovered, field) == getattr(config, field), field

    if config.use_attention:
        assert recovered.enc_gsa_blocks == config.enc_gsa_blocks
        assert recovered.dec_gsa_blocks == config.dec_gsa_blocks
        assert recovered.num_heads == config.num_heads
    else:
        # `use_attention=False` leaves the depths in the config but builds no blocks;
        # what the state dict records — and all inference needs — is the built shape.
        assert recovered.enc_gsa_blocks == (0, 0, 0)
        assert recovered.dec_gsa_blocks == (0, 0)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_loads_strictly_and_reproduces_its_output(tmp_path, name: str) -> None:
    config = VARIANTS[name]
    path, reference = _checkpoint(tmp_path, config, f"{name}.pt")
    loaded = load_model(path, torch.device("cpu"))

    assert loaded.parameters == sum(p.numel() for p in reference.parameters())
    assert loaded.epoch == 7

    x = torch.rand(1, 1, config.input_size, config.input_size)
    reference.eval()
    with torch.inference_mode():
        assert torch.max((loaded.model(x) - reference(x)).abs()).item() == 0.0


def test_full_base_checkpoint_recovers_the_contract_parameter_count(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, sparc_base())
    assert load_model(path, torch.device("cpu")).parameters == 2_345_650


def test_a_foreign_checkpoint_is_rejected_not_partially_loaded(tmp_path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"model": {"backbone.conv.weight": torch.zeros(3, 3)}}, path)
    with pytest.raises(KeyError, match="SPARC-Net"):
        load_model(path, torch.device("cpu"))


def test_a_truncated_checkpoint_is_rejected(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"], with_ema=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"].pop("head.to_subbands.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="does not match"):
        load_model(path, torch.device("cpu"))


def test_ema_weights_are_preferred_and_can_be_disabled(tmp_path) -> None:
    config = VARIANTS["pre-attention"]
    model = SPARCNet(config)
    live = model.state_dict()
    shadow = {k: v.clone() for k, v in live.items()}
    shadow["head.to_subbands.bias"] += 1.0
    path = tmp_path / "ema.pt"
    torch.save({"model": live, "ema": {"module": shadow}}, path)

    assert extract_state_dict(torch.load(path, weights_only=False), True)[1] == "ema"
    assert extract_state_dict(torch.load(path, weights_only=False), False)[1] == "model"

    with_ema = load_model(path, torch.device("cpu"), prefer_ema=True)
    without = load_model(path, torch.device("cpu"), prefer_ema=False)
    delta = (
        with_ema.model.head.to_subbands.bias - without.model.head.to_subbands.bias
    )
    assert torch.allclose(delta, torch.ones_like(delta))


# -------------------------------------------------------------------- input rules
def test_npy_input_is_passed_through_unmodified(tmp_path) -> None:
    """Contract Part 2.8 reads statistics off the raw input; rescaling breaks it."""
    array = np.random.RandomState(0).uniform(-0.2, 1.7, (128, 128)).astype(np.float32)
    path = tmp_path / "lr.npy"
    np.save(path, array)
    loaded, meta = read_image(path)
    assert np.array_equal(loaded, array)
    assert meta["clipped_by_format"] is False


def test_png_input_is_scaled_by_the_dtype_maximum(tmp_path) -> None:
    from PIL import Image

    data = np.array([[0, 128, 255]] * 3, dtype=np.uint8)
    path = tmp_path / "lr.png"
    Image.fromarray(data).save(path)
    loaded, meta = read_image(path)
    assert np.allclose(loaded, data / 255.0)
    assert meta["clipped_by_format"] is True


def test_rejects_colour_input(tmp_path) -> None:
    path = tmp_path / "rgb.npy"
    np.save(path, np.zeros((16, 16, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="grayscale"):
        read_image(path)


def test_rejects_unsupported_format(tmp_path) -> None:
    path = tmp_path / "lr.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported input format"):
        read_image(path)


def test_size_that_is_not_divisible_by_eight_is_rejected() -> None:
    config = VARIANTS["pre-attention"]
    with pytest.raises(ValueError, match="divisible by 8"):
        check_input_size(np.zeros((130, 130), dtype=np.float32), config)


def test_attention_model_refuses_a_size_it_was_not_built_for() -> None:
    """The rel-pos index is built for one grid; resizing silently would be wrong."""
    with pytest.raises(ValueError, match="attention enabled"):
        check_input_size(np.zeros((64, 64), dtype=np.float32), sparc_base())
    check_input_size(np.zeros((128, 128), dtype=np.float32), sparc_base())


# ------------------------------------------------------------------------ output
def test_restore_doubles_the_resolution_and_stays_in_range(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    loaded = load_model(path, torch.device("cpu"))
    array = np.random.RandomState(1).uniform(-0.2, 1.7, (128, 128)).astype(np.float32)
    restored, elapsed = restore(loaded.model, array, torch.device("cpu"))
    assert restored.shape == (256, 256)
    assert restored.dtype == np.float32
    assert restored.min() >= 0.0 and restored.max() <= 1.0
    assert elapsed > 0.0


def test_restore_does_not_train_or_mutate_the_model(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    loaded = load_model(path, torch.device("cpu"))
    before = {k: v.clone() for k, v in loaded.model.state_dict().items()}

    array = np.random.RandomState(2).rand(128, 128).astype(np.float32)
    restore(loaded.model, array, torch.device("cpu"))

    assert not loaded.model.training
    assert not any(p.requires_grad for p in loaded.model.parameters())
    assert all(p.grad is None for p in loaded.model.parameters())
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("bit_depth,peak", [(8, 255), (16, 65535)])
def test_write_image_round_trips_within_quantisation(tmp_path, bit_depth, peak) -> None:
    array = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "out.png"
    write_image(path, array, bit_depth)
    recovered, _ = read_image(path)
    assert np.max(np.abs(recovered - array)) <= 0.5 / peak + 1e-6


def test_write_npy_is_lossless(tmp_path) -> None:
    array = np.random.RandomState(3).rand(32, 32).astype(np.float32)
    path = tmp_path / "out.npy"
    write_image(path, array)
    assert np.array_equal(read_image(path)[0], array)


# ------------------------------------------------------------------ device / amp
def test_resolve_device_falls_back_to_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_bf16_autocast_matches_fp32_within_tolerance(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    loaded = load_model(path, torch.device("cpu"))
    array = np.random.RandomState(4).rand(128, 128).astype(np.float32)
    reference, _ = restore(loaded.model, array, torch.device("cpu"))
    reduced, _ = restore(loaded.model, array, torch.device("cpu"), torch.bfloat16)
    assert reduced.shape == reference.shape
    assert np.isfinite(reduced).all()
    assert np.abs(reduced - reference).mean() < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_matches_cpu(tmp_path) -> None:
    """fp32 on CUDA must reproduce fp32 on CPU to 1e-4.

    ``restore`` disables TF32 for the fp32 path (see ``infer.true_float32``). Without
    that, Ampere runs every convolution with a 10-bit mantissa and this measured
    2.415e-04 on an RTX A400 — a reduced-precision path nobody asked for, which is a
    defect in the inference path rather than in this tolerance.
    """
    path, _ = _checkpoint(tmp_path, VARIANTS["full-base"])
    array = np.random.RandomState(5).rand(128, 128).astype(np.float32)
    cpu, _ = restore(load_model(path, torch.device("cpu")).model, array, torch.device("cpu"))
    gpu, _ = restore(
        load_model(path, torch.device("cuda")).model, array, torch.device("cuda")
    )
    assert np.max(np.abs(cpu - gpu)) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="TF32 requires Ampere or newer",
)
def test_tf32_is_what_breaks_fp32_parity(tmp_path) -> None:
    """Pin the diagnosis: with TF32 re-enabled the same comparison fails.

    This is the control for `test_cuda_inference_matches_cpu`. If TF32 ever stops being
    the explanation — a driver change, a PyTorch default flip — this test fails and
    tells us the diagnosis has gone stale, instead of leaving a fix in place for a
    cause that no longer exists.
    """
    path, _ = _checkpoint(tmp_path, VARIANTS["full-base"])
    array = np.random.RandomState(5).rand(128, 128).astype(np.float32)
    cpu, _ = restore(load_model(path, torch.device("cpu")).model, array, torch.device("cpu"))
    model = load_model(path, torch.device("cuda")).model

    strict, _ = restore(model, array, torch.device("cuda"), allow_tf32=False)
    loose, _ = restore(model, array, torch.device("cuda"), allow_tf32=True)

    assert np.max(np.abs(cpu - strict)) < np.max(np.abs(cpu - loose))
    assert np.max(np.abs(cpu - strict)) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_restore_leaves_the_global_tf32_state_untouched(tmp_path) -> None:
    """`true_float32` must not leak: it is global backend state."""
    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    loaded = load_model(path, torch.device("cuda"))
    array = np.random.RandomState(11).rand(128, 128).astype(np.float32)

    for setting in (True, False):
        torch.backends.cudnn.allow_tf32 = setting
        torch.backends.cuda.matmul.allow_tf32 = setting
        restore(loaded.model, array, torch.device("cuda"))
        assert torch.backends.cudnn.allow_tf32 is setting
        assert torch.backends.cuda.matmul.allow_tf32 is setting


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bf16_inference_is_finite(tmp_path) -> None:
    path, _ = _checkpoint(tmp_path, VARIANTS["full-base"])
    loaded = load_model(path, torch.device("cuda"))
    array = np.random.RandomState(6).rand(128, 128).astype(np.float32)
    restored, _ = restore(loaded.model, array, torch.device("cuda"), torch.bfloat16)
    assert np.isfinite(restored).all()
    assert restored.min() >= 0.0 and restored.max() <= 1.0


# --------------------------------------------------------------------------- cli
def test_cli_single_image(tmp_path) -> None:
    from scripts.infer import main

    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    source = tmp_path / "lr.npy"
    np.save(source, np.random.RandomState(7).rand(128, 128).astype(np.float32))
    destination = tmp_path / "out" / "restored.png"

    assert main([
        "--weights", str(path), "--input", str(source),
        "--output", str(destination), "--device", "cpu",
    ]) == 0
    assert destination.exists()
    assert read_image(destination)[0].shape == (256, 256)


def test_cli_folder(tmp_path) -> None:
    from scripts.infer import main

    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    source = tmp_path / "in"
    source.mkdir()
    for index in range(3):
        np.save(
            source / f"{index:03d}.npy",
            np.random.RandomState(index).rand(128, 128).astype(np.float32),
        )
    destination = tmp_path / "out"

    assert main([
        "--weights", str(path), "--input-dir", str(source),
        "--output-dir", str(destination), "--device", "cpu",
    ]) == 0
    assert sorted(p.name for p in destination.iterdir()) == [
        "000.png", "001.png", "002.png"
    ]


def test_cli_rejects_ambiguous_arguments(tmp_path) -> None:
    from scripts.infer import main

    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    with pytest.raises(SystemExit):
        main(["--weights", str(path)])
    with pytest.raises(SystemExit):
        main(["--weights", str(path), "--input", "a.npy"])


def test_visualize_cli_writes_a_three_panel_strip(tmp_path) -> None:
    from scripts.visualize_restoration import main

    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    lr = tmp_path / "lr.npy"
    gt = tmp_path / "gt.npy"
    np.save(lr, np.random.RandomState(8).rand(128, 128).astype(np.float32))
    np.save(gt, np.random.RandomState(9).rand(256, 256).astype(np.float32))
    out = tmp_path / "compare.png"

    assert main([
        "--weights", str(path), "--input", str(lr), "--gt", str(gt),
        "--output", str(out), "--device", "cpu",
    ]) == 0
    image, _ = read_image(out)
    assert image.shape[0] == 256 + 22  # panel + caption strip
    assert image.shape[1] == 3 * 256 + 2 * 8  # three panels plus two gaps


def test_visualize_cli_without_ground_truth(tmp_path) -> None:
    from scripts.visualize_restoration import main

    path, _ = _checkpoint(tmp_path, VARIANTS["pre-attention"])
    lr = tmp_path / "lr.npy"
    np.save(lr, np.random.RandomState(10).rand(128, 128).astype(np.float32))
    out = tmp_path / "compare2.png"

    assert main([
        "--weights", str(path), "--input", str(lr),
        "--output", str(out), "--device", "cpu",
    ]) == 0
    assert read_image(out)[0].shape[1] == 2 * 256 + 8
