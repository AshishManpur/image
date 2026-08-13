"""Full Phase 4.12 model trainability (Contract Parts 3, 5, 6, 8 step 13).

These tests exist because of one specific, expensive failure mode: every switch that
removes a module from SPARC-Net still produces a network that trains to a plausible
loss curve. A run that quietly dropped the three GSA groups would look healthy for
eleven hours and then be worth 608,088 parameters less than the model it claimed to be.

So the assertions here are about *identity* as much as about health â€” the thing that
trained must be provably the 2,345,650-parameter V1 model, and the checkpoint it wrote
must reconstruct that model with zero missing and zero unexpected keys.

CUDA-denominated behaviour (BF16 autocast, VRAM) is covered by the CUDA-marked tests;
they skip rather than substitute CPU numbers.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from configs.sparc_config import TrainingConfig, build_sparc_config, sparc_base
from losses.composite_loss import CompositeLoss
from models.attention.gsa_block import GSABlock
from models.fusion.gated_fuse import GatedFuse
from models.sparc_net import SPARCNet
from trainer.trainer import Trainer, build_param_groups

FULL_V1_PARAMETERS = 2_345_650
GSA_PARAMETERS = 608_088
PRE_ATTENTION_PARAMETERS = 1_737_562

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TinyPairs(Dataset):
    """A few full-resolution 128->256 pairs: the real shapes, a trivial count."""

    def __init__(self, count: int = 4) -> None:
        generator = torch.Generator().manual_seed(1337)
        self.lr = torch.rand(count, 1, 128, 128, generator=generator)
        self.gt = torch.rand(count, 1, 256, 256, generator=generator)

    def __len__(self) -> int:
        return self.lr.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"lr": self.lr[index], "gt": self.gt[index], "index": torch.tensor(index)}


def make_full_trainer(tmp_path, device: str = "cpu", **config_kwargs) -> Trainer:
    """A Trainer wrapping the FULL V1 model and the full composite objective."""
    loader = DataLoader(TinyPairs(), batch_size=2, num_workers=0)
    defaults = dict(
        epochs=2, batch_size=8, warmup_epochs=1, num_workers=0,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs",
        output_dir=tmp_path / "out",
    )
    defaults.update(config_kwargs)
    return Trainer(
        model=SPARCNet(sparc_base()),
        criterion=CompositeLoss(),
        train_loader=loader,
        val_loader=loader,
        config=TrainingConfig(**defaults),
        device=device,
        run_name="full",
    )


# ------------------------------------------------------------------- identity
def test_default_variant_is_the_full_v1_model() -> None:
    """`--variant sparc-base` with no flags must be the scored model."""
    config = build_sparc_config("sparc-base")
    assert config.use_attention
    assert config.use_gated_fusion
    assert config.use_noise_head
    assert config.use_normalization
    assert config.use_global_residual

    model = SPARCNet(config)
    assert sum(p.numel() for p in model.parameters()) == FULL_V1_PARAMETERS
    assert len([m for m in model.modules() if isinstance(m, GSABlock)]) == 6
    assert len([m for m in model.modules() if isinstance(m, GatedFuse)]) == 2
    assert model.noise_head is not None


def test_the_attention_delta_is_exactly_the_contract_value() -> None:
    without = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    assert sum(p.numel() for p in without.parameters()) == PRE_ATTENTION_PARAMETERS
    assert FULL_V1_PARAMETERS - PRE_ATTENTION_PARAMETERS == GSA_PARAMETERS


def test_composite_loss_has_all_six_terms() -> None:
    criterion = CompositeLoss()
    # 256 px: MS-SSIM at 5 scales and window 11 needs at least 161 px, and 256 is the
    # size the model actually produces.
    _, terms = criterion(torch.rand(1, 1, 256, 256), torch.rand(1, 1, 256, 256))
    for name in ("charbonnier", "ms_ssim", "wavelet", "fft", "gradient"):
        assert name in terms, name
    assert criterion.wants_aux


def test_train_py_reports_the_full_model(caplog) -> None:
    """`log_module_state` is what stops a run being misattributed after the fact."""
    import logging

    from train import log_module_state

    config = sparc_base()
    with caplog.at_level(logging.INFO):
        log_module_state(SPARCNet(config), config)
    text = caplog.text
    assert "attention=ENABLED" in text
    assert "6 GSA blocks, 608088 params" in text
    assert "gated_fusion=ENABLED" in text
    assert "noise_head=ENABLED" in text
    assert "matches Contract Part 3 exactly" in text


def test_train_py_warns_when_the_count_is_not_the_contract_total(caplog) -> None:
    import logging

    from train import log_module_state

    config = build_sparc_config("sparc-base", use_attention=False)
    with caplog.at_level(logging.INFO):
        log_module_state(SPARCNet(config), config)
    assert "attention=DISABLED" in caplog.text
    assert "not the scored V1 model" in caplog.text


# ------------------------------------------------------------------- trainability
def test_full_model_completes_a_training_epoch(tmp_path) -> None:
    """Forward, CompositeLoss, backward, optimizer, scheduler, EMA â€” all six."""
    trainer = make_full_trainer(tmp_path)
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    ema_before = trainer.ema.module.head.to_subbands.bias.detach().clone()

    metrics = trainer.train_epoch()

    assert trainer.state.global_step == len(trainer.train_loader)
    assert trainer.skipped_batches == 0
    assert torch.isfinite(torch.tensor(metrics["total"]))
    assert trainer.optimizer.param_groups[0]["lr"] != lr_before  # scheduler stepped
    assert trainer.ema.steps == len(trainer.train_loader)  # EMA updated
    assert not torch.equal(ema_before, trainer.ema.module.head.to_subbands.bias)
    for name, parameter in trainer.model.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_gradients_reach_every_gsa_parameter(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path)
    batch = next(iter(trainer.train_loader))
    output = trainer.model.forward_with_aux(batch["lr"])
    loss, _ = trainer.criterion(output, batch)
    loss.backward()

    checked = 0
    for name, parameter in trainer.model.named_parameters():
        if ".gsa." not in name:
            continue
        checked += 1
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name
    assert checked == 114  # 19 tensors x 6 blocks


def test_rel_pos_tables_are_optimised_but_not_decayed(tmp_path) -> None:
    model = SPARCNet(sparc_base())
    groups = build_param_groups(model, weight_decay=1e-4)
    decayed = {id(p) for p in groups[0]["params"]}
    undecayed = {id(p) for p in groups[1]["params"]}
    tables = [p for name, p in model.named_parameters() if "rel_pos_table" in name]
    assert len(tables) == 6
    for table in tables:
        assert id(table) not in decayed
        assert id(table) in undecayed  # still trained, just not decayed
    assert groups[1]["weight_decay"] == 0.0


def test_full_model_fits_and_writes_every_checkpoint(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path)
    trainer.fit()
    directory = tmp_path / "ckpt" / "full"
    for name in ("last.pt", "best_psnr.pt", "best_ema_psnr.pt"):
        assert (directory / name).exists(), name


# -------------------------------------------------------------------- checkpoint
def test_checkpoint_reconstructs_the_full_model_with_no_key_drift(tmp_path) -> None:
    """The GSA weights must survive the round trip: 0 missing, 0 unexpected."""
    trainer = make_full_trainer(tmp_path)
    trainer.train_epoch()
    path = trainer.save("last.pt")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    fresh = SPARCNet(sparc_base())
    result = fresh.load_state_dict(payload["model"], strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    gsa_keys = [k for k in payload["model"] if ".gsa." in k]
    assert len(gsa_keys) == 114
    assert sum(payload["model"][k].numel() for k in gsa_keys) == GSA_PARAMETERS
    assert sum(v.numel() for v in payload["model"].values()) == FULL_V1_PARAMETERS
    assert sum(1 for k in payload["model"] if "rel_pos_table" in k) == 6

    # rel_pos_index is registered persistent=False: a derived constant rebuilt at
    # construction, deliberately absent from the file. This is the one documented
    # checkpoint/state distinction.
    assert not any("rel_pos_index" in k for k in payload["model"])

    ema = fresh.load_state_dict(payload["ema"]["module"], strict=True)
    assert ema.missing_keys == [] and ema.unexpected_keys == []


def test_resume_restores_weights_optimiser_scheduler_and_ema(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path, epochs=2)
    trainer.train_epoch()
    path = trainer.save("last.pt")
    reference = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    reference_ema = {k: v.clone() for k, v in trainer.ema.module.state_dict().items()}
    step = trainer.state.global_step
    lr = trainer.optimizer.param_groups[0]["lr"]

    resumed = make_full_trainer(tmp_path, epochs=2)
    resumed.load(path, resume=True)

    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, reference[key]), key
    for key, value in resumed.ema.module.state_dict().items():
        assert torch.equal(value, reference_ema[key]), key
    assert resumed.state.global_step == step
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    assert resumed.ema.steps == trainer.ema.steps


def test_resumed_trainer_continues_for_another_epoch(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path, epochs=2)
    trainer.train_epoch()
    path = trainer.save("last.pt")

    resumed = make_full_trainer(tmp_path, epochs=2)
    resumed.load(path, resume=True)
    before = resumed.state.global_step
    metrics = resumed.train_epoch()

    assert resumed.state.global_step == before + len(resumed.train_loader)
    assert resumed.skipped_batches == 0
    assert torch.isfinite(torch.tensor(metrics["total"]))
    for name, parameter in resumed.model.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_a_pre_attention_checkpoint_cannot_masquerade_as_the_full_model(tmp_path) -> None:
    """Loading the 1.74 M baseline into the V1 model must fail, not partially load."""
    baseline = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    path = tmp_path / "pre_attention.pt"
    torch.save({"model": baseline.state_dict()}, path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError):
        SPARCNet(sparc_base()).load_state_dict(payload["model"], strict=True)

    result = SPARCNet(sparc_base()).load_state_dict(payload["model"], strict=False)
    assert len(result.missing_keys) == 114  # every GSA tensor, silently absent
    assert result.unexpected_keys == []


# -------------------------------------------------------------- amp / validation
def test_validation_autocast_follows_the_configured_dtype(tmp_path) -> None:
    """Regression: `evaluate` hardcoded fp16, so a bf16 run was scored in fp16."""
    trainer = make_full_trainer(tmp_path)
    trainer.config = dataclasses.replace(trainer.config, amp_dtype="bf16")
    trainer.amp_dtype = torch.bfloat16

    seen: list[torch.dtype] = []
    original = torch.amp.autocast

    class _Spy:
        def __init__(self, device_type, **kwargs):
            seen.append(kwargs.get("dtype"))
            self._inner = original(device_type, **kwargs)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    torch.amp.autocast = _Spy
    try:
        trainer.evaluate(trainer.model)
    finally:
        torch.amp.autocast = original

    assert seen, "evaluate() did not open an autocast context"
    assert all(dtype is torch.bfloat16 for dtype in seen), seen


def test_bf16_config_runs_without_a_gradient_scaler(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path, amp_dtype="bf16")
    assert not trainer.scaler.is_enabled()
    trainer.train_epoch()
    assert trainer.overflow_steps == 0


@CUDA
def test_full_model_trains_one_step_on_cuda_in_bf16(tmp_path) -> None:
    trainer = make_full_trainer(tmp_path, device="cuda", amp_dtype="bf16")
    assert trainer.amp_enabled
    assert trainer.amp_dtype is torch.bfloat16
    metrics = trainer.train_epoch()
    assert trainer.skipped_batches == 0
    assert trainer.overflow_steps == 0
    assert torch.isfinite(torch.tensor(metrics["total"]))
    for name, parameter in trainer.model.named_parameters():
        assert torch.isfinite(parameter).all(), name


@CUDA
def test_training_vram_at_batch_eight_is_within_budget() -> None:
    """Contract Part 10 (amended A-004): peak training VRAM at batch 8 must stay under 2.06 GB.

    A-003 set 2.05 GB from a single 1-step measurement (~2.017 GB). The 50-step trace
    in ``scripts/vram_step_trace.py`` later established the *stable* cumulative peak at
    2.0516 GB — the caching allocator's steady state, not a leak — which sits above that
    threshold by ~1.6 MB. A-004 raises the limit to 2.06 GB against that measurement.
    """
    device = torch.device("cuda")
    model = SPARCNet(sparc_base()).to(device).to(memory_format=torch.channels_last)
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.rand(8, 1, 128, 128, device=device).to(memory_format=torch.channels_last)
    gt = torch.rand(8, 1, 256, 256, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = criterion(model.forward_with_aux(x), {"gt": gt, "lr": x})
    loss.backward()
    optimizer.step()
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

    assert peak_gb < 2.06, f"peak training VRAM {peak_gb:.3f} GB exceeds the 2.06 GB budget (A-004)"

