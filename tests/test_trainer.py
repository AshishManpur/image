"""Training framework tests (Contract Part 8, step 7; Part 9)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from configs.sparc_config import TrainingConfig, build_sparc_config
from losses.charbonnier import CharbonnierLoss
from models.sparc_net import SPARCNet
from trainer.ema import ModelEma
from trainer.tb_layout import GROUPS, LOSS_TERMS, build_layout, should_log_group
from trainer.trainer import (
    DivergenceError,
    Trainer,
    build_param_groups,
    warmup_cosine_lambda,
)

STAGE6 = {"use_noise_head": False, "use_gated_fusion": False, "use_attention": False}


class TinyPairs(Dataset):
    """A handful of deterministic LR/GT pairs for loop-level tests."""

    def __init__(self, count: int = 4) -> None:
        generator = torch.Generator().manual_seed(0)
        self.lr = torch.rand(count, 1, 32, 32, generator=generator)
        self.gt = torch.rand(count, 1, 64, 64, generator=generator)

    def __len__(self) -> int:
        return self.lr.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"lr": self.lr[index], "gt": self.gt[index], "index": torch.tensor(index)}


class TinyNet(nn.Module):
    """Minimal 2x upsampler standing in for SPARC-Net in loop tests."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, 4)
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.out = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.pixel_shuffle(self.norm(self.conv(x)), 2))


def make_trainer(tmp_path, **config_kwargs) -> Trainer:
    dataset = TinyPairs()
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    defaults = dict(
        epochs=2, batch_size=8, warmup_epochs=1, num_workers=0,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs",
        output_dir=tmp_path / "out",
    )
    defaults.update(config_kwargs)
    return Trainer(
        model=TinyNet(),
        criterion=CharbonnierLoss(),
        train_loader=loader,
        val_loader=loader,
        config=TrainingConfig(**defaults),
        device="cpu",
        run_name="unit",
    )


# ------------------------------------------------------------------ param groups
def test_param_groups_exclude_norms_scales_and_biases() -> None:
    model = SPARCNet(build_sparc_config("sparc-tiny"))
    groups = build_param_groups(model, weight_decay=1e-4)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["weight_decay"] == 0.0

    decayed = {id(p) for p in groups[0]["params"]}
    for name, param in model.named_parameters():
        excluded = param.ndim <= 1 or any(
            key in name.lower() for key in ("norm", "gamma", "rel_pos", "bias")
        )
        assert (id(param) not in decayed) == excluded, name


def test_param_groups_cover_every_trainable_parameter() -> None:
    model = SPARCNet(build_sparc_config("sparc-tiny"))
    groups = build_param_groups(model, 1e-4)
    counted = sum(p.numel() for g in groups for p in g["params"])
    assert counted == sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------- schedule
def test_warmup_cosine_shape() -> None:
    schedule = warmup_cosine_lambda(warmup_steps=10, total_steps=110, min_ratio=0.01)
    assert schedule(0) == pytest.approx(0.01)
    assert schedule(10) == pytest.approx(1.0)
    assert schedule(110) == pytest.approx(0.01, abs=1e-6)
    assert schedule(60) < 1.0


def test_warmup_cosine_is_monotone_after_warmup() -> None:
    schedule = warmup_cosine_lambda(10, 110, 0.01)
    values = [schedule(step) for step in range(10, 111)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_warmup_cosine_clamps_beyond_total() -> None:
    schedule = warmup_cosine_lambda(10, 110, 0.01)
    assert schedule(500) == pytest.approx(0.01, abs=1e-6)


def test_schedule_without_warmup_starts_at_one() -> None:
    assert warmup_cosine_lambda(0, 100, 0.0)(0) == pytest.approx(1.0)


# -------------------------------------------------------------------------- EMA
def test_ema_tracks_the_live_model() -> None:
    model = TinyNet()
    ema = ModelEma(model, decay=0.5, warmup_steps=0)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update(model)
    for live, shadow in zip(model.parameters(), ema.parameters()):
        assert not torch.allclose(live, shadow)
        assert (shadow - live).abs().mean() > 0.0


def test_ema_converges_to_the_live_weights() -> None:
    model = TinyNet()
    ema = ModelEma(model, decay=0.5, warmup_steps=0)
    for _ in range(200):
        ema.update(model)
    for live, shadow in zip(model.parameters(), ema.parameters()):
        assert torch.allclose(live, shadow, atol=1e-5)


def test_ema_parameters_do_not_require_grad() -> None:
    ema = ModelEma(TinyNet())
    assert all(not p.requires_grad for p in ema.parameters())


def test_ema_state_round_trip() -> None:
    model = TinyNet()
    source = ModelEma(model, decay=0.9, warmup_steps=5)
    for _ in range(3):
        source.update(model)
    target = ModelEma(TinyNet(), decay=0.9, warmup_steps=5)
    target.load_state_dict(source.state_dict())
    assert target.steps == source.steps
    for a, b in zip(source.parameters(), target.parameters()):
        assert torch.allclose(a, b)


def test_ema_rejects_bad_decay_and_state() -> None:
    with pytest.raises(ValueError):
        ModelEma(TinyNet(), decay=1.0)
    with pytest.raises(KeyError):
        ModelEma(TinyNet()).load_state_dict({})


# ------------------------------------------------------------------------- loop
def test_one_epoch_runs_without_nan(tmp_path) -> None:
    trainer = make_trainer(tmp_path)
    metrics = trainer.train_epoch()
    assert math.isfinite(metrics["total"])
    assert math.isfinite(metrics["grad_norm"])
    assert trainer.state.global_step == len(trainer.train_loader)


def test_evaluate_returns_finite_metrics(tmp_path) -> None:
    trainer = make_trainer(tmp_path)
    summary = trainer.evaluate(trainer.model)
    assert math.isfinite(summary["psnr_mean"])
    assert math.isfinite(summary["ssim_mean"])
    assert summary["count"] == len(trainer.val_loader.dataset)


def test_fit_writes_all_checkpoints_and_logs(tmp_path) -> None:
    trainer = make_trainer(tmp_path)
    state = trainer.fit()
    run_dir = tmp_path / "ckpt" / "unit"
    for name in ("last.pt", "best_psnr.pt", "best_ema_psnr.pt"):
        assert (run_dir / name).exists(), name
    assert (tmp_path / "logs" / "unit" / "metrics.jsonl").exists()
    assert (tmp_path / "logs" / "unit" / "metrics.csv").exists()
    assert len(state.history) == 2


def test_checkpoint_resume_restores_state_exactly(tmp_path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.train_epoch()
    trainer.state.epoch = 0
    path = trainer.save("resume.pt")

    restored = make_trainer(tmp_path)
    restored.load(path, resume=True)
    assert restored.state.global_step == trainer.state.global_step
    assert restored.state.epoch == trainer.state.epoch + 1
    for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(trainer.ema.parameters(), restored.ema.parameters()):
        assert torch.equal(a, b)


def test_load_without_resume_keeps_optimiser_fresh(tmp_path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.train_epoch()
    path = trainer.save("weights.pt")
    fresh = make_trainer(tmp_path)
    fresh.load(path, resume=False)
    assert fresh.state.global_step == 0


def test_missing_checkpoint_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        make_trainer(tmp_path).load(tmp_path / "nope.pt")


def test_early_stopping_triggers(tmp_path) -> None:
    """An improvement threshold no epoch can clear must stop at the first opportunity.

    Epoch 0 always counts as an improvement because ``best_ema_psnr`` starts at
    ``-inf`` and ``-inf + min_delta`` is still ``-inf``: the first epoch establishes
    the baseline rather than beating one. With ``patience=1`` the earliest legitimate
    stop is therefore epoch 1.
    """
    trainer = make_trainer(
        tmp_path, epochs=6, early_stopping_patience=1, early_stopping_min_delta=1e6
    )
    state = trainer.fit()
    assert state.epoch == 1
    assert state.epochs_without_improvement == 1


def test_early_stopping_does_not_fire_while_improving(tmp_path) -> None:
    trainer = make_trainer(
        tmp_path, epochs=3, early_stopping_patience=1, early_stopping_min_delta=-1e6
    )
    state = trainer.fit()
    assert state.epoch == 2
    assert state.epochs_without_improvement == 0


def test_trainer_rejects_unsanctioned_batch_size(tmp_path) -> None:
    with pytest.raises(ValueError):
        make_trainer(tmp_path, batch_size=12)


def test_amp_disabled_on_cpu(tmp_path) -> None:
    """fp16 autocast is CUDA-only; the trainer must not enable it on CPU."""
    assert make_trainer(tmp_path).amp_enabled is False


# --------------------------------------------------------------- Phase 4.8 additions
def test_training_config_accepts_cli_overrides() -> None:
    """`train.py` must be able to derive a modified config from CLI flags.

    Regression test for audit finding T-1. ``TrainingConfig`` is a ``slots=True``
    dataclass, so instances have no ``__dict__``; the old
    ``TrainingConfig(**config.__dict__)`` idiom raised ``AttributeError`` and killed
    every documented invocation that passed ``--epochs``, ``--batch-size``, ``--lr``,
    ``--num-workers`` or ``--seed``.
    """
    base = TrainingConfig()
    # epochs must stay above warmup_epochs (default 5) or validate() rightly rejects it.
    overrides = {"epochs": 50, "batch_size": 8, "learning_rate": 1e-4, "seed": 7}
    updated = dataclasses.replace(base, **overrides)

    assert updated.epochs == 50
    assert updated.learning_rate == pytest.approx(1e-4)
    assert updated.seed == 7
    # Untouched fields must survive unchanged.
    assert updated.weight_decay == base.weight_decay
    assert updated.betas == base.betas
    assert updated.ema_decay == base.ema_decay
    updated.validate()

    with pytest.raises(AttributeError):
        _ = base.__dict__  # the trap this test exists to prevent


def test_per_step_loss_terms_are_logged(tmp_path, monkeypatch) -> None:
    """Contract Part 6: every loss term is logged separately every step."""
    trainer = make_trainer(tmp_path)
    recorded: list[tuple[str, float, int]] = []

    class _Spy:
        def add_scalar(self, tag, value, step):
            recorded.append((tag, float(value), int(step)))

        def add_custom_scalars(self, layout):
            pass

        def close(self):
            pass

    trainer.writer = _Spy()
    trainer.train_epoch()

    steps = len(trainer.train_loader)
    assert steps > 0
    total_points = [r for r in recorded if r[0] == "step_loss/total"]
    assert len(total_points) == steps, "one point per optimisation step"
    assert [r[2] for r in total_points] == list(range(1, steps + 1))

    for tag in ("grad/global_norm", "optim/lr", "grad/clipped_fraction"):
        assert len([r for r in recorded if r[0] == tag]) == steps

    # Memory is CUDA-only and must not be emitted as a misleading flat zero on CPU.
    assert not [r for r in recorded if r[0].startswith("memory/")]


def test_scheduler_advances_on_skipped_batch(tmp_path) -> None:
    """A non-finite loss must not stall the LR schedule against ``global_step``.

    Regression test for audit finding T-4: skipping the batch without stepping the
    scheduler stretches the cosine relative to the step counter for the rest of the run.
    """

    class NanLoss(nn.Module):
        def forward(self, pred, target):
            return pred.sum() * float("nan")

    trainer = make_trainer(tmp_path)
    trainer.criterion = NanLoss()
    steps = len(trainer.train_loader)

    trainer.train_epoch()

    assert trainer.skipped_batches == steps
    assert trainer.state.global_step == steps
    assert trainer.scheduler.last_epoch == steps


def test_composite_loss_receives_aux_and_logs_every_term(tmp_path) -> None:
    """Phase 4.10 / T-3: `wants_aux` routes the full SparcOutput and batch through.

    Also checks Contract Part 6's logging requirement end to end — every term the
    criterion returns must reach TensorBoard at per-step resolution.
    """
    from losses import CompositeLoss

    dataset = TinyPairs()
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    )
    trainer = Trainer(
        model=model,
        criterion=CompositeLoss(enabled={"ms_ssim": False}),  # 64px < MS-SSIM minimum
        train_loader=loader,
        val_loader=loader,
        config=TrainingConfig(
            epochs=2, batch_size=8, warmup_epochs=1, num_workers=0,
            checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs",
            output_dir=tmp_path / "out",
        ),
        device="cpu",
        run_name="composite",
    )

    recorded: list[str] = []

    class _Spy:
        def add_scalar(self, tag, value, step):
            recorded.append(tag)

        def add_custom_scalars(self, layout):
            pass

        def close(self):
            pass

    trainer.writer = _Spy()
    metrics = trainer.train_epoch()

    for term in ("charbonnier", "wavelet", "fft", "gradient", "noise", "total"):
        assert term in metrics, f"{term} missing from epoch metrics"
        assert f"step_loss/{term}" in recorded, f"{term} not logged per step"
    assert all(math.isfinite(v) for v in metrics.values())


def test_reconstruction_only_criterion_is_unaffected(tmp_path) -> None:
    """Backward compatibility: a criterion without `wants_aux` sees the old signature."""
    trainer = make_trainer(tmp_path)
    assert getattr(trainer.criterion, "wants_aux", False) is False
    metrics = trainer.train_epoch()
    assert math.isfinite(metrics["total"])


def test_tb_layout_groups_are_guarded() -> None:
    """Layout groups are gated on device and model flags, and typos are rejected."""
    layout = build_layout()
    assert "Loss terms (per step)" in layout
    for term in LOSS_TERMS:
        assert f"step_loss/{term}" in layout["Loss terms (per step)"][
            "Every term, every step"
        ][1]

    assert should_log_group("memory", cuda=False, flags={}) is False
    assert should_log_group("memory", cuda=True, flags={}) is True
    assert should_log_group("noise", cuda=True, flags={"use_noise_head": False}) is False
    assert should_log_group("attention", cuda=True, flags={"use_attention": True}) is True
    assert should_log_group("train", cuda=False, flags={}) is True

    with pytest.raises(ValueError):
        should_log_group("typo", cuda=True, flags={})
    assert "step_loss" in GROUPS and "loss" in GROUPS


# ------------------------------------------------- Phase 4.10.1: divergence guard
class NanLoss(nn.Module):
    """A criterion that always returns a non-finite loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.sum() * float("nan")


def test_divergence_aborts_after_the_threshold(tmp_path) -> None:
    """Phase 4.10.1: consecutive non-finite batches must stop the run, not be skipped.

    Phase 4.10 skipped every batch from step 423 to the end of a three-epoch run and
    exited reporting success. The skip path never touches the weights, so a diverged
    model fails every subsequent batch identically.
    """
    trainer = make_trainer(tmp_path, max_consecutive_nonfinite=1)
    trainer.criterion = NanLoss()

    with pytest.raises(DivergenceError) as excinfo:
        trainer.train_epoch()

    assert trainer.consecutive_nonfinite > trainer.config.max_consecutive_nonfinite
    assert excinfo.value.report_path is not None
    assert excinfo.value.report_path.exists()
    assert excinfo.value.checkpoint_path.exists()


def test_divergence_report_records_the_diagnosis(tmp_path) -> None:
    """The dump must answer 'where did it break' without a rerun."""
    trainer = make_trainer(tmp_path, max_consecutive_nonfinite=1)
    trainer.criterion = NanLoss()

    with pytest.raises(DivergenceError) as excinfo:
        trainer.train_epoch()

    report = json.loads(excinfo.value.report_path.read_text(encoding="utf-8"))
    for key in (
        "epoch", "global_step", "batch_index", "consecutive_nonfinite", "loss_terms",
        "batch_stats", "parameters", "first_nonfinite_module", "largest_activations",
        "amp", "learning_rate", "checkpoint",
    ):
        assert key in report, f"divergence report is missing {key!r}"
    assert report["parameters"]["nonfinite_count"] == 0
    assert "lr" in report["batch_stats"] and "gt" in report["batch_stats"]


def test_below_threshold_still_skips_without_aborting(tmp_path) -> None:
    """A rare bad batch must not end a run; only a sustained run of them does."""
    trainer = make_trainer(tmp_path, max_consecutive_nonfinite=1000)
    trainer.criterion = NanLoss()
    steps = len(trainer.train_loader)

    trainer.train_epoch()

    assert trainer.skipped_batches == steps
    assert trainer.consecutive_nonfinite == steps


def test_a_successful_step_resets_the_consecutive_counter(tmp_path) -> None:
    """Alternating failures must not accumulate toward the abort threshold."""

    class Flaky(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            base = (pred - target).abs().mean()
            return base * float("nan") if self.calls % 2 else base

    trainer = make_trainer(tmp_path, max_consecutive_nonfinite=1)
    trainer.criterion = Flaky()

    trainer.train_epoch()

    assert trainer.skipped_batches > 0
    assert trainer.consecutive_nonfinite == 0


# ---------------------------------------------------------- Phase 4.10.1: AMP
def test_bf16_runs_without_a_gradient_scaler(tmp_path) -> None:
    """bf16 carries fp32's exponent range, so loss scaling is pointless overhead."""
    trainer = make_trainer(tmp_path, amp_dtype="bf16")
    assert trainer.amp_dtype is torch.bfloat16
    assert not trainer.scaler.is_enabled()


def test_fp16_is_the_default_amp_dtype(tmp_path) -> None:
    """Task 7 requires the default to be unchanged."""
    assert TrainingConfig().amp_dtype == "fp16"
    assert make_trainer(tmp_path).amp_dtype is torch.float16


def test_invalid_amp_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="amp_dtype"):
        TrainingConfig(amp_dtype="fp8").validate()


def test_invalid_divergence_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_consecutive_nonfinite"):
        TrainingConfig(max_consecutive_nonfinite=0).validate()


def test_epoch_metrics_report_overflow_steps(tmp_path) -> None:
    """Task 5: GradScaler overflows must be observable, not inferred from the scale."""
    trainer = make_trainer(tmp_path)
    metrics = trainer.train_epoch()
    assert "overflow_steps" in metrics
    assert metrics["overflow_steps"] == 0.0


# -------------------------------------------------- Phase 4.10.1: numerics trace
def test_debug_numerics_writes_a_per_step_trace(tmp_path) -> None:
    """Task 1: the trace must name the tensors, not just the total loss."""
    trainer = make_trainer(tmp_path, debug_numerics=True)
    trainer.train_epoch()

    path = tmp_path / "logs" / "unit" / "numerics.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(trainer.train_loader)

    record = json.loads(lines[0])
    for key in (
        "step", "grad_norm", "prediction_min", "prediction_max", "prediction_absmax",
        "input_absmax", "target_absmax", "param_absmax", "param_nonfinite",
    ):
        assert key in record, f"numerics trace is missing {key!r}"


def test_debug_numerics_respects_the_start_step(tmp_path) -> None:
    """The trace forces a device sync, so it must be scopeable to a step window."""
    start = 2
    trainer = make_trainer(
        tmp_path, debug_numerics=True, debug_numerics_from_step=start
    )
    trainer.train_epoch()

    path = tmp_path / "logs" / "unit" / "numerics.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert records, "expected at least one record past the start step"
    assert all(record["step"] >= start for record in records)
    assert len(records) < len(trainer.train_loader) + 1


def test_numerics_trace_is_off_by_default(tmp_path) -> None:
    """The file must not even be created: the trace costs a sync per step."""
    trainer = make_trainer(tmp_path)
    trainer.train_epoch()
    assert not (tmp_path / "logs" / "unit" / "numerics.jsonl").exists()


def test_detect_anomaly_is_off_by_default_and_runnable(tmp_path) -> None:
    """Task 8: default disabled, and enabling it must not break the step."""
    assert TrainingConfig().detect_anomaly is False
    trainer = make_trainer(tmp_path, detect_anomaly=True)
    metrics = trainer.train_epoch()
    assert math.isfinite(metrics["total"])
