"""Training loop for SPARC-Base V1.0 (Contract Parts 5 and 6).

Every hyperparameter is taken from :class:`configs.sparc_config.TrainingConfig` and is
frozen by the contract. The loop implements: AdamW with decay excluded from norm,
scale and bias parameters; linear warmup into cosine decay; fp16 AMP with a gradient
scaler; global-norm gradient clipping; an EMA shadow model evaluated separately;
per-epoch validation, checkpointing and early stopping; and CSV/JSONL/TensorBoard
telemetry with every loss term logged individually.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.sparc_config import TrainingConfig
from datasets.degradation import forward_operator
from evaluation.metrics import BandCorrelationAccumulator, MetricAccumulator
from trainer.ema import ModelEma
from trainer.tb_layout import build_layout, should_log_group
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logging_utils import CsvLogger, JsonlLogger, get_logger
from utils.numerics import ModuleTracer, detect_anomaly, tensor_stats

_LOGGER = get_logger(__name__)

NO_DECAY_KEYWORDS = ("norm", "gamma", "rel_pos", "bias")

AMP_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
"""Autocast dtypes selectable through ``TrainingConfig.amp_dtype``."""


class DivergenceError(RuntimeError):
    """Raised when training has produced non-finite losses beyond the threshold.

    Phase 4.10 treated a non-finite loss as a transient bad batch and skipped it. That
    is right for a rare data artefact and wrong for a diverged model: the guard never
    touches the weights, so once the weights are bad *every* subsequent batch fails and
    the skip is permanent. The Phase 4.10 shakedown skipped every batch from step 423
    to the end of the run and reported success.

    Attributes:
        report_path: Path to the JSON diagnostic report written before raising.
        checkpoint_path: Path to the checkpoint written before raising.
    """

    def __init__(
        self, message: str, report_path: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.report_path = report_path
        self.checkpoint_path = checkpoint_path


def build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups.

    Weight decay is applied only to convolution and linear weights. LayerNorm affine
    parameters, LayerScale gammas, relative-position tables and all biases are
    excluded: decaying them shrinks the network toward a degenerate map rather than
    toward a simpler one.

    Args:
        model: Model whose parameters to group.
        weight_decay: Decay applied to the first group.

    Returns:
        Two parameter groups suitable for ``torch.optim.AdamW``.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim <= 1 or any(key in lowered for key in NO_DECAY_KEYWORDS):
            no_decay.append(param)
        else:
            decay.append(param)
    _LOGGER.info(
        "Parameter groups: %d decayed, %d not decayed", len(decay), len(no_decay)
    )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine_lambda(
    warmup_steps: int, total_steps: int, min_ratio: float
) -> Any:
    """Build the LR multiplier schedule: linear warmup into cosine decay.

    Args:
        warmup_steps: Steps of linear warmup.
        total_steps: Total optimisation steps.
        min_ratio: Floor as a fraction of the base learning rate.

    Returns:
        A callable suitable for ``torch.optim.lr_scheduler.LambdaLR``.
    """

    def _schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return min_ratio + (1.0 - min_ratio) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _schedule


@dataclass
class TrainState:
    """Mutable training state, checkpointed and restored verbatim."""

    epoch: int = 0
    global_step: int = 0
    best_psnr: float = -float("inf")
    best_ema_psnr: float = -float("inf")
    epochs_without_improvement: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Drives training, validation, checkpointing and early stopping.

    Args:
        model: The model to train.
        criterion: Loss module returning either a scalar or ``(scalar, term_dict)``.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Frozen training configuration.
        device: Device to train on.
        run_name: Sub-directory name for checkpoints and logs.

    Raises:
        ValueError: If the training configuration violates the contract.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig | None = None,
        device: torch.device | str = "cpu",
        run_name: str = "sparc-base",
    ) -> None:
        self.config = config or TrainingConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.run_name = run_name

        self.model = model.to(self.device)
        if self.config.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Derived from the model and criterion rather than added to the constructor
        # signature, so every existing caller keeps working unchanged. The getattr
        # defaults cover criteria that are not `CompositeLoss` (e.g. a bare Charbonnier
        # in the unit tests) and models without a `config`.
        model_config = getattr(model, "config", None)
        self.model_scale = int(getattr(model_config, "scale", 2))
        loss_config = getattr(criterion, "config", None)
        self.clean_lr_blur_sigma = float(
            getattr(loss_config, "clean_lr_blur_sigma", 0.4)
        )
        self.clean_lr_full_weight = float(getattr(loss_config, "clean_lr", 0.0))
        self.clean_lr_warmup_epochs = int(
            getattr(loss_config, "clean_lr_warmup_epochs", 0)
        )

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, self.config.weight_decay),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        steps_per_epoch = max(1, len(train_loader))
        self.total_steps = steps_per_epoch * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            warmup_cosine_lambda(
                warmup_steps=steps_per_epoch * self.config.warmup_epochs,
                total_steps=self.total_steps,
                min_ratio=self.config.min_learning_rate / self.config.learning_rate,
            ),
        )

        # AMP is CUDA-only here; CPU autocast has a different op policy and is not the
        # training path.
        self.amp_enabled = bool(self.config.amp and self.device.type == "cuda")
        self.amp_dtype = AMP_DTYPES[self.config.amp_dtype]
        if self.amp_enabled and self.amp_dtype is torch.bfloat16:
            self._require_bf16_support()

        # The GradScaler exists to stop fp16 *gradients* underflowing to zero. bf16 has
        # fp32's exponent range, so there is nothing to rescale and enabling the scaler
        # would only add loss-scale search to a run that cannot benefit from it.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=self.config.amp_init_scale,
            enabled=self.amp_enabled and self.amp_dtype is torch.float16,
        )
        self.ema = ModelEma(self.model, decay=self.config.ema_decay).to(self.device)
        self.state = TrainState()
        self.skipped_batches = 0
        """Batches dropped for a non-finite loss. Diagnostic only; see ``train_epoch``."""
        self.consecutive_nonfinite = 0
        """Run length of consecutive non-finite batches; reset by any successful step.

        This, not ``skipped_batches``, is what distinguishes an unlucky batch from a
        diverged model. See :class:`DivergenceError`.
        """
        self.overflow_steps = 0
        """Steps where the GradScaler found non-finite gradients and skipped the update."""

        if self.amp_enabled:
            _LOGGER.info(
                "AMP enabled: dtype=%s, GradScaler=%s, validation dtype=%s",
                self.config.amp_dtype,
                "on" if self.scaler.is_enabled() else "off (bf16 needs no scaling)",
                self.config.amp_dtype,
            )

        self.checkpoint_dir = Path(self.config.checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.config.log_dir) / run_name
        self.jsonl = JsonlLogger(log_dir / "metrics.jsonl")
        self.numerics_log = JsonlLogger(log_dir / "numerics.jsonl")
        """Per-step Task 1 numerics trace; written only under ``--debug-numerics``."""
        self.csv = CsvLogger(
            log_dir / "metrics.csv",
            ["epoch", "step", "lr", "train_loss", "val_psnr", "val_ssim",
             "ema_psnr", "ema_ssim", "seconds"],
        )
        self.writer = self._build_tensorboard(log_dir)
        if self.writer is not None:
            self.writer.add_custom_scalars(build_layout())

        model_config = getattr(self.model, "config", None)
        self.log_flags = {
            "use_noise_head": bool(getattr(model_config, "use_noise_head", False)),
            "use_attention": bool(getattr(model_config, "use_attention", False)),
        }

    def _require_bf16_support(self) -> None:
        """Fail fast if bf16 was requested on hardware that cannot do it natively.

        CUDA will emulate bf16 on pre-Ampere cards rather than error, at a large speed
        penalty and with no accuracy benefit. Refusing is more useful than silently
        running a slow job.

        Raises:
            RuntimeError: If the device does not support bf16.
        """
        if not torch.cuda.is_bf16_supported():
            capability = torch.cuda.get_device_capability(self.device)
            raise RuntimeError(
                "amp_dtype='bf16' requires a GPU with native bfloat16 support "
                f"(compute capability 8.0+); this device reports {capability}. "
                "Use --amp-dtype fp16."
            )

    def autocast(self) -> torch.amp.autocast:
        """The one autocast context this trainer opens, anywhere.

        Training, validation, the debug-numerics probe and the divergence trace all go
        through here. They used to construct their own contexts, and one of the four —
        ``evaluate`` — carried a hardcoded ``torch.float16``: a ``--amp-dtype bf16`` run
        trained in bf16 but was *scored* in fp16, silently reintroducing the 65504
        activation ceiling into the reported PSNR only. Four call sites meant four
        chances to get it wrong; there is now one.

        Returns:
            An autocast context using the configured dtype, disabled off CUDA.
        """
        return torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.amp_enabled
        )

    @staticmethod
    def _build_tensorboard(log_dir: Path) -> Any:
        """Create a TensorBoard writer, or ``None`` if the package is absent."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:  # pragma: no cover - optional dependency
            _LOGGER.warning("tensorboard unavailable; skipping TensorBoard logging.")
            return None
        return SummaryWriter(log_dir=str(log_dir))

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write one scalar, honouring the group guards in :mod:`trainer.tb_layout`.

        Args:
            tag: Fully-qualified ``<group>/<name>`` tag.
            value: Scalar value.
            step: Step or epoch index, matching the tag's group.
        """
        if self.writer is None:
            return
        group = tag.split("/", 1)[0]
        if not should_log_group(
            group, cuda=self.device.type == "cuda", flags=self.log_flags
        ):
            return
        self.writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------ batching
    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            tensor = value.to(self.device, non_blocking=self.config.pin_memory)
            if (
                self.config.channels_last
                and self.device.type == "cuda"
                and tensor.dim() == 4
            ):
                tensor = tensor.to(memory_format=torch.channels_last)
            moved[key] = tensor
        return moved

    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the model and the criterion, returning the loss and its terms.

        A criterion opts in to the auxiliary path by declaring ``wants_aux = True``
        (:class:`losses.composite_loss.CompositeLoss` does). It then receives the whole
        ``SparcOutput`` and the whole batch, because the noise term's target is derived
        from ``(D(gt), lr)`` and so needs the input image as well as the target.

        Criteria without the flag — ``CharbonnierLoss`` and every existing training
        script — keep receiving ``(prediction, gt)`` exactly as before. This is what
        makes the change backward compatible rather than merely compatible-looking.

        Args:
            batch: Device-resident batch with ``lr`` and ``gt`` entries.

        Returns:
            ``(loss, terms)`` where ``terms`` maps each logged name to a float.
        """
        wants_aux = getattr(self.criterion, "wants_aux", False)
        forward_with_aux = getattr(self.model, "forward_with_aux", None)

        if wants_aux and forward_with_aux is not None:
            result = self.criterion(forward_with_aux(batch["lr"]), batch)
        else:
            result = self.criterion(self.model(batch["lr"]), batch["gt"])

        if isinstance(result, tuple):
            loss, terms = result
            return loss, {k: float(v) for k, v in terms.items()}
        return result, {"total": float(result.detach())}

    # ----------------------------------------------------------------- one epoch
    def _apply_clean_lr_warmup(self) -> None:
        """Ramp the ``clean_lr`` loss weight linearly over the first epochs.

        Uses the project's existing epoch counter (``state.epoch``) and the same linear
        ramp shape as the LR warmup rather than introducing a second scheduler. With
        ``clean_lr_warmup_epochs=3`` and a full weight of 0.5 this gives epoch 0 -> 0,
        epoch 1 -> 1/6, epoch 2 -> 1/3, epoch 3+ -> 0.5.

        No-op unless the criterion exposes ``set_clean_lr_weight``, so non-composite
        criteria are unaffected.
        """
        setter = getattr(self.criterion, "set_clean_lr_weight", None)
        if setter is None or self.clean_lr_full_weight <= 0.0:
            return
        warmup = self.clean_lr_warmup_epochs
        if warmup <= 0:
            weight = self.clean_lr_full_weight
        else:
            ratio = min(1.0, self.state.epoch / warmup)
            weight = self.clean_lr_full_weight * ratio
        setter(weight)
        _LOGGER.info(
            "clean_lr weight for epoch %d: %.4f (full %.4f)",
            self.state.epoch,
            weight,
            self.clean_lr_full_weight,
        )

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        A non-finite loss skips the batch, as before — a rare data artefact should not
        end a run. What is new in Phase 4.10.1 is that consecutive failures are counted
        and abort the run once they exceed ``config.max_consecutive_nonfinite``. The
        skip path never modifies the weights, so if the *model* has diverged rather
        than the batch being unlucky, every subsequent batch fails identically and
        skipping is no longer recovery, it is a silent no-op that burns the rest of the
        budget. See :class:`DivergenceError`.

        Returns:
            Mean value of every logged loss term over the epoch.

        Raises:
            DivergenceError: If ``max_consecutive_nonfinite`` consecutive batches
                produce a non-finite loss. A checkpoint and a JSON diagnostic report
                are written before raising.
        """
        self.model.train()
        self._apply_clean_lr_warmup()
        totals: dict[str, float] = {}
        batches = 0
        start = time.perf_counter()

        progress = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"epoch {self.state.epoch + 1}/{self.config.epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with detect_anomaly(self.config.detect_anomaly):
                with self.autocast():
                    loss, terms = self._compute_loss(batch)

                if not torch.isfinite(loss):
                    self._handle_nonfinite(batch, batch_index, terms)
                    progress.set_postfix_str("non-finite batch skipped")
                    continue

                self.scaler.scale(loss).backward()
                # unscale_ before clipping so the clip threshold is measured against
                # true gradients, and so the scaler's own inf/NaN check runs before
                # clip_grad_norm_ turns an Inf gradient into a NaN one (clip_coef is
                # zero for an infinite norm, and 0 * inf is NaN).
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # The scaler skips the optimiser step when unscale_ found a non-finite
            # gradient, and signals it only by lowering the scale. Nothing else in the
            # loop can observe that, so an fp16 overflow storm was previously invisible.
            overflowed = (
                self.scaler.is_enabled() and self.scaler.get_scale() < scale_before
            )
            self.overflow_steps += int(overflowed)

            self.scheduler.step()
            if not overflowed:
                self.ema.update(self.model)

            self.state.global_step += 1
            self.consecutive_nonfinite = 0
            batches += 1
            terms["grad_norm"] = float(grad_norm)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value

            # Contract Part 6: every loss term is logged separately every step.
            self._log_step(terms, float(grad_norm), overflowed=overflowed)
            if self._debug_active():
                self._log_numerics(batch, terms, float(grad_norm), overflowed)

            progress.set_postfix(
                loss=f"{terms.get('total', float('nan')):.4f}",
                grad=f"{float(grad_norm):.2f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                step=self.state.global_step,
            )
        progress.close()

        averaged = {key: value / max(1, batches) for key, value in totals.items()}
        averaged["seconds"] = time.perf_counter() - start
        averaged["skipped_batches"] = float(self.skipped_batches)
        averaged["overflow_steps"] = float(self.overflow_steps)
        return averaged

    def _debug_active(self) -> bool:
        """Whether per-step numerics telemetry should be emitted right now."""
        return (
            self.config.debug_numerics
            and self.state.global_step >= self.config.debug_numerics_from_step
        )

    # --------------------------------------------------------------- divergence
    def _handle_nonfinite(
        self, batch: dict[str, torch.Tensor], batch_index: int,
        terms: dict[str, float],
    ) -> None:
        """Account for one non-finite batch and abort if the run has diverged.

        Args:
            batch: The offending batch, already on device.
            batch_index: Index of the batch within the current epoch.
            terms: Loss decomposition for the failed step; identifies which term went
                non-finite first.

        Raises:
            DivergenceError: Once the consecutive-failure threshold is exceeded.
        """
        # Advance the schedule anyway: the LR curve is defined against the optimisation
        # step index, and silently stalling it here would stretch the cosine relative
        # to global_step for the rest of training.
        self.scheduler.step()
        self.state.global_step += 1
        self.skipped_batches += 1
        self.consecutive_nonfinite += 1

        bad_terms = sorted(
            name for name, value in terms.items()
            if not math.isfinite(value) and not name.startswith("raw_")
        )
        _LOGGER.error(
            "Non-finite loss at step %d (consecutive %d/%d); non-finite terms: %s",
            self.state.global_step,
            self.consecutive_nonfinite,
            self.config.max_consecutive_nonfinite,
            ", ".join(bad_terms) or "none reported",
        )

        if self.consecutive_nonfinite <= self.config.max_consecutive_nonfinite:
            return

        report_path, checkpoint_path = self._dump_divergence(batch, batch_index, terms)
        raise DivergenceError(
            f"Training diverged: {self.consecutive_nonfinite} consecutive non-finite "
            f"losses ending at step {self.state.global_step} (epoch "
            f"{self.state.epoch}, batch {batch_index}). The weights are not being "
            f"updated on these steps, so this will not recover on its own. "
            f"Diagnostics: {report_path}",
            report_path=report_path,
            checkpoint_path=checkpoint_path,
        )

    def _dump_divergence(
        self, batch: dict[str, torch.Tensor], batch_index: int,
        terms: dict[str, float],
    ) -> tuple[Path, Path]:
        """Write everything needed to diagnose the divergence offline.

        Captures the full training state plus the statistics that say *where* the
        failure is: per-term loss values, input and target ranges, parameter and
        gradient health, and the first module in the forward pass whose output is
        non-finite.

        Args:
            batch: The offending batch.
            batch_index: Index of the batch within the epoch.
            terms: Loss decomposition for the failed step.

        Returns:
            ``(report_path, checkpoint_path)``.
        """
        checkpoint_path = self.save("divergence.pt")

        # Which parameters are bad matters: non-finite *weights* mean an optimiser step
        # applied a bad update, whereas finite-but-large weights with a non-finite
        # forward pass mean the arithmetic overflowed rather than the training.
        bad_params = [
            name for name, param in self.model.named_parameters()
            if not bool(torch.isfinite(param).all())
        ]
        param_absmax = max(
            (float(p.detach().float().abs().max()) for p in self.model.parameters()),
            default=0.0,
        )

        activations: list[dict[str, Any]] = []
        first_bad: dict[str, Any] | None = None
        try:
            tracer = ModuleTracer(self.model)
            with tracer, torch.no_grad(), self.autocast():
                self.model(batch["lr"])
            first_bad = tracer.first_bad()
            activations = tracer.worst_headroom(20)
        except Exception as exc:  # pragma: no cover - diagnostics must never mask
            _LOGGER.warning("Activation trace failed during dump: %s", exc)

        report = {
            "reason": "consecutive non-finite losses exceeded threshold",
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "batch_index": batch_index,
            "consecutive_nonfinite": self.consecutive_nonfinite,
            "threshold": self.config.max_consecutive_nonfinite,
            "skipped_batches_total": self.skipped_batches,
            "overflow_steps_total": self.overflow_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "amp": {
                "enabled": self.amp_enabled,
                "dtype": self.config.amp_dtype,
                "scaler_enabled": self.scaler.is_enabled(),
                "scale": float(self.scaler.get_scale())
                if self.scaler.is_enabled() else None,
            },
            "loss_terms": terms,
            "batch_stats": {
                key: tensor_stats(value).to_dict()
                for key, value in batch.items()
                if torch.is_tensor(value) and value.is_floating_point()
            },
            "parameters": {
                "nonfinite_count": len(bad_params),
                "nonfinite_names": bad_params[:32],
                "absmax": param_absmax,
            },
            "first_nonfinite_module": first_bad,
            "largest_activations": activations,
            "checkpoint": str(checkpoint_path),
        }

        report_path = self.checkpoint_dir / "divergence_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        _LOGGER.error(
            "DIVERGENCE at epoch %d step %d. Non-finite parameters: %d. "
            "Parameter absmax: %.4g. First non-finite module: %s. "
            "Wrote %s and %s.",
            self.state.epoch,
            self.state.global_step,
            len(bad_params),
            param_absmax,
            first_bad["name"] if first_bad else "none (forward pass was finite)",
            checkpoint_path,
            report_path,
        )
        return report_path, checkpoint_path

    def _log_step(
        self, terms: dict[str, float], grad_norm: float, overflowed: bool = False
    ) -> None:
        """Emit per-step telemetry (Contract Part 6).

        Loss terms go to the ``step_loss`` group so they never share an axis with the
        per-epoch ``loss`` group — the two use different step bases and plotting them
        together would give a misleading x-axis.

        Args:
            terms: Loss decomposition for this step, plus ``grad_norm``.
            grad_norm: Global gradient norm before clipping.
            overflowed: Whether the GradScaler skipped this optimiser step because it
                found non-finite gradients.
        """
        if self.writer is None:
            return
        step = self.state.global_step
        for key, value in terms.items():
            if key == "grad_norm":
                continue
            self._log_scalar(f"step_loss/{key}", value, step)
        self._log_scalar("grad/global_norm", grad_norm, step)
        self._log_scalar(
            "grad/clipped_fraction",
            1.0 if grad_norm > self.config.grad_clip_norm else 0.0,
            step,
        )
        self._log_scalar("optim/lr", self.optimizer.param_groups[0]["lr"], step)
        if self.scaler.is_enabled():
            # Task 5: the scale trace is the cheapest divergence early-warning there
            # is. A healthy fp16 run ratchets the scale upward every 2000 steps; a
            # collapsing scale means the gradients are overflowing repeatedly, which
            # precedes loss divergence rather than following it.
            self._log_scalar("optim/amp_scale", float(self.scaler.get_scale()), step)
            self._log_scalar("optim/amp_overflow", float(overflowed), step)
            self._log_scalar("optim/amp_overflow_total", float(self.overflow_steps), step)
        self._log_scalar("grad/skipped_batches_total", float(self.skipped_batches), step)
        if self.device.type == "cuda":
            self._log_scalar(
                "memory/peak_allocated_gb",
                torch.cuda.max_memory_allocated(self.device) / 1e9,
                step,
            )

    def _log_numerics(
        self,
        batch: dict[str, torch.Tensor],
        terms: dict[str, float],
        grad_norm: float,
        overflowed: bool,
    ) -> None:
        """Emit the Task 1 per-step numerics trace to ``numerics.jsonl``.

        This is what the Phase 4.10 run lacked: the loss log said only that the total
        was non-finite, so there was no way to tell which term, which tensor, or which
        module went first. Enabled by ``--debug-numerics`` and normally scoped to a
        window around the suspected step with ``--debug-numerics-from-step``, because
        every entry forces a device synchronisation.

        Args:
            batch: The current batch, on device.
            terms: Loss decomposition for this step.
            grad_norm: Global gradient norm before clipping.
            overflowed: Whether the GradScaler skipped the optimiser step.
        """
        with torch.no_grad(), self.autocast():
            output = self._forward_for_debug(batch)

        record: dict[str, Any] = {
            "epoch": self.state.epoch,
            "step": self.state.global_step,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "overflowed": overflowed,
            "amp_scale": float(self.scaler.get_scale())
            if self.scaler.is_enabled() else None,
            **{f"loss_{k}": v for k, v in terms.items()},
        }
        for name, tensor in output.items():
            if tensor is None:
                continue
            stats = tensor_stats(tensor)
            record[f"{name}_min"] = stats.min
            record[f"{name}_max"] = stats.max
            record[f"{name}_mean"] = stats.mean
            record[f"{name}_std"] = stats.std
            record[f"{name}_absmax"] = stats.absmax
            record[f"{name}_finite"] = stats.finite
            if self.amp_dtype is torch.float16:
                # Headroom below ~1 means this tensor is at the fp16 ceiling. Tracking
                # it turns divergence from a surprise into a countdown.
                record[f"{name}_fp16_headroom"] = stats.fp16_headroom

        # Parameter and gradient health, which is what separates "the arithmetic
        # overflowed" from "an optimiser step wrote garbage into the weights".
        record["param_absmax"] = max(
            (float(p.detach().float().abs().max()) for p in self.model.parameters()),
            default=0.0,
        )
        record["param_nonfinite"] = sum(
            int(not bool(torch.isfinite(p).all())) for p in self.model.parameters()
        )
        self.numerics_log.log(record)

    def _forward_for_debug(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        """Collect the tensors the Task 1 trace reports on.

        Args:
            batch: The current batch, on device.

        Returns:
            Mapping of trace name to tensor. Entries are ``None`` when the model does
            not produce them, e.g. ``sigma`` with the noise head disabled.
        """
        forward_with_aux = getattr(self.model, "forward_with_aux", None)
        collected: dict[str, torch.Tensor | None] = {
            "input": batch.get("lr"),
            "target": batch.get("gt"),
        }
        if forward_with_aux is not None:
            output = forward_with_aux(batch["lr"])
            collected["prediction"] = output.image
            collected["sigma"] = output.sigma
            noise = getattr(output, "noise", None)
            if noise is not None:
                collected["sigma_gauss"] = noise.sigma_gauss
                collected["sigma_speckle"] = noise.sigma_speckle
        else:
            collected["prediction"] = self.model(batch["lr"])
        return collected

    @torch.no_grad()
    def evaluate(self, module: nn.Module, desc: str = "validating") -> dict[str, float]:
        """Evaluate a module on the validation loader.

        Args:
            module: Model to evaluate (live weights or the EMA shadow).
            desc: Progress-bar label, so the live model and the EMA shadow passes are
                distinguishable on the terminal.

        Returns:
            Metric summary from :class:`MetricAccumulator`.
        """
        module.eval()
        accumulator = MetricAccumulator()
        bands = BandCorrelationAccumulator()
        clean_sse = 0.0
        clean_count = 0
        forward_with_aux = getattr(module, "forward_with_aux", None)
        for batch in tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            batch = self._to_device(batch)
            # Validation must run in the same arithmetic as training; see
            # `Trainer.autocast` for why this is a shared helper and not a local
            # context.
            with self.autocast():
                if forward_with_aux is not None:
                    output = forward_with_aux(batch["lr"])
                    prediction = output.image
                    clean_lr = getattr(output, "clean_lr", None)
                else:
                    prediction = module(batch["lr"])
                    clean_lr = None
            target = batch["gt"].float()
            prediction = prediction.float()
            accumulator.update(prediction, target)

            # Diagnostic pair for the Phase 6 hypothesis: `clean_lr_psnr` says whether
            # the branch is learning the denoising task at all, and the band rhos say
            # whether that is translating into the bands the failure was measured in.
            base = F.interpolate(
                batch["lr"].float(),
                scale_factor=float(self.model_scale),
                mode="bicubic",
                align_corners=False,
            )
            bands.update(prediction, target, base)
            if clean_lr is not None:
                clean_target = forward_operator(target, self.clean_lr_blur_sigma)
                diff = (clean_lr.float().double() - clean_target.double()) ** 2
                clean_sse += float(diff.sum())
                clean_count += int(diff.numel())

        summary = accumulator.summary()
        summary.update(bands.summary())
        if clean_count:
            mse = clean_sse / clean_count
            summary["clean_lr_psnr"] = (
                float("inf") if mse <= 0.0 else float(10.0 * math.log10(1.0 / mse))
            )
        return summary

    # --------------------------------------------------------------------- loop
    def fit(self) -> TrainState:
        """Train until the epoch budget or the early-stopping rule is reached.

        Returns:
            The final training state.
        """
        _LOGGER.info(
            "Training %s for %d epochs on %s (AMP=%s, %d train steps/epoch, %d val "
            "steps/epoch, %d total optimizer steps)",
            self.run_name,
            self.config.epochs,
            self.device,
            self.amp_enabled,
            max(1, len(self.train_loader)),
            max(1, len(self.val_loader)),
            self.total_steps,
        )
        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.model, desc="validating (live)")
            ema_metrics = self.evaluate(self.ema.module, desc="validating (EMA)")
            self._record(epoch, train_metrics, val_metrics, ema_metrics)

            improved = ema_metrics["psnr_mean"] > (
                self.state.best_ema_psnr + self.config.early_stopping_min_delta
            )
            if val_metrics["psnr_mean"] > self.state.best_psnr:
                self.state.best_psnr = val_metrics["psnr_mean"]
                self.save("best_psnr.pt")
            if ema_metrics["psnr_mean"] > self.state.best_ema_psnr:
                self.state.best_ema_psnr = ema_metrics["psnr_mean"]
                self.save("best_ema_psnr.pt")
            self.save("last.pt")

            self.state.epochs_without_improvement = (
                0 if improved else self.state.epochs_without_improvement + 1
            )
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                _LOGGER.info(
                    "Early stopping at epoch %d: no EMA improvement for %d epochs.",
                    epoch,
                    self.state.epochs_without_improvement,
                )
                break

        if self.writer is not None:
            self.writer.close()
        return self.state

    def _record(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        ema_metrics: dict[str, float],
    ) -> None:
        """Emit one epoch's telemetry to every configured sink."""
        lr = self.optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "step": self.state.global_step,
            "lr": lr,
            "train_loss": train_metrics.get("total", float("nan")),
            "val_psnr": val_metrics["psnr_mean"],
            "val_psnr_median": val_metrics["psnr_median"],
            "val_psnr_pooled": val_metrics["psnr_pooled"],
            "val_ssim": val_metrics["ssim_mean"],
            "ema_psnr": ema_metrics["psnr_mean"],
            "ema_psnr_median": ema_metrics["psnr_median"],
            "ema_psnr_pooled": ema_metrics["psnr_pooled"],
            "ema_ssim": ema_metrics["ssim_mean"],
            "seconds": train_metrics.get("seconds", 0.0),
        }
        # Phase 6 diagnostics. `psnr_mean` and `psnr_pooled` are both logged from here
        # on: the two reductions are not comparable, and mixing them is what produced
        # the false "27.36 dB ceiling" that stalled the project for several cycles.
        for key in ("clean_lr_psnr", "rho_low", "rho_mid", "rho_high", "rho_nyq"):
            if key in val_metrics:
                record[f"val_{key}"] = val_metrics[key]
            if key in ema_metrics:
                record[f"ema_{key}"] = ema_metrics[key]
        record["clean_lr_weight"] = float(
            getattr(self.criterion, "weights", {}).get("clean_lr", 0.0)
        )
        for key, value in train_metrics.items():
            if key not in ("seconds",):
                record[f"train_{key}"] = value

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(self.device) / 1e9
            if self.device.type == "cuda"
            else float("nan")
        )
        record["peak_vram_gb"] = peak_vram_gb

        _LOGGER.info(
            "epoch %3d | loss %.5f | val %.3f dB / %.4f | EMA %.3f dB / %.4f | "
            "lr %.2e | %.1f s%s",
            epoch,
            record["train_loss"],
            record["val_psnr"],
            record["val_ssim"],
            record["ema_psnr"],
            record["ema_ssim"],
            lr,
            record["seconds"],
            f" | peak VRAM {peak_vram_gb:.2f} GB" if self.device.type == "cuda" else "",
        )
        self.jsonl.log(record)
        self.csv.log(record)
        self.state.history.append(record)
        self._log_epoch(epoch, train_metrics, val_metrics, ema_metrics)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        ema_metrics: dict[str, float],
    ) -> None:
        """Emit per-epoch telemetry under the :mod:`trainer.tb_layout` groups.

        Args:
            epoch: Epoch index (the step base for every tag written here).
            train_metrics: Averaged training metrics.
            val_metrics: Validation metrics for the live weights.
            ema_metrics: Validation metrics for the EMA weights.
        """
        if self.writer is None:
            return
        for key, value in val_metrics.items():
            self._log_scalar(f"val/{key}", value, epoch)
        for key, value in ema_metrics.items():
            self._log_scalar(f"ema/{key}", value, epoch)
        for key, value in train_metrics.items():
            if key in ("seconds", "skipped_batches", "grad_norm"):
                continue
            self._log_scalar(f"loss/{key}", value, epoch)
            self._log_scalar(f"train/{key}", value, epoch)
        # `optim/lr` is deliberately NOT written here: it is already logged every step
        # at `global_step`, and writing the same tag against `epoch` would mix two step
        # bases on one plot.
        seconds = train_metrics.get("seconds", 0.0)
        self._log_scalar("throughput/seconds_per_epoch", seconds, epoch)
        if seconds > 0:
            images = len(self.train_loader) * self.config.batch_size
            self._log_scalar("throughput/images_per_second", images / seconds, epoch)

    # -------------------------------------------------------------- checkpoints
    def save(self, filename: str) -> Path:
        """Write a checkpoint containing everything needed to resume exactly.

        Args:
            filename: File name inside the run's checkpoint directory.

        Returns:
            The path written.
        """
        path = self.checkpoint_dir / filename
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "state": {
                    "epoch": self.state.epoch,
                    "global_step": self.state.global_step,
                    "best_psnr": self.state.best_psnr,
                    "best_ema_psnr": self.state.best_ema_psnr,
                    "epochs_without_improvement": self.state.epochs_without_improvement,
                },
            },
            path,
        )
        return path

    def load(self, path: Path, resume: bool = True) -> None:
        """Restore from a checkpoint.

        Args:
            path: Checkpoint path.
            resume: When ``True``, also restore optimiser, scheduler, scaler and the
                training state so that training continues bit-identically.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}.")
        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.ema.load_state_dict(payload["ema"])
        if not resume:
            return
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        saved = payload["state"]
        self.state.epoch = int(saved["epoch"]) + 1
        self.state.global_step = int(saved["global_step"])
        self.state.best_psnr = float(saved["best_psnr"])
        self.state.best_ema_psnr = float(saved["best_ema_psnr"])
        self.state.epochs_without_improvement = int(saved["epochs_without_improvement"])
        _LOGGER.info("Resumed from %s at epoch %d.", path, self.state.epoch)
