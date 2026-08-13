"""Frozen configuration for SPARC-Base V1.0.

Every constant here is fixed by ``SPARC_BASE_V1_IMPLEMENTATION_CONTRACT.md``.
Changing a value in this file is a contract amendment (Part 16), not a tuning
decision. The boolean ``use_*`` flags exist solely to support the staged build
order (Part 8) and the ablation contract (Part 11); they must all be ``True`` in
the final V1 model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTENTION_HEAD_DIM = 16
"""Contract Part 5: attention head dimension is invariant at 16."""


@dataclass(frozen=True, slots=True)
class NoiseHeadConfig:
    """Contract Part 2.9 — blind two-parameter noise estimator."""

    trunk_channels: tuple[int, ...] = (16, 24, 32, 32)
    """Post-SimpleGate widths of the four strided stages."""
    smoother_kernel: int = 5
    sigma_gauss_init: float = 0.024
    sigma_speckle_init: float = 0.165
    sigma_min: float = 1e-4
    sigma_max: float = 2.0


@dataclass(frozen=True, slots=True)
class SparcConfig:
    """SPARC-Net architecture configuration.

    Level index 0 is the finest trunk level (64x64 for a 128x128 input); the trunk
    begins one Haar level below the input because of the DWT stem.
    """

    name: str = "sparc-base"

    in_channels: int = 1
    out_channels: int = 1
    scale: int = 2

    widths: tuple[int, int, int] = (48, 96, 160)
    enc_naf_blocks: tuple[int, int, int] = (4, 4, 4)
    enc_gsa_blocks: tuple[int, int, int] = (0, 2, 3)
    dec_naf_blocks: tuple[int, int] = (4, 4)
    """Coarse-to-fine: ``(D1, D0)``."""
    dec_gsa_blocks: tuple[int, int] = (1, 0)
    """Coarse-to-fine: ``(D1, D0)``."""
    num_heads: tuple[int, int, int] = (0, 3, 5)

    head_width: int = 32
    head_naf_blocks: int = 3
    head_project_kernel_size: int = 3
    """Kernel size of the reconstruction head's pre-IDWT projection (``widths[0] ->
    4*head_width``). Frozen at 3 for every V1-lineage variant. ``sparc-restoration-v2``
    is the only config that sets this to 1: the decoder's own ``Upsample`` already uses
    a 1x1 projection ahead of its inverse Haar step, and the head's 3x3 version was
    measured to cost 10.4 % of sparc-xl-moderate's total FLOPs for the same job — the
    single most wasteful op in the network relative to its architectural justification.
    """

    attn_dim_ratio: float = 0.5
    naf_expansion: int = 2
    ffn_expansion: int = 2
    fusion_reduction: int = 4
    layer_scale_init: float = 1e-2
    layer_norm_eps: float = 1e-6

    normalization_scale_floor: float = 0.02
    noise: NoiseHeadConfig = field(default_factory=NoiseHeadConfig)

    use_normalization: bool = True
    use_noise_head: bool = True
    use_gated_fusion: bool = True
    use_attention: bool = True
    use_attention_branch: bool = True
    """Ablation switch: keep the GSA blocks but drop their self-attention sub-branch.

    ``use_attention=False`` removes the six GSA blocks outright, which deletes the
    attention *and* the GDFN feed-forward inside them — 608,088 parameters of which only
    274,776 are attention. The pre-attention baseline is therefore not a clean attention
    ablation, and the measured null result cannot be attributed to attention alone.

    Setting this to ``False`` keeps every GSA block, its two LayerNorms are reduced to
    one, the GDFN branch and its LayerScale are untouched, and only
    ``norm1``/``qkv``/``qkv_dwconv``/``project``/``scale1``/``rel_pos`` are dropped. That
    isolates the attention sub-branch as the single independent variable and yields a
    2,070,874-parameter arm sitting between the two existing checkpoints.

    ``True`` is the V1 behaviour and leaves every existing checkpoint loading unchanged.
    """
    use_global_residual: bool = True
    clamp_output: bool = True
    output_min: float = 0.0
    output_max: float = 1.0

    use_clean_lr_branch: bool = False
    """Phase 6: predict a supervised clean-LR estimate and use it as the residual base.

    Measured motivation: with the V1 formulation ``out = head + bicubic(noisy_LR)`` the
    network must synthesise an exact cancelling signal for the noise the residual path
    injects. On the 320-image validation split the trained sparc-base checkpoint removes
    only 34 % of the low-frequency error energy of ``bicubic(noisy_LR)``, against 66-69 %
    in the mid and high bands, and ``corr(R, R*)`` is 0.599 at low frequency versus
    0.831/0.815 at mid/high — the network is *worst* in the band that should be easiest.

    When ``True`` a lightweight branch reads the decoder trunk and predicts ``Delta`` at
    LR resolution; ``clean_norm = y_hat + Delta`` becomes the residual base (subject to
    ``residual_source``) and is supervised against ``A(GT)``. The branch's output
    convolution is zero-initialised, so at step 0 ``Delta = 0`` and the model is
    **bit-identical** to the V1 path — verified ``max|new - old| = 0``.

    Defaults to ``False`` so every frozen variant and every existing checkpoint is
    unaffected.
    """
    clean_branch_width: int = 32
    """Channel width of the clean-LR branch at 128x128. Matches ``head_width`` rather
    than being tuned independently: the branch mirrors the reconstruction head's proven
    ``project -> IDWT`` structure."""
    clean_branch_naf_blocks: int = 2
    """NAF blocks in the clean-LR branch. One fewer than ``head_naf_blocks`` because the
    LR-domain read-out is a strictly better-conditioned task than HR detail synthesis."""
    clean_branch_zero_init: bool = True
    """Zero-initialise the branch's output convolution. This is what makes the modified
    network start from the existing trained behaviour instead of a random residual."""
    residual_source: str = "noisy"
    """Which tensor feeds the global residual: ``"noisy"`` (V1, ``y_hat``) or ``"clean"``
    (``y_hat + Delta``). ``"noisy"`` is retained as an exact compatibility mode and is
    the default; it also permits training the clean-LR branch as a pure auxiliary task
    without changing the output path."""

    use_vst: bool = False
    """Phase 5: stabilise the input variance before the trunk (``models/vst.py``).

    Phase 1 measured ``Var(y | I) = sigma_g^2 + sigma_s^2 I^2``, which makes the noise
    standard deviation vary **9.50x** with intensity inside every image. ``True`` applies
    the exact asinh stabiliser driven by the NoiseHead's own ``(sigma_g, sigma_s)`` and
    inverts it after de-normalisation, which measures the residual spread down to 1.48x.
    Requires ``use_noise_head``. Defaults to ``False`` so every frozen variant and every
    existing checkpoint is unaffected.
    """
    use_vst_bias_correction: bool = True
    """Register the learnable second-order correction on the inverse transform.

    Initialised at exactly zero, so training starts from the naive inverse. Inert when
    ``use_vst`` is ``False``.
    """
    use_film: bool = False
    """Phase 5: FiLM-condition the trunk on ``(sigma_g, sigma_s)`` (``models/film.py``).

    The noise level currently reaches the trunk once, as an input channel. ``True``
    re-asserts it as a per-channel affine modulation after the NAF stack at L0, L1, L2,
    D1 and D0. Identity at initialisation (gamma = beta = 0). Requires
    ``use_noise_head``. Defaults to ``False``.
    """
    film_hidden: int = 64
    """Pre-SimpleGate width of the shared FiLM trunk. Inert when ``use_film`` is False."""

    use_sdpa: bool = True
    """Contract Part 2.6: SDPA is mandatory in training (saves 20.8 MB/image)."""
    use_relative_position_bias: bool = True

    input_size: int = 128
    """Spatial size the relative-position-bias tables are constructed for."""

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ helpers
    @property
    def num_levels(self) -> int:
        return len(self.widths)

    @property
    def trunk_size(self) -> int:
        """Spatial size of trunk level 0 (one Haar level below the input)."""
        return self.input_size // 2

    def level_size(self, level: int) -> int:
        """Spatial size at trunk ``level``."""
        if not 0 <= level < self.num_levels:
            raise IndexError(f"level {level} out of range [0, {self.num_levels}).")
        return self.trunk_size // (2**level)

    def attention_dim(self, level: int) -> int:
        """qkv projection width at ``level``."""
        return int(self.widths[level] * self.attn_dim_ratio)

    @property
    def film_sites(self) -> tuple[int, ...]:
        """Channel width of each FiLM conditioning site, in application order.

        Encoder levels fine-to-coarse, then decoder stages coarse-to-fine: for
        ``widths=(96, 144, 192)`` this is ``(96, 144, 192, 144, 96)`` — L0, L1, L2,
        D1, D0. Decoder stage ``s`` outputs trunk level ``num_levels - 2 - s``.
        """
        encoder = tuple(self.widths)
        decoder = tuple(
            self.widths[self.num_levels - 2 - stage]
            for stage in range(self.num_levels - 1)
        )
        return encoder + decoder

    def uses_attention_at(self, level: int) -> bool:
        """Whether any encoder or decoder stage at ``level`` instantiates a GSA block."""
        if self.enc_gsa_blocks[level] > 0:
            return True
        decoder_index = self.num_levels - 2 - level
        return 0 <= decoder_index < len(self.dec_gsa_blocks) and (
            self.dec_gsa_blocks[decoder_index] > 0
        )

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Verify every structural invariant of the contract.

        Raises:
            ValueError: If any invariant is violated.
        """
        n = self.num_levels
        if n != 3:
            raise ValueError(f"SPARC-Base V1 is specified with 3 trunk levels, got {n}.")
        for name in ("enc_naf_blocks", "enc_gsa_blocks", "num_heads"):
            value = getattr(self, name)
            if len(value) != n:
                raise ValueError(f"{name} must have {n} entries, got {len(value)}.")
        for name in ("dec_naf_blocks", "dec_gsa_blocks"):
            value = getattr(self, name)
            if len(value) != n - 1:
                raise ValueError(f"{name} must have {n - 1} entries, got {len(value)}.")
        if self.scale != 2:
            raise ValueError("SPARC-Net V1 supports scale factor 2 only.")
        if self.input_size % (2**n) != 0:
            raise ValueError(
                f"input_size {self.input_size} must be divisible by {2**n} "
                f"(stem DWT plus {n - 1} encoder DWT stages)."
            )
        if not 0.0 < self.attn_dim_ratio <= 1.0:
            raise ValueError(f"attn_dim_ratio must be in (0, 1], got {self.attn_dim_ratio}.")
        if self.residual_source not in ("noisy", "clean"):
            raise ValueError(
                f"residual_source must be 'noisy' or 'clean', got {self.residual_source!r}."
            )
        if self.residual_source == "clean" and not self.use_clean_lr_branch:
            raise ValueError(
                "residual_source='clean' requires use_clean_lr_branch=True; there is no "
                "clean-LR estimate to feed the residual otherwise."
            )
        if self.use_clean_lr_branch:
            if self.clean_branch_width <= 0 or self.clean_branch_width % 2 != 0:
                raise ValueError(
                    "clean_branch_width must be a positive even number, got "
                    f"{self.clean_branch_width}."
                )
            if self.clean_branch_naf_blocks < 0:
                raise ValueError(
                    "clean_branch_naf_blocks must be non-negative, got "
                    f"{self.clean_branch_naf_blocks}."
                )

        for level in range(n):
            width = self.widths[level]
            if width % 2 != 0:
                raise ValueError(f"Level {level} width {width} must be even (SimpleGate).")
            if not self.uses_attention_at(level):
                continue
            heads = self.num_heads[level]
            dim = self.attention_dim(level)
            if heads <= 0:
                raise ValueError(
                    f"Level {level} instantiates attention but num_heads is {heads}."
                )
            if dim % heads != 0:
                raise ValueError(
                    f"Level {level}: attention dim {dim} is not divisible by {heads} heads."
                )
            head_dim = dim // heads
            if head_dim != ATTENTION_HEAD_DIM:
                raise ValueError(
                    f"Level {level}: head_dim is {head_dim}, contract requires "
                    f"{ATTENTION_HEAD_DIM} (width={width}, ratio={self.attn_dim_ratio}, "
                    f"heads={heads})."
                )
        if self.head_width % 2 != 0:
            raise ValueError(f"head_width {self.head_width} must be even.")
        if self.head_project_kernel_size <= 0 or self.head_project_kernel_size % 2 == 0:
            raise ValueError(
                "head_project_kernel_size must be a positive odd number, got "
                f"{self.head_project_kernel_size}."
            )
        if self.use_vst and not self.use_noise_head:
            raise ValueError(
                "use_vst requires use_noise_head: the transform is parameterised by "
                "the head's (sigma_g, sigma_s) and has no other source for them."
            )
        if self.use_film:
            if not self.use_noise_head:
                raise ValueError(
                    "use_film requires use_noise_head: the conditioner takes the "
                    "head's (sigma_g, sigma_s) as its only input."
                )
            if self.film_hidden <= 0 or self.film_hidden % 2 != 0:
                raise ValueError(
                    f"film_hidden must be a positive even number, got {self.film_hidden}."
                )
            for width in self.film_sites:
                if width % 2 != 0:
                    raise ValueError(
                        f"FiLM site width {width} must be even (SimpleGate halves it)."
                    )

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: Any) -> "SparcConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Frozen training contract (Contract Parts 5 and 6)."""

    seed: int = 1337
    epochs: int = 400
    batch_size: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.9)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    amp_init_scale: float = 2.0**14
    amp_dtype: str = "fp16"
    """Autocast dtype: ``"fp16"`` (contract default) or ``"bf16"``.

    Phase 4.10.1 operational setting, not a contract hyperparameter — it selects the
    arithmetic the frozen model is evaluated in, and changes no weight, shape or
    schedule. bf16 carries fp32's exponent range, which removes the 65504 activation
    ceiling that the Phase 4.10 divergence ran into; it requires Ampere or newer and
    runs without a ``GradScaler``.
    """
    max_consecutive_nonfinite: int = 25
    """Consecutive non-finite batches tolerated before training aborts.

    Phase 4.10 skipped 300+ consecutive non-finite batches and ran to completion having
    learned nothing after step 423. Beyond a handful, consecutive failures mean the
    weights are diverged rather than the batch unlucky, and continuing only burns GPU
    time. See :meth:`trainer.trainer.Trainer.train_epoch`.
    """
    detect_anomaly: bool = False
    """Run the step inside ``torch.autograd.detect_anomaly``. Debugging only."""
    debug_numerics: bool = False
    """Log per-term losses and tensor ranges every step. Debugging only; forces a
    device synchronisation per step and roughly halves throughput."""
    debug_numerics_from_step: int = 0
    """First global step at which ``debug_numerics`` telemetry is emitted."""
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    channels_last: bool = True
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 0.01

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    val_block_size: int = 32
    """Group-aware split: contiguous ID blocks of this size."""
    val_every_n_blocks: int = 10
    """Every n-th block goes to validation (320 val / 2880 train)."""

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    def validate(self) -> None:
        """Raises:
        ValueError: If a training constant is outside the contract."""
        if self.batch_size not in (8, 16):
            raise ValueError(
                "Contract Part 5 permits batch size 8, or 16 only under the sanctioned "
                f"override; got {self.batch_size}."
            )
        if self.batch_size == 16 and abs(self.learning_rate - 4.2e-4) > 1e-6:
            raise ValueError(
                "The sanctioned batch-16 override requires learning_rate 4.2e-4."
            )
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs.")
        if self.amp_dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"amp_dtype must be 'fp16' or 'bf16', got {self.amp_dtype!r}."
            )
        if self.max_consecutive_nonfinite < 1:
            raise ValueError(
                "max_consecutive_nonfinite must be at least 1, got "
                f"{self.max_consecutive_nonfinite}."
            )
        if self.debug_numerics_from_step < 0:
            raise ValueError(
                "debug_numerics_from_step must be non-negative, got "
                f"{self.debug_numerics_from_step}."
            )


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Frozen loss weights (Contract Part 6)."""

    charbonnier: float = 1.00
    charbonnier_eps: float = 1e-6
    ms_ssim: float = 0.15
    wavelet: float = 0.10
    fft: float = 0.05
    gradient: float = 0.05
    noise_aux: float = 0.02
    wavelet_band_weights: tuple[float, float, float, float] = (0.25, 1.0, 1.0, 1.5)
    """``(LL, LH, HL, HH)``."""
    wavelet_levels: int = 2
    ms_ssim_scales: int = 5
    ms_ssim_window: int = 11
    ms_ssim_sigma: float = 1.5

    clean_lr: float = 0.5
    """Phase 6 weight on ``Charbonnier(predicted_clean_LR, A(GT))``. Inert unless the
    model produces ``clean_lr``, so every existing configuration is unchanged.

    Applied through a warmup (see ``clean_lr_warmup_epochs``) rather than flat. At step 0
    the branch is zero-initialised, so ``predicted_clean_LR`` is the *noisy* LR: measured
    PSNR 22.26 dB, Charbonnier ~0.07, against a converged ``train_charbonnier`` of ~0.031.
    A flat 0.5 would make the new term ~40 % of the objective on the first step of a
    warm-started run."""
    clean_lr_blur_sigma: float = 0.4
    """Blur sigma of the supervision target ``A(GT) = bicubic_down2(g_sigma * GT)``.

    Fixed at the Phase 1 global estimate rather than the per-sample value, because the
    loader does not return the sampled degradation parameters and Requirement 8 forbids
    changing the dataset. Measured cost of the approximation: PSNR between ``A_0.4(GT)``
    and ``A_sigma(GT)`` is 46-55 dB across the sampled range sigma in [0.3, 0.5], far
    above the ~30-35 dB regime the branch operates in. sigma=0.4 is also the best fit to
    the real LR on that grid (22.256 dB, maximal)."""
    clean_lr_warmup_epochs: int = 3
    """Epochs over which ``clean_lr`` ramps linearly from 0 to its full value:
    epoch 0 -> 0, epoch 1 -> 1/6, epoch 2 -> 1/3, epoch 3+ -> 0.5."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset layout and degradation-synthesis constants."""

    raw_root: Path = PROJECT_ROOT / "Data"
    packed_root: Path = PROJECT_ROOT / "data" / "packed"
    train_lr_subpath: str = "train/train/NoisyLR"
    train_gt_subpath: str = "train/train/GT"
    test_lr_subpath: str = "Test_NoisyLR (1)/NoisyLR"
    expected_train: int = 3200
    expected_test: int = 400
    lr_size: int = 128
    gt_size: int = 256
    store_dtype: str = "float16"

    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.75
    resynth_prob: float = 0.5

    blur_sigma_range: tuple[float, float] = (0.3, 0.5)
    speckle_looks_range: tuple[float, float] = (26.0, 51.0)
    """Amendment A-001: Phase 1 measured sigma_s p10/p90 = 0.140/0.194, i.e.
    L = 26.6/51.0. The contract's (25, 60) centred sigma_s at 0.155 instead of 0.162."""
    gauss_sigma_range: tuple[float, float] = (0.0, 0.04)
    """Amendment A-001: narrowed from (0.0, 0.08); see ``AMENDMENTS.md``."""

    noise_at_lr: bool = True
    """Amendment A-001: inject speckle and additive noise after decimation.

    ``False`` reproduces the frozen contract (noise at 256x256, then decimate), which
    delivers only 50 % of the measured speckle variance. See ``AMENDMENTS.md``.
    """


def sparc_tiny() -> SparcConfig:
    """Debug variant (Contract Part 8, step 6). No attention, no noise head."""
    return SparcConfig(
        name="sparc-tiny",
        widths=(24, 48, 96),
        enc_naf_blocks=(1, 1, 2),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(1, 1),
        dec_gsa_blocks=(0, 0),
        num_heads=(0, 0, 0),
        head_width=16,
        head_naf_blocks=1,
        use_noise_head=False,
        use_gated_fusion=False,
        use_attention=False,
    )


def sparc_base() -> SparcConfig:
    """The frozen V1.0 competition model."""
    return SparcConfig()


def sparc_clean_lr() -> SparcConfig:
    """sparc-base plus the supervised clean-LR base (Phase 6).

    Identical to :func:`sparc_base` in every backbone dimension — same widths, same
    block counts, same attention setting, same reconstruction head. The only change is
    the residual base: ``bicubic(y_hat + Delta)`` instead of ``bicubic(y_hat)``, where
    ``Delta`` comes from a zero-initialised branch supervised against ``A(GT)``.

    Measured cost against sparc-base: +72,161 parameters (+3.08 %) and +18.8 % MACs
    (2.4055 -> 2.8569 GMAC at 1x1x128x128).
    """
    return SparcConfig(
        name="sparc-clean-lr",
        use_clean_lr_branch=True,
        residual_source="clean",
    )


def sparc_fine() -> SparcConfig:
    """Capacity reallocated coarse-to-fine, at slightly *lower* parameter count.

    Phase 1 measured 96.3 % of target energy below LR Nyquist and spatially white
    noise, which makes this a local, high-resolution restoration problem. The frozen
    V1 allocation is the opposite shape: 44.6 % of convolution parameters sit at
    16x16 (doing 10.7 % of the MACs) and only 12.0 % at 64x64 or finer. Three
    completed ablations added 333 k (GDFN), 275 k (attention) and 608 k (both)
    parameters to those coarse levels for a net **-0.018 dB** on 320 held-out
    images, so coarse capacity is empirically saturated and moving it is free.

    This variant moves it: the bottleneck narrows 160 -> 128 and loses one NAF
    block, level 0 doubles to 8 blocks at width 64, the fine decoder stage grows
    4 -> 6 blocks, and the reconstruction head widens 32 -> 48 with a fourth NAF
    block. The result is **1,673,518 parameters - 64,044 fewer than sparc-base's
    pre-attention arm (1,737,562)** - with the share at >=64x64 lifted from 12.0 %
    to 33.4 %. Compute rises 1.82 -> 3.57 GMACs because the work moves to larger
    feature maps.

    Everything else is untouched: RobustNormalizer, NoiseHead, the Haar DWT/IDWT
    resampling, the NAFBlock internals, GatedFuse, and the global bicubic residual
    are all inherited unchanged from :class:`SparcConfig`.

    ``num_heads`` is unused while ``use_attention=False`` but is set to values that
    divide each level's attention dimension into head_dim 16, so the same backbone
    can be re-run with attention enabled without editing this function.
    """
    return SparcConfig(
        name="sparc-fine",
        widths=(64, 96, 128),
        enc_naf_blocks=(8, 4, 3),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(3, 6),
        dec_gsa_blocks=(0, 0),
        num_heads=(2, 3, 4),
        head_width=48,
        head_naf_blocks=4,
        use_attention=False,
    )


def sparc_xl() -> SparcConfig:
    """Phase 5 candidate: variance-stabilised, FiLM-conditioned, moderately widened.

    Three mechanisms, each tied to a measurement rather than to a scaling intuition:

    1. **VST** (``use_vst``). ``Var(y | I) = sigma_g^2 + sigma_s^2 I^2`` makes the noise
       standard deviation span **9.50x** with intensity inside every image, so the trunk
       currently has to represent an intensity-indexed family of denoisers. The exact
       asinh stabiliser collapses that to one member (measured residual spread 1.48x).
    2. **FiLM** (``use_film``). The noise level reaches the trunk once, as an input
       channel that must then survive the DWT stem and three levels of resampling. FiLM
       re-asserts it at L0, L1, L2, D1, D0, where the denoising decisions are made.
    3. **Moderate width scaling** to ``(96, 144, 192)``. Ranked *third* deliberately:
       three ablations at 1.7M-2.35M landed within 0.03 dB, so width alone is not
       expected to carry this variant. It is here to give the two conditioning
       mechanisms something to modulate.

    Attention stays off: three controlled experiments measured the GSA arm at a net
    **-0.018 dB** on 320 held-out images. No high-frequency or FFT branch is added; the
    model was measured to pass *more* high-frequency energy than the Wiener-optimal
    estimator already, so it is under-smoothing rather than over-smoothing.

    Backbone is 4,505,458 parameters; FiLM adds 46,656 and the VST bias correction 1,
    for **4,552,115** total at **9.070 GMACs / 18.14 GFLOP**.
    """
    return SparcConfig(
        name="sparc-xl",
        widths=(96, 144, 192),
        enc_naf_blocks=(8, 6, 4),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(4, 8),
        dec_gsa_blocks=(0, 0),
        num_heads=(3, 4, 6),
        head_width=64,
        head_naf_blocks=6,
        use_attention=False,
        use_vst=True,
        use_film=True,
    )


def sparc_xl_moderate() -> SparcConfig:
    """Phase 5 moderate-memory variant preserving the conditioning mechanisms.

    This is the approved activation-focused reduction of :func:`sparc_xl`. It keeps the
    same core architectural ideas â€” VST, FiLM, the noise-aware multi-scale trunk, and
    the reconstruction head â€” while shrinking the dominant high-resolution path:

    * finest trunk width: ``96 -> 80``
    * fine decoder stage depth: ``8 -> 6``
    * reconstruction head width: ``64 -> 48``
    * reconstruction head depth: ``6 -> 4``
    * moderate mid/coarse trim: ``(144, 192) -> (136, 176)``

    The goal is to reduce training activation memory at batch 8 without deleting the
    mechanisms that are most likely to matter for PSNR/SSIM.
    """
    return SparcConfig(
        name="sparc-xl-moderate",
        widths=(80, 136, 176),
        enc_naf_blocks=(8, 6, 4),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(4, 6),
        dec_gsa_blocks=(0, 0),
        num_heads=(3, 4, 6),
        head_width=48,
        head_naf_blocks=4,
        use_attention=False,
        use_vst=True,
        use_film=True,
    )


def sparc_restoration_v2() -> SparcConfig:
    """Coarse-to-fine compute reallocation (PRIMARY candidate, Phase 6).

    Measured on ``sparc-xl-moderate`` with ``torch.utils.flop_counter``: the 64x64
    trunk level (encoder L0 + decoder D0) alone consumes **41.8 %** of total FLOPs,
    while the 16x16 bottleneck — the cheapest resolution, where contextual reasoning
    is least expensive per unit of receptive field — consumes only **3.5 %**. The
    128x128 reconstruction head, which is the network's only genuinely
    highest-resolution compute, is actually *cheaper* (17.6 %) than the 64x64 level.
    That allocation is backwards for a network whose Phase 1 measurements show the
    noise is spatially white and restoration is dominated by local statistics at fine
    scale but needs long-range context to disambiguate structure at coarse scale.

    This variant does not add new mechanisms; it moves existing, validated ones:

    * ``widths`` ``80,136,176 -> 64,160,224``: the finest trunk level narrows (it is
      the most expensive per channel, being H*W=4096), the mid and bottleneck levels
      widen (cheapest compute per channel, H*W=1024 and 256).
    * ``enc_naf_blocks`` ``8,6,4 -> 3,8,8``: block count follows the same inversion —
      light at 64x64, heaviest at 32x32 and 16x16 where a NAF block's receptive field
      covers proportionally more of the image per FLOP spent.
    * ``dec_naf_blocks`` ``4,6 -> 6,3``: mirrors the encoder change; D1 (32x32) grows,
      D0 (64x64) shrinks.
    * ``head_project_kernel_size`` ``3 -> 1``: the reconstruction head's pre-IDWT
      projection was the single most wasteful op (10.4 % of total FLOPs for a 3x3
      kernel doing a job the decoder's own ``Upsample`` does with a 1x1 everywhere
      else). Shrinking it funds part of the bottleneck growth for free.
    * ``head_width``/``head_naf_blocks`` unchanged at 48/4: the lightweight
      high-resolution detail path near the output is preserved exactly as in
      sparc-xl-moderate — this is deliberately the one place compute is *not* pulled
      from, since it is what recovers fine texture right before the final IDWT.

    VST, FiLM, gated fusion, the noise head, global residual and normalisation are
    all inherited unchanged: three controlled ablations already established each of
    them is net-positive or architecturally required, and attention was measured at
    a net -0.015 to -0.018 dB, so none of that is revisited here.

    At measurement time: ~6.4 M parameters, ~5.0 GMAC (see
    ``reports/inspect_full.py`` or ``utils.complexity.measure_complexity`` for the
    exact figures for the installed code, since block/width edits change both).
    """
    return SparcConfig(
        name="sparc-restoration-v2",
        widths=(64, 160, 224),
        enc_naf_blocks=(3, 8, 8),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(6, 3),
        dec_gsa_blocks=(0, 0),
        num_heads=(0, 0, 0),
        head_width=48,
        head_naf_blocks=4,
        head_project_kernel_size=1,
        use_attention=False,
        use_vst=True,
        use_film=True,
    )


def sparc_restoration_v2_fallback() -> SparcConfig:
    """Conservative fallback: the same reallocation direction, half the magnitude.

    If ``sparc-restoration-v2`` trains unstably or its wider bottleneck (224 channels,
    8 blocks at 16x16) turns out to need more epochs to converge than the 3-day budget
    allows, this variant applies the identical coarse-to-fine principle with a smaller
    step size from ``sparc-xl-moderate``, so it carries less risk and less deviation
    from an already-measured configuration.

    ``widths`` moves only ``80,136,176 -> 72,144,192`` (vs. v2's 64,160,224), and block
    counts shift by 1-2 rather than v2's 3-5. The 1x1 head projection is kept — it is
    a free efficiency win with no architectural downside, independent of how
    aggressively the rest of the budget is reallocated.
    """
    return SparcConfig(
        name="sparc-restoration-v2-fallback",
        widths=(72, 144, 192),
        enc_naf_blocks=(4, 7, 6),
        enc_gsa_blocks=(0, 0, 0),
        dec_naf_blocks=(5, 4),
        dec_gsa_blocks=(0, 0),
        num_heads=(0, 0, 0),
        head_width=48,
        head_naf_blocks=4,
        head_project_kernel_size=1,
        use_attention=False,
        use_vst=True,
        use_film=True,
    )


SPARC_VARIANTS = {
    "sparc-tiny": sparc_tiny,
    "sparc-base": sparc_base,
    "sparc-clean-lr": sparc_clean_lr,
    "sparc-fine": sparc_fine,
    "sparc-xl": sparc_xl,
    "sparc-xl-moderate": sparc_xl_moderate,
    "sparc-restoration-v2": sparc_restoration_v2,
    "sparc-restoration-v2-fallback": sparc_restoration_v2_fallback,
}


def build_sparc_config(variant: str = "sparc-base", **overrides: Any) -> SparcConfig:
    """Build a named SPARC configuration.

    Args:
        variant: ``sparc-tiny`` or ``sparc-base``.
        **overrides: Fields to replace on the resulting config.

    Returns:
        The requested configuration, validated.

    Raises:
        KeyError: If ``variant`` is unknown.
    """
    key = variant.lower()
    if key not in SPARC_VARIANTS:
        raise KeyError(
            f"Unknown SPARC variant '{variant}'. Available: {sorted(SPARC_VARIANTS)}"
        )
    config = SPARC_VARIANTS[key]()
    return config.with_overrides(**overrides) if overrides else config
