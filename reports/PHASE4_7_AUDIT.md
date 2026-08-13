# Phase 4.7 — Codebase Audit and Forward Plan

**Date:** 2026-08-06
**Status of Phase 4.7:** **OPEN** — the overfit gate is still running.
**Scope:** read-only audit. No module was designed, redesigned or added while the gate runs.

Reference: `SPARC_BASE_V1_IMPLEMENTATION_CONTRACT.md` (frozen), `AMENDMENTS.md` (A-001).

**Hardware note (affects several later phases):** this machine is **CPU-only** —
`torch 2.10.0+cpu`, `torch.cuda.is_available() == False`, 12 logical cores. Every
GPU-denominated acceptance number in the contract (Part 10 VRAM limits, latency limits,
7–13 h training estimate) is **unmeasurable here**. See Phase 4.13/4.14 in Task 5.

---

## TASK 1 — Module-by-module audit against the contract

Legend: **I** implemented and contract-conformant · **P** partially implemented ·
**M** missing · **D** deviates from contract.

### configs/

| File | Status | Notes |
|---|---|---|
| `__init__.py` | I | |
| `default_config.py` | I | Pre-SPARC project config, retained per Part 14. |
| `sparc_config.py` | I | All Part 5 constants present. `validate()` enforces head_dim==16, 3 levels, even widths, scale==2. `sparc_tiny()` / `sparc_base()` both present. |

### datasets/

| File | Status | Notes |
|---|---|---|
| `packed_dataset.py` | I | memmap-backed, contract shapes verified by 7 tests. |
| `transforms.py` | I | Paired geometric only; no photometric jitter (Part 6 compliant). |
| `degradation.py` | I **D** | Deviates from Part 6 ranges **by sanctioned Amendment A-001**: `speckle_looks (26,51)` not `(25,60)`; `gauss_sigma (0,0.04)` not `(0,0.08)`; `noise_at_lr=True`. Justified by Phase 1 measurement, recorded in `AMENDMENTS.md`. **Legitimate.** |
| `splits.py` | I | Group-aware, 32-ID blocks, every 10th → val. Zero-overlap test present. |

### models/

| File | Status | Notes |
|---|---|---|
| `sparc_net.py` | I | Full Part 1 data flow. `forward_with_aux` already returns `(image, sigma, stats)` — the hook the noise-aux loss needs. Deferred imports let step-6 build before attention/noise exist. |
| `encoder.py` | I | Stem `Conv3x3(8→48)` = **3,504 params, matches Part 3 stage 3 exactly**. `Downsample Conv1x1(192→96)` = **18,528, exact**. |
| `normalization.py` | I | mean/std, `clamp_min(0.02)`, stats on raw unclamped input. Round-trip exactness tested. |
| `blocks/layer_norm.py` | I | Channel-wise, eps 1e-6, `2C` params, identity at init. |
| `blocks/simple_gate.py` | I | `SimpleGate` + `LayerScale(gamma, init 1e-2)`. Part 14 places both here — conformant. |
| `blocks/naf_block.py` | I **D** | See erratum note below. Params **exact** vs Part 3. SCA implemented as `mean(dim=[2,3])` ≡ `AdaptiveAvgPool2d(1)`, no sigmoid — conformant and more export-friendly. |
| `wavelet/haar.py` | I | Forward/inverse coefficients match Part 2.4; reshape/slice/add/mul only; invertibility + ONNX tested. |
| `decoder/decoder.py` | I | Mirror structure, GSA only at D1, fusion pluggable via `build_fusion`. |
| `decoder/reconstruction_head.py` | I | `Conv3x3(48→128)` = **55,424 exact**; `Conv3x3(32→4)` = **1,156 exact**. No conv at 256². `predict_subbands()` already exposed for the wavelet loss. |
| `fusion/concat_fuse.py` | I | A3 control arm. |
| `fusion/gated_fuse.py` | **M** | Phase 4.11. `build_fusion` already dispatches to it behind `use_gated_fusion`. |
| `attention/rel_pos.py` | **M** | Phase 4.12. |
| `attention/gsa_block.py` | **M** | Phase 4.12. Directory exists but is **completely empty — no `__init__.py`**. |
| `noise/noise_head.py` | **M** | Phase 4.9. |
| `noise/noise_map.py` | **M** | Phase 4.9. Directory empty, **no `__init__.py`**. |

**NAFBlock parameter erratum (accepted, already documented in the source).**
Part 2.5 prose prints `7C² + 8C`; the Part 3 table implies `7C² + 33C`
(C=48 → 17,712 × 4 = 70,848 = Part 3 stage 4 exactly). The implementation follows the
**table**, which is authoritative and internally consistent. The extra `25C` is
2×LayerNorm (4C) + 2×LayerScale (2C) + depthwise 3×3 with bias (20C) − overlap.
This is a documentation erratum in the contract, **not** an implementation deviation.
No amendment needed; recommend a one-line correction to Part 2.5 prose at the next
amendment cycle.

### losses/ — see Task 4

### trainer/, evaluation/, utils/ — see Tasks 2 and 3

### tests/

| Contract file (Part 14) | Status | Notes |
|---|---|---|
| `test_haar.py` | I | 15 tests |
| `test_blocks.py` | I | 23 tests |
| `test_model.py` | I | 32 tests — also **absorbs** `test_normalization.py` (6 normalizer tests) and part of `test_fusion.py` (2 concat tests) |
| `test_dataset.py` | I | 30 tests |
| `test_attention.py` | **M** | Phase 4.12 |
| `test_fusion.py` | **M** | Phase 4.11 (concat covered inside `test_model.py`) |
| `test_noise.py` | **M** | Phase 4.9 |
| `test_normalization.py` | **M** as a file | Content exists inside `test_model.py`. Cosmetic. |
| `test_losses.py` | **M** | Phase 4.10 |
| `test_export.py` | **M** | Phase 4.15 (per-module TorchScript/ONNX tests do exist in `test_blocks.py`/`test_haar.py`) |
| `test_config.py`, `test_utils.py`, `test_trainer.py` | I (extra) | 10 / 10 / 21 tests. Not in Part 14 but strictly additive. |

**Current suite: 159 tests, 159 passing.**

### scripts/ and root

| File | Status | Notes |
|---|---|---|
| `pack_dataset.py` | I | |
| `baselines.py` | I | bicubic 21.67 dB reproduced (`outputs/baselines.json`) |
| `overfit.py` | I (extra) | Not in Part 14 but required by Part 8 step 6. Extended this session with val/SSIM/trace reporting. |
| `benchmark.py` | **M** | Phase 4.14 |
| `export_onnx.py` | **M** | Phase 4.15 |
| `export_trt.py` | **M** | Phase 4.15 (checklist 23; **not runnable on this CPU-only host**) |
| `train.py` | I **D** | **Latent crash — see Task 2, finding T-1.** |
| `README.md` | **M** | Checklist item 25 |

### Repository hygiene — stale artefacts from the aborted Phase 4

| Path | Size | Verdict |
|---|---|---|
| `checkpoints/best.pt`, `checkpoints/latest.pt` | 316 MB total | **Legacy Restormer**, not SPARC. Verified: 410 tensors, **13.14 M params**, keys `shallow.weight`, `enc1.blocks.0.attn.temperature` (MDTA). The current `Trainer` writes `checkpoints/<run_name>/{best_psnr,best_ema_psnr,last}.pt` instead — these two files can never be produced by it. |
| `outputs/history.jsonl` | 1 record | Same legacy run (val PSNR **2.98 dB**). Actively misleading if read as a SPARC result. |
| `outputs/logs/events.out.tfevents.*` (Aug 3–4) | 12 files | Legacy TensorBoard runs, 11 of them empty (88 bytes). |

Part 14's migration note requires legacy Phase-4 files to be deleted. **Recommend
deleting all three groups** once you confirm nothing else depends on them. I have not
deleted anything — that is your call, and it is outside a read-only audit.

---

## TASK 2 — Trainer audit

`trainer/trainer.py` (435 lines) and `trainer/ema.py` (108 lines) are **substantially
complete**. Verify, do not rewrite.

| Requirement | Status | Evidence / verdict |
|---|---|---|
| **AdamW** | ✅ correct | `betas=(0.9, 0.9)`, `eps=1e-8`, `lr=3e-4`, `wd=1e-4` — all match Part 5. |
| — decay exclusions | ✅ correct | `build_param_groups` excludes `ndim<=1` plus name-matched `norm`/`gamma`/`rel_pos`/`bias`. Contract requires LayerNorm, LayerScale, bias, rel-pos excluded. `LayerScale`'s parameter is literally named `gamma` (shape `(1,C,1,1)`, ndim 4) so the keyword is load-bearing and correct. Two tests cover this. |
| **Cosine scheduler** | ✅ correct | `LambdaLR`, **per-step** (not per-epoch), floor `min_ratio = 1e-6/3e-4`. Reaches exactly 1e-6 at the end. |
| **Warmup** | ✅ correct | Linear from `min_ratio` over `steps_per_epoch × 5` steps → matches "5 epochs, linear from 1e-6". Step 0 yields exactly 1e-6. 4 tests. |
| **EMA** | ✅ correct | decay 0.999, updated **every step**, evaluated separately, params `requires_grad_(False)`, buffers copied not averaged. 5 tests. |
| **AMP** | ✅ correct | fp16 + `GradScaler(init_scale=2**14)`, **auto-disabled off CUDA** — correct, since CPU fp16 autocast is unsupported. Means AMP is **untestable on this host beyond the disabled path**. |
| **Gradient clipping** | ✅ correct | 1.0 global norm, applied **after `scaler.unscale_()` and before `scaler.step()`** — the correct and easy-to-get-wrong ordering. Norm is logged. |
| **Checkpoint saving** | ✅ correct | Every epoch; keeps `best_psnr.pt`, `best_ema_psnr.pt`, `last.pt` — exactly Part 6. |
| **Resume** | ✅ correct | Restores model, EMA, optimizer, scheduler, scaler, and all counters; `epoch+1` on reload so `fit()` resumes on the next epoch. Bit-exactness test passes. |
| **TensorBoard** | ⚠️ partial | Writer present, degrades gracefully if absent (tensorboard **is** installed). **But it logs per-epoch only.** |
| Early stopping | ✅ correct | EMA val PSNR, patience 40, min-delta 0.01 dB. |
| Non-finite loss guard | ✅ present | Skips the batch and logs. |
| channels-last | ✅ correct | Applied to model and 4-D tensors, CUDA-only guard. |

### Required changes (only these — no rewrite)

**T-1 · `train.py` crashes on any CLI override — BLOCKING for Phase 4.8 acceptance.**
`TrainingConfig` is `@dataclass(frozen=True, slots=True)`, so instances have **no
`__dict__`**. Line 124 does `TrainingConfig(**{**train_config.__dict__, **overrides})`.
Verified by direct execution:
`AttributeError: 'TrainingConfig' object has no attribute '__dict__'`.
Every documented invocation that passes `--epochs`, `--batch-size`, `--lr`,
`--num-workers` or `--seed` — including the header's own
`python train.py --variant sparc-base --epochs 400` — dies before training starts.
*Fix:* use `dataclasses.replace(train_config, **overrides)`. One line. Add a test that
constructs the config with an override.
*(I hit the identical `slots=True` class-attribute trap in `scripts/overfit.py` this
session and fixed it there. Worth grepping for other `.__dict__` / class-attribute
access on these configs.)*

**T-2 · Per-step loss-term logging missing.** Part 6: *"Every term must be logged
separately every step."* The trainer accumulates terms and logs **epoch means**. The
plumbing exists (`_compute_loss` already returns a term dict).
*Fix:* in `train_epoch`, write each term to TensorBoard at `global_step`. ~5 lines.
Required before Phase 4.10 is meaningful — the composite loss is exactly what needs
per-term visibility.

**T-3 · `_compute_loss` discards the auxiliary outputs.** It calls `self.model(...)`,
which returns only `.image`. The noise-aux term needs `σ̂`.
*Fix:* call `forward_with_aux()` and pass the `SparcOutput` to the criterion.
**Do not do this now** — it is a Phase 4.10 change and requires the composite loss
signature to exist first. Recorded so it is not forgotten.

**T-4 · (minor) Skipped batch desynchronises the schedule.** On non-finite loss the
loop `continue`s without stepping the scheduler, so the LR schedule shifts relative to
`global_step`. Harmless if it never fires; if it fires often, the schedule silently
stretches. *Fix:* step the scheduler on the skip path, or count skips. Low priority.

**T-5 · (minor) `ModelEma.warmup_steps` is accepted but unused.** `effective_decay()`
computes `1 - 1/(1+steps)` and ignores `warmup_steps` except for the `==0` short-circuit.
The behaviour is the standard timm ramp and is not wrong, but the parameter is
misleading. *Fix:* either use it or drop it from the signature.

**T-6 · (minor) `utils/checkpoint.py` uses bare `torch.load`.** Under torch ≥2.6 the
`weights_only` default flipped to `True`. The current payload is plain
tensors/dicts/scalars so it loads fine (resume tests pass), but a future payload
carrying a config object would break. *Fix:* pass `weights_only=False` explicitly with
a comment, since we only ever load our own checkpoints.

**Verdict: Phase 4.8 is ~85 % complete.** Remaining work is T-1 (one line + test), T-2
(~5 lines), and the Part 14 files that do not exist yet. It does **not** need rewriting.

---

## TASK 3 — Evaluation pipeline audit

| Item | Status | Verdict |
|---|---|---|
| **PSNR** | ✅ complete | Two reductions, deliberately non-interchangeable and documented: `psnr_pooled` (matches Phase 1 baselines) and `psnr_per_image`. `data_range=1.0`. Computed in float64. |
| **SSIM** | ✅ complete | 11×11 Gaussian, σ=1.5, K=(0.01,0.03), valid region only. Matches Part 6. |
| **LPIPS** | ⚠️ code ready, **dependency absent** | `LpipsMetric` (AlexNet, grayscale→3ch, eval-only, lazily constructed, `available` property). **`import lpips` fails — package not installed.** Not in `requirements.txt` as an installed dep. Needed for the Part 12 promotion rule (condition 3) and checklist 20. **Not blocking** Phases 4.8–4.15. |
| **Mean + median reporting** | ✅ complete | `MetricAccumulator` reports mean, median and pooled — the median matters because Phase 1 found 34 near-featureless images that inflate the mean. |
| **Validation loop** | ✅ complete | `Trainer.evaluate()`, every epoch, run separately for live and EMA weights. |
| **Best-checkpoint logic** | ✅ complete | `best_psnr` on live val PSNR, `best_ema_psnr` on EMA val PSNR, `last` every epoch. Early stopping keys on EMA. |
| **`evaluation/evaluate.py`** | ❌ **missing** | Required by Part 14. Standalone evaluation + submission assembly (checklist 24: correct test-ID order over the 400 test images). |
| **Image saving** | ❌ **missing** | Nothing anywhere writes a PNG/NPY prediction. No qualitative-inspection path. |
| **σ-stratified / texture-stratified breakdown** | ❌ **missing** | Explicitly required by Part 6 and checklist 20. Phase 1 data needed to define the strata exists in `reports/report_deg.json`. |

**Verdict: metrics are complete and correct; the evaluation *pipeline* is not.**
Three artefacts missing: `evaluate.py`, image saving, stratified reporting.

---

## TASK 4 — Loss system audit

| Loss | Contract weight | Status | Tests |
|---|---|---|---|
| Charbonnier | 1.00 | ✅ implemented | No dedicated `test_losses.py`; exercised indirectly by `test_trainer.py` and `scripts/overfit.py`. Shape-mismatch and reduction paths are guarded but untested. |
| MS-SSIM | 0.15 | ❌ missing | — |
| Wavelet | 0.10 | ❌ missing | — |
| FFT | 0.05 | ❌ missing | — |
| Gradient | 0.05 | ❌ missing | — |
| Noise aux | 0.02 | ❌ missing | — |
| **Composite** | — | ❌ missing | — |

**What already exists to build on (no new design needed):**
- `LossConfig` holds **every** weight and hyperparameter already — band weights
  `(0.25, 1.0, 1.0, 1.5)`, `wavelet_levels=2`, `ms_ssim_scales=5`, window 11, σ 1.5.
  Verified against the contract by `test_config.py::test_loss_weights_match_contract`.
- `HaarDWT` is available and tested → the wavelet loss is a composition, not new maths.
- `evaluation.metrics.ssim` already implements the single-scale SSIM the MS-SSIM
  pyramid needs (note: the metric is eval-only; the **loss** needs its own
  differentiable path — the metric implementation is already differentiable).
- `ReconstructionHead.predict_subbands()` exposes `[LL,LH,HL,HH]` directly.
- `datasets/degradation.py` already contains `fit_noise_parameters` and
  `analytic_sigma_map` **with passing tests** — the noise-aux target is done.

**No loss was implemented this session, as instructed.**

---

## TASK 5 — Remaining phases

| Phase | Item | Status | Est. effort | Dependencies | Files | Blocking issues |
|---|---|---|---|---|---|---|
| **4.7** | Overfit gate | **OPEN — running** | ~30 min remaining | — | `scripts/overfit.py`, `outputs/overfit_gate_*` | Gate is trending to ~33–35 dB vs a 45 dB target. See Task 7. |
| **4.8** | Training framework | **~85 % complete** | 0.5–1 day | 4.7 closed | `trainer/trainer.py` (T-2), `train.py` (**T-1, blocking**), `tests/test_trainer.py` | T-1 crashes every documented `train.py` invocation with an override. |
| **4.9** | Noise Head | Not started | 1–1.5 days | 4.8 | `models/noise/{__init__,noise_head,noise_map}.py`, `tests/test_noise.py` | Directory has no `__init__.py`. σ̂ correlation > 0.9 gate needs the analytic target — already implemented and tested. |
| **4.10** | Composite loss | Not started | 2–3 days | 4.9 (for noise-aux) | `losses/{ms_ssim,wavelet,fft,gradient,noise_aux,composite}.py`, `tests/test_losses.py` | Requires T-3 (trainer must pass `SparcOutput`). 6 losses × (shape + stability + gradient) tests before combining. |
| **4.11** | Gated fusion | Not started | 0.5 day | 4.8 | `models/fusion/gated_fuse.py`, `tests/test_fusion.py` | Lowest-risk phase: `build_fusion` dispatch already exists; A3 control arm already implemented. |
| **4.12** | Global sparse attention | Not started | 2–3 days | 4.11 | `models/attention/{__init__,rel_pos,gsa_block}.py`, `tests/test_attention.py` | Highest-risk phase. Needs SDPA↔explicit parity to 1e-4, rel-pos index buffer, exact param match. Empty directory. |
| **4.13** | Full integration | Not started | 1 day | 4.9–4.12 | `models/sparc_net.py` (already wired), `tests/test_model.py` | **Params 2.346 M ±2 % and MACs 2.449 G ±5 % are checkable on CPU. VRAM is NOT** — no CUDA device. |
| **4.14** | Profiling | Not started | 0.5 day | 4.13 | `scripts/benchmark.py` | **Partially blocked:** params/MACs/latency measurable on CPU; **peak GPU memory, images/sec and the Part 10 VRAM budget are not measurable on this host.** Needs a CUDA machine or an explicit contract waiver. |
| **4.15** | Export | Not started | 1–1.5 days | 4.13 | `scripts/export_onnx.py`, `tests/test_export.py` | ONNX/TorchScript are CPU-testable. **TensorRT (checklist 23) is not** — no CUDA. Per-module export tests already pass for NAF and Haar. |
| — | Docs | Not started | 0.5 day | 4.15 | `README.md` | Checklist item 25. |
| — | Eval pipeline | Not started | 1 day | 4.10 | `evaluation/evaluate.py` | Task 3 gaps: submission assembly, image saving, stratified reporting. |

**Critical-path note:** 4.10 depends on 4.9 only for the noise-aux term. The other five
losses can be built and unit-tested in parallel with 4.9 if you want to compress the
schedule. I have **not** assumed that reordering — the instruction is sequential.

---

## TASK 6 — Implementation plans (no code)

### Phase 4.8 — Complete the training framework

- **Modify:** `train.py` (T-1: `dataclasses.replace`), `trainer/trainer.py` (T-2:
  per-step term logging; optionally T-4, T-5), `utils/checkpoint.py` (T-6),
  `tests/test_trainer.py` (+2 tests).
- **Create:** none.
- **Functions:** no new classes. `Trainer.train_epoch` gains a per-step
  `writer.add_scalar(f"step/{term}", value, global_step)` block.
- **Shapes:** unchanged.
- **Acceptance:** 1 full epoch NaN-free; AMP stable (**CUDA-only — record as
  unverifiable here**); checkpoint resume bit-exact (already passing).
- **Unit tests:** `test_train_py_accepts_cli_overrides`,
  `test_per_step_terms_are_logged`.

### Phase 4.9 — Noise Head

- **Create:** `models/noise/__init__.py`, `noise_head.py`, `noise_map.py`,
  `tests/test_noise.py`.
- **Classes:** `NoiseHead(NoiseHeadConfig)`, plus σ-map assembly in `noise_map.py`
  (the analytic target already lives in `datasets/degradation.py`).
- **Shapes:** `(B,1,128,128)` → trunk `(16,24,32,32)` at strides 2/4/8/16 →
  GAP → MLP → `(B,2)` → σ-map `(B,1,128,128)`.
- **Init:** final layer weight `= 0`, bias `= (-3.718, -1.718)` so softplus gives
  exactly `(0.024, 0.165)`. **Assert this numerically in a test** — it is the one
  init in the contract with an exact required output.
- **Range:** clamp σ̂ to `[1e-4, 2.0]`.
- **Acceptance:** params **exactly 42,050** (Part 3 stage 1); MACs 52.3 M ±5 %;
  σ̂ correlation > 0.9 vs analytic σ.
- **Unit tests:** shape (B∈{1,2,8}), gradient reaches every parameter, stability over
  100 batches incl. ×1e-3/×1e3, exact param count, softplus-at-init equals
  `(0.024, 0.165)`, clamp bounds, TorchScript, ONNX, memory.

### Phase 4.10 — Composite loss

- **Create:** `losses/{ms_ssim,wavelet,fft,gradient,noise_aux,composite}.py`,
  `tests/test_losses.py`.
- **Modify:** `losses/__init__.py`; `trainer/trainer.py` (T-3 — pass `SparcOutput`).
- **Classes:** `MSSSIMLoss`, `WaveletLoss`, `FFTLoss`, `GradientLoss`, `NoiseAuxLoss`,
  `CompositeLoss` returning `(scalar, {term: value})` — the tuple form
  `Trainer._compute_loss` **already handles**.
- **Shapes:** all take `(B,1,256,256)` pred/target and return a scalar; `NoiseAuxLoss`
  takes σ̂ `(B,1,128,128)` and the analytic σ target of the same shape.
- **Per-loss detail:** MS-SSIM 5 scales/window 11/σ 1.5 → `1 - MS_SSIM`;
  Wavelet 2-level Haar, L1 per band, weights `(0.25, 1.0, 1.0, 1.5)`;
  FFT `mean(abs(|rfft2(x̂)| - |rfft2(x)|))`, amplitude only;
  Gradient L1 on Sobel-x and Sobel-y; Noise-aux L1 on `log σ̂`.
- **Critical ordering constraint (Part 6):** all terms are computed **after
  de-normalisation and clamping**, against unmodified GT.
- **Acceptance:** each loss passes shape + numerical-stability + gradient-propagation
  **before** combination (your explicit instruction, and Part 9's rule); composite
  weights match `LossConfig` exactly; every term logged separately.
- **Unit tests per loss:** shape B∈{1,2,8}; finite over 100 random batches incl.
  ×1e-3/×1e3; non-zero finite grads; identical inputs → loss ≈ 0 (MS-SSIM, wavelet,
  FFT, gradient); TorchScript. Plus `test_composite_equals_manual_weighted_sum`.
  MS-SSIM needs a **minimum-size guard** — 5 scales of an 11×11 window needs ≥161 px;
  at 256² that is fine, but the test must not use a small dummy tensor.

### Phase 4.11 — Gated fusion

- **Create:** `models/fusion/gated_fuse.py`, `tests/test_fusion.py`.
- **Class:** `GatedFuse(channels, reduction=4)` — `build_fusion` already constructs it
  with exactly this signature.
- **Shapes:** `(skip, dec)` each `(B,C,H,W)` → `(B,C,H,W)`; gate is `(B,C,1,1)`.
- **Acceptance:** params **exactly 23,256 at C=96** and **5,868 at C=48** (Part 3
  stages 15 and 20); gate ∈ (0,1); output is a convex combination.
- **Benchmark vs concat (your Task-4.11 requirement, = ablation A3):** identical seed,
  schedule and split; report mean **and** median PSNR/SSIM, latency, memory.
- **Unit tests:** shape, gradient, stability, exact params, gate range, convexity
  (output between the inputs), TorchScript, ONNX, memory.

### Phase 4.12 — Global sparse attention

> Contract terminology note: Part 2.6 specifies **exact, unrestricted** global
> self-attention — *"No windows, no sparsity, no approximation."* "Sparse" in the phase
> name refers to it being instantiated at only 3 of the 5 stages, never to sparsifying
> the attention matrix. **Implement exactly as Part 2.6 specifies.**

- **Create:** `models/attention/__init__.py`, `rel_pos.py`, `gsa_block.py`,
  `tests/test_attention.py`.
- **Classes:** `RelativePositionBias(heads, n)` (table `(heads,(2n-1)²)`,
  `trunc_normal_(std=0.02)`, non-trainable int64 `(N,N)` index **buffer**);
  `GSABlock(channels, heads, spatial_size, ...)` — signature already fixed by
  `encoder.build_gsa_blocks`.
- **Shapes:** `(B,C,H,W)` → qkv `(B,3d,H,W)` → `(B,heads,HW,16)` → bias
  `(heads,N,N)` → out `(B,C,H,W)`. head_dim **must** be 16.
- **Instantiation:** Enc L1 (C=96,h=3,d=48), Enc L2 (C=160,h=5,d=80),
  Dec D1 (C=96,h=3,d=48). Never at 64² or 128².
- **Acceptance:** params exactly **124,902 / 420,735 / 62,451** (Part 3 stages 8/12/16);
  **SDPA path == explicit matmul path to 1e-4**; SDPA mandatory in training.
- **Unit tests:** shape, gradient, stability, exact params, MACs ±5 %,
  SDPA↔explicit parity, rel-pos index symmetry, TorchScript, ONNX (explicit path),
  memory, latency.

### Phase 4.13 — Integration

- **Modify:** `models/sparc_net.py` (already wired — expect flag flips only),
  `tests/test_model.py`.
- **Acceptance:** full tensor flow per Part 4; **params 2,345,650 ±2 %**;
  **MACs 2.449 G ±5 %**; gradients reach every parameter; constant input → constant
  output. **VRAM < 2.0 GB: NOT MEASURABLE — no CUDA device.**
- **Unit tests:** `test_base_model_matches_contract_parameter_total`,
  `test_base_model_macs_within_budget`, full-model gradient and stability at B∈{1,2,8}.

### Phase 4.14 — Profiling

- **Create:** `scripts/benchmark.py`.
- **Measurable here:** params, MACs/FLOPs, CPU latency, CPU throughput, model size on
  disk (budget 12 MB fp32 — 2.346 M params ≈ 9.38 MB, will pass).
- **Not measurable here:** peak GPU memory, GPU images/sec, GPU latency budgets
  (35 ms batch-1, 10 ms/image batch-16), the 16 h/400-epoch training budget.
- **Recommendation:** emit a report that marks GPU rows `UNMEASURED (no CUDA)` rather
  than silently reporting CPU numbers against GPU budgets.

### Phase 4.15 — Export

- **Create:** `scripts/export_onnx.py`, `tests/test_export.py`.
- **Acceptance:** TorchScript matches eager to 1e-5; ONNX opset 17, onnxruntime
  matches eager to 1e-3; GSA exports via the **explicit** path.
- **Blocked:** `export_trt.py` / checklist 23 (TensorRT parity 1e-2) — no CUDA.
- **Unit tests:** round-trip parity for the full model at B∈{1,8}; constant-input
  invariance preserved after export.

---

## ADDENDUM — Verification pass (2026-08-06, executable re-check)

Every Task-1/2 finding was re-verified by execution, not by re-reading.

| Finding | Method | Result |
|---|---|---|
| T-1 `train.py` CLI override crash | executed the exact expression | **CONFIRMED** `AttributeError: 'TrainingConfig' object has no attribute '__dict__'`; `dataclasses.replace()` verified as the fix |
| Decay exclusions correct | ran `build_param_groups` on the real model | **CONFIRMED** zero `gamma`/`norm`/`bias` params leaked into the decayed group (51 decayed / 93 not, 144 total) |
| Warmup starts at exactly 1e-6 | evaluated the lambda at step 0 | **CONFIRMED** 1.000e-06; end-of-schedule also exactly 1.000e-06 |
| `models/attention`, `models/noise` empty | `ls -a` | **CONFIRMED** no `__init__.py` |
| `lpips` absent | import | **CONFIRMED** `ModuleNotFoundError` |
| Legacy checkpoints | loaded and inspected | **CONFIRMED** 13.14 M params, Restormer MDTA keys |

### New pre-implementation findings

**V-1 · GatedFuse and GSA parameter budgets are exactly reproducible.** Derived
analytically from the Part 2.6/2.7 structures and compared with Part 3:

| Module | Computed | Part 3 | |
|---|---|---|---|
| GatedFuse C=96 | 23,256 | 23,256 | **MATCH** |
| GatedFuse C=48 | 5,868 | 5,868 | **MATCH** |
| GSABlock C=96, h=3, n=32 | 62,451 | 62,451 (×2 = 124,902) | **MATCH** |
| GSABlock C=160, h=5, n=16 | 140,245 | 140,245 (×3 = 420,735) | **MATCH** |

Both head_dims are exactly 16. **Phases 4.11 and 4.12 are structurally de-risked** —
the exact composition that hits the budget is now known before a line is written.

**V-2 · NoiseHead structure is under-specified by the contract; two variants hit
42,050.** Part 2.9 fixes the trunk widths and the output behaviour but not the
LayerNorm placement or the MLP hidden width. Exhaustive search over plausible
structures found exactly two that hit the target:

1. LN after SimpleGate in **all four** stages, MLP `32 → 64 → 2` with the **first
   Linear bias-free** (trunk 39,872 + MLP 2,178).
2. LN in the **first three** stages only, MLP `32 → 64 → 2` fully biased
   (trunk 39,808 + MLP 2,242).

**Resolution:** adopt variant 1. Part 3 stage 1 writes the trunk as
`4×[Conv3×3 s2+LN+SG]` — literally four stages each containing an LN — which is
textual evidence for LN in all four. Both agree on hidden width 64. The final
`Linear(64→2)` keeps its bias, as Part 2.9 requires (`bias = (-3.718, -1.718)`).
This will be documented in the module docstring.

**V-3 · Part 3's NoiseHead MAC figure is over-counted 4× (documentation erratum).**
The stage-1 cell reads 52.32 MMAC. Reconciliation:

| Accounting | Value |
|---|---|
| Strided convs counted at **output** resolution (what `FlopCounterMode` measures) | 12.98 M |
| Strided convs counted at **input** resolution (4× over-count) | 51.90 M |
| σ-map smoother, 5×5 depthwise at 128² | 0.41 M |
| **input-res + smoother** | **52.31 M** ← contract says 52.32 M |
| **output-res + smoother (what we will actually measure)** | **13.39 M** |

The 0.01 M residual is rounding. The contract's figure counts each strided
convolution at its input resolution. **Consequence:** the Part 9 per-module MAC test
(±5 % of Part 3) will *fail* for the NoiseHead at ~13.4 M vs 52.32 M, through no fault
of the implementation. The top-level budget is unaffected (2,449 → ~2,410 MMAC, still
inside ±5 %). Same class as the NAFBlock `7C²+8C` erratum: **the code will be right and
the table cell wrong.** Recommend a Part-16 documentation amendment; assert the
measured value in the test and reference this finding.

---

## TASK 7 — Overfit gate report

**Completed 2026-08-06 00:43:54. Result: FAIL.**
Artefacts: `outputs/overfit_gate.log`, `outputs/overfit_gate_trace.jsonl` (81 records),
`outputs/overfit_gate_report.json`, `reports/figures/overfit_gate.png`.

### 1–6. Final measurements

| Metric | Train (8 memorised) | Held-out (8 unseen) |
|---|---|---|
| **Loss** (Charbonnier) | **0.010649** | **0.100144** |
| **PSNR** | **33.164 dB** | **17.720 dB** |
| **SSIM** | **0.9344** | **0.2251** |

Run: SPARC-Tiny, 250,452 params, 8 images, 2000/2000 steps, lr 3e-3 cosine,
AdamW β=(0.9,0.9), grad-clip 1.0, CPU, 2046.5 s (1.023 s/step).
Best PSNR 33.164 dB at step 1999.

### 7–8. Convergence

![overfit gate](figures/overfit_gate.png)

| Step | LR | Train PSNR | Δ dB / 100 steps | Held-out PSNR | Train loss |
|---|---|---|---|---|---|
| 100 | 2.982e-03 | 28.68 | — | 21.78 | 0.020030 |
| 250 | 2.886e-03 | 30.28 | +1.070 | 20.54 | 0.015833 |
| 500 | 2.561e-03 | 31.53 | +0.499 | 19.65 | 0.013548 |
| 750 | 2.074e-03 | 32.21 | +0.270 | 19.15 | 0.012393 |
| 1000 | 1.500e-03 | 32.63 | +0.171 | 18.73 | 0.011645 |
| 1250 | 9.260e-04 | 32.89 | +0.103 | 18.33 | 0.011177 |
| 1500 | 4.393e-04 | 33.04 | +0.060 | 18.03 | 0.010877 |
| 1750 | 1.142e-04 | 33.13 | +0.034 | 17.81 | 0.010712 |
| 2000 | 0.000e+00 | 33.16 | +0.015 | 17.72 | 0.010649 |

Training was monotone and stable throughout: no NaN, no loss spike, no divergence.
Held-out PSNR peaked at **21.78 dB at step 100** and then declined monotonically to
17.72 dB — textbook memorisation, and the correct behaviour for this test.

### 9. Gate result

**FAIL — 33.16 dB against a 45.0 dB threshold. Shortfall 11.84 dB.**

### 10. Root-cause analysis

**Classification: CAPACITY LIMIT** (single cause).

**Ruled out — implementation bug.** The same code, optimiser, data path and loss
reached **45.07 dB on 1 image** and **43.17 dB on 2 images**
(`outputs/diag_overfit.log`). A defect in the data path, gradient flow, reconstruction
head or optimiser configuration could not produce 45 dB on any input count. The
backbone can demonstrably represent and reach the target.

**Ruled out — data pipeline.** Identical loader and identical packed arrays produced
the passing 1-image result. Train SSIM 0.9344 confirms the model fits the supplied
pairs; held-out collapse confirms the pairs are genuinely distinct images, not
duplicates.

**Ruled out — loss design.** Charbonnier alone reached 45.07 dB on 1 image. A loss
that can drive one image to the threshold is not what blocks eight.

**Ruled out — learning rate.** 3e-3 is already 10× the contract's training LR.
Decisively: **the convergence rate halves between steps 100–250 and 250–500
(+1.070 → +0.499 dB/100) while the LR is essentially unchanged at near-peak
(2.982e-03 → 2.561e-03, a 14 % change).** The deceleration is therefore intrinsic, not
schedule-induced. The diagnostic run with `layer_scale_init=1.0` was *worse*
(42.66 vs 45.07 dB), so this is not a residual-scale initialisation pathology either.

**Ruled out — optimisation budget.** 5× more steps (400 → 2000) bought only 2.85 dB.
Fitting train PSNR against log-steps over the region where the LR is near peak gives
**3.43 dB/decade**; reaching 45 dB needs **≈4.5 × 10⁶ steps — 2250× the contract
budget**. That fit is *optimistic*: panel 4 shows the measured curve bending below the
log-linear trend, so the true requirement is larger and the asymptote may lie below
45 dB entirely.

**Established — capacity.** Two controlled comparisons, identical optimiser, LR,
schedule and step count:

| Control | Model | Params | Images | params/output px | PSNR @ 400 steps |
|---|---|---|---|---|---|
| Vary data | SPARC-Tiny | 0.250 M | 1 | 3.82 | **45.07 dB** |
| | SPARC-Tiny | 0.250 M | 2 | 1.91 | 43.17 dB |
| | SPARC-Tiny | 0.250 M | 8 | 0.48 | 30.31 dB |
| Vary model | SPARC-Tiny | 0.250 M | 8 | 0.48 | 30.31 dB |
| | SPARC-Base | 1.688 M | 8 | 3.22 | **36.34 dB** |

Holding data and steps fixed and increasing parameters 6.75× yields **+6.03 dB**.
Holding the model fixed and increasing images 1→8 costs **−14.76 dB**. Both point the
same way: the binding constraint is representational capacity per output pixel, not
optimisation. At 0.48 params per output pixel SPARC-Tiny cannot memorise 8 images to
45 dB.

### Secondary finding — the gate threshold is mis-specified

`scripts/overfit.py` records the gate's purpose (Contract Part 8, step 6): prove *the
data path is correct, gradients reach every parameter, the reconstruction head can
represent full-resolution detail, and the optimiser configuration is sane.*

**All four propositions are established** by the 1-image (45.07 dB) and 2-image
(43.17 dB) results. What the 8-image run adds is not a correctness signal but a
capacity measurement — and it is asking a deliberately-undersized 250 k debug model to
encode 8 × 256² outputs at 0.48 params/pixel.

The gate's *diagnostic intent* is satisfied; its specific numeric combination
(Tiny × 8 images × 45 dB) is not achievable. **This requires a Part 16 amendment and
must be reviewed before any contract change is made.** Options, in order of fidelity
to intent:

| Option | Change | Evidence | Cost |
|---|---|---|---|
| **A** | Run the gate on **SPARC-Base**, 8 images | 36.34 dB at only 400 steps, still climbing, 3.22 params/px | High on CPU (~18 s/step); cheap on the A400 |
| **B** | Keep Tiny, **1–2 images** | 45.07 / 43.17 dB already measured at 400 steps | ~10 min CPU to confirm at 2000 steps |
| **C** | Keep Tiny × 8 images, **lower the threshold** | 33.16 dB measured, vs bicubic 21.67 dB | Free — but weakens the gate |

### Resolution — Amendment A-002 approved, gate re-run, **PASSED**

Option B was approved on review (2026-08-06) and recorded as **A-002** in
`AMENDMENTS.md`. The gate was re-run at 2 images:

| | Train (2 memorised) | Held-out (2 unseen) |
|---|---|---|
| Loss | 0.003900 | 0.024609 |
| PSNR | 44.930 dB | 29.842 dB |
| SSIM | 0.9886 | 0.7628 |

**Target reached at step 517 of 2000 — best 45.01 dB. GATE PASSED**, inside a quarter
of the step budget. 176.5 s, 0.341 s/step. Plot: `reports/figures/overfit_regate.png`.

**Methodological cross-check.** The same log-step extrapolation used to diagnose the
8-image failure predicts **497 steps** to reach 45 dB on this run; the measured value
was **517** — a 4 % error. That the method is accurate where it can be checked
materially strengthens the 4.5 × 10⁶-step estimate it produced for the 8-image case.

**Phase 4.7 is CLOSED.** Contract Part 15 checklist item 10 is satisfied under A-002.
