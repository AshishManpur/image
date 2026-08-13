# Phase 4.8–4.15 — Unit-Test Specifications and Implementation Plans

**Written:** 2026-08-06, during the Phase 4.7 gate run.
**Status:** specification only. **No functionality implemented.**

Governed by Contract Part 9 (unit-test contract) and Part 3 (parameter table).
Parameter targets marked **VERIFIED** were reproduced analytically during the Phase 4.7
verification pass — see `PHASE4_7_AUDIT.md` addendum V-1.

Universal requirements applied to **every** module below (Part 9):

| Test | Requirement |
|---|---|
| Shape | Output shape matches the contract for `B ∈ {1, 2, 8}` |
| Gradient | `loss.backward()` gives non-zero, finite grads for **every** parameter |
| Stability | No NaN/Inf over 100 random batches, including inputs scaled ×1e-3 and ×1e3 |
| Parameter count | Matches Part 3 **exactly** — not a tolerance |
| MACs | Within ±5 % of Part 3, via `FlopCounterMode` |
| TorchScript | `torch.jit.script` succeeds, matches eager to 1e-5 |
| ONNX | opset 17 export succeeds, `onnxruntime` matches eager to 1e-3 |

**GPU-readiness requirements** (target: RTX A400, 4 GB). Every module must:
run under `torch.amp.autocast(dtype=float16)` without overflow; accept
`memory_format=torch.channels_last`; survive `torch.compile` (checked, not enabled in
V1 per Part 5); export cleanly; contain **no CPU-specific branches**.

---

## Phase 4.8 — Trainer completion

**Files modified:** `train.py`, `trainer/trainer.py`, `utils/checkpoint.py`,
`tests/test_trainer.py`.
**Files created:** none (`trainer/tb_layout.py` already prepared).
**No new classes.**

### Changes

| ID | Change | Contract clause |
|---|---|---|
| T-1 | `dataclasses.replace()` instead of `**config.__dict__` | — (crash bug) |
| T-2 | Per-step logging of every loss term | Part 6: *"Every term must be logged separately every step"* |
| T-4 | Step the scheduler on the non-finite-loss skip path | Part 5 (schedule integrity) |
| T-5 | Use or remove `ModelEma.warmup_steps` | — (dead parameter) |
| T-6 | Explicit `weights_only=False` in `torch.load` | — (forward compatibility) |

### Test specification

| Test | Expected behaviour | Failure case caught |
|---|---|---|
| `test_train_py_accepts_cli_overrides` | Building `TrainingConfig` with `epochs=5` returns a config with `epochs == 5` and all other fields unchanged | The `slots=True` `AttributeError` (T-1) |
| `test_per_step_terms_are_logged` | After one epoch of `n` steps, each of the 7 tags in `LOSS_TERMS` has `n` recorded points at `global_step` resolution | Silent regression to epoch-only logging |
| `test_scheduler_advances_on_skipped_batch` | With a criterion returning NaN once, `global_step` and the scheduler step count stay consistent | Schedule drift (T-4) |
| `test_tb_layout_groups_are_guarded` | `should_log_group("memory", cuda=False)` is `False`; unknown group raises `ValueError` | Zero-valued memory plots on CPU |

**Acceptance (Part 8 step 7):** 1 full epoch, no NaN, AMP stable, checkpoint resume
bit-exact. *AMP stability is **CUDA-only** — record as unverifiable on this host and
re-run on the A400.*
**Regression guard:** the existing 21 trainer tests must still pass unchanged.

---

## Phase 4.9 — Noise Head

**Files created:** `models/noise/noise_head.py`, `models/noise/noise_map.py`,
`tests/test_noise.py`. (`models/noise/__init__.py` already created.)
**Files modified:** `trainer/trainer.py` (T-3, deferred to 4.10).

### Structure (resolved in verification finding V-2)

```
(B,1,128,128)
  4 × [ Conv3×3 stride 2 (cin → 2w) → LayerNorm2d(2w) → SimpleGate → w ]
      w = 16, 24, 32, 32   at 64², 32², 16², 8²
  → GlobalAvgPool → (B,32)
  → Linear(32 → 64, bias=False) → Linear(64 → 2, bias=True)
  → softplus → (σ_g, σ_s)
  → σ-map assembly → (B,1,128,128), clamped to [1e-4, 2.0]
```

**Parameter target: exactly 42,050** (Part 3 stage 1).
Trunk 39,872 + MLP 2,178. **VERIFIED** by exhaustive search: only two structures hit
42,050 and both use hidden width 64; LN-in-all-four-stages is selected on Part 3's
literal `4×[Conv3×3 s2+LN+SG]` notation.

**MAC expectation: ~13.39 M measured, NOT the 52.32 M in Part 3.**
Erratum V-3 — Part 3 counts strided convolutions at input resolution (4× over-count):
51.90 M (input-res) + 0.41 M (smoother) = 52.31 M. **Do not contort the module to hit
the wrong number.** The test asserts the measured value and cites V-3.

### Test specification

| Test | Expected behaviour | Failure case caught |
|---|---|---|
| `test_noise_head_output_shape` | `(B,1,128,128)` → σ-map `(B,1,128,128)` for `B ∈ {1,2,8}` | Stride/resolution error |
| `test_noise_head_parameter_count_is_exact` | **exactly 42,050** | Any structural drift |
| `test_softplus_at_init_gives_contract_sigmas` | With weight `= 0` and bias `= (-3.718, -1.718)`, softplus outputs `(0.024, 0.165)` to 1e-4 | Wrong bias, wrong activation, wrong ordering of the two channels |
| `test_sigma_is_clamped_to_contract_range` | σ̂ ∈ `[1e-4, 2.0]` for inputs scaled ×1e-3 and ×1e3 | Missing clamp; overflow under fp16 |
| `test_sigma_correlates_with_analytic_target` | Pearson **r > 0.9** against `analytic_sigma_map` on synthetic pairs | The Part 8 step-10 acceptance gate |
| `test_noise_head_gradients_reach_every_parameter` | All grads finite and non-zero | Detached branch; dead zero-init layer never receiving gradient |
| `test_noise_head_no_batchnorm` | No `BatchNorm*` anywhere | Part 7 forbids BatchNorm |
| `test_noise_head_autocast_fp16` | Finite output under `autocast(float16)` | fp16 overflow in the variance path |
| `test_noise_head_torchscript` / `_onnx` | Parity 1e-5 / 1e-3 | Export-hostile ops |

**Note on the zero-initialised final layer:** its weight is `0` by contract, so the
gradient test must assert the weight *receives* a non-zero gradient, not that it *is*
non-zero. This is the single most likely false-failure in the suite.

---

## Phase 4.10 — Composite loss

**Files created:** `losses/{ms_ssim,wavelet,fft,gradient,noise_aux,composite}.py`,
`tests/test_losses.py`.
**Files modified:** `losses/__init__.py`, `trainer/trainer.py` (T-3).

| Loss | Weight | Definition | Reuses |
|---|---|---|---|
| Charbonnier | 1.00 | `mean(sqrt((x̂-x)² + 1e-6))` | implemented |
| MS-SSIM | 0.15 | `1 - MS_SSIM`, 5 scales, window 11, σ 1.5 | `evaluation.metrics.ssim` |
| Wavelet | 0.10 | 2-level Haar, L1 per band, weights `(0.25, 1.0, 1.0, 1.5)` | `HaarDWT` |
| FFT | 0.05 | `mean(abs(\|rfft2(x̂)\| - \|rfft2(x)\|))`, amplitude only | — |
| Gradient | 0.05 | L1 on Sobel-x and Sobel-y | — |
| Noise aux | 0.02 | L1 on `log σ̂` vs analytic σ | `analytic_sigma_map` |

**Ordering constraint (Part 6):** every term is computed **after de-normalisation and
clamping**, against unmodified GT.
**`CompositeLoss` returns `(scalar, {term: value})`** — the tuple form
`Trainer._compute_loss` already handles.

### Test specification — per loss, before any combination

| Test | Expected behaviour | Failure case caught |
|---|---|---|
| `test_<loss>_shape` | Scalar output for `B ∈ {1,2,8}`, input `(B,1,256,256)` | Missing reduction |
| `test_<loss>_is_zero_for_identical_inputs` | `loss(x, x) ≈ 0` (MS-SSIM, wavelet, FFT, gradient) | Sign error; wrong normalisation |
| `test_<loss>_is_finite_over_100_batches` | No NaN/Inf, including ×1e-3 and ×1e3 | `sqrt(0)` and `log(0)` gradients |
| `test_<loss>_gradients_propagate` | Non-zero finite grad w.r.t. the prediction | Accidental `detach()`; `torch.no_grad` leakage |
| `test_<loss>_autocast_fp16` | Finite under fp16 autocast | FFT/MS-SSIM fp16 overflow — **the highest-risk item**; both may need an explicit `float32` cast internally |
| `test_composite_equals_manual_weighted_sum` | Composite total == Σ weight × term to 1e-6 | Weight transcription errors |
| `test_composite_terms_match_loss_config` | Weights read from `LossConfig`, not literals | Config/implementation divergence |

**MS-SSIM size guard:** 5 scales of an 11×11 window needs ≥161 px. At 256² this is
fine, but tests must not use small dummy tensors — they would fail spuriously.

**Gradient/Sobel note:** implement as a fixed non-trainable `register_buffer` kernel,
not an `nn.Conv2d`, so no parameters leak into the model's count.

---

## Phase 4.11 — Gated fusion

**Files created:** `models/fusion/gated_fuse.py`, `tests/test_fusion.py`.
`build_fusion` already dispatches on `use_gated_fusion` — **no wiring needed**.

```
u = Conv1x1(2C → C)(concat([skip, dec]))
g = sigmoid(Conv1x1(C//4 → C)(Conv1x1(C → C//4)(GAP(u))))
return g * skip + (1 - g) * dec
```

**Parameter targets: exactly 23,256 at C=96 and 5,868 at C=48** — **VERIFIED** (V-1).

| Test | Expected behaviour | Failure case caught |
|---|---|---|
| `test_gated_fuse_shape` | Two `(B,C,H,W)` → one `(B,C,H,W)` | — |
| `test_gated_fuse_parameter_count_is_exact` | 23,256 / 5,868 | Wrong reduction ratio |
| `test_gate_is_in_unit_range` | `g ∈ (0,1)` for extreme inputs | Missing sigmoid |
| `test_output_is_convex_combination` | Output between `min(skip,dec)` and `max(skip,dec)` elementwise | Broadcast error |
| `test_gated_fuse_reduces_to_skip_when_gate_saturates` | Forcing `g → 1` returns `skip` exactly | Operand order swapped — silently trains but inverts the semantics |
| Universal set | shape/grad/stab/params/TS/ONNX/memory | |

**Benchmark vs concatenation (= ablation A3):** identical seed, schedule and split;
report mean **and** median PSNR/SSIM, latency, peak memory.

---

## Phase 4.12 — Global self-attention (highest risk)

**Files created:** `models/attention/rel_pos.py`, `models/attention/gsa_block.py`,
`tests/test_attention.py`. (`__init__.py` already created.)
**Signature is already fixed** by `models.encoder.build_gsa_blocks` — match it exactly.

> **Terminology:** Part 2.6 requires attention that is **exact and unrestricted** —
> *"No windows, no sparsity, no approximation."* "Sparse" refers to instantiating the
> block at only 3 of 5 stages, never to sparsifying the attention matrix.

**Parameter targets — VERIFIED (V-1):**

| Instance | Per block | Part 3 |
|---|---|---|
| Enc L1, C=96, h=3, d=48, n=32 | 62,451 | ×2 = 124,902 |
| Enc L2, C=160, h=5, d=80, n=16 | 140,245 | ×3 = 420,735 |
| Dec D1, C=96, h=3, d=48, n=32 | 62,451 | ×1 = 62,451 |

Composition per block: `2C` + `Conv1x1(C→3d)` + `dw3x3(3d)` + `Conv1x1(d→C)` + `C` +
`2C` + `Conv1x1(C→2C)` + `dw3x3(2C)` + `Conv1x1(C→C)` + `C` + `heads·(2n-1)²`.
Rel-pos table sizes: `3·63² = 11,907` and `5·31² = 4,805`. **head_dim = 16 in both.**

| Test | Expected behaviour | Failure case caught |
|---|---|---|
| `test_gsa_shape` | `(B,C,H,W)` → `(B,C,H,W)`, `B ∈ {1,2,8}` | Reshape error in the head split |
| `test_gsa_parameter_count_is_exact` | 62,451 / 140,245 | Any structural drift |
| `test_head_dim_is_sixteen` | `d // heads == 16` at every instantiation | Part 5 invariant |
| `test_sdpa_matches_explicit_path` | **max abs diff ≤ 1e-4** | The single most important test in the phase |
| `test_rel_pos_index_is_symmetric_and_in_range` | Index buffer ∈ `[0, (2n-1)²)`; distance-0 maps to the centre | Off-by-one in the index construction |
| `test_rel_pos_table_is_trainable_index_is_not` | Table has `requires_grad`; index is an int64 **buffer** | Index accidentally a parameter — would corrupt the count and the decay groups |
| `test_gsa_is_permutation_equivariant_without_bias` | With bias zeroed, shuffling tokens shuffles outputs identically | Attention applied along the wrong axis |
| `test_gsa_autocast_fp16` | Finite under fp16 | Softmax overflow without scaling |
| `test_gsa_onnx_uses_explicit_path` | Export succeeds with `use_sdpa=False`, parity 1e-3 | SDPA not exportable at opset 17 |
| `test_gsa_never_instantiated_at_64_or_128` | Config-level assertion | Part 2.6 prohibition; a 64² block costs 16× the 32² matmul |

**Memory note:** Part 2.6 records SDPA as mandatory in training — the naive path costs
**+20.8 MB/image**. On a 4 GB A400 at batch 8 that is ~166 MB, material but not fatal;
the explicit path is for export only.

---

## Phase 4.13 — Integration

**Acceptance:** params **2,345,650 ±2 %**; MACs **2.449 G ±5 %**; full Part 4 tensor
flow; gradients reach every parameter; constant input → constant output.
**Expected measured MAC shortfall:** ~39 M below the table because of erratum V-3
(noise head). Result ≈ 2,410 M — still inside ±5 %. **Anticipated, not a defect.**
**VRAM < 2.0 GB: NOT MEASURABLE on this host.**

## Phase 4.14 — Profiling

`scripts/benchmark.py` **prepared and CLI-verified**. Measures params, MACs, GFLOPs,
disk size, latency, throughput, and both VRAM figures; emits `UNMEASURED (no CUDA)`
for GPU-denominated budgets rather than substituting CPU numbers.

## Phase 4.15 — Export

`scripts/export_onnx.py` and `scripts/export_trt.py` **prepared and CLI-verified**.
TorchScript 1e-5, ONNX 1e-3 (opset 17), TensorRT 1e-2. Export forces `use_sdpa=False`
so attention takes the explicit path, and deep-copies the model so exporting cannot
mutate training state. TensorRT fails fast with a clear message on a CPU-only host.
