# Phase 4.10.1 — Numerical stability investigation and AMP hardening

**Status:** investigation complete, fixes landed, GPU shakedown (Task 9) pending — it
requires the training box and could not be run here.

**Scope guard:** no architecture change, no contract hyperparameter change. Parameter
count, MACs and FLOPs verified identical (§11).

---

## 1. Root cause analysis

The Phase 4.10 divergence has **two independent defects**. The first produces the NaN;
the second is why a three-epoch run reported success after learning nothing for two of
them.

### 1.1 Primary — fp16 activation overflow in the NAF spatial branch

`NAFBlock` grows activation magnitude to roughly the **fourth power** of its first
convolution's output before anything reduces it:

```
t = conv1(norm1(x))          # magnitude m
t = dwconv(t)                # magnitude ~m
t = SimpleGate(t)            # t1 * t2          -> m^2
t = t * sca(t.mean(2,3))     # multiplied again -> m^4 (SCA has no sigmoid)
t = conv3(t)                 # <- consumes the overflowed tensor
```

`SimpleGate` is a multiply, and Simplified Channel Attention is a second multiply with
no squashing nonlinearity between them. fp16 saturates at **65504**. Measured on the
real model with real packed data, one NAF block at the point of failure:

| stage | dtype | absmax |
| --- | --- | --- |
| block input | fp16 | 8.72 |
| `norm1` | fp32 | 10.97 |
| `conv1` | fp16 | 27.89 |
| `dwconv` | fp16 | 92.81 |
| `gate1` (SimpleGate) | fp16 | **3 738** |
| `sca` | fp16 | 147.9 |
| `t * sca` | fp16 | **inf** |
| `conv3` | fp16 | **NaN** |
| *same block, fp32* | fp32 | 2.486e+05 — perfectly finite |

`conv3` receives `inf` and its mixed-sign weights compute `inf - inf`, which is NaN.
The fp32 reference peaks at 2.49e5, only **3.8x** above the fp16 ceiling — so this is a
dynamic-range failure, not a mathematical one. Nothing is wrong with the arithmetic;
fp16 simply cannot hold the number.

**Why the output is NaN rather than clipped.** `SPARCNet.forward_with_aux` clamps to
`[0,1]`, and `clamp(inf) == 1.0` would have produced a *finite* loss. Only a NaN
survives the clamp. That is consistent with the observed log and rules out a merely
saturating output.

### 1.2 Why it never recovered — the skip path is not a recovery path

`train_epoch` skipped a non-finite batch and continued. That guard **never touches the
weights**. So:

- If the model has diverged, the next batch fails identically. Forever.
- The forward pass is stateless (no BatchNorm, no running buffers — confirmed), so
  nothing self-heals.
- Epoch 0 finished at 23.47 dB; every batch from step 423 to the end was skipped. The
  run performed roughly **one** epoch of work, then burned two more doing nothing, and
  exited **0**.

Step 423 is step 63 of epoch 1. Warmup was 1 epoch = 360 steps, so the failure landed
about 60 steps after the LR first reached its peak of 3e-4 — the classic post-warmup
divergence window.

### 1.3 Contributing — moments computed in fp16

`LayerNorm2d` is hand-rolled from `mean`/`var`/`rsqrt` to avoid two permutes.
Measured autocast policy (§4): CUDA promotes `layer_norm` to fp32, but `mean` and `var`
are on **no policy list at all** and follow their input dtype. The hand-rolled layer
therefore forfeited a protection the built-in has.

A variance is a sum of squares, so it saturates fp16 once the channel standard
deviation reaches ~256. The resulting failure is *silent*: `rsqrt(inf) == 0`, so the
layer returns zeros and deletes the signal without raising.

8 of 50 `LayerNorm2d` sites were receiving fp16 input:

```
noise_head.stages.{0,1,2,3}.norm
encoder.levels.{0,1,2}.naf.0.norm1
head.blocks.0.norm1
```

This was **not** the first thing to fail in the traced run, but it is a real hazard on
the same failure path and is fixed.

### 1.4 Contributing — `.float()` in the losses was not honoured

`losses/ms_ssim.py:178` carries the comment *"float32 throughout"*. Measured: its
internal `F.conv2d` ran in **fp16**. Autocast re-casts convolution arguments to the
reduced dtype regardless of what the caller passes, so an explicit `.float()` on the
input is decorative. Same for `GradientLoss` (no `.float()` at all) and for
`models/noise/noise_map.py`, whose docstring promised float32 for the squared terms
that were landing around 1e-6 — **subnormal** in fp16.

---

## 2. Evidence

Every claim above is reproducible with the scripts added in this phase.

### 2.1 The mechanism, before and after the fixes

`python -m scripts.trace_divergence --dtype {fp16,bf16} --weight-scale 1 2 3 4`

Weights are scaled to walk a freshly initialised model toward the failure; real packed
LR data; peak activation is the largest absmax over all 460 traced module outputs.

| weight scale | fp16 peak act | fp16 verdict | bf16 peak act | bf16 verdict |
| --- | --- | --- | --- | --- |
| 1.0 | 21.1 | finite | 20.8 | finite |
| 2.0 | inf | **NaN at `encoder.levels.0.naf.0.conv3`** | 6.45e+04 | finite |
| 3.0 | 1.09e+04 | **NaN** at the same module | 2.16e+08 | finite |
| 4.0 | 6.12e+04 | **NaN** at the same module | 9.23e+10 | finite |

bf16 stays finite carrying activations **1.4 million times** the fp16 ceiling. This is
the decisive evidence: the failure is dynamic range, and bf16 removes it.

The failure localises to the *same module* at every scale, which is what a structural
range problem looks like — as opposed to a data-dependent one, which would move around.

### 2.2 Which tensor goes first

Answered by `utils.numerics.ModuleTracer`, now also wired into the trainer's divergence
dump. In every fp16 trace the first non-finite module is
`encoder.levels.0.naf.0.conv3`, a `Conv2d` executing in fp16, whose **input** is the
already-infinite SimpleGate×SCA product.

---

## 3. Before/after activation statistics

Freshly initialised `sparc-base --no-attention`, fp16 autocast, real data, batch 2.

| metric | before | after |
| --- | --- | --- |
| peak activation at init | 21.09 | 21.09 |
| fp16 headroom at init | 3 105x | 3 105x |
| `LayerNorm2d` sites with fp16 moments | **8** | **0** |
| `LayerNorm2d` output at channel std 500 | **all zeros** (signal deleted) | correctly normalised, unit variance |
| LayerNorm error vs float64 reference | 3.5e-03 | **4.9e-07** (7 000x better) |
| loss terms executing in fp16 | 2 of 5 (`ms_ssim`, `gradient`) | **0 of 5** |
| first non-finite module, fp16, scale 2.0 | `naf.0.conv3` | `naf.0.conv3` — **unchanged, see below** |

**The LayerNorm fix does not by itself save fp16.** That is the honest result and it is
why §7 matters: the binding constraint is the SimpleGate×SCA product, which is
architectural and must not be changed under this phase's mandate. bf16 is the fix for
it.

---

## 4. AMP audit report

Full machine-generated tables: `reports/AMP_AUDIT_fp16_cpu.md`,
`reports/amp_audit_fp16_cpu.json`. Regenerate on the GPU box with:

```bash
python -m scripts.amp_audit --device cuda --dtype fp16 \
    --json reports/amp_audit_fp16_cuda.json --markdown reports/AMP_AUDIT_fp16_cuda.md
```

> **Autocast policy is per device type.** The CPU and CUDA lists differ — `layer_norm`
> is promoted on CUDA but not on CPU. The authoritative CUDA lists are embedded in
> `scripts/amp_audit.py` (`CUDA_FP32_POLICY`, `CUDA_REDUCED_POLICY`), extracted from
> `torch.testing._internal.autocast_test_lists`.

### 4.1 Operations, CUDA policy

| class | operations | consequence |
| --- | --- | --- |
| **Demoted to fp16** | `conv2d`, `conv_transpose2d`, `linear`, `matmul`, `mm`, `bmm`, `einsum`, `addmm`, `prelu`, SDPA | **an explicit `.float()` on the input does not survive** |
| **Promoted to fp32** | `layer_norm`, `group_norm`, `softmax`, `log_softmax`, `softplus`, `log`, `pow`, `prod`, `sum`, `rsqrt`, `reciprocal`, `norm`, `l1_loss`, `mse_loss` | safe whatever you pass |
| **No policy — follows input** | `mean`, `var`, elementwise `mul`, `avg_pool2d`, `interpolate`, `sqrt` | **a hand-rolled LayerNorm is built entirely from these** |

`rsqrt` being promoted is worth noting: it does not help, because it is promoted only
*after* the fp16 `var` that feeds it has already overflowed to `inf`.

### 4.2 Modules still running in fp16 — status after the fixes

| site | before | after |
| --- | --- | --- |
| `LayerNorm2d` moments (8 sites) | fp16 | **fp32** |
| `MSSSIMLoss` internal `conv2d` | fp16 | **fp32** (island) |
| `GradientLoss` Sobel `conv2d` | fp16 | **fp32** (island) |
| `NoiseAuxLoss` `forward_operator` | fp16 | **fp32** (island) |
| `noise_map` squared terms | fp16 (subnormal) | **fp32** |
| `WaveletLoss`, `FFTLoss` | already fp32 | fp32 |
| NAF `conv1`/`dwconv`/`conv3`, all trunk convolutions | fp16 | fp16 — **intended**, this is the AMP speedup |

---

## 5. LayerNorm numerical analysis

`python -m scripts.layernorm_analysis` → `reports/layernorm_analysis.json`.

fp16 max = 65504, so an elementwise square overflows at |x| > 255.9. Channel variance
(a *mean* of squares over C=48) overflows once the channel standard deviation reaches
~256, which for Gaussian data is an absmax around 900:

| input absmax | var fp16 | var fp32 | overflowed | output all zeros |
| --- | --- | --- | --- | --- |
| 292 | 6 696 | 6 695 | no | no |
| 678.9 | 5.25e+04 | 5.25e+04 | no | no |
| 883.6 | 6.27e+04 | 6.27e+04 | no | no |
| **1 056** | **inf** | 1.21e+05 | **yes** | no |
| 2 010 | inf | 4.09e+05 | yes | **yes — signal deleted** |
| 16 950 | inf | 2.60e+07 | yes | **yes** |

Accuracy against a float64 reference, in the regime where fp16 works at all:

| input magnitude | fp32 moments | fp16 moments |
| --- | --- | --- |
| 1 | 4.93e-07 | 3.54e-03 |
| 10 | 5.00e-07 | 3.20e-03 |
| 100 | 4.39e-07 | 4.03e-03 |

**The maths is unchanged.** The fix casts to float32, computes the same three
expressions, and returns float32 — which is what the layer already returned in practice,
because the fp32 affine parameters promoted every affine site anyway. `x - mean` is also
done in fp32, which matters: an fp16 input already at the ceiling gives `inf - inf` and
NaNs before the variance is ever consulted.

`autocast(enabled=False)` was **not** used here — it does not survive `torch.jit.script`,
and Contract Part 9 requires `NAFBlock` and `NoiseHead` to script. Plain casts do.
Verified: both still script and match eager to 1e-5.

---

## 6. Loss stability report

`python -m scripts.loss_stability` → `reports/loss_stability_cpu.json`.

5 terms x 5 adversarial input cases x 3 dtypes, forward and backward, plus TorchScript,
determinism and optional `torch.compile`.

**Result: 75/75 configurations finite in both forward and backward. All 5 terms are
deterministic and scriptable with eager agreement.** The losses were never the problem.

Adversarial cases covered — `identical` (MS-SSIM contrast at its clamp, FFT magnitude at
the `sqrt` singularity), `constant` (zero variance everywhere, zero Sobel response),
`saturated` (0 vs 1, reachable because the model clamps), `near_identical` (fp16
subtraction cancellation), `random`.

The one real finding is precision, not finiteness — `near_identical`, MS-SSIM:

| dtype | value |
| --- | --- |
| fp32 | 8.94e-07 |
| fp16 | 1.22e-03 |
| bf16 | 5.86e-03 |

A ~1000x error on small residuals, i.e. exactly the regime a converging restoration
model lives in. This is what the fp32 island in §7.2 fixes, and it is a correctness
argument for the island independent of the stability one.

---

## 7. Changes made

### 7.1 `LayerNorm2d` moments in float32 — `models/blocks/layer_norm.py`

Cast to fp32, compute mean/var/rsqrt/affine there, return fp32. Same expressions, same
epsilon, same parameters. Still scriptable.

### 7.2 float32 island around the objective — `losses/composite_loss.py`

`CompositeLoss.forward` now casts its inputs and evaluates every term inside
`autocast(enabled=False)` (`utils.numerics.fp32_island`). The `.float()` calls already
in the term bodies become true instead of decorative. `CompositeLoss` is not scripted,
so the context manager is safe here.

`models/noise/noise_map.py` re-casts the *result* of the box filter, putting the squared
terms in fp32 without an autocast context (scriptability).

### 7.3 Trainer robustness — `trainer/trainer.py`

- `DivergenceError`, raised once `consecutive_nonfinite` exceeds
  `max_consecutive_nonfinite` (default **25**).
- On abort: writes `divergence.pt` (model, EMA, optimiser, scheduler, scaler, state) and
  `divergence_report.json` — epoch, global step, batch index, per-term loss values,
  input/target/prediction ranges, parameter absmax and non-finite parameter names, the
  **first non-finite module in the forward pass**, and the 20 largest activations.
- `train.py` exits **2** on divergence instead of 0.
- A successful step resets the counter, so an isolated bad batch still just skips.
- GradScaler telemetry: `optim/amp_scale`, `optim/amp_overflow`,
  `optim/amp_overflow_total`, `grad/skipped_batches_total`; `overflow_steps` in the
  epoch metrics.
- EMA is no longer updated on a step the scaler skipped — previously the EMA absorbed a
  step the live weights never took.
- Per-step numerics trace to `numerics.jsonl` under `--debug-numerics`.

### 7.4 Config — `configs/sparc_config.py`

Five new **operational** fields, none a contract hyperparameter: `amp_dtype`,
`max_consecutive_nonfinite`, `detect_anomaly`, `debug_numerics`,
`debug_numerics_from_step`. All validated.

---

## 8. BF16 support summary

`--amp-dtype {fp16,bf16}`, **default `fp16` — unchanged**.

- bf16 selects `autocast(dtype=torch.bfloat16)` and **disables the GradScaler**: bf16
  has fp32's exponent range, so there is nothing to rescale.
- Fails fast with a clear message on pre-Ampere hardware rather than silently emulating.
- Training logic is otherwise identical — same schedule, same clipping, same EMA.

Per §2.1, this is the fix for the primary root cause. The SimpleGate×SCA fourth-power
growth is architectural and out of scope to change; bf16 gives it the exponent range it
needs.

---

## 9. Debug mode

`--detect-anomaly` runs each step inside `torch.autograd.detect_anomaly(check_nan=True)`
via `utils.numerics.detect_anomaly`, a conditional wrapper so `train_epoch` stays
single-pathed. **Default disabled.** Roughly halves throughput.

`--debug-numerics` / `--debug-numerics-from-step N` write per-step ranges for input,
target, prediction, sigma map and both per-image sigmas, plus grad norm, AMP scale,
overflow flag, parameter absmax, non-finite parameter count, and — under fp16 — the
**fp16 headroom** of every traced tensor. Off by default; the file is not even created.

---

## 10. GPU shakedown (Task 9) — NOT RUN

**This could not be run here.** This machine has `torch 2.10.0+cpu` and no CUDA; the
Phase 4.10 run was on a different box. Task 9 is the one deliverable still outstanding.

### 10.1 What was verified here instead

`python -m scripts.cpu_smoke --images 64 --epochs 3` — the full `Trainer.fit` path
(composite loss, noise head, EMA, validation, checkpointing, divergence guard, numerics
trace) over real packed data. Log: `reports/cpu_smoke_run.log`.

| epoch | train loss | val PSNR | val SSIM | EMA PSNR |
| --- | --- | --- | --- | --- |
| 0 | 0.14519 | 20.294 dB | 0.3890 | 19.303 dB |
| 1 | 0.11757 | 21.668 dB | 0.4557 | 20.616 dB |
| 2 | 0.10665 | 21.986 dB | 0.4698 | 21.150 dB |

`skipped=0 overflow=0 consecutive_nonfinite=0`, completed in 270 s. Loss decreasing,
PSNR increasing, no batch dropped.

**This proves the loop is intact; it proves nothing about fp16**, because AMP is
CUDA-only and this run had `AMP=False`. The absolute PSNR is low only because it trains
on 64 images.

### 10.2 What still has to run on the GPU box

Run these in order:

```bash
# 1. Regenerate the audit against real CUDA policy (2 minutes).
python -m scripts.amp_audit --device cuda --dtype fp16 \
    --json reports/amp_audit_fp16_cuda.json --markdown reports/AMP_AUDIT_fp16_cuda.md
python -m scripts.layernorm_analysis --device cuda --json reports/layernorm_cuda.json
python -m scripts.loss_stability --device cuda --json reports/loss_stability_cuda.json

# 2. Confirm the mechanism on the real device (2 minutes).
python -m scripts.trace_divergence --device cuda --dtype fp16 --weight-scale 1 2 3 4
python -m scripts.trace_divergence --device cuda --dtype bf16 --weight-scale 1 2 3 4

# 3. The shakedown that is expected to pass.
python train.py --variant sparc-base --no-attention --epochs 3 --warmup-epochs 1 \
    --amp-dtype bf16 --run-name shakedown_bf16

# 4. The fp16 control. May still diverge — that is the predicted result, and it now
#    aborts loudly at ~25 consecutive failures with a full diagnostic dump instead of
#    burning two silent epochs.
python train.py --variant sparc-base --no-attention --epochs 3 --warmup-epochs 1 \
    --amp-dtype fp16 --run-name shakedown_fp16
```

**Prediction, recorded before the fact:** bf16 completes clean. fp16 may still diverge —
the LayerNorm and loss fixes are necessary and correct but do not address the
SimpleGate×SCA range problem, and §2.1 shows fp16 breaking at a weight scale where bf16
is comfortable. If fp16 does diverge, `divergence_report.json` will name
`encoder.levels.N.naf.M.conv3` as the first non-finite module, and that confirms the
diagnosis rather than contradicting it.

If both diverge, the next lever — a separate decision, needing a contract amendment —
is an fp32 island around the SimpleGate→SCA→conv3 product, or a `--lr` reduction.

Acceptance criteria for step 3: zero skipped batches, zero GradScaler overflows,
monotonically decreasing training loss, validation PSNR above the 23.47 dB that epoch 0
reached before, and exit code 0.

---

## 11. Architecture invariance — verified

| variant | parameters | trainable | GMAC | GFLOP |
| --- | --- | --- | --- | --- |
| `sparc-base --no-attention` | **1 737 562** (1.7376 M) | 1 737 562 | **1.8190** | **3.638** |
| `sparc-tiny` | 250 452 (0.2505 M) | 250 452 | 0.2189 | 0.438 |

Identical to the figures in the Phase 4.10 shakedown log (`1.7376 M params | 1.8190
GMAC | 3.638 GFLOP`). `state_dict` has 460 entries; shape signature sha256 `21a1574a…`.

No module was added, removed or resized. No contract hyperparameter changed: learning
rate 3e-4, betas (0.9, 0.9), weight decay 1e-4, grad clip 1.0, EMA 0.999, batch size 8
all untouched.

---

## 12. Files modified

**Changed (6)**

| file | change |
| --- | --- |
| `models/blocks/layer_norm.py` | float32 moments |
| `models/noise/noise_map.py` | float32 squared terms |
| `losses/composite_loss.py` | float32 island around all terms |
| `trainer/trainer.py` | `DivergenceError`, divergence dump, scaler telemetry, bf16, anomaly mode, numerics trace, EMA-on-overflow fix |
| `configs/sparc_config.py` | 5 operational fields + validation |
| `train.py` | `numerical stability` CLI group, exit code 2 on divergence |

**Added (7)**

| file | purpose |
| --- | --- |
| `utils/numerics.py` | `TensorStats`, `ModuleTracer`, `first_nonfinite`, `fp32_island`, `detect_anomaly`, `fp16_headroom` |
| `scripts/amp_audit.py` | Task 3 |
| `scripts/layernorm_analysis.py` | Task 2 |
| `scripts/loss_stability.py` | Task 4 |
| `scripts/trace_divergence.py` | Task 1 |
| `scripts/cpu_smoke.py` | end-to-end loop check on real data without a GPU |
| `tests/test_numerics.py` | 21 tests |

---

## 13. Tests added

**`tests/test_numerics.py` — 21 tests.** `TensorStats` range/health reporting;
fp32-computed statistics so a near-ceiling fp16 tensor is not misreported; mean over
finite elements only; `fp16_headroom`; `ModuleTracer` leaf coverage, hook detachment and
non-finite-first ordering; `first_nonfinite` attribution; **`fp32_island` defeating
autocast demotion** (the measurement behind the loss fix); `detect_anomaly` no-op when
disabled; LayerNorm fp32 moments under autocast, survival in the overflow regime with
unit output variance, agreement with a float64 reference, and continued scriptability;
composite loss fp32 under autocast, autocast/fp32 agreement, finite gradients; and a
model-level assertion that initialisation has >10x fp16 headroom.

**`tests/test_trainer.py` — 13 added.** Divergence aborts past threshold; report
contains every diagnostic key; below-threshold still skips without aborting; a
successful step resets the counter; bf16 disables the scaler; fp16 remains the default;
invalid `amp_dtype` and `max_consecutive_nonfinite` rejected; `overflow_steps` reported;
numerics trace written, start-step honoured, off by default; `detect_anomaly` off by
default and runnable.

**Full suite: 370 passed, 2 skipped, no regressions.**
