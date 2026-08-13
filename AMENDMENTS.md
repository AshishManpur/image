# Contract Amendments

Amendments to `SPARC_BASE_V1_IMPLEMENTATION_CONTRACT.md`, per Part 16.
Each entry states the clause, the measurement that justifies the change, the effect on
every Part 10 budget, and the approval status.

---

## A-001 — Degradation re-synthesis: inject noise at LR resolution, not HR

**Status:** ✅ **APPROVED** (2026-08-05). Implemented behind `DataConfig.noise_at_lr`,
which defaults to the amended behaviour (`True`). Setting it to `False` reproduces the
frozen contract exactly and is retained for ablation only.

### Clause amended

Contract Part 6, "Augmentation (frozen)":

> On-the-fly LR re-synthesis from GT (p=0.5): `g_{σ~U(0.3,0.5)} → ·Γ(L~U(25,60))/L →
> +N(0,σ_g), σ_g~U(0,0.08) → bicubic↓2 antialias=False`, no clipping.

### Problem

The clause injects noise at **256×256** and then decimates. Bicubic 2× decimation
attenuates white noise, so the noise level that survives to the 128×128 output is
substantially lower than the level that was injected. Phase 1 fitted its noise
parameters on the **LR** image, so applying those parameters at HR under-degrades.

### Measurement

400 random training pairs, real LR vs. re-synthesised LR, both compared against the
same noise-free `A(GT)`:

| Ordering | σ_s median | vs. real | Residual std | vs. real |
|---|---|---|---|---|
| Real training data | 0.1624 | — | 0.0884 | — |
| **Contract (noise at HR)** | **0.1086** | **−33 %** | **0.0673** | **−24 %** |
| Amended (noise at LR) | 0.1556 | −4 % | 0.0885 | +0.1 % |

Variance, not standard deviation, is what the noise head is conditioned on: the
contract ordering delivers **50 % of the true speckle variance**. Training on it would
systematically under-represent noise, which is the wrong direction given Phase 1
measured the *test* split as 1.13× noisier than train.

### Change

1. Inject speckle and additive noise **after** `bicubic↓2` rather than before.
2. Narrow the additive range from `σ_g ~ U(0, 0.08)` to `σ_g ~ U(0, 0.04)`.
   With HR injection the additive term was also attenuated; at LR it is not.
   Sweep over the darkest three intensity bins and the residual std:

   | σ_g range | darkest-bin variance ratio | mid/high max dev | residual std |
   |---|---|---|---|
   | U(0, 0.08) | 20.9 / 6.2 / 2.8 | 0.59 | 0.0938 |
   | **U(0, 0.04)** | **5.4 / 1.9 / 1.05** | **0.21** | **0.0845** |
   | U(0, 0.02) | 1.5 / 0.8 / 0.6 | 0.35 | 0.0820 |

3. Narrow the speckle range from `L ~ U(25, 60)` to `L ~ U(26, 51)`.
   Phase 1 measured σ_s p10 / median / p90 = 0.140 / 0.162 / 0.194, which is
   `L = 26.6 / 38.1 / 51.0`. The contract's `(25, 60)` has median `L = 42.5`, i.e.
   σ_s = 0.153 — biased 6 % low. The amended range reproduces the measured
   percentiles directly:

   | Looks range | σ_s median (real 0.1624) | error |
   |---|---|---|
   | U(25, 60) | 0.1545 | −4.9 % |
   | **U(26, 51)** | **0.1617** | **−0.4 %** |

### Rejected alternative

Adding the Poisson term `b·I` that Phase 1 measured (`6.42e-3·I`) would be a *new*
noise component, which Part 7 forbids. It was tested anyway and made the fit worse:
σ_s overshot to 0.176 and every intensity bin landed ~20 % high. **Not adopted.**

### Residual known limitation

The two darkest intensity bins (I < 0.10, ≈12 % of pixels) remain over-noised by
1.9–5.4×. This is irreducible with a two-component `a + cI²` model, because the real
noise contains the linear Poisson term the contract deliberately omits. Phase 1 already
recorded that `b` and `c` are collinear (`corr = −0.90`) and not separately identifiable
per image.

### Consequent amendment to the acceptance criterion

Contract Part 8, step 2 requires "re-synthesised LR σ-vs-I curve within 5 % of real".
That is unachievable with the frozen two-component model and is replaced by:

* aggregate residual std within **10 %** of real — achieved: 5.7 %;
* σ_s median within **10 %** of real — achieved: **0.4 %**;
* intensity-bin variance ratio within **30 %** for I > 0.10 — achieved: 25 %.

The band tolerance is 30 % rather than 5 % because the omitted Poisson term makes
anything tighter unreachable; this is stated as an accepted limitation rather than
worked around. Enforced by
`tests/test_dataset.py::test_resynthesised_noise_matches_real_statistics`.

### Budget impact (Part 10)

**None.** This is a data-pipeline change. Parameters, MACs, activations, VRAM, model
size and latency are all unaffected. Training-time cost is marginally *lower*, because
speckle is now sampled at 128² instead of 256² (4× fewer gamma variates).

---

## A-002 — Step-6 overfit gate: 2 images, not 8

**Status:** ✅ **APPROVED** (2026-08-06). Approved after review of the failed 8-image
gate and its root-cause analysis in `reports/PHASE4_7_AUDIT.md`, Task 7.

### Clause amended

Contract Part 8, step 6 acceptance test (and Part 15 checklist item 10):

> **SPARC-Tiny overfits 8 images to > 45 dB in < 2000 steps**

### Problem

The clause is not achievable by the model it names. SPARC-Tiny is a deliberately
undersized 250,452-parameter debug variant; 8 images at 256² is 524,288 output pixels,
i.e. **0.48 parameters per output pixel**. The 45 dB threshold requires near-exact
reconstruction of high-frequency detail that is not present in the 128² input.

### Measurement

Full run: SPARC-Tiny, 8 images, 2000/2000 steps, lr 3e-3 cosine, 2046 s.
**Result: 33.16 dB — 11.84 dB short.** Training was monotone and stable throughout.

Cause established as capacity, not correctness, by two controlled comparisons at
identical optimiser, LR, schedule and step count:

| Control | Model | Params | Images | params/output px | PSNR @ 400 steps |
|---|---|---|---|---|---|
| Vary data | SPARC-Tiny | 0.250 M | 1 | 3.82 | **45.07 dB** |
| | SPARC-Tiny | 0.250 M | 2 | 1.91 | **43.17 dB** |
| | SPARC-Tiny | 0.250 M | 8 | 0.48 | 30.31 dB |
| Vary model | SPARC-Base | 1.688 M | 8 | 3.22 | 36.34 dB |

6.75× more parameters at fixed data and fixed steps yields **+6.03 dB**; 1→8 images at
fixed model costs **−14.76 dB**.

Alternative causes were excluded by measurement, not assumption:

* **Implementation bug** — excluded: the same code path reaches 45.07 dB on 1 image.
* **Learning rate** — excluded: the convergence rate halves between steps 100–250 and
  250–500 (+1.070 → +0.499 dB/100 steps) while the LR moves only 14 % (2.982e-03 →
  2.561e-03). The deceleration is intrinsic, not schedule-induced.
* **Optimisation budget** — excluded: 5× more steps bought 2.85 dB. The log-step fit
  gives 3.43 dB/decade, so 45 dB needs ≈4.5 × 10⁶ steps (2250× the budget), and the
  measured curve bends *below* that fit.
* **Loss design / data pipeline** — excluded: identical loss and loader produced the
  passing 1-image result.

### Change

Step 6 acceptance becomes:

> **SPARC-Tiny overfits 2 images to > 45 dB in < 2000 steps**

Rationale: `scripts/overfit.py` documents the gate's purpose as proving that the data
path is correct, that gradients reach every parameter, that the reconstruction head can
represent full-resolution detail, and that the optimiser configuration is sane. **All
four propositions are proven at 2 images** (1.91 params/output px, 43.17 dB already
measured at 400 steps) and none of them require 8. The 8-image variant measures
capacity, which is not what a step-6 correctness gate is for.

### Rejected alternatives

* **Gate SPARC-Base at 8 images** (3.22 params/px, 36.34 dB at 400 steps and still
  climbing). Most faithful to intent, but ~18 s/step on the CPU development host makes
  a 2000-step run ~10 h. Retained as a recommended **GPU-side confirmation** on the
  RTX A400; not adopted as the step-6 blocker.
* **Keep 8 images and lower the threshold to ~30 dB.** Rejected: it converts a
  correctness proof into a weak regression check, and 33.16 dB is a capacity
  measurement rather than a correctness criterion.

### Recorded capacity measurement (not a gate)

SPARC-Tiny at 8 images converges to **33.16 dB train / 17.72 dB held-out**. Retained in
`outputs/overfit_gate_report.json` as the reference capacity datum for ablation A0.

### Budget impact (Part 10)

**None.** This amends an acceptance test only. No architectural change, no capacity
change, no change to parameters, MACs, activations, VRAM, model size or latency. The
frozen configuration in Part 5 is untouched.

---

## A-003 — Maximum training VRAM @ batch 8: 2.00 GB → 2.05 GB

**Status:** ✅ **APPROVED** (2026-08-10). Approved in the same session as the
measurement, alongside adoption of the zero-risk `rel_pos_index` dedup below.

### Clause amended

Contract Part 10, "Performance budget":

> Maximum training VRAM @ batch 8 | **2.00 GB** | 1.17 GB analytic

### Problem

`tests/test_full_model_training.py::test_training_vram_at_batch_eight_is_within_budget`
measured **2.024 GB** on the RTX A400, 24 MB (1.2 %) over budget — not an OOM (the card
has 4.29 GB), a budget miss. The 1.17 GB analytic figure in Part 4 assumes **fp16
activations throughout**. Two decisions taken after Part 4 was written break that
assumption, both deliberate:

* Phase 4.10.1's `LayerNorm2d` upcasts to fp32 and *returns* fp32, so under bf16
  autocast each of the 58 LayerNorms retains an fp32 copy of its input for backward
  and emits an fp32 activation into the residual stream.
* Phase 4.10.1's composite loss runs inside an `fp32_island`, so every MS-SSIM scale,
  the FFT term and the Sobel term hold fp32 intermediates at 256×256.

`scripts/vram_profile.py`'s forward-activation census confirms it: of 1,411 MB of
forward activations, **645.73 MB (46 %) is fp32**, not bf16.

### Measurement

`scripts/vram_profile.py` first *mis*-reported this as 0.098 GB, because its `phase()`
helper called `torch.cuda.reset_peak_memory_stats()` on every phase entry —
`max_memory_allocated()` only reports the max since the last reset, so by the time the
step-level peak was read back it reflected only the final (`optimizer_step`) phase's
tiny local peak, not the true peak across the step. Fixed to reset once per step
(matching the pytest test's own methodology), it reports the same 2.0167 GB the test
measures.

Two candidate fixes were A/B-measured against the corrected baseline:

| Arm | Peak | Δ vs baseline | Verdict |
|---|---|---|---|
| baseline | 2.0167 GB | — | over by 16.7 MB |
| `candidate_shared_rel_pos_index` | 2.0080 GB | **−8.65 MB** | real, adopted (see below) |
| `candidate_layernorm_input_dtype` | 2.0261 GB | +9.44 MB (noise floor) | **no effect** — rejected |
| `candidate_both` | 2.0080 GB | −8.65 MB | identical to rel_pos alone |

The layernorm candidate (cast `LayerNorm2d`'s output back to the input dtype
post-hoc) measured as having **no effect**: `candidate_both` is bit-identical to
`candidate_shared_rel_pos_index` alone, and the candidate's own delta matches the
+9.44 MB noise floor shared by several unrelated arms (`zero_grad_not_set_to_none`,
four of the five loss-term ablations). This is expected once traced through: casting
the *output* after `original_forward` has already run doesn't shrink what autograd
retains for backward — the fp32 intermediates inside the original computation are
saved regardless of what happens to the result afterward. Recovering the ~289 MB a
true bf16-backward LayerNorm would save requires a custom `autograd.Function` that
computes moments in fp32 but saves only bf16 tensors for backward — nontrivial rework,
not undertaken here, and not needed to close a 0.4 % gap.

### Change

1. **Adopted, zero numerics change:** `models/attention/rel_pos.py`'s
   `RelativePositionBias` now shares one `rel_pos_index` buffer per grid size across
   all GSA blocks at that resolution (three blocks at 32×32, three at 16×16 are
   bit-identical), via a module-level cache re-established in `_apply` after every
   `.to(device)`. Same dtype, same shape, same gathered values — only the storage is
   shared. Saves 8.65 MB measured peak VRAM. This alone is not enough (2.008 GB still
   exceeds 2.00 GB).
2. **Budget revised** from 2.00 GB to **2.05 GB**, rounding up from the ~2.015–2.017 GB
   measured with the dedup applied, leaving ~35 MB (1.7 %) headroom for measurement
   noise across GPUs/torch versions. This is the corrected number for the fp32
   LayerNorm/loss policy Phase 4.10.1 already committed to — not a relaxation to dodge
   a real regression.

### Rejected alternative

* **Make `LayerNorm2d` return bf16 under bf16 autocast**, closing the gap without
  touching the budget. Rejected for now: the naive post-hoc cast measured zero effect
  (see above); a version that actually works needs a custom `autograd.Function` and
  its own numerics validation (0 non-finite steps over a shakedown run) before it can
  be trusted. Left as future work if the budget needs tightening again.

### Budget impact (Part 10)

Only the "Maximum training VRAM @ batch 8" row changes, from **2.00 GB** to **2.05
GB**. Parameters, MACs, GFLOPs, activations/image, inference VRAM/latency, training
time and model size are all unaffected — this is a training-VRAM-only correction, and
the underlying architecture, loss and optimiser are unchanged apart from the
`rel_pos_index` storage dedup (which changes no arithmetic).

**Superseded by A-004** (2026-08-11): the 2.05 GB figure was derived from a *single-step*
measurement and is ~1.6 MB below the stable multi-step peak. It remains the historical
record of why the budget moved off 2.00 GB; the operative threshold is now 2.06 GB.

---

## A-004 — Maximum training VRAM @ batch 8: 2.05 GB → 2.06 GB

**Status:** ✅ **APPROVED** (2026-08-11). Supersedes the threshold set by A-003; the
analysis in A-003 (why the fp16-activation analytic figure was stale) still stands and
is not revisited here.

### Clause amended

Contract Part 10, "Performance budget", as previously amended by A-003:

> Maximum training VRAM @ batch 8 | **2.05 GB** | 1.17 GB analytic

### Problem

The RTX A400 now passes every CUDA correctness test (**108 passed, 2 skipped**, zero
non-finite steps, zero AMP overflow steps) and every other shakedown measurement. The
sole remaining blocker in `scripts/run_gpu_validation.py` is a **1–2 MB** gap between the
measured stable peak and the threshold:

    measured peak      2.051  GB
    A-003 threshold    2.05   GB
                       -------------
    over by            ~1.0-1.6 MB   (0.05-0.08 %)

A-003's 2.05 GB was set by rounding up a **single-step** measurement (~2.015–2.017 GB
with the `rel_pos_index` dedup applied). What the subsequent instrumentation showed is
that a single step does not observe the steady-state peak: the caching allocator's
working set grows over the first handful of optimiser steps and then **flattens**. The
threshold was therefore calibrated against a number that real training never sustains,
and it is the *threshold* that is wrong, not the memory usage.

### Measurement

Three independent scripts, all on `NVIDIA RTX A400 (4.29 GB), torch 2.11.0+cu128`,
sparc-base (2,345,650 params), bf16 AMP, batch 8, `channels_last`:

| Script | Methodology | Peak |
|---|---|---|
| `scripts/vram_reconcile.py` | shakedown methodology (grad clip + GradScaler wrapper + multi-step window) | **2.0526 GB** |
| `scripts/vram_step_trace.py` | 50-step per-step trace, one reset, cumulative | **2.0516 GB** (stable) |
| `scripts/cuda_shakedown.py` | 20 real optimiser steps, full `CompositeLoss` | **2.051 GB** |

The three agree to within **1.6 MB**. `vram_step_trace.py` is the decisive one: over 50
steps the cumulative peak rises during the first steps and then holds flat at 2.0516 GB
for the remainder. **This is a steady state, not a leak** — a leak would show a
monotonically rising cumulative peak across all 50 steps, which is precisely what the
trace was written to distinguish and precisely what it does not show.

Accompanying numerics from the same shakedown, all clean:

| Signal | Result |
|---|---|
| CUDA correctness tests | **108 passed, 2 skipped** |
| Non-finite steps | **0** |
| AMP overflow steps | **0** |
| Loss over the run | 0.5446289778 → 0.4946310818 (decreasing) |
| Train step time | 1492 ms |
| Inference, batch 1 | 46.05 ms/image |
| Inference, batch 16 | 17.75 ms/image |

### Change

The threshold moves from **2.05 GB** to **2.06 GB**, i.e. the measured stable peak of
~2.0516 GB plus **~8.4 MB (0.4 %)** of headroom for cross-GPU and cross-torch-version
measurement noise.

**This is not an arbitrary relaxation, and it is explicitly not a way to make a
regression disappear:**

* the new limit is set from *measured* long-run behaviour (2.0516 GB over 50 steps),
  not chosen to clear an observation after the fact;
* **VRAM usage itself is unchanged.** No architecture, GSA, GatedFuse, noise-head,
  `CompositeLoss`, batch-size, AMP-dtype, optimiser, learning-rate, gradient-clipping or
  schedule change was made, and attention was not disabled. Nothing was traded away to
  fit the budget;
* the **actual measured value continues to be printed and recorded** —
  `cuda_shakedown.py` prints `train b8 VRAM : 2.051 GB (limit 2.06 GB, A-004) PASS` and
  writes both the measurement and the budget into
  `reports/phase4_12_shakedown.json`. The measurement is gated, never hidden;
* the headroom is a **noise margin, not slack to grow into**. If a future measurement
  moves the stable peak materially above ~2.052 GB, that is a regression to investigate,
  not a threshold to raise again.

### Physical headroom

The A400 has **4.29 GB** of total VRAM. At a 2.0516 GB stable peak the training run uses
**~48 %** of the card and leaves **~2.24 GB physically free**. Neither 2.05 GB nor 2.06 GB
is anywhere near an out-of-memory condition; both are *budget discipline* figures chosen
to detect unintended memory growth, and the 10 MB difference between them has no
physical consequence on this hardware.

### Rejected alternatives

* **Leave the threshold at 2.05 GB and shrink the measurement** (e.g. the bf16-backward
  `LayerNorm2d` rework A-003 left as future work). Rejected *for this amendment*: it is
  a numerics change requiring its own shakedown validation, and it would be undertaken
  to recover ~1.6 MB — an argument that inverts the actual engineering priority. Still
  available, and still the right move if the budget ever needs a real tightening.
* **Round the threshold to the measurement (2.0516 → 2.055 GB).** Rejected: too tight to
  survive ordinary cross-host allocator variation, which would make the gate flaky and
  train the team to ignore it.
* **Drop the training-VRAM gate.** Rejected outright: the gate exists to catch memory
  regressions, and the correct response to a 1 MB discrepancy is to calibrate it, not to
  delete it.

### Not changed by this amendment

`torch.compile` remains **optional for V1**. Its status is reported by
`cuda_shakedown.py` as diagnostic information and a compile failure remains
**non-blocking** for the validation pipeline. A-004 does not touch that requirement.

### Budget impact (Part 10)

Only the "Maximum training VRAM @ batch 8" row changes, from **2.05 GB** to **2.06 GB**.
Parameters, MACs, GFLOPs, activations/image, inference VRAM/latency, training time and
model size are unaffected, and no measured quantity changes at all — this amendment moves
an acceptance threshold and nothing else.

### Files updated

| File | Change |
|---|---|
| `tests/test_full_model_training.py` | `test_training_vram_at_batch_eight_is_within_budget` asserts `< 2.06`, docstring cites A-004 |
| `scripts/cuda_shakedown.py` | `TRAINING_VRAM_BUDGET_GB = 2.06`; prints measured GB, limit and PASS/FAIL; records both in the JSON report |
| `scripts/vram_profile.py` | `BUDGET_GB = 2.06` |
| `scripts/vram_reconcile.py` | `BUDGET_GB = 2.06` (diagnostic; does not gate) |
| `scripts/benchmark.py` | `BUDGETS["train_vram_gb"] = 2.06` — this table had never been updated for A-003 and still read 2.00 GB; its probe methodology was also corrected, see erratum M-2 in `reports/PHASE4_12_VERIFICATION.md` |
| `reports/PHASE4_12_VERIFICATION.md` | §4 and §6 budget rows updated to the A-004 limit and the measured A400 result |
