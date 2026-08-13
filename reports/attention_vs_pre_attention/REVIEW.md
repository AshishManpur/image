# Attention vs Pre-Attention — Strict Experiment Review

**Date:** 2026-08-11
**Question:** does the Phase 4.12 GSA attention module provide a meaningful restoration
benefit over the pre-attention baseline?

**Verdict: C — ATTENTION CURRENTLY SHOWS NO MEANINGFUL BENEFIT.**

On 320 held-out validation images the attention model is **−0.018 dB PSNR** and
**−0.0015 SSIM** versus the baseline. Both differences are statistically significant and
**negative**, while costing **+35 % parameters**, **+46 % inference latency** and
**+55 % training time**. The quality difference is too small to matter in either
direction; the cost difference is not.

Nothing in the model, training code, loss, dataset or checkpoints was modified.

---

## 1. COMPARABILITY CHECK

This is a genuinely well-controlled ablation. Verified item by item:

| Question | Evidence | Verdict |
| --- | --- | --- |
| Same architecture family, one variable? | `infer_config_from_state_dict` reconstructs both: `use_attention=False` (1,737,562 params) vs `use_attention=True` (2,345,650). Only attention differs. | **PASS** |
| Same dataset split? | `datasets/splits.py::group_aware_split` is **deterministic — no RNG**. With `n=3200`, `block_size=32`, `every_n=10` it always yields the same 2880/320 partition. | **PASS** |
| Same schedule? | `metrics.csv` for both runs: 50 epochs, `step` column identical at every row, `lr` column **bit-identical** at every epoch (e.g. both `2.4549237231352307e-06` at epoch 47). | **PASS** |
| Same optimiser state point? | Both `best_ema_psnr.pt` are `epoch 29, global_step 10800`. Identical. | **PASS** |
| Same preprocessing/normalisation? | No dataset-level normalisation exists. `RobustNormalizer` is **inside** the model (`models/normalization.py`) and de-normalises before returning. Both consume and emit identical units. | **PASS** |
| Same GT definition? | Both read `Data/packed/train_gt.npy`, packed once by `scripts/pack_dataset.py`, fp16 round-trip verified at 77.38 dB. Single source. | **PASS** |
| EMA compared consistently? | `trainer/trainer.py:777-779` saves `best_ema_psnr.pt` on improvement in **EMA** `psnr_mean` — same rule for both. Both files resolve `source='ema'`; the harness **refuses to run** if the two sources differ. | **PASS** |
| PSNR/SSIM identical? | Both scored by `evaluation/metrics.py` (PSNR `data_range=1.0`; SSIM 11×11 Gaussian, σ=1.5, K=(0.01,0.03), valid region). Same code path for both. | **PASS** |
| Inference settings identical? | Single `--device`/`--amp-dtype` applied to both; fp32 with TF32 disabled (`true_float32`). Irrelevant here — CPU only on this machine. | **PASS** |
| Materially different training conditions? | None detectable. Identical steps, identical LR trace, identical split. | **PASS** |

### Independent validation of the reported numbers

All 10 originally reported per-image PSNRs **reproduce bit-for-bit** through this
harness (largest discrepancy 0.01 dB, i.e. rounding):

```
000000 31.53/31.32   000001 32.15/32.05   000002 32.46/32.46   000003 33.15/33.15
000004 21.28/21.37   000005 28.31/28.33   000006 28.30/28.35   000007 28.23/28.23
000008 21.69/21.70   000009 23.70/23.68        10-image mean delta -0.0149 dB
```

**Correction to the request, in your favour:** those 10 images were described as
*training* images. They are not — `group_aware_split` places IDs 0–31 in the
**validation** partition. They are legitimate held-out samples. The problem with them
was never leakage; it was that n=10 all drawn from a single 32-ID block, where Phase 1
found 14.1 % of images have a near-duplicate twin within ±2 IDs.

**Conclusion: the experiment is comparable. Verdict E is ruled out.**

### The one real gap

Checkpoints store weights, optimiser, scheduler and `state`, but **not the resolved
`SparcConfig`/`TrainingConfig`, git SHA or seed**. Comparability above was reconstructed
from the metric logs, which happened to survive. That is luck, not process — see the
recommendation.

---

## 2. DATASET / EVALUATION SIZE

| Array | Shape | dtype | Size |
| --- | --- | --- | ---: |
| `train_lr` | (3200, 128, 128) | fp16 | 105 MB |
| `train_gt` | (3200, 256, 256) | fp16 | 419 MB |
| `test_lr` | (400, 128, 128) | fp16 | 13 MB |

- **3200 paired LR/GT images**, split **2880 train / 320 validation** by contiguous ID
  blocks (near-duplicates never straddle the boundary).
- **400 test LR images with NO GT** — test is visual-only, never a metric.
- The original evidence used **10 of 320 available validation images (3.1 %)**.

Evaluation performed here:

| Category | Split | n | Role |
| --- | --- | ---: | --- |
| **B. Validation** | `val` | **320** | **the decisive evidence** |
| A. Training | `train` | 2880 | diagnostic only — not generalisation |
| C. Test visual | `test` | 5 rendered | no GT; qualitative only |

---

## 3. METRIC TABLE — validation set, 320 images, EMA weights

| Metric | Pre-attention | Attention | Δ (attn − pre) |
| --- | ---: | ---: | ---: |
| PSNR mean | 27.458 dB | 27.440 dB | **−0.018 dB** |
| PSNR median | 27.500 dB | 27.521 dB | +0.021 dB |
| PSNR std | 4.100 | 4.105 | — |
| SSIM mean | 0.7518 | 0.7503 | **−0.0015** |
| SSIM median | 0.7974 | 0.7985 | +0.0011 |
| Parameters | 1,737,562 | 2,345,650 | +35.0 % |
| Latency (CPU fp32, median of 20) | 157.8 ms | 229.9 ms | +45.7 % |
| Training wall-clock (50 ep) | 2.75 h | 4.25 h | +55 % |

Note the sign flip between mean and median: attention is very slightly ahead on the
*typical* image and behind on the *mean*, i.e. it loses more on the tail than it gains in
the bulk. Neither movement is large enough to matter.

For scale, the Phase 1 baselines: bicubic 22.69 dB mean, oracle clean-LR bicubic
31.90 dB mean. Both models sit at ~27.45 dB — the −0.018 dB difference is **0.4 % of the
gap between the two models and bicubic**.

---

## 4. STATISTICAL ANALYSIS — paired per-image PSNR deltas (n = 320)

| Statistic | Value |
| --- | ---: |
| mean Δ | −0.0182 dB |
| median Δ | −0.0132 dB |
| std Δ | 0.0997 dB |
| **95 % CI** | **[−0.0292, −0.0072] dB** |
| paired t-test | t = −3.264, **p = 0.0012** |
| Wilcoxon signed-rank | W = 19652, **p = 0.00027** |
| Cohen's d_z | **−0.182** (small) |
| improved | 131 (40.9 %) |
| degraded | 189 (59.1 %) |
| largest improvement | +0.454 dB |
| largest degradation | −0.444 dB |

SSIM deltas agree: mean −0.00151, median −0.00041, Wilcoxon **p = 0.0067**, 43.1 %
improved.

**Reading this correctly — the distinction that matters:**

- The result **is statistically significant**. Both a parametric and a rank-based test
  agree at p < 0.005, so this is not noise. With n=320 the harness can resolve a
  0.018 dB difference.
- The result is **practically negligible**. The entire 95 % CI lies within
  **[−0.03, −0.01] dB**. For scale, 8-bit PNG quantisation alone costs ~0.1 dB — roughly
  **five times** the effect being measured. Cohen's d_z = −0.18 is a small effect, and
  the near 41/59 improved/degraded split is close to a coin flip.

**This is a precisely measured null result, not a large negative one.** Statistical
significance here is a statement about sample size, not about importance. The honest
summary is: *the two architectures are equivalent in quality to within ±0.03 dB, and
attention is on the wrong side of that by a hair.*

### Why the original 10-image evidence could not have decided this

The 10-image sample gave mean Δ = −0.0149 dB with std ≈ 0.086 → 95 % CI ≈
**[−0.077, +0.046] dB**, spanning zero (p ≈ 0.6). The point estimate was almost exactly
right; the uncertainty was ~6× too wide to act on. It took 320 images to shrink the CI
enough to exclude zero — and even then, only to confirm the difference is negligible.

---

## 5. VISUAL ANALYSIS

Figures in `outputs/attention_vs_pre_attention/val_320_ema/visuals/`
(`INPUT LR | PRE-ATTENTION | ATTENTION | GROUND TRUTH`).

| Trait | Finding |
| --- | --- |
| Fine detail preservation | **No advantage.** On the median case (`000009`, palm fronds) both models smooth the fine frond texture away identically; neither recovers GT detail. |
| Noise removal | Equivalent. Both suppress speckle to the same visible degree. |
| Oversmoothing | Both oversmooth on high-frequency texture; **attention is marginally less smooth**, which is what costs it PSNR on textured images. |
| Artifacts | None specific to attention. No block, grid or attention-window artifacts observed — the GSA implementation is clean. |
| Edges/textures | On the largest-degradation case (`001299`, diagonal streak texture, −0.444 dB) attention produces slightly **more contrasty, blotchier** texture that does not match GT — amplified rather than recovered structure. |
| Halos / ringing | None observed in either model. |
| Brightness/contrast | No shift. `RobustNormalizer` inverts correctly in both. |
| Difficult images | On the largest-improvement case (`001612`, near-featureless smooth gradient, +0.454 dB) attention is slightly cleaner. Its wins cluster on **flat, low-texture** content. |

Quantitative confirmation of that last row — mean ΔPSNR by GT-texture quartile:

| GT std quartile | n | ΔPSNR | ΔSSIM | % improved |
| --- | ---: | ---: | ---: | ---: |
| Q1 (flattest, 0.034–0.135) | 80 | −0.008 | −0.0032 | 51 % |
| Q2 (0.135–0.179) | 80 | −0.008 | −0.0001 | 42 % |
| Q3 (0.179–0.223) | 80 | −0.036 | −0.0011 | 32 % |
| Q4 (most textured, 0.223–0.415) | 80 | −0.021 | −0.0016 | 38 % |

Attention is behind in **every** quartile. Spearman corr(Δ, GT std) = −0.127 (p = 0.023):
a weak tendency to do relatively worse on textured content — the opposite of the usual
motivation for adding attention.

### C. Test-set visual inference (no GT)

5 random test images rendered to `outputs/attention_vs_pre_attention/test_visual/` with
an `|ATTN − PRE| ×10` difference panel. Mean absolute difference between the two models'
outputs: **0.0015–0.0098** (0.15–1 % of dynamic range). Differences concentrate on edges.
The two models are, for practical purposes, **visually indistinguishable** on unseen data.

---

## 6. COMPUTATIONAL COST

| | Pre-attention | Attention | Overhead |
| --- | ---: | ---: | ---: |
| Parameters | 1,737,562 | 2,345,650 | **+35.0 %** |
| Inference latency (CPU fp32, median of 20) | 157.8 ms | 229.9 ms | **+45.7 %** |
| Training time, 50 epochs (RTX A4000) | 2.75 h | 4.25 h | **+55 %** |

Latency measured on this machine, which is **CPU-only** (`torch.cuda.is_available()`
is `False`). The reported 128.4 / 160.5 ms are the same order; absolute values vary with
load, but the **ratio** is the meaningful quantity and it reproduces (+40–46 %). On GPU
attention's relative cost would likely be lower — worth re-measuring on the A4000, but it
cannot change the quality conclusion.

**Is the cost justified?** No. A defensible bar for +35 % parameters and +46 % latency is
roughly **+0.3 dB or better**, with the CI excluding zero. The measured result is
**−0.018 dB with the CI entirely below zero** — not merely short of the bar, but on the
wrong side of it. There is no operating point at which paying 46 % more latency for
−0.018 dB is correct.

---

## 7. FINAL VERDICT

### **C. ATTENTION CURRENTLY SHOWS NO MEANINGFUL BENEFIT**

Measured justification:

1. On 320 held-out images, ΔPSNR = **−0.018 dB**, 95 % CI **[−0.029, −0.007]**, and
   ΔSSIM = **−0.0015** (Wilcoxon p = 0.0067). Both metrics move against attention.
2. The effect is **statistically significant but practically negligible** — the whole CI
   is within ±0.03 dB, roughly one-fifth of what 8-bit quantisation costs.
3. Improved/degraded is **41 % / 59 %**, near a coin flip; d_z = −0.18.
4. Visually the two models are **indistinguishable** on validation and test data
   (mean absolute output difference 0.15–1 % of range), with no attention-specific
   artifacts in either direction.
5. Costs are large and certain: **+35 % parameters, +46 % latency, +55 % training time.**

Not **D** ("hurts quality"): −0.018 dB is not meaningful damage, and attention is
slightly *ahead* on median PSNR and median SSIM. Not **B** ("probably helps, needs more
training"): both runs plateaued at **epoch 25** and peaked at **epoch 29**, with 20
subsequent epochs producing no improvement in either model — there is no upward trend
left to extrapolate. Not **E**: the comparability audit passed on every item.

### The diagnostic finding — why attention is not helping

Attention's **training loss is consistently lower**, and the gap **widens monotonically**:

| Epoch | Pre train loss | Attn train loss | Gap |
| ---: | ---: | ---: | ---: |
| 10 | 0.05271 | 0.05231 | −0.00040 |
| 29 | 0.04780 | 0.04729 | −0.00051 |
| 49 | 0.04644 | 0.04562 | −0.00082 |

The attention model uses its extra 35 % capacity to fit the training set **strictly
better**, while its validation PSNR stays equal-or-worse throughout. That is the textbook
signature of **capacity that is not generalising**. The module is working — it just is
not learning anything that transfers.

This makes the mechanism, not the schedule, the thing to investigate. More epochs would
widen that gap, not close it.

---

## 8. RECOMMENDED NEXT EXPERIMENT

**Do not train to 100/200/400 epochs.** Both models plateaued at epoch 25 and drifted
*down* over the final 10 epochs (pre −0.018 dB, attn −0.007 dB). Compute spent extending
this schedule buys nothing.

**Do not delete the attention code either** — it is clean, artifact-free, and the null
result is informative.

Recommended, in priority order:

1. **Ship the pre-attention baseline as the V1 default.** It matches attention's quality
   within 0.02 dB at 74 % of the parameters and 69 % of the latency. This is the
   evidence-backed default today.

2. **Fix checkpoint provenance (cheap, blocking for all future ablations).** Serialise
   the resolved `SparcConfig`/`TrainingConfig`, git SHA, seed and epoch budget into every
   checkpoint. This review only succeeded because `metrics.csv` happened to survive
   alongside the weights; the next one may not be so lucky.

3. **Then, if attention is still worth pursuing — diagnose the mechanism, do not retrain
   the same thing.** The specific question raised by the widening train-loss gap is
   *whether the attention maps carry useful signal at all*. Concretely:
   - instrument the GSA blocks and inspect the learned attention distributions — if they
     are near-uniform, the module is degenerating to an expensive convolution;
   - check whether the relative-position bias tables have moved from initialisation;
   - ablate *where* attention sits (encoder-only vs decoder-only vs both) rather than
     scaling the same configuration up.
   The 128×128 fixed-grid constraint that attention imposes (`check_input_size`) is a
   real deployment cost too, and it is only worth paying for a module that earns it.

4. **Regularisation, if you want to test the overfitting hypothesis directly.** The extra
   capacity is being absorbed by the training set. A single run with stronger
   augmentation or weight decay on the attention variant would show whether the capacity
   can be redirected — one 50-epoch run (~4.25 h on the A4000), not 400 epochs.

**What I would not spend compute on:** longer schedules, larger attention widths, or more
evaluation. The evaluation question is now closed — 320 images with agreeing parametric
and rank tests is sufficient to conclude equivalence.

---

## 9. ASSUMPTIONS AND LIMITS

- **Assumed:** the two runs used this working copy's `group_aware_split` constants
  (`block_size=32`, `every_n=10`). Supported by identical `step` counts (10800 at epoch
  29 → 360 steps/epoch → 2880 train images at batch 8), but not independently recorded in
  the checkpoints.
- **Assumed:** `best_ema_psnr.pt` for both is the intended comparison point. Both are
  epoch 29; `best_psnr.pt` and `last.pt` were not evaluated.
- **Not verifiable:** loss weights and augmentation settings of the two runs are not
  stored in the checkpoints. Identical LR traces and step counts make a materially
  different configuration unlikely but do not prove it.
- **Measured on CPU.** The A4000 latency ratio may differ; quality results are
  device-independent (fp32, TF32 disabled).
- **Single seed per architecture.** Run-to-run variance is unmeasured, so the −0.018 dB
  cannot be decomposed into architecture effect vs seed effect. Given the effect is
  negligible either way, this does not change the verdict — but a second seed per arm
  would be required to claim the *direction* is real.
- **Unmodified:** architecture, loss, dataset, checkpoints, hyperparameters. Added only
  `scripts/eval_attention_ablation.py` and artifacts under
  `reports/attention_vs_pre_attention/` and `outputs/attention_vs_pre_attention/`.
