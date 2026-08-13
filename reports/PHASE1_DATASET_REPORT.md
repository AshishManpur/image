# Phase 1 — Dataset Forensics Report

**Project:** Degraded grayscale image restoration (denoise + deblur + ×2 super-resolution)
**Data root:** `Data/`
**Date:** 2026-08-04
**Reproducibility:** all numbers below are produced by `reports/inspect_full.py`, `inspect_deg.py`,
`inspect_stage3.py`, `inspect_stage4.py`, `inspect_stage5.py`; raw outputs in `reports/report_stage*.json`;
figures in `reports/figures/`.

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Domain | **Natural grayscale photographs (DIV2K/Flickr2K-like crops) — NOT semiconductor imagery** |
| Samples | 3200 paired train, 400 unpaired test |
| Task | `(128,128) float32` → `(256,256) float32`, ×2, grayscale |
| Pair integrity | **100 % — zero missing, zero duplicate, zero corrupt** |
| Dominant degradation | **Multiplicative speckle** (σ≈0.166), not blur |
| Forward operator | Gaussian blur σ≈0.4 px → speckle+Gaussian → **bicubic ×0.5 without antialiasing** |
| Input SNR | **median 7.1 dB** (p10 0.7 dB) — a low-SNR denoising problem |
| Bicubic baseline | **21.67 dB PSNR** |
| Denoise-only oracle | **27.36 dB PSNR** |
| Critical risk #1 | **Near-duplicate crops at adjacent IDs ⇒ random val split leaks** |
| Critical risk #2 | **Test set is measurably higher-frequency than train (KS D=0.215, p=8e-15)** |
| Critical risk #3 | Current augmentation double-degrades and photometrically jitters the input |

---

## 1. Directory analysis

### 1.1 Tree

```
Data/                                                  1 076 780 216 B (1.027 GB)
│                                                      13 600 .npy + 5 extension-less + 1 .DS_Store
├── Test_NoisyLR (1)/
│   ├── NoisyLR/                    400 × *.npy        26 265 600 B     ← TEST INPUT (no GT)
│   └── __MACOSX/                   1 × "._NoisyLR"
│       └── NoisyLR/                400 × "._*.npy"    65 200 B         ← ARTEFACT
└── train/
    ├── train/
    │   ├── GT/                     3200 × *.npy       839 270 400 B    ← TRAIN TARGET
    │   └── NoisyLR/                3200 × *.npy       210 124 800 B    ← TRAIN INPUT
    └── __MACOSX/
        └── train/                  .DS_Store + 2 × "._{GT,NoisyLR}"
            ├── GT/                 3200 × "._*.npy"   521 600 B        ← ARTEFACT
            └── NoisyLR/            3200 × "._*.npy"   521 600 B        ← ARTEFACT
```

### 1.2 Filesystem artefacts

| Check | Result |
|---|---|
| Symbolic links | **0** |
| NTFS reparse points / junctions | **0** |
| Zero-byte files | **0** |
| Windows artefacts (`Thumbs.db`, `desktop.ini`) | **0** |
| macOS artefacts | **6805** — 6800 AppleDouble `._*.npy` + 4 `._<dirname>` + 1 `.DS_Store` |
| Hidden (dot-prefixed) entries | 6805 (all of the above) |
| File-size uniformity | Perfect: every LR file **65 664 B**, every GT file **262 272 B** |

**The `__MACOSX` tree is a trap.** Each `._*.npy` is exactly 163 bytes with header
`00 05 16 07 00 02 00 00 "Mac OS X"` — an AppleDouble resource fork produced by unzipping on macOS.
They carry a `.npy` extension and mirror every real basename with a `._` prefix.

Consequence: `Path(root).rglob("*.npy")` returns **13 600 paths instead of 6 800**, and `np.load` on any
of them raises. Every consumer must filter `"__MACOSX" not in p.parts` (or `not name.startswith("._")`).
`datasets/semiconductor_dataset.py:24,30` already does this correctly — this guard must be preserved
in any refactor.

Size uniformity confirms 4-byte C-order arrays with a 128-byte NPY v1 header:
`128 + 128·128·4 = 65 664` and `128 + 256·256·4 = 262 272`. No object arrays, no pickles, no F-order.

### 1.3 Naming conventions

Strict zero-padded 6-digit decimal, `%06d.npy`:

* `train/train/{GT,NoisyLR}`: `000000.npy … 003199.npy` — 3200 IDs, **fully contiguous, no gaps**
* `Test_NoisyLR (1)/NoisyLR`: `000000.npy … 000399.npy` — 400 IDs, contiguous
* Train and test ID spaces **overlap numerically** (both start at `000000`). Any flat ID-keyed cache or
  submission map must therefore be namespaced by split, or IDs will silently collide.

### 1.4 Is the organisation appropriate?

Adequate but not good. Discussed in full in §9.

---

## 2. File analysis

| Directory | Files | Format | Size each | Readable | Corrupt | MD5 duplicates |
|---|---|---|---|---|---|---|
| `train/train/NoisyLR` | 3200 | NPY v1, float32 | 65 664 B | 3200/3200 | 0 | **0** |
| `train/train/GT` | 3200 | NPY v1, float32 | 262 272 B | 3200/3200 | 0 | **0** |
| `Test_NoisyLR (1)/NoisyLR` | 400 | NPY v1, float32 | 65 664 B | 400/400 | 0 | **0** |
| `__MACOSX/**` | 6800 | AppleDouble (not NPY) | 163 B | n/a | n/a | n/a |

No `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` files exist anywhere in the tree. The dataset is
100 % NumPy. Every array loads with `allow_pickle=False`, so there is no deserialisation risk.

MD5 was computed over the raw `ascontiguousarray().tobytes()` of all 6800 real arrays.
**Zero collisions within any split, and zero collisions between train-LR and test-LR** — the test inputs
are not recycled training inputs.

---

## 3. Pair validation

Pairing rule (inferred, then verified): **same filename stem across `NoisyLR/` and `GT/` within the same
split root.**

| Check | Result |
|---|---|
| LR stems | 3200 |
| GT stems | 3200 |
| Matched pairs | **3200 (100 %)** |
| LR without GT | 0 |
| GT without LR | 0 |
| Duplicated IDs | 0 |
| Inconsistent filenames | 0 (all match `^\d{6}\.npy$`) |
| Corrupted pairs | 0 |
| Distinct `(LR.shape, GT.shape)` combinations | **1** — `((128,128),(256,256))` for all 3200 |

Scale factor is exactly 2 for every pair, with no exceptions. Test has no GT by design.

---

## 4. NumPy analysis

Full scan of all 6800 arrays (not a sample).

| Property | train/NoisyLR | train/GT | Test/NoisyLR |
|---|---|---|---|
| ndim | 2 (all) | 2 (all) | 2 (all) |
| shape | `(128,128)` ×3200 | `(256,256)` ×3200 | `(128,128)` ×400 |
| dtype | `float32` | `float32` | `float32` |
| Memory / array | 64 KiB | 256 KiB | 64 KiB |
| Total in-RAM | 200 MiB | 800 MiB | 25 MiB |
| Global min | −0.27856 | **0.000000** | −0.22488 |
| Global max | +2.15800 | **1.000000** | +2.15802 |
| Mean of per-image means | 0.43354 | 0.43353 | 0.44274 |
| Mean of per-image stds | 0.20580 | 0.18763 | 0.22027 |
| NaN | **0** | **0** | **0** |
| Inf | **0** | **0** | **0** |
| Channels | 1 (2-D, grayscale) | 1 | 1 |

Per-image ranges (train):

| | min | max | mean | std |
|---|---|---|---|---|
| NoisyLR | −0.279 … +0.404 | 0.658 … 2.158 | 0.016 … 0.959 | 0.016 … 0.451 |
| GT | **0.0 … 0.0** | **1.0 … 1.0** | 0.016 … 0.959 | 0.015 … 0.440 |

### 4.1 GT is per-image min–max normalised — exactly

Every one of the 3200 GT arrays has `min == 0.0` and `max == 1.0` bit-exactly, and the fraction of pixels
equal to each extreme is `1.526e-5 = 1/65536` — i.e. **exactly one pixel at 0 and one at 1 per image**.

$$\text{GT}_i = \frac{X_i - \min X_i}{\max X_i - \min X_i}$$

This is a hard, exploitable invariant: the target's dynamic range is known a priori. A final
`clamp(0,1)`, and optionally a per-image affine re-normalisation of the prediction, is free PSNR.

### 4.2 LR is NOT clipped

| | fraction |
|---|---|
| LR pixels `< 0` | 0.300 % |
| LR pixels `> 1` | 3.360 % |
| LR pixels exactly 0 or 1 | 0 % |

Therefore **do not clamp the input** (that would destroy 3.7 % of the samples and bias bright regions),
but **do clamp the output**.

### 4.3 Unique values and histogram

Median unique values per GT image: **65 390** out of 65 536 pixels. The data is continuous-tone with no
quantisation lattice — it is not an 8-bit image cast to float, and it is not a discrete layout mask.

GT intensity histogram (16 coarse bins over [0,1], % of pixels):

```
7.60 8.32 7.95 7.66 7.30 7.06 6.99 6.74 6.72 6.35 5.93 5.20 4.66 4.51 3.89 3.12
```

Mildly dark-skewed, broadly uniform, **unimodal-flat — not bimodal**. 62.6 % of pixels lie in the
[0.2, 0.8] midtone band; only 5.6 % below 0.05 and 2.8 % above 0.95.

LR histogram over [−0.4, 1.6] shows symmetric spill past both ends, consistent with unclipped
zero-mean noise added on top of a [0,1] signal.

---

## 5. Dataset statistics

* **Samples:** 3200 train pairs / 400 test inputs. No official validation split is provided.
* **Resolution distribution:** exactly 2 resolutions — `128²` (inputs) and `256²` (targets). Zero variance.
* **Categories / class labels:** none. This is a pure regression dataset; class imbalance is not applicable.
  The relevant analogue is *content* imbalance, addressed below.
* **Dynamic range:** GT fixed at [0,1] by construction; LR effective range ≈ [−0.28, 2.16].
* **Per-image mean spans 0.016 → 0.959** — extremely wide exposure diversity, from near-black to near-white scenes.

### 5.1 Outliers

| Condition | Count | Example IDs |
|---|---|---|
| GT std < 0.05 (near-flat) | **34** | 2224, 2227, 2226, 2225, 361 |
| GT std < 0.03 (essentially featureless) | **7** | 2224 (σ=0.0145), 2227, 2226, 2225 |
| GT mean outside [0.1, 0.9] | **87** | dark: 2224, 1827, 1826 · bright: 643, 1193, 1194 |

Note the ID clustering (2224–2227, 1826–1827, 1193–1195) — outliers arrive in consecutive runs, which is
itself evidence for the crop-grouping structure established in §5.2.

These 34 near-flat images are a **metric hazard**: PSNR on a near-constant image is dominated by the
noise floor and can reach 40 dB+ trivially, inflating any mean-PSNR validation number. Report median
PSNR alongside mean, or stratify.

### 5.2 Repeated scenes and data leakage — **critical**

I computed a 16×16 mean-pooled, mean-centred, L2-normalised signature for all 3200 GT images and took the
full 3200×3200 cosine similarity matrix.

| Threshold | Images with a twin | Twin is within ±2 IDs | Median ID gap |
|---|---|---|---|
| cos > 0.99 | 450 (14.1 %) | **98.9 %** | 1 |
| cos > 0.95 | 730 (22.8 %) | 95.9 % | 1 |
| cos > 0.90 | 969 (30.3 %) | 90.1 % | 1 |

The closest pair (2121, 2122) has **pixel MSE 5.1×10⁻⁶** — visually the same crop, shifted by a few pixels.
Visual confirmation in `reports/figures/dups.png`: pairs (2121,2122), (985,986), (2321,2322), (1922,1923),
(2111,2109), (651,649) are all near-identical scenes at adjacent IDs.

**Conclusion:** the dataset was constructed by taking *multiple overlapping crops from each source
photograph* and writing them to consecutive IDs. Connected-component analysis gives
**2749 distinct scenes at cos>0.95** (2554 at cos>0.9), not 3200 independent samples.

**Leakage consequence:** `split_samples()` in `datasets/semiconductor_dataset.py:39` shuffles indices
uniformly at random. With a 10 % split, ~10 % of validation images have a near-identical twin in training.
**Validation PSNR will be optimistically biased and will not track the held-out test set.** This is the
single most damaging issue in the current pipeline — it corrupts every model-selection decision downstream.

Fix: split by **contiguous ID blocks** (e.g. blocks of 32 consecutive IDs assigned wholesale) or by
**connected component** of the cos>0.95 graph.

### 5.3 Train ↔ test distribution shift — **significant**

Blind (no-GT) features over all 3200 train-LR vs all 400 test-LR, two-sample Kolmogorov–Smirnov:

| Feature | train median | test median | ratio | KS D | p-value |
|---|---|---|---|---|---|
| mean intensity | 0.4338 | 0.4331 | 1.00 | 0.084 | 1.3e-02 |
| std | 0.2016 | 0.2152 | 1.07 | 0.107 | 5.0e-04 |
| Immerkær σ estimate | 0.0668 | 0.0752 | **1.13** | 0.130 | 1.0e-05 |
| high-pass std (5×5) | 0.1051 | 0.1215 | **1.16** | **0.215** | **7.8e-15** |
| mean \|∂x\| | 0.1029 | 0.1170 | **1.14** | 0.147 | 4.1e-07 |
| histogram entropy | 3.204 | 3.251 | 1.01 | 0.078 | 2.6e-02 |

Radially averaged power spectra confirm it: test-LR carries **1.2–1.6× more energy at every non-DC
frequency**.

Interpretation — the two splits are the same *domain* but not the same *distribution*. Test content is
visibly dominated by man-made high-frequency texture (brick walls, window grids, tiling, masonry;
see `figures/test.png`), whereas train contains substantially more sky, cloud, blur-of-field and foliage
(`figures/diversity.png`). **Generalisation pressure is toward dense, fine, repetitive texture.**

Practical implication for model selection: a validation set drawn uniformly from train will
*under-represent* the hardest test content. Consider a texture-stratified validation set, weighting
high-`hp_std` images.

---

## 6. Image analysis — visual and statistical

`reports/figures/samples.png` (6 pairs × {GT, LR, bicubic×2, residual, log-FFT, crop}),
`figures/diversity.png` (16 random pairs), `figures/test.png` (16 test inputs).

### 6.1 Content — **the dataset is not semiconductor imagery**

The images are grayscale natural photographs: clouds, tree branches, rocks, brick and masonry walls,
building façades, foliage, water reflections, people, and a market stall bearing legible German signage
("Unser Spar…", "Familient…", "10 Lose 50"). The content, the 256×256 crop size, and the multiple-crops-
per-source structure are all consistent with **DIV2K / Flickr2K converted to grayscale and per-image
min–max normalised**.

Corroborating statistics, independent of my visual reading:

* **Gradient isotropy**: mean|∂x| / mean|∂y| = **0.939**. Semiconductor layouts are strongly axis-anisotropic
  (Manhattan geometry); this is near-isotropic, as natural images are.
* **Continuous tone**: 65 390 unique values per image, flat-unimodal histogram, 62.6 % midtones. Layout
  imagery is characteristically bimodal with quantised levels.
* No periodic lattice peaks survive windowed spectral analysis.

> ⚠️ **Caveat on an earlier reading.** The raw (unwindowed) 2-D FFT of these images shows a bright
> horizontal/vertical cross, which superficially suggests Manhattan structure. That cross is **spectral
> leakage from non-periodic image boundaries**, not content. The pixel-domain gradient isotropy of 0.939
> is the reliable measurement. I flag this because it is exactly the kind of artefact that would justify
> a wrong architectural choice (an oriented/periodic-structure branch) if taken at face value.

### 6.2 Noise characterisation

Residual `r = LR − D(GT)` under the best-fit operator, 32 intensity quantile bins, 300 images:

| clean intensity I | 0.10 | 0.35 | 0.65 | 0.90 |
|---|---|---|---|---|
| std(r) | 0.0256 | 0.0675 | 0.1118 | 0.1499 |
| std(r/I) | 0.259 | 0.192 | 0.172 | 0.167 |
| skew | +1.54 | +0.54 | +0.31 | +0.30 |
| excess kurtosis | +13.9 | +2.39 | +0.62 | +0.36 |

`std(r)` grows almost linearly in `I`, and `std(r/I)` converges to **0.167** — the defining signature of
**multiplicative noise**. Fitting the variance model over the full intensity range:

$$\operatorname{Var}(r \mid I) \;=\; \underbrace{-1.15{\times}10^{-4}}_{a\ (\text{additive})} \;+\; \underbrace{6.42{\times}10^{-3}}_{b\ (\text{Poisson})}\,I \;+\; \underbrace{2.09{\times}10^{-2}}_{c\ (\text{speckle})}\,I^{2}$$

The `I²` term dominates for all `I > 0.31`, i.e. over the large majority of the image.

**The noise is not Gaussian.** Mid-band skew +0.35 and excess kurtosis +0.64, both decaying toward 0 as
`I → 1`. For a multi-look gamma speckle multiplier `g ~ Γ(L, 1/L)`:

$$\sigma = L^{-1/2}, \qquad \gamma_1 = 2L^{-1/2}$$

Measured σ = 0.167 ⇒ **L ≈ 36**; measured γ₁ = 0.35 ⇒ **L ≈ 33**. Two independent moments agreeing on
L ≈ 33–36 is strong evidence for a **gamma / multi-look speckle model**, not multiplicative Gaussian.
(The rising skew/kurtosis at low `I` is the additive Gaussian floor dominating a vanishing multiplicative
term, plus quantile-bin edge effects — expected, not contradictory.)

Per-image two-parameter fits (`Var = a + cI²`), 200 images:

| Parameter | p10 | median | p90 |
|---|---|---|---|
| σ_speckle | 0.140 | **0.165** | 0.194 |
| σ_gaussian | 0.000 | **0.024** | 0.060 |

Per-image total residual std ranges **0.022 → 0.188** (median 0.083, CV 0.34).
⇒ **The noise level varies per image and is not signalled anywhere. The restorer must be blind.**

**Noise is spatially white**, with a small negative first lag:

```
autocorrelation lags:  (0,1) −0.043   (1,0) −0.047   (1,1) +0.006   (0,2) +0.003   (2,0) +0.005
```

Zero beyond lag 1. The mild negative lobe is the fingerprint of noise injected at 256² and carried through
a bicubic decimation kernel (which has negative side lobes) — i.e. **noise was added before downsampling**.

Verdict on the degradation type asked for in the brief:

| Candidate | Present? | Evidence |
|---|---|---|
| **Speckle (multiplicative)** | **YES — dominant** | `Var ∝ I²`, coefficient 2.09e-2, σ=0.165 |
| **Gaussian (additive)** | **YES — secondary** | σ ≈ 0.024 median, up to 0.060 |
| Poisson | weak/ambiguous | `b I` term = 6.4e-3, but `b` and `c` are collinear (r = −0.90) in per-image fits and cannot be separated reliably |
| Non-Gaussian speckle shape | **YES** | skew +0.35, kurtosis +0.64 ⇒ Γ(L≈35) |
| Mixed | **YES** | speckle + additive Gaussian, both per-image random |

### 6.3 Blur, aliasing, frequency content

* **Effective MTF** (cross-spectral estimate `H(f)=E[Y·X*]/E[|X|²]` on the LR grid, radially averaged)
  falls monotonically from **1.00 at DC to 0.89 at LR Nyquist** — an ~11 % droop.
  **This is not a meaningful blur.** There is no defocus, no motion blur, no PSF worth deconvolving.
  Fitting a free 9×9 FIR kernel recovers an essentially delta-like kernel (centre tap 0.914, all other
  taps < 0.025), and reduces RMSE by only 0.0002 (0.0897 → 0.0895) — indistinguishable from noise.
* **Aliasing is present.** The winning downsampler applies **no antialias filter** (§7). Energy above LR
  Nyquist folds back into the LR band. This is a double-edged property: it is a corruption that must be
  undone, but it also carries genuine sub-pixel information that a well-designed SR model can exploit.
* **Only 3.7 % of GT spectral energy lies above LR Nyquist (f > 0.25 cyc/px).** The remaining 96.3 % of
  the target's energy is *present but noise-corrupted* in the input.
  ⇒ **This is a denoising problem with a super-resolution tail, not the reverse.** Capacity allocation
  should follow that ratio.
* GT energy in the top decile of its own band (f > 0.4) is 1.14 % of AC energy, and
  `P(Nyquist)/P(half-Nyquist) = 0.158` — the GT itself is genuinely sharp, not an upsampled source.
* **No JPEG blocking.** 8×8 grid gradient ratio = **0.993** for GT and 0.993 for LR (1.0 = no blocking).
  The blocky appearance in small crops of `figures/samples.png` is matplotlib nearest-neighbour display,
  not compression.

### 6.4 Contrast and brightness

The LR↔GT relationship is photometrically clean. Per-image OLS of LR on `D(GT)`:

* gain = **0.99 ± 0.02**, offset = **0.005 ± 0.009** (raw operator)
* gain = **1.001 ± 0.006**, offset = **−0.0006 ± 0.0034** (after absorbing the mild blur into the FIR fit)

The residual sub-unity gain in the raw fit is regression attenuation from the σ≈0.4 px pre-blur, not a real
photometric transform. **There is no brightness, contrast, or gamma shift between input and target.**
Any augmentation that jitters the input's brightness/contrast/gamma without applying the same transform to
the target is therefore *introducing* a train/test mismatch that does not exist in the data (see §8).

### 6.5 SNR

Defined per image as `10·log₁₀( Var(clean_LR) / Var(noise) )`:

| | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| SNR (dB) | −13.5 | 0.7 | **7.1** | 11.1 | 13.9 |

Equivalent input PSNR against the clean LR: median **21.2 dB** (p10 18.6, p90 26.2).

**Median SNR of 7 dB is low.** Ten percent of images are at or below 0.7 dB — noise power comparable to
signal power. This decisively confirms the capacity-allocation conclusion of §6.3.

---

## 7. Reverse-engineering the forward operator

### 7.1 Method

Naive RMSE ranking cannot identify the operator here: the noise floor (σ≈0.09) is an order of magnitude
larger than the differences between candidate operators (≈0.002–0.007). I therefore used a second,
noise-orthogonal statistic — **structure leakage**:

$$\mathcal{L}(D) \;=\; \operatorname{corr}\!\big(\, \text{LR} - D(\text{GT}),\; \nabla^{2} D(\text{GT}) \,\big)$$

If `D` is the true operator, the residual is pure noise and is uncorrelated with image structure, so
`L(D) = 0`. If `D` under-smooths relative to the truth, the residual retains a `−∇²` component (L < 0);
if it over-smooths, L > 0. Zero-crossing of `L` localises the operator; RMSE breaks ties.

I searched a 50-point grid: pre-blur σ ∈ {0, 0.3, 0.4, 0.5, 0.7} × decimator ∈ {bicubic-noAA, bicubic-AA,
bilinear-noAA, bilinear-AA, area, stride, nearest-exact, Lanczos-2, Lanczos-3, Lanczos-4}.

### 7.2 Results

| Candidate | leak `L` | RMSE |
|---|---|---|
| **blur σ=0.4 → bicubic, no antialias** | **+0.0129** | **0.09104** |
| blur σ=0.3 → bicubic, no AA | +0.0548 | 0.09114 |
| blur σ=0.5 → bicubic, no AA | −0.0559 | 0.09134 |
| blur σ=0.0 → bicubic, no AA | +0.0594 | 0.09116 |
| blur σ=0.7 → nearest / stride | −0.044 | 0.0978 |
| blur σ=0.0 → area ≡ bilinear-noAA | −0.0953 | 0.09158 |
| blur σ=0.0 → bicubic **with** AA | −0.1134 | 0.09320 |
| Lanczos-2/3/4 (all σ) | −0.15 … −0.33 | 0.093 … 0.101 |

**`blur0.4_bicubic_noAA` wins on both criteria simultaneously** — it is the unique candidate that is
best-by-RMSE *and* best-by-leak. The leak sign flips between σ=0.3 (+0.055) and σ=0.5 (−0.056), bracketing
the true pre-blur at **σ ≈ 0.4 px** by linear interpolation of the zero-crossing.

Antialiased, area, Lanczos and nearest variants are all rejected with strongly negative leak — they
over-smooth relative to the truth.

### 7.3 Recovered pipeline

$$\text{LR} \;=\; \mathcal{D}_{\text{bicubic}}^{\downarrow 2,\,\text{noAA}}\Big(\; \big(\text{GT} * g_{\sigma=0.4}\big)\cdot \frac{\Gamma(L{\approx}35)}{L} \;+\; \mathcal{N}(0,\sigma_g^2) \;\Big), \qquad \sigma_g \sim \mathcal{U}(0,\,0.06)$$

with **no output clipping**, and per-image random `σ_speckle ∈ [0.14, 0.19]`, `σ_gauss ∈ [0, 0.06]`.

### 7.4 Was normalisation applied before or after degradation?

**Before.** Three independent lines of evidence:

1. GT is exactly [0,1] per image; **LR is not** (min −0.279, max +2.158, 3.36 % of pixels > 1). If min–max
   normalisation had been applied to the LR after degradation, LR would also be exactly [0,1].
2. The photometric gain/offset between LR and `D(GT)` is 1.00 / 0.00 — no rescaling was applied to LR to
   compensate for the noise, so the noise was added *into* an already-[0,1]-normalised signal.
3. The additive Gaussian component has a per-image σ of 0–0.06 that is **not** proportional to the image's
   own scale — which would be impossible to observe if normalisation followed noise injection.

So the generator was: `source photo → grayscale → 256² crop → per-image min–max to [0,1] → degrade → save`.

This matters: **the additive noise floor is in absolute [0,1] units**, so dark, low-contrast images
(§5.1) have catastrophically worse SNR than bright ones — consistent with the observed −13.5 dB SNR minimum.

---

## 8. Dataset quality report

### Strengths

1. **Flawless integrity.** 6800/6800 arrays load; zero NaN, zero Inf, zero corruption, zero MD5 duplicates,
   zero missing pairs, zero shape anomalies. This is rare and removes an entire class of debugging.
2. **Perfectly regular structure.** One dtype, two shapes, fixed file sizes, contiguous zero-padded IDs.
   Enables memory-mapped packing (§10) and trivially correct indexing.
3. **Known target range.** GT ∈ [0,1] exactly, per image — a free, exploitable output constraint.
4. **Exact photometric consistency.** No brightness/gamma nuisance between input and target.
5. **Fully characterised, reproducible degradation** (§7.3) — enables unlimited synthetic augmentation from
   external clean data (DIV2K, Flickr2K, BSD) with a *distribution-matched* forward model.
6. **Wide content and exposure diversity** (per-image mean 0.016 → 0.959).

### Weaknesses

1. **Only ~2749 independent scenes**, not 3200 samples (§5.2).
2. **No official validation split.**
3. **34 near-featureless images** (σ<0.05) that distort mean-PSNR.
4. **Very low SNR** (median 7.1 dB) with a long bad tail (p10 = 0.7 dB) — some images are close to
   information-theoretically unrecoverable at fine scale.
5. **Small by modern SR standards** — 3200×256² ≈ 210 M pixels of target. A 20 M-parameter transformer will
   overfit without aggressive augmentation or external data.
6. Filesystem hygiene: doubled `train/train` nesting, a directory name containing a space and parentheses
   (`Test_NoisyLR (1)`), and 6800 booby-trapped artefact files.

### Data leakage

**Present and severe under the current splitting code.** See §5.2. Random splitting places near-identical
twins on both sides. Required fix before any model comparison is meaningful.

There is *no* leakage between train and the provided test set (0 MD5 collisions, and test content is
distributionally distinct).

### Class imbalance

Not applicable — regression task, no labels. The analogue, **content imbalance**, is real: smooth
low-texture content (sky, cloud, defocus) is over-represented in train relative to test (§5.3).

### Distribution shift / OOD risk

**Confirmed, quantified, and directional.** Test-LR has 1.13–1.16× higher high-frequency energy than
train-LR, KS D = 0.215 at p = 7.8e-15 on high-pass std. The shift is toward *harder* content.

OOD risks in priority order:
1. **Texture-frequency shift** (measured, above) — the model will be evaluated on denser texture than it
   was trained on.
2. **Noise-level extrapolation** — test Immerkær σ is 1.13× train's; the tail of the test noise
   distribution may exceed anything seen in training.
3. **Content-type shift** — test skews to man-made repetitive structure (brickwork, grids, tiling).
4. **Unknown final evaluation domain** — if the true competition set really is semiconductor imagery, this
   entire training set is out-of-domain and the shift is far larger than anything measured here.

### Generalisation risks

* Training on 2749 effective scenes with a high-capacity model ⇒ memorisation. Strong geometric
  augmentation and/or external clean data via the §7.3 synthesiser are close to mandatory.
* Over-fitting the specific speckle level: σ_speckle is tightly concentrated (0.140–0.194). A model trained
  only on this band may degrade sharply outside it. Widen the synthetic range deliberately.
* Over-fitting the specific downsampler: bicubic-noAA is a *particular* aliasing pattern. Blind-SR
  literature is unanimous that operator-specific models collapse off-operator.

### Potential preprocessing mistakes

1. **Applying the input normaliser to the target.** `semiconductor_dataset.py:127-129` runs the same
   `Preprocessor` over both input and target, and `configs/default_config.py:21` sets
   `normalization="percentile"`. This remaps a GT that is already exactly [0,1] into `(x−p1)/(p99−p1)`.
   Metrics then live in a shifted space, and predictions must be inverse-mapped before submission.
   **PSNR/SSIM must be computed against the original, unmodified GT.**
2. **Clamping the input to [0,1]** — would destroy 3.7 % of pixels. Don't.
3. **Not clamping the output** — leaves free PSNR on the table.
4. **Global (dataset-wide) mean/std normalisation** — poorly suited here, because per-image means span
   0.016–0.959. Per-image or no normalisation is more appropriate; the network should be given the raw
   signal plus, ideally, an explicit noise-level estimate.
5. **fp16 storage of GT** — safe in principle (fp16 resolution ≈ 5e-4 in [0,1], far below the 0.083 median
   noise floor), but must be verified by round-trip PSNR before adoption (§10).

### Potential augmentation mistakes

Examining `datasets/transforms.py:150-177` against the measured statistics:

| Current augmentation | Probability | Verdict |
|---|---|---|
| Extra Gaussian noise | 0.4 | ❌ Double-degrades an input that is already at 7 dB SNR |
| Extra speckle | 0.4 | ❌ Same |
| Extra Poisson | 0.2 | ❌ Same |
| Extra Gaussian blur σ∈[0.5,1.5] | 0.3 | ❌ Adds blur far beyond the measured σ≈0.4 |
| Extra motion blur | 0.2 | ❌ **No motion blur exists in this data at all** |
| Brightness ×U(a,b), input only | 0.3 | ❌❌ **Breaks the exact 1.00/0.00 photometric consistency — makes the mapping ill-posed** |
| Contrast, input only | 0.3 | ❌❌ Same |
| Gamma, input only | 0.2 | ❌❌ Same |
| H/V flip, rot90 (paired) | 0.5/0.2/0.5 | ✅ Correct and valuable |
| Random crop, patch=128 | 1.0 | ⚠️ **Silent no-op** — input is already 128×128, so `randint(0,0)=0` returns the full image |

The three photometric augmentations are the serious ones: they apply a transform `T` to the input while
leaving the target fixed, so the network is asked to learn `T⁻¹` from a signal that contains no information
about which `T` was applied. The four extra-degradation augmentations push the training input distribution
*away* from the test distribution rather than toward it.

The correct augmentation strategy given §7.3 is the opposite: **regenerate LR from GT** with the measured
forward model and randomised parameters, rather than piling further degradation onto the provided LR.

### Potential metric pitfalls

1. **Mean PSNR is dominated by the 34 near-flat images.** Report median and per-quartile PSNR.
2. **PSNR must be computed on the native GT scale** (`data_range = 1.0`), against un-normalised GT.
3. **SSIM parameters must be fixed and declared** (window 11, σ=1.5, K₁=0.01, K₂=0.03, `data_range=1.0`);
   values are not comparable across implementations otherwise.
4. **LPIPS is defined on 3-channel natural images.** Grayscale must be replicated to 3 channels, and the
   backbone (AlexNet vs VGG) must be declared — the two differ by a large margin.
5. **PSNR and LPIPS pull in opposite directions.** MSE-optimal outputs are over-smooth and score badly on
   LPIPS; perceptually sharp outputs lose PSNR. With 3.7 % of energy genuinely unrecoverable, this
   trade-off is unavoidable and must be managed explicitly via loss weighting, with the operating point
   chosen against the actual competition scoring formula.
6. **Leaky validation** (§5.2) invalidates all of the above regardless of how carefully the metrics are computed.

### Baselines to beat

Measured on 300 random train pairs:

| Predictor | PSNR (dB) |
|---|---|
| Per-image mean (degenerate) | 13.91 |
| Nearest ×2 | 20.38 |
| **Bicubic ×2** | **21.67** |
| **Oracle: noise-free LR → bicubic ×2** | **27.36** |

The 21.67 → 27.36 span (5.7 dB) is attainable by denoising alone. Everything above 27.36 dB requires
genuine high-frequency synthesis. This decomposition should drive the architecture's capacity budget.

---

## 9. Assessment of the current dataset structure

### Why it works

* Contiguous zero-padded IDs give O(1) indexing and a stem-based pairing rule that is trivially verifiable.
* Physical separation of `GT/`, `NoisyLR/`, and the test root maps cleanly onto a `Dataset` abstraction —
  the existing implementation is ~30 lines and correct.
* Uniform dtype and shape mean no per-sample branching, no ragged collation, no resize logic.
* Directory-per-role scales naturally to additional roles (e.g. a future `GT_val/`).

### Advantages

1. Human-inspectable; any single sample can be examined without tooling.
2. Git-LFS/DVC friendly at file granularity.
3. Zero coupling between the loader and any index/manifest format.
4. Adding or removing samples requires no rebuild step.

### Disadvantages

1. **6800 individual file opens per epoch.** On Windows/NTFS, per-file `open`/`stat` overhead dominates the
   cost of reading a 64 KiB array. This is the primary I/O bottleneck.
2. **6800 artefact files** that every consumer must remember to filter — a latent correctness bug.
3. **No manifest.** Splits live in code (`split_samples`), so they are not reproducible, not auditable, and
   cannot express the scene-grouping required to fix the leak in §5.2.
4. **No place for per-sample metadata** — noise-level estimates, scene group, texture stratum.
5. **Awkward paths**: doubled `train/train`, and `Test_NoisyLR (1)` contains a space and parentheses,
   which breaks naive shell interpolation and some config parsers.
6. **GT at float32 costs 4× the necessary storage** given the ~0.08 noise floor.
7. Train and test ID spaces overlap numerically — a flat cache keyed on ID silently collides.

### Maintainability and scalability

Acceptable at 3200 samples; degrades poorly. At 10× scale, 68 000 small files makes directory listing,
backup, and copy operations slow, and the missing manifest makes reproducing an experiment from six months
prior effectively impossible.

### PyTorch compatibility and dataloader efficiency

Compatible, and correctly implemented today. But: each `__getitem__` performs 2 `np.load` calls
(2 opens, 2 header parses, 2 allocations). With `num_workers=8`, `pin_memory=True` and
`persistent_workers=True` this is survivable at 128×128, yet it remains pure overhead — the entire training
set is 1.0 GB and fits in RAM/page-cache. Memory-mapped packing converts every read into a page-fault
against cached memory, typically **3–8× faster on Windows**, and eliminates worker startup cost entirely.

---

## 10. Proposed structure (PROPOSAL ONLY — nothing has been modified)

```
data/
├── raw/                          ← the current Data/ tree, copied or symlinked, NEVER modified
└── packed/
    ├── train_lr.npy              (3200,128,128) float16   memmap    105 MB
    ├── train_gt.npy              (3200,256,256) float16   memmap    420 MB
    ├── test_lr.npy               (400,128,128)  float16   memmap     13 MB
    ├── manifest.parquet          one row per sample:
    │                               id, split, stem, source_path, md5,
    │                               mean, std, p1, p99,
    │                               scene_group,              ← from cos>0.95 components
    │                               sigma_speckle_est, sigma_gauss_est, snr_est,
    │                               hp_std, texture_stratum
    └── splits/
        ├── blocked_v1.json       train/val by contiguous 32-ID blocks
        ├── group_v1.json         train/val by scene_group (leak-free, recommended)
        └── stratified_v1.json    group-safe + texture-stratified to match test distribution
```

### Why it is better

| Problem today | Fixed by |
|---|---|
| 6800 file opens/epoch | 3 memmaps; reads become page-faults on cached memory (3–8× faster on Windows) |
| `__MACOSX` trap | Eliminated by construction — the packer never emits them |
| Leaky random split (§5.2) | `scene_group` in the manifest + `group_v1.json` |
| Train/test frequency shift (§5.3) | `texture_stratum` enables a validation set that matches test |
| Near-flat outliers skew metrics (§5.1) | `std` in the manifest enables stratified metric reporting |
| Blind noise level (§6.2) | `sigma_*_est`, `snr_est` enable curriculum sampling and noise-conditioned models |
| Splits not reproducible | Declarative JSON, versioned, diffable |
| Awkward paths, ID collisions | Flat `split`-namespaced index |
| GT storage 4× larger than needed | float16 (verify first, see below) |

### float16 justification and safety gate

fp16 resolution in [0.5, 1.0] is 2⁻¹¹ ≈ 4.9×10⁻⁴, and finer below. The median per-image noise floor is
0.083 — **170× larger**. Quantisation error is therefore far below the noise. But this must be *verified*,
not assumed: the packer will compute round-trip PSNR over all 3200 GT arrays and **abort if it is below
70 dB**. If you prefer zero risk, keep GT at float32 for +420 MB — the I/O win comes from packing, not from
the dtype.

### Migration steps

1. `scripts/pack_dataset.py` — idempotent, read-only w.r.t. `raw/`:
   a. Discover files with the `__MACOSX` filter; **assert counts are exactly 3200/3200/400**.
   b. Record per-file MD5 of the source arrays.
   c. Write the three memmapped arrays in ID order.
   d. Re-read every packed slice and **verify MD5 (float32 path) or round-trip PSNR ≥ 70 dB (float16 path)**;
      abort on any mismatch.
   e. Compute per-sample statistics, scene-group components, and noise estimates → `manifest.parquet`.
   f. Emit the three split JSONs from the manifest.
2. `scripts/verify_pack.py` — standalone re-verification, runnable at any time.
3. `PackedRestorationDataset` alongside the existing class, selected by a config flag.

### Backward compatibility

`SemiconductorRestorationDataset` keeps its exact current constructor and behaviour. The new
`PackedRestorationDataset` returns an **identical** item contract — `{"input": Tensor, "target": Tensor,
"stem": str}` — so `trainer/`, `evaluation/` and `inference/` require no changes. A single config flag
(`dataset.backend: "files" | "packed"`) selects between them, and `raw/` remains the always-available
fallback. Rolling back is deleting `packed/`.

### Benefits

* **Training:** faster epochs, leak-free splits, stratified validation matched to test, metadata-driven
  sampling.
* **Inference:** batched memmap reads, deterministic ordering for submission assembly, no artefact-filter
  bug surface.
* **Scalability:** adding external clean data (DIV2K/Flickr2K synthesised via the §7.3 model) is a new
  shard plus manifest rows — no change to the loader.
* **Reproducibility:** every experiment references a split file by name and hash.

**Nothing above has been executed. Awaiting approval.**

---

## 11. Open question that gates Phase 2 and 3

The brief specifies semiconductor inspection imagery, and the data is natural photographs. This is not a
cosmetic mismatch — it changes concrete architectural decisions:

| Module in the requested pipeline | Justified by *this* data? |
|---|---|
| Frequency decomposition | **Yes**, but for noise/detail separation (speckle is signal-dependent and broadband), not for periodic-lattice recovery |
| Dedicated edge branch | **Partially** — edges are isotropic and natural, not Manhattan; oriented/axis-aligned kernels would not pay off |
| Texture / periodic-structure branch | **Weakly** — no periodic layouts exist; but test content *is* texture-heavy, so a texture branch is justified on different grounds |
| Heavy deblurring capacity | **No** — measured MTF droop is only 11 %; this capacity is nearly wasted |
| Heavy denoising capacity | **Yes, strongly** — 7 dB median SNR, 5.7 of the ~6+ dB headroom is denoising |
| Speckle-specific (log-domain / variance-stabilising) processing | **Yes, strongly** — `Var ∝ I²`, Γ(L≈35) |
| Blind noise-level conditioning | **Yes** — per-image σ varies 0.022–0.188 with no side information |
| Aliasing-aware upsampling | **Yes** — the operator has no antialias filter |

**Decision 1 — design target.**

* (a) Design for the data that is actually present (natural images, speckle-dominated, low SNR); or
* (b) Design for semiconductor imagery regardless, treating this set as a proxy; or
* (c) Design a core network validated on this data, with semiconductor-specific priors as switchable,
  separately-ablated modules.

My recommendation is **(c)**: the measured degradation physics (speckle, aliasing, blind noise level) is
almost certainly shared with the real target domain even if the *content* statistics are not, so a
degradation-driven core transfers while content-driven priors stay optional and testable.

**Decision 2 — approve, amend, or reject the packed dataset structure in §10.**

Phase 2 (literature review) begins only after both are settled.
