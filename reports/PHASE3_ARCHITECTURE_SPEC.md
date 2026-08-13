# Phase 3 — Architecture Specification: **SPARC-Net**

**S**peckle-**P**rior **A**daptive **R**estoration with **C**onsistency

**Status:** design specification. No implementation. Phase 4 implements exactly this document.
**Provenance:** every decision traces to a Phase 1 measurement, a Phase 2 literature finding, or a
stated engineering trade-off. Parameter and FLOP figures are computed by `reports/budget_sparc.py`
(analytic layer arithmetic, no network constructed).

---

## 0. Specification summary

| | SPARC-S | **SPARC-B (default)** | SPARC-L |
|---|---|---|---|
| Widths `C1..C4` | 48/96/144/160 | **64/128/192/192** | 80/160/224/224 |
| Parameters | 2.75 M | **5.17 M** | 8.26 M |
| MACs (128²→256²) | 7.09 G | **14.90 G** | 26.94 G |
| GFLOPs | 14.2 | **29.8** | 53.9 |
| Peak activations, fp16, B=1 | ~19 MB | **~30 MB** | ~44 MB |
| Exact-MHSA matmul share | 7.0 % | **4.4 %** | 2.8 % |

For reference, Restormer is 26.1 M params / ~141 GMAC at 256², and NAFNet-32 is 17.1 M / ~16 GMAC at
256². SPARC-B does the harder 128→256 task at **1/10 of Restormer's compute and 1/5 of its parameters**.

---

## STEP 1 — Design philosophy

The architecture follows one principle: **the network's structure should mirror the measured structure of
the inverse problem.** Phase 1 established that this problem is 84 % denoising and 16 % high-frequency
synthesis, that the noise is multiplicative and per-image blind, and that the forward operator is known.
Each module below exists to address one of those measured facts, and each is paired with the simpler
alternative it displaces.

| # | Module | Problem it solves | Simpler alternative rejected — and why |
|---|---|---|---|
| 1 | **Robust per-image normalisation** | Per-image means span 0.016–0.959 and stds 0.016–0.451 (P1 §4). An un-normalised network wastes capacity modelling exposure. | *Global dataset mean/std*: a single global affine cannot centre images whose means differ by 60×. *No normalisation*: leaves a nuisance dimension in the input manifold. |
| 2 | **Noise-map generator (2-parameter, blind)** | `Var(r\|I) = σ_g² + σ_s²I²`, with σ_s ∈ [0.14, 0.19] and σ_g ∈ [0, 0.06] varying per image and **unsignalled** (P1 §6.2). | *Fixed-σ network*: fails across a 8.5× per-image σ range. *Implicit estimation inside the trunk*: forces the backbone to spend depth on a quantity we can compute in closed form (FFDNet's result, P2 §2.1). |
| 3 | **NAF local blocks** | 84 % of headroom is denoising; noise is white so it is primarily a local-statistics problem. | *ConvNeXt / RDB / dynamic conv*: see Step 7 — all lose on the accuracy-per-FLOP and stability axes. |
| 4 | **Haar DWT / IDWT resampling** | Strided conv and pooling discard information before the network has denoised it; at 7 dB SNR that loss is unrecoverable. DWT is lossless, orthogonal and parameter-free. | *Strided conv*: lossy and costs parameters. *Max-pool*: lossy and noise-selecting (picks the noisiest pixel of each 2×2). |
| 5 | **LKA regional blocks (level 2)** | Bridges the receptive-field gap between 3×3 local blocks and level-3 attention, at CNN cost and with CNN inductive bias — valuable with 2749 effective scenes. | *More 3×3 blocks*: receptive field grows too slowly. *Attention at 64×64*: 3.2 GMAC/block, 5× the level-3 cost (P2 §3.1). |
| 6 | **Exact spatial MHSA at ≤32×32 only** | Test split is repetitive man-made texture at 1.16× train's HF energy (P1 §5.3); non-local patch matching is how you denoise repetitive texture without blurring it. Costs 4.4 % of total MACs (P2 §3.1). | *Mamba/SSM*: solves a cost problem we do not have and blocks ONNX/TensorRT (P2 §2.4). *Channel attention alone*: cannot match distant patches (P2 §3.2). *Full-res attention*: 25.8 GMAC at level 1. |
| 7 | **Gated skip fusion** | Optimal encoder/decoder mix is input-dependent because the noise level is input-dependent. A static concat cannot express "trust the skip more when this image is clean". | *Concatenation*: fixed mixing. *Addition*: fixed and lossier. |
| 8 | **Sub-band Reconstruction Module (SRM)** | Realises the edge (12 %) and texture (14 %) functions as three band-specialised paths predicting the Haar sub-bands of the output, at 128² rather than 256² — 4× cheaper than post-upsample branches. | *Two deep parallel branches at 256²*: 4× the FLOPs for the same function (Step 10). *Single unified head*: no explicit gradient signal separating edge from texture. |
| 9 | **Soft, noise-weighted data consistency** | The forward operator is **known** (P1 §7.3). Almost no restoration backbone can use this. | *Hard projection*: at 7 dB SNR it re-injects the noise it is meant to remove. *No consistency*: discards a free physical constraint that also improves OOD behaviour. |
| 10 | **Global residual from bicubic ×2** | The network then models only the noise correction plus the 3.7 % missing band, not the whole image. | *Direct prediction*: worse conditioning; the DnCNN residual result is the most reproduced finding in denoising. |

---

## STEP 2 — Complete network diagram

```
 y : (B,1,128,128)  float32, UNCLIPPED  (3.36 % of pixels > 1, 0.30 % < 0 — P1 §4.2)
 │
 ├─ STAGE 0 ── ROBUST NORMALISATION  (parameter-free, invertible)
 │     m = median(y)                                   per-image scalars
 │     s = 1.4826 · MAD(y) , clamped to [0.02, ∞)      stored for inversion
 │     ŷ = (y − m) / s
 │
 ├─ STAGE 1 ── NOISE-MAP GENERATOR                      (0.021 M par | 0.026 G)
 │     Î   = box₅(y)                            local clean-intensity proxy
 │     (σ̂_g, σ̂_s) = NoiseHead(ŷ)                4 strided conv → GAP → MLP → softplus
 │                                              init to P1 medians (0.024, 0.165)
 │     σ̂  = sqrt(σ̂_g² + σ̂_s²·Î²) / s           per-pixel map, normalised units
 │     ── aux supervision available: true σ computable from GT during training
 │
 ├─ STAGE 2 ── SHALLOW STEM      concat[ŷ, σ̂] → Conv3×3(2→C1)      (B,64,128,128)
 │
 ╞═ ENCODER ══════════════════════════════════════════════════════════════════════
 │
 │  L1  128×128×64   ── 4 × NAF                                    ──┐ skip₁
 │       │ Haar DWT ↓2   (64→256, lossless)  →  1×1 (256→128)       │
 │  L2   64×64×128   ── 4 × NAF  →  2 × LKA (≈21×21 ERF)           ──┤ skip₂
 │       │ Haar DWT ↓2   (128→512)           →  1×1 (512→192)       │
 │  L3   32×32×192   ── 3 × NAF  →  2 × GSA (exact MHSA, 1024 tok) ──┤ skip₃
 │       │ Haar DWT ↓2   (192→768)           →  1×1 (768→192)       │
 │  L4   16×16×192   ── 4 × GSA (exact MHSA, 256 tokens)            │
 │       BOTTLENECK — global receptive field, 0.245 GMAC total      │
 │                                                                  │
 ╞═ DECODER ═══════════════════════════════════════════════════════════════════════
 │       │ 1×1 (192→768) → Haar IDWT ↑2                             │
 │  D3   32×32×192  ←── GatedFuse(skip₃, ·) ←──────────────────────┤
 │                   ── 1 × GSA  →  2 × NAF                         │
 │       │ 1×1 (192→512) → IDWT ↑2                                  │
 │  D2   64×64×128  ←── GatedFuse(skip₂, ·) ←──────────────────────┤
 │                   ── 4 × NAF          ←── CrossScale(D3↑)        │
 │       │ 1×1 (128→256) → IDWT ↑2                                  │
 │  D1  128×128×64  ←── GatedFuse(skip₁, ·) ←──────────────────────┘
 │                   ── 4 × NAF          ←── CrossScale(D2↑)
 │
 ├─ STAGE 8 ── SUB-BAND RECONSTRUCTION MODULE  (at 128², predicts 256² sub-bands)
 │       shared trunk: 3 × NAF @128²×64
 │       ├── LL  "structure" path : 1×1→32 , 2 × NAF , Conv3×3→1   ← low band
 │       ├── LH+HL "edge" path    : 1×1→48 , 4 × NAF , Conv3×3→2   ← oriented edges
 │       └── HH  "texture" path   : 1×1→48 , 4 × NAF , Conv3×3→1   ← fine texture
 │       stack (LL,LH,HL,HH) : (B,4,128,128)
 │
 ├─ STAGE 9 ── ORTHOGONAL IDWT ×2  (fixed Haar; ≡ PixelShuffle ∘ orthogonal 4×4 mix)
 │       → x̂₀ : (B,1,256,256)   in normalised units
 │
 ├─ STAGE 10 ── GLOBAL RESIDUAL      x̂₀ ← x̂₀ + bicubic↑2(ŷ)
 │
 ├─ STAGE 11 ── SOFT DATA CONSISTENCY  (one step, λ initialised to 0)
 │       A(x)  = bicubic↓2( g_{σ=0.4} * x )          ← P1-recovered operator
 │       r     = A(x̂₀) − ŷ                          (B,1,128,128)
 │       w     = (σ̂⁻²) / mean(σ̂⁻²)                  noise-weighting
 │       b     = g_{σ=0.4} * bicubic↑2( w ⊙ r )      surrogate adjoint Aᵀ
 │       x̂     = x̂₀ − λ · b  +  Refine([x̂₀, b, σ̂↑2])   3 convs, 16 ch
 │
 ├─ STAGE 12 ── DE-NORMALISATION      x = x̂ · s + m
 │
 └─ STAGE 13 ── CLAMP to [0,1]        (GT is exactly [0,1] for all 3200 images — P1 §4.1)

 OUTPUT : (B,1,256,256)
```

---

## STEP 3 — Tensor shapes

| # | Operation | Input | Output |
|---|---|---|---|
| 0 | Robust normalisation | `(B,1,128,128)` | `(B,1,128,128)` + `(m,s)` |
| 1 | Noise head (4 strided conv → GAP → MLP) | `(B,1,128,128)` | `(B,2)` |
| 1b | σ-map assembly | `(B,1,128,128)` | `(B,1,128,128)` |
| 2 | Stem Conv3×3(2→64) | `(B,2,128,128)` | `(B,64,128,128)` |
| 3 | Enc L1 · 4 × NAF | `(B,64,128,128)` | `(B,64,128,128)` → skip₁ |
| 4 | Haar DWT ↓2 | `(B,64,128,128)` | `(B,256,64,64)` |
| 5 | 1×1 proj | `(B,256,64,64)` | `(B,128,64,64)` |
| 6 | Enc L2 · 4 × NAF + 2 × LKA | `(B,128,64,64)` | `(B,128,64,64)` → skip₂ |
| 7 | Haar DWT ↓2 → 1×1 | `(B,128,64,64)` | `(B,192,32,32)` |
| 8 | Enc L3 · 3 × NAF + 2 × GSA | `(B,192,32,32)` | `(B,192,32,32)` → skip₃ |
| 9 | Haar DWT ↓2 → 1×1 | `(B,192,32,32)` | `(B,192,16,16)` |
| 10 | **Bottleneck** · 4 × GSA (256 tokens) | `(B,192,16,16)` | `(B,192,16,16)` |
| 11 | 1×1 → IDWT ↑2 | `(B,192,16,16)` | `(B,192,32,32)` |
| 12 | GatedFuse(skip₃) · 1 × GSA + 2 × NAF | `2×(B,192,32,32)` | `(B,192,32,32)` |
| 13 | 1×1 → IDWT ↑2 | `(B,192,32,32)` | `(B,128,64,64)` |
| 14 | GatedFuse(skip₂) + CrossScale · 4 × NAF | `2×(B,128,64,64)` | `(B,128,64,64)` |
| 15 | 1×1 → IDWT ↑2 | `(B,128,64,64)` | `(B,64,128,128)` |
| 16 | GatedFuse(skip₁) + CrossScale · 4 × NAF | `2×(B,64,128,128)` | `(B,64,128,128)` |
| 17 | SRM trunk · 3 × NAF | `(B,64,128,128)` | `(B,64,128,128)` |
| 18a | LL path · 1×1→32, 2 × NAF, Conv3×3 | `(B,64,128,128)` | `(B,1,128,128)` |
| 18b | LH+HL path · 1×1→48, 4 × NAF, Conv3×3 | `(B,64,128,128)` | `(B,2,128,128)` |
| 18c | HH path · 1×1→48, 4 × NAF, Conv3×3 | `(B,64,128,128)` | `(B,1,128,128)` |
| 19 | Sub-band stack | 3 tensors | `(B,4,128,128)` |
| 20 | **Haar IDWT ×2** | `(B,4,128,128)` | `(B,1,256,256)` |
| 21 | Global residual + bicubic↑2 | `(B,1,256,256)` | `(B,1,256,256)` |
| 22 | Data consistency (A, Aᵀ, Refine) | `(B,1,256,256)` | `(B,1,256,256)` |
| 23 | De-normalise + clamp | `(B,1,256,256)` | `(B,1,256,256)` |

Compact resolution/channel trace:

```
128²×1 → 128²×2 → 128²×64 → 64²×128 → 32²×192 → 16²×192
                                  ↓ decoder ↓
       16²×192 → 32²×192 → 64²×128 → 128²×64 → 128²×4 → 256²×1
```

---

## STEP 4 — Parameter budget

Computed, not estimated (`reports/budget_sparc.py`). SPARC-B:

| Functional group | Params | par % | MACs | MAC % | Phase 2 target |
|---|---|---|---|---|---|
| Local restoration (NAF, all levels + SRM trunk) | 2.608 M | 50.5 % | 9.103 G | 61.1 % | 38 % |
| Context (LKA + GSA) | 1.602 M | 31.0 % | 2.067 G | 13.9 % | 15 % |
| Edge path (LH+HL) | 0.074 M | 1.4 % | 1.013 G | 6.8 % | 12 % |
| Texture path (HH) | 0.074 M | 1.4 % | 1.013 G | 6.8 % | 14 % |
| Structure path (LL) | 0.018 M | 0.4 % | 0.254 G | 1.7 % | — |
| Frequency (DWT/IDWT projections) | 0.559 M | 10.8 % | 0.545 G | 3.7 % | 8 % |
| SR tail (sub-band heads + IDWT) | 0.002 M | 0.0 % | 0.026 G | 0.2 % | 7 % |
| Fusion (gated skips + cross-scale) | 0.206 M | 4.0 % | 0.665 G | 4.5 % | — |
| Noise estimation | 0.021 M | 0.4 % | 0.026 G | 0.2 % | — |
| Data consistency | 0.003 M | 0.1 % | 0.192 G | 1.3 % | — |
| **Total** | **5.166 M** | | **14.904 G** | | |

### 4.1 Reconciliation with Phase 2 — a correction

The Phase 2 §6.2 percentages **cannot be satisfied on both the parameter and the FLOP axis
simultaneously**, and this is a structural property of hierarchical networks rather than a defect of the
design:

* **Parameters concentrate at low resolution.** Block cost scales as `C²`, and `C` doubles per level, so
  the bottleneck dominates parameters while contributing almost nothing to FLOPs (level 4 is 15.4 % of
  parameters but 1.6 % of MACs).
* **FLOPs concentrate at high resolution.** Block cost scales as `C²·HW`; halving resolution while doubling
  width leaves FLOPs flat, so the 128² levels dominate compute while being parameter-cheap (level 1 is
  2.4 % of parameters but 11.3 % of MACs).

A single percentage therefore cannot govern both axes. **The correct operationalisation is: allocate
resolution-bound functions (local restoration, edge, texture, SR) by FLOPs, and representation-bound
functions (global context) by parameters.** Phase 2 stated "capacity" without making this distinction; this
is the refinement.

Judged on the correct axis, the design lands well:

| Phase 1 physics split | Target | SPARC-B achieved |
|---|---|---|
| Denoising-serving compute (local + context + freq + fusion) | 84 % | **83.2 % of MACs** |
| HF-synthesis-serving compute (edge + texture + structure + SR tail) | 16 % | **15.5 % of MACs** |

The top-level 84/16 split derived from measured dB headroom is reproduced to within 0.8 points. The
finer Phase 2 sub-percentages are superseded by this two-axis analysis.

Two deviations are stated explicitly rather than hidden:

1. **Context is 31 % of parameters against a 15 % target.** Attention is parameter-dense. It is held to
   13.9 % of MACs, and low-rank attention (qkv projected to `C/2`, GDFN expansion 2 instead of 4) already
   halves what a Restormer/SwinIR-style block would cost. Further reduction would mean deleting the
   non-local mechanism that Phase 1 §5.3 justifies. With 2749 effective scenes this remains the largest
   overfitting surface in the model and is the subject of ablation A3.
2. **"Local restoration" over-attributes at 61 % of MACs**, because the encoder/decoder trunk at 128²/64²
   also serves edge and texture reconstruction. The tag is a module label, not a clean functional
   partition.

---

## STEP 5 — Computational analysis

| Quantity | SPARC-B, 128²→256², batch 1 |
|---|---|
| MACs | 14.90 G |
| FLOPs | 29.8 G |
| Parameters | 5.17 M (10.3 MB fp16) |
| Peak feature activations (fp16) | ~30 MB (B=1) · ~239 MB (B=8) · ~478 MB (B=16) |
| Largest materialised attention map | 16.8 MB (B=1) — **0 MB with SDPA/flash** |
| Exact-MHSA matmul share | 0.654 GMAC = 4.4 % |

### 5.1 Bottleneck analysis

| Rank | Bottleneck | Share | Nature | Mitigation |
|---|---|---|---|---|
| 1 | 128² NAF stacks (enc L1, dec D1, SRM trunk) | 31 % MACs | compute-bound, well-fused | Reduce block count (SPARC-S); this is the primary speed knob |
| 2 | 64² NAF stacks | 22 % MACs | compute-bound | — |
| 3 | Depthwise convolutions everywhere | ~8 % MACs, disproportionate wall-clock | **memory-bandwidth-bound** | Channels-last (NHWC) memory format; fuse LN+conv |
| 4 | Attention at 32² | 4.4 % MACs | fine at fp16 | SDPA / TensorRT MHA fusion; never materialise the map |
| 5 | Data consistency at 256² | 1.3 % MACs | bandwidth-bound at output res | Keep refine width at 16 channels |

**GPU utilisation.** The model is small and shallow-per-level, so at batch 1 it is launch-latency- and
bandwidth-bound rather than FLOP-bound; expect low utilisation. Throughput improves strongly with batching.
**Batch inference is the correct deployment mode** — the 400-image test set should be run at batch 16–32.

**Latency (estimates requiring measurement).** On a mid-range modern GPU at fp16, order 8–15 ms at batch 1
and 1.5–4 ms/image at batch 16–32; with TensorRT fusion, roughly 1.5–2.5× better than eager PyTorch. These
are extrapolations from FLOP counts and typical utilisation, **not measurements**, and must be benchmarked
in Phase 4 before any claim is made.

### 5.2 Export

Every operator has standard ONNX coverage: `Conv` (incl. grouped/depthwise), `LayerNormalization`, `Mul`,
`Add`, `Split`, `ReduceMean`, `MatMul`/`Softmax` (or fused `Attention`), `Resize` (bicubic), `Reshape`,
`Transpose`, `Clip`. The Haar DWT/IDWT are fixed 2×2 orthogonal transforms expressible as a stride-2 group
convolution and its transpose. **No custom kernels, no plugins** — this is the direct consequence of
rejecting Mamba in Phase 2.

### 5.3 The 256→512 case

The brief mentions a 256²→512² mode. Levels then become 256/128/64/32, and level-3 attention rises from
1024 to 4096 tokens — a 16× increase in that term (0.79 G → 12.7 G), which would dominate. Two options,
decided by measurement in Phase 4:

* **(a) Tiled inference** — process overlapping 128×128 tiles with 16 px overlap and blend. Keeps cost
  exactly linear, needs no retraining, but forfeits global context across tile boundaries.
* **(b) Native, with the attention level demoted** — run exact MHSA only at ≤32×32 (i.e. levels 4 and 5),
  and bilinearly interpolate the relative-position-bias table. Preserves global context; requires the bias
  interpolation path to be exported.

**(a) is the recommended default**; (b) is the quality option. This must not be left implicit — it is a
real gap in a fixed-resolution design.

---

## STEP 6 — Module-by-module specification

Throughout: `LN` = LayerNorm over channels; `SG(x)` = SimpleGate, `x₁ ⊙ x₂` after a channel split;
`dw_k` = depthwise `k×k`. All blocks are residual with a learnable per-block scale initialised to a small
value, which makes every block an identity at initialisation and is the main reason NAFNet-style stacks
train without warmup.

> **Note on the ΔPSNR / ΔSSIM / ΔLPIPS columns below:** these are **prior hypotheses**, stated so that the
> ablation plan in Step 14 can falsify them. They are not measurements and must not be reported as results.

### 6.1 Robust Normalisation (Stage 0)

*Purpose:* remove per-image exposure and contrast as nuisance variables.
*In:* `y (B,1,128,128)`. *Out:* `ŷ`, and `(m,s)` retained for exact inversion.
*Ops:* `m = median(y)`, `s = max(1.4826·MAD(y), 0.02)`, `ŷ = (y−m)/s`. No parameters, no normalisation
layers, no activation.
*Why median/MAD rather than mean/std:* the input is unclipped and contains heavy-tailed speckle
(excess kurtosis +13.9 at low intensity, P1 §6.2); mean/std are contaminated by exactly the noise we are
removing, whereas MAD has a 50 % breakdown point.
*Why not min–max:* the input min/max are pure noise outliers.
*Complexity:* O(HW). *Hypothesis:* +0.10–0.25 dB, mainly on the 87 extreme-exposure images (P1 §5.1).
*Failure case:* images with `MAD → 0` (the 7 near-featureless images with σ<0.03) — handled by the 0.02
clamp on `s`.

### 6.2 Noise-Map Generator (Stage 1)

*Purpose:* convert a blind problem into a conditional one.
*In:* `y`. *Out:* `σ̂ (B,1,128,128)`.
*Ops:* `Î = box₅(y)`; `NoiseHead` = 4 × [Conv3×3 stride 2 → LN → SG] (1→16→24→32→32) → GAP →
Linear(32→32) → Linear(32→2) → softplus, giving `(σ̂_g, σ̂_s)` per image, biases initialised so the outputs
start at the Phase 1 medians **(0.024, 0.165)**. Then

$$\hat{\sigma}(p) \;=\; \frac{1}{s}\sqrt{\hat{\sigma}_g^{2} + \hat{\sigma}_s^{2}\,\hat{I}(p)^{2}}$$

*Why the 2-parameter form and not 3:* the Phase 1 three-parameter per-image fits had `corr(b,c) = −0.90`
— `b` and `c` are collinear over the observable intensity range and cannot be separately identified. The
2-parameter fit `a + cI²` was well conditioned (P1 §7).
*Why a per-image scalar pair rather than a dense prediction:* Phase 1 measured that the per-image variation
is captured by two scalars; predicting a dense field invites the head to absorb signal structure.
*Auxiliary supervision:* during training GT is available, so `D(GT)` and hence the true per-pixel σ are
computable exactly — this head can be supervised directly. That is unusual and worth exploiting.
*Complexity:* 0.021 M par, 0.026 G MAC (0.2 %).
*Hypothesis:* **+0.4–0.8 dB** — the single highest-value module per parameter in the design.
*Failure case:* if test noise falls outside the training σ range, the softplus head extrapolates poorly;
mitigated by widening the synthesis range (Step 13).

### 6.3 NAF Block — local restoration operator

*Purpose:* the 38 % workhorse; local denoising.
*Ops:*
`x ← x + γ₁·[Conv1×1(C→2C) → dw₃(2C) → SG → SCA → Conv1×1(C→C)](LN(x))`
`x ← x + γ₂·[Conv1×1(C→2C) → SG → Conv1×1(C→C)](LN(x))`
where `SCA(z) = z ⊙ Conv1×1(GAP(z))` — no sigmoid, no hidden nonlinearity.
*Normalisation:* LayerNorm. *Activation:* none — SimpleGate replaces it.
*Complexity:* `≈ 7C²` params, `≈ (6C² + 18C)·HW` MACs.
*Hypothesis:* the dominant PSNR/SSIM contributor; little direct LPIPS effect.
*Failure case:* over-smoothing fine texture at high σ — the reason the SRM and attention exist.

### 6.4 LKA Block — regional context (level 2 only)

*Ops:* `attn = Conv1×1( dw₇^{d=3}( dw₅(x) ) )`; `x ← x + γ·(x ⊙ attn)`; then a gated FFN.
*Effective receptive field:* ≈21×21 at 64×64 = 42×42 input pixels, for ~0.08 M params per block.
*Why here and only here:* level 1 is too expensive, level 3 already has true attention. Level 2 is exactly
the gap where the receptive field must grow but attention is still 5× the level-3 price.
*Complexity:* 0.158 M par, 0.633 G MAC for 2 blocks.
*Failure case:* fixed, content-independent aggregation — cannot substitute for attention on self-similar
texture (P2 §2.6). Never used as the sole global mechanism.

### 6.5 GSA Block — exact global self-attention (levels 3, 4)

*Ops:* `LN → Conv1×1(C→3C/2) → dw₃(3C/2) → split(q,k,v) → heads → softmax(qkᵀ/√d + B_rel)v →
Conv1×1(C/2→C)`, residual; then `LN → GDFN`, residual, where
`GDFN = Conv1×1(C→2C) → dw₃(2C) → SG → Conv1×1(C→C)`.
*Low-rank choice:* qkv is projected to `C/2` rather than `C`, and GDFN expansion is 2 rather than 4. This
halves the parameter cost of attention versus a Restormer/SwinIR-style block. Justified by the §4.1
finding that attention was otherwise consuming 52–81 % of parameters, and by the 2749-scene overfitting
constraint.
*Position encoding:* learnable relative-position bias table, `heads × (2n−1)²` — 23.8 k params at 32×32.
Fixed input size makes this trivially exportable.
*Attention is exact and unrestricted* — no windows, no sparsity, no scan. At 1024 and 256 tokens this is
affordable (P2 §3.1) and is the only mechanism in the network that performs content-based non-local
matching.
*Complexity:* `≈ 5C²` params; `≈ 5C²·HW + 2·(HW)²·(C/2)` MACs.
*Hypothesis:* **+0.3–0.6 dB and the largest single LPIPS improvement**, concentrated on the repetitive
man-made texture that dominates the test split.
*Failure cases:* content with no self-similar support (unique, non-repeating structure) — attention
degenerates toward a learned low-pass. Also the main overfitting surface.

### 6.6 Gated Skip Fusion

*Purpose:* input-dependent mixing of encoder and decoder features.
*Ops:* `u = Conv1×1(2C→C)([e,d])`; local branch `L = Conv1×1(C/4→C)(Conv1×1(C→C/4)(u))`; global branch
`G = Conv1×1(C/4→C)(Conv1×1(C→C/4)(GAP(u)))`; `g = σ(L+G)`; output `g⊙e + (1−g)⊙d`.
*Why AFF-style (local + global) rather than SKNet:* SKNet's gate is purely global (from GAP), so it cannot
express "trust the skip in the flat sky region, distrust it on the noisy brickwork" within one image —
which is exactly the case here, because the noise is signal-dependent and therefore spatially varying.
*Complexity:* 0.173 M par, 0.430 G MAC for three skips.
*Hypothesis:* +0.1–0.2 dB; larger effect on OOD noise levels than on in-distribution PSNR.

### 6.7 Sub-band Reconstruction Module (SRM)

*Purpose:* realises the edge and texture functions with an explicit frequency interpretation.
*In:* `(B,64,128,128)`. *Out:* `(B,4,128,128)` = (LL, LH, HL, HH) of the 256² output.
*Structure:* shared trunk (3 NAF @64) → three paths, each `Conv1×1 → n × NAF → Conv3×3`:
LL (32 ch, 2 blocks), LH+HL (48 ch, 4 blocks), HH (48 ch, 4 blocks).
*Why the asymmetric widths:* LL is already well determined by the input (96.3 % of the target's energy is
below LR Nyquist), so it needs little capacity; LH/HL and HH carry the 3.7 % that must be synthesised.
*Supervision:* the wavelet loss term supervises each band directly, which is precisely the gradient signal
a dedicated edge branch would provide — obtained without duplicating a deep branch.
*Complexity:* 0.166 M par, 2.28 G MAC (15.3 %).
*Hypothesis:* +0.2–0.4 dB; the principal LPIPS contributor alongside attention.
*Failure case:* Haar's poor frequency selectivity can leave 2×2 blocking if the HH path is under-trained —
monitored by inspecting the HH residual during training.

### 6.8 Orthogonal IDWT ×2 upsampler

*Purpose:* the SR tail. Mathematically **PixelShuffle composed with a fixed orthogonal 4×4 mixing matrix**
(the Haar synthesis matrix). Parameter-free, exactly invertible, no checkerboard artefacts by construction.
*Complexity:* 0 params, 0.026 G MAC including the sub-band heads.

### 6.9 Soft Data-Consistency Module

Specified fully in Step 11. 0.003 M par, 0.192 G MAC.

---

## STEP 7 — Local operator selection

Phase 2 nominated the NAF block but explicitly required this comparison rather than an automatic choice.

| Operator | Params @C | MACs @C,HW | Speckle | Low SNR | Small data | Stability | Export | Verdict |
|---|---|---|---|---|---|---|---|---|
| **NAF block** | `7C²` | `6C²·HW` | ●●● | ●●● | ●●● | ●●● | ●●● | **Selected** |
| ConvNeXt V2 | `~9C²` | `8C²·HW` | ●● | ●● | ●● | ●● | ●●● | Rejected |
| RepLKNet block | `~8C² + k²C` | `6C²·HW + k²C·HW` | ●● | ●● | ●●● | ●●● | ●● | Rejected as the *primary* operator |
| Residual Dense Block | `~4·g·C + …` | high | ●●● | ●●● | ● | ●● | ●●● | Rejected |
| Dynamic convolution | `n·C²·k²` | `C²k²·HW` | ●● | ● | ● | ● | ● | Rejected |
| Deformable convolution | `~C²k² + offsets` | high | ● | ● | ● | ● | ✗ | Rejected |

**Reasoning against each:**

* **ConvNeXt V2.** GRN normalises by global feature statistics — with per-image means spanning 0.016–0.959
  and signal-dependent noise, that global statistic is itself noise-contaminated. GELU also reintroduces the
  nonlinearity NAFNet demonstrated is unnecessary, at ~30 % more cost for no measured restoration gain.
* **RepLKNet block.** Excellent inductive bias, but a 31×31 depthwise kernel is memory-bandwidth-bound and
  its wall-clock cost far exceeds its FLOP count. It is retained in a *reduced* form (LKA, level 2) where
  receptive field is the binding constraint, not as the primary operator.
* **Residual Dense Block.** Dense concatenation grows channels linearly with depth; excellent at fitting,
  poor at generalising from 2749 scenes, and the growth pattern is awkward to export cleanly.
* **Dynamic convolution.** Predicting per-pixel kernels from a 7 dB SNR input means predicting kernels from
  noise. The kernel-prediction branch sees the same corrupted signal and has no privileged information —
  this is the specific reason to reject it *here* rather than in general.
* **Deformable convolution.** Offsets are learned from noisy input; `grid_sample` is poorly supported in
  TensorRT; training is unstable at low SNR. Fails the deployment requirement outright.

**Decision: NAF block**, on three measured grounds — best accuracy per FLOP (P2 §2.2), most stable training
(no activation nonlinearity, identity-initialised residual scales), and trivial export. The SCA inside it
supplies channel recalibration for free.

---

## STEP 8 — Global context selection

| Candidate | Non-local matching | Cost @32²,C=192 | Export | Verdict |
|---|---|---|---|---|
| **Exact MHSA (low-rank qkv)** | **Yes, unrestricted** | **0.40 G/block** | ●●● | **Selected** |
| Sparse / dilated attention (ART) | Yes, subsampled | ~0.2 G | ●● | Rejected — saves 0.2 G on a 14.9 G budget |
| Shifted-window (Swin) | Within window only | ~0.15 G | ● | Rejected — export-hostile, weaker |
| Token dictionary (ATD) | Via codebook | ~0.1 G | ●● | Deferred to ablation A3c |
| Cross-scale attention | Between levels | moderate | ●● | Partially adopted as CrossScale exchange |
| Mamba / SSM | Along scan only | low | ✗ | Rejected (P2 §2.4) |

**Decision: exact MHSA at levels 3 and 4 only.**

The entire justification is the cost calculation. At 32×32 with low-rank `d = C/2 = 96`, attention costs
`2·1024²·96 = 0.20 GMAC` of matmul per block; across all seven GSA blocks the attention matmuls total
**0.654 GMAC = 4.4 % of the model**. Every approximation scheme — sparsity, windows, dictionaries, state
spaces — exists to reduce a term that is already 4.4 % of our budget. Approximating it would trade measured
accuracy for unmeasurable savings, while adding export risk.

**Where attention is forbidden:** levels 1 and 2. At 128² the same block would cost 25.8 GMAC — larger than
the entire network. The rule is *where*, not *whether*.

**Heads:** 6 at both levels (head dim 16 at level 3, 16 at level 4). **Position:** learnable relative bias.

---

## STEP 9 — Multi-scale fusion

| Mechanism | Expressiveness | Cost | Verdict |
|---|---|---|---|
| Concatenation + conv | Fixed mixing | low | Rejected — cannot adapt to per-image noise level |
| SKNet | Global (GAP) gate | low | Rejected — cannot vary the gate spatially |
| **AFF (local + global gate)** | **Per-channel and per-pixel** | **low** | **Selected** |
| Cross-attention between scales | Very high | high | Rejected — cost at 128² is prohibitive |
| Learnable scalar weights | One scalar per skip | ~0 | Rejected — too coarse; retained as ablation baseline |

**What is retained, suppressed, and how information flows.** The encoder skip carries high-resolution
detail *and* the noise that has not yet been removed; the decoder path carries denoised but
lower-resolution content. The optimal mix is therefore a function of the local noise level, which Phase 1
established is **spatially varying** (because `σ ∝ I`) and **per-image varying** (σ_s ∈ [0.14, 0.19]).
A gate driven only by a global pooled statistic (SKNet) cannot express the spatial part. AFF's dual local +
global branch can, at 0.173 M parameters total.

**Cross-scale exchange** is deliberately minimal — a single 1×1 projection of the upsampled coarser
decoder level into the finer one, between adjacent levels only. Full HRNet-style all-pairs exchange was
rejected: at 128² each additional cross-scale path costs more than the entire attention budget.

---

## STEP 10 — Edge and texture strategy

**Decision: no separate deep branches. Band-specialised paths inside a shared sub-band module.**

The brief requests dedicated edge and texture branches. The evidence says the *function* is needed but the
*two-parallel-deep-branch* realisation is not:

1. **Cost.** A dedicated branch operating at output resolution costs 4× a branch at 128² for identical
   function. Placing the split before the upsampler and predicting sub-bands captures the same
   specialisation at a quarter of the compute.
2. **The oriented-kernel prior does not apply.** Phase 1 measured gradient isotropy of **0.939** — edges
   here are natural and isotropic, not Manhattan. A dedicated edge branch built around oriented or
   axis-aligned kernels has no structural prior to exploit. (This was the specific reading corrected in
   Phase 1 §6.1: the axis-aligned FFT cross was boundary leakage, not content.)
3. **The energy budget does not justify it.** Edge and texture together address the 3.7 % of energy above
   LR Nyquist — worth ~16 % of the dB headroom. Two deep parallel branches would be a large fraction of the
   model for that share.
4. **Multi-branch designs have been superseded.** MPRNet/HINet-style multi-path architectures cost ~2×
   inference for gains that single-stage designs have since matched (P2 §2.2).
5. **The supervision, not the topology, is what matters.** The benefit usually attributed to an edge branch
   is the explicit gradient signal on edges. The wavelet band loss supplies exactly that signal to the
   LH/HL heads without a second deep branch.

**What is built instead.** Three shallow paths off a shared trunk, each predicting a distinct Haar
sub-band of the output: LL (structure), LH+HL (oriented edges), HH (texture). This *is* a branch structure —
it simply lives at 128² and has an explicit frequency meaning rather than a heuristic one, and the bands are
recombined by a fixed orthogonal transform rather than a learned fusion.

**This decision is falsifiable and is tested by ablation A7**, which compares it against (a) a unified
single head and (b) genuine dual branches at 256². If A7 shows dual branches win by more than the cost, the
decision reverses.

---

## STEP 11 — Soft data-consistency module

**Forward operator** (Phase 1 §7.3, identified by joint RMSE + structure-leakage minimisation):

$$A(x) \;=\; \mathcal{D}^{\downarrow 2,\,\text{noAA}}_{\text{bicubic}}\big( g_{\sigma=0.4} * x \big)$$

Implemented as a fixed 5×5 Gaussian depthwise convolution followed by `Resize(bicubic, antialias=False)`.
Zero parameters, fully differentiable.

**Backward correction.** The exact adjoint of a bicubic decimation is not a bicubic interpolation; we use
the standard surrogate `Aᵀ ≈ g_{σ=0.4} * bicubic↑2(·)`, which is the transpose of the blur composed with an
interpolation of matched support. **This is an approximation and is stated as such** — it is the same
surrogate used throughout the back-projection and unfolding literature, and the learnable `λ` absorbs the
resulting scale mismatch.

**Noise-aware weighting.** The residual `r = A(x̂₀) − ŷ` is not equally trustworthy across the image: at
7 dB median SNR it is dominated by noise, and by `Var ∝ I²` that noise is worst in bright regions.
Weighting by inverse variance,

$$w = \frac{\hat{\sigma}^{-2}}{\overline{\hat{\sigma}^{-2}}}, \qquad
b = A^{\top}(w \odot r), \qquad
\hat{x} = \hat{x}_0 - \lambda\, b + \mathrm{Refine}\big([\hat{x}_0,\, b,\, \hat{\sigma}^{\uparrow 2}]\big)$$

is exactly the Gauss–Newton step for a Gaussian likelihood with heteroscedastic variance — i.e. the
weighting is derived from the measured noise model, not chosen heuristically.

**Placement.** After the global residual, before de-normalisation. It must see the completed
reconstruction, and it must operate in normalised units so that `σ̂` and the residual share a scale.

**Gradient flow.** `A` and `Aᵀ` are fixed linear operators, so gradients propagate to `x̂₀` through both the
direct path and the correction path, and to the noise head through `w`. This gives the noise estimator a
second, physically-grounded training signal beyond its auxiliary loss.

**Why `λ` is initialised to 0.** The module starts as an exact identity, so it cannot destabilise early
training; `λ` grows only if the correction reduces the loss. This makes the module strictly non-harmful in
expectation and makes ablation A5 clean — if `λ` stays near zero, the module has falsified itself.

**Why one step, not an unfolded loop.** Three reasons: (i) at 7 dB SNR each projection re-injects noise, so
the fixed-point of hard iteration is *worse* than one damped step; (ii) `k` steps multiply inference cost by
roughly `k`, conflicting with the scored speed criterion; (iii) the trailing `Refine` block is a learned
correction that subsumes what additional analytic steps would contribute.

---

## STEP 12 — Reconstruction head

| Method | Params | Cost | Artefacts | Export | Verdict |
|---|---|---|---|---|---|
| **Orthogonal IDWT ×2 (Haar)** | **0** | **~0** | **none by construction** | ●●● | **Selected** |
| PixelShuffle + ICNR | `4C·C_out·k²` | low | none with ICNR | ●●● | Equivalent; retained as ablation A6 baseline |
| Nearest + Conv | moderate | 4× (works at 256²) | none | ●●● | Rejected — 4× cost |
| CARAFE | moderate | high | none | ●● | Rejected |
| Dynamic upsampling (DySample) | low | low | none | ●● | Deferred to ablation A6 |

**Decision: orthogonal Haar IDWT.**

The key observation is that **IDWT ×2 and PixelShuffle ×2 are the same operator up to a fixed 4×4 mixing
matrix.** PixelShuffle reshapes 4 channels into a 2×2 spatial block; Haar IDWT does the same after applying
an orthogonal 4×4 synthesis matrix. Choosing the orthogonal version costs nothing and buys three things:

1. **The four predicted channels acquire a defined meaning** (LL/LH/HL/HH), which is what makes the
   band-specialised SRM paths of Step 10 coherent rather than arbitrary.
2. **A frequency-band loss becomes directly applicable** to the pre-upsample tensor.
3. **Checkerboard artefacts are impossible by construction** — the orthogonal basis has no preferred
   sub-pixel position, so no ICNR initialisation trick is needed.

CARAFE and DySample predict content-adaptive upsampling kernels. That is valuable when upsampling must
resolve ambiguity — but here the ambiguity is only 3.7 % of the energy, and the kernel-prediction branch
would be predicting from a 7 dB SNR signal. The same objection as dynamic convolution (Step 7) applies.
A6 tests this rather than assuming it.

---

## STEP 13 — Failure analysis

| Failure mode | Why it occurs | Frequency in data | Mitigation |
|---|---|---|---|
| **Extremely low SNR** | 10 % of images are at ≤0.7 dB SNR (P1 §6.5); at that level fine detail is information-theoretically absent | ~10 % | Noise-map conditioning tells the network to fall back to smoothing rather than hallucinating; σ-stratified validation to monitor separately |
| **Novel textures (OOD)** | Test HF energy is 1.16× train, KS D=0.215 (P1 §5.3); attention may find no learned analogue | affects the whole test set | CNN-dominant design; LR re-synthesis augmentation with widened σ; texture-stratified validation |
| **Very smooth regions** | 34 images have σ<0.05; the denoiser may leave residual low-frequency texture (the classic DnCNN failure) | ~1 % of train | Global residual + data consistency both anchor the low frequencies; report **median** PSNR alongside mean |
| **Near-zero-MAD images** | Robust normalisation divides by a vanishing scale | 7 images | `s` clamped to ≥0.02 |
| **Noise level outside training range** | Softplus head extrapolates poorly | unknown on test | Widen synthetic σ range well beyond [0.14, 0.19]; monitor predicted σ̂ distribution on test as an OOD detector |
| **Attention overfitting** | 31 % of parameters, 2749 effective scenes | training-wide | Low-rank qkv already halves it; ablation A3; stochastic depth on GSA blocks if validation diverges |
| **Haar blocking in HH** | Poor frequency selectivity if the HH path is under-trained | possible | Band-wise loss weighting; visual monitoring of the HH residual |
| **Tile seams at 256→512** | Tiled inference forfeits cross-tile context | inference-time | 16 px overlap with cosine blending; or option (b) of §5.3 |
| **Aliasing not undone** | The operator has no antialias filter, so folded energy must be actively resolved | whole dataset | This is what the SRM's LH/HL/HH paths and the data-consistency step are for; a failure here shows as directional ringing |

---

## STEP 14 — Ablation plan

Every experiment: same data, same leak-free group split (P1 §5.2), same schedule, same seed set (3 seeds),
metrics reported as **mean and median** PSNR / SSIM / LPIPS plus latency, with a σ-stratified and a
texture-stratified breakdown.

| ID | Experiment | Hypothesis | Primary metric | Expected outcome |
|---|---|---|---|---|
| **A1** | Remove noise-map conditioning (drop σ̂ channel and the aux loss) | Blind conditioning is the highest-value module per parameter | PSNR, and PSNR vs. per-image σ̂ | **−0.4 to −0.8 dB**, degradation concentrated at the σ extremes |
| **A2** | Replace NAF with ConvNeXt V2 / RDB / RepLKNet block at matched FLOPs | NAF wins on accuracy-per-FLOP and stability | PSNR, wall-clock, training curve variance | NAF ≥ others by 0.1–0.3 dB at equal cost |
| **A3a** | Remove all GSA (attention) blocks | Non-local matching drives texture and LPIPS | LPIPS, PSNR on the texture-stratified subset | **−0.3 to −0.6 dB**, LPIPS worse, largest loss on repetitive texture |
| **A3b** | Full-rank qkv (`C` instead of `C/2`) | Low-rank loses little and reduces overfitting | val−train gap | ≤0.05 dB gain, larger overfitting gap |
| **A3c** | Replace exact MHSA with ATD token-dictionary attention | Approximation is unnecessary at 4.4 % cost | PSNR, latency | Exact ≥ approximate; negligible speed gain |
| **A4** | Replace gated fusion with concatenation, then with SKNet | Spatial gating matters because noise is spatially varying | PSNR at high σ | AFF > SKNet > concat, gap widening with σ |
| **A5** | Disable data consistency (`λ = 0`, no Refine) | Known-operator consistency is worth 0.3–1.0 dB | PSNR; also inspect learned `λ` | If `λ` converges near 0, **the module is falsified — remove it** |
| **A6** | IDWT vs. PixelShuffle+ICNR vs. DySample vs. CARAFE | The orthogonal reparameterisation is free and enables the band loss | PSNR, LPIPS, latency | IDWT ≈ PixelShuffle; both beat CARAFE on cost-adjusted score |
| **A7** | SRM band paths vs. unified head vs. true dual branches at 256² | Band specialisation captures the function at 1/4 the cost | LPIPS, cost-adjusted PSNR | Band paths ≥ unified; dual branches ≤ band paths per FLOP |
| **A8** | Haar DWT resampling vs. strided conv vs. avg-pool | Lossless resampling matters at 7 dB SNR | PSNR | DWT > strided conv > pool |
| **A9** | Remove robust normalisation; try global mean/std | Per-image statistics are a nuisance dimension | PSNR on the 87 extreme-exposure images | Robust > global > none |
| **A10** | LR re-synthesis augmentation vs. the current double-degradation augmentation | Phase 1 §8 predicts the current scheme is actively harmful | PSNR on test-like stratum | Re-synthesis wins clearly |
| **A11** | Loss ablation: drop each of MS-SSIM / wavelet / FFT / gradient / LPIPS | Each term targets a distinct measured deficit | its own metric | Each term improves its target metric; LPIPS term costs PSNR |
| **A12** | Model scale S / B / L | Diminishing returns given 2749 scenes | PSNR vs. latency Pareto | B on the knee; L overfits without more data |

**Ordering.** A1, A5, A10 first — they are the highest-variance hypotheses and A5/A10 can *remove* work.

---

## STEP 15 — Implementation roadmap

Each step is independently testable and gated on its own acceptance criteria. No step begins before the
previous one passes.

| # | Deliverable | Acceptance criteria |
|---|---|---|
| 1 | **Data layer** — packed memmap, manifest, group-aware split, LR re-synthesis | Round-trip MD5/PSNR verification; zero group overlap between train and val; re-synthesised LR statistically matches real LR (σ vs I curve within 5 %) |
| 2 | **Metrics and baselines** — PSNR/SSIM/LPIPS with declared parameters | Reproduces the Phase 1 baselines exactly: bicubic 21.67 dB, nearest 20.38 dB |
| 3 | **Haar DWT/IDWT** | `IDWT(DWT(x)) == x` to 1e-6; orthogonality check; shape tests |
| 4 | **Noise-map generator** | Predicted σ̂ correlates > 0.9 with the analytic σ on held-out data |
| 5 | **NAF / LKA / GSA blocks** | Shape, gradient-flow, identity-at-init, and FLOP-count tests; measured params match §4 within 2 % |
| 6 | **Encoder + gated fusion + decoder** | End-to-end shape test; overfit 8 images to > 45 dB (capacity sanity check) |
| 7 | **SRM + IDWT reconstruction head** | Band outputs have the expected energy distribution; no checkerboard in a constant-input test |
| 8 | **Data-consistency module** | With `λ=0` output is bit-identical to the no-DC path; adjoint test `⟨Ax,y⟩ ≈ ⟨x,Aᵀy⟩` within the stated surrogate tolerance |
| 9 | **Composite loss** | Each term computed and logged separately; gradient magnitudes within 2 orders of magnitude of each other |
| 10 | **Training pipeline** — AMP, EMA, cosine schedule, checkpointing | Deterministic resume; a 200-step run reproduces to 1e-4 |
| 11 | **Full training + ablations A1–A12** | Per §14 |
| 12 | **Evaluation and submission assembly** | Stratified reporting; ID-order-correct output |
| 13 | **ONNX export** | Numerical parity with PyTorch within 1e-3; opset ≥ 17 |
| 14 | **TensorRT / fp16 optimisation** | Parity within 1e-2; measured latency at batch 1 / 8 / 16 / 32 |

### Loss composition (specified here, implemented at step 9)

$$\mathcal{L} = \mathcal{L}_{\text{Charb}} + 0.15\,\mathcal{L}_{\text{MS-SSIM}} + 0.10\,\mathcal{L}_{\text{wavelet}} + 0.05\,\mathcal{L}_{\text{FFT}} + 0.05\,\mathcal{L}_{\text{grad}} + 0.02\,\mathcal{L}_{\sigma} \;[\; +\; 0.03\,\mathcal{L}_{\text{LPIPS}}\;]$$

| Term | Weight | Why it exists — traced to a measurement |
|---|---|---|
| Charbonnier (ε=1e-3) | 1.00 | Primary fidelity; robust to the heavy-tailed speckle residual (excess kurtosis +13.9 at low I) that would give L2 outsized gradients |
| MS-SSIM | 0.15 | SSIM is scored directly; MS-SSIM is its differentiable multi-scale surrogate |
| Wavelet band | 0.10 | Supplies the band-specific gradient signal that replaces a dedicated edge branch (Step 10); weighted toward LH/HL/HH |
| FFT amplitude | 0.05 | L1/L2 training systematically under-predicts high frequencies (spectral bias); this directly penalises the missing 3.7 % band. **A loss, not a layer** (P2 §2.5) |
| Gradient (Sobel L1) | 0.05 | Sharpness at edges, where denoising does most of its damage at 7 dB |
| Noise-map aux | 0.02 | Supervises the σ head against the analytically computable true σ — available because GT is present in training |
| LPIPS (AlexNet, gray→3ch) | 0.03, **last 15 % of training only** | LPIPS is scored, but with 3.7 % of energy genuinely absent, a large weight buys perceptual score by fabricating detail and costs PSNR/SSIM (P2 §8) |

---

## Comparison with the reference architectures

| | Params | MACs (this task) | Blind σ | Non-local | Known-operator | Export | Data need |
|---|---|---|---|---|---|---|---|
| **SPARC-B** | **5.17 M** | **14.9 G** | **✓ explicit** | **✓ exact MHSA** | **✓ soft DC** | **✓ trivial** | **low** |
| Restormer | 26.1 M | ~35 G | ✗ | channel only | ✗ | moderate | medium |
| SwinIR | 11.8 M | ~50 G | ✗ | window only | ✗ | poor | medium |
| NAFNet-32 | 17.1 M | ~16 G | ✗ | ✗ | ✗ | trivial | low |
| HAT | 20.8 M | ~90 G | ✗ | window + OCA | ✗ | poor | **high** |
| Uformer-B | 50.9 M | ~25 G | ✗ | window only | ✗ | poor | medium |
| MambaIR | ~20 M | ~20 G | ✗ | scan only | ✗ | **blocked** | medium |

MACs for competitors are scaled estimates for a 128²→256² task from their published settings; treat as
±30 %. The three columns that matter for this dataset — blind σ conditioning, true non-local matching, and
exploitation of the known operator — are held by **none** of the reference architectures simultaneously.
That, and not raw capacity, is the basis for expecting SPARC-Net to win here.

---

**Phase 3 ends here. Awaiting approval before Phase 4 implementation.**

Open items requiring a decision:
1. Model scale: **SPARC-B (5.17 M)** recommended; S and L specified for the Pareto sweep.
2. The 256→512 mode: **tiled inference** recommended as default (§5.3).
3. Whether to run the full A1–A12 ablation suite or a reduced set (A1, A3a, A5, A7, A10 are the
   decision-critical ones).
