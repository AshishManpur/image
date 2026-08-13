# Phase 3.5 — Critical Architecture Review under a 4 GB Hardware Constraint

**Scope:** decide what actually gets built for the first competition model.
**Verdict up front:** the Phase 3 SPARC-B specification **does not fit the target hardware** and must be
revised. The revision is not cosmetic — it removes four modules, restructures the resolution schedule, and
cuts compute by 6.6× and training memory by 4.9×.

---

## 0. The error that forces this revision

Phase 3 §5 reported peak activation memory of "~239 MB at batch 8". **That figure was wrong by roughly
30×.** It counted about eight feature tensors *per level*. Autograd saves the output of essentially every
operation inside every *block* — a NAF block alone retains ~15 tensors of size `C·H·W`.

Recomputed properly (`reports/vram_budget.py`, fp16 activations):

| Model | Params | MACs | Activations/image | **B=4** | **B=8** |
|---|---|---|---|---|---|
| **Phase 3 SPARC-B** | 4.90 M | 15.95 G | **881 MB** | **3.52 GB** | **7.05 GB** |
| **Revised SPARC-Base** | **2.38 M** | **2.41 G** | **181 MB** | **0.72 GB** | **1.45 GB** |

The RTX A400 has 4 GB total, of which roughly 3.3–3.5 GB is usable once the driver and display surface are
accounted for. **Phase 3 SPARC-B cannot train at batch 4**, and at batch 2 it would be both memory-tight and
throughput-starved. This is a hard failure, not a tuning problem.

**Hardware planning assumptions** (entry-level Ampere workstation card; exact figures to be measured in
Phase 4, not assumed): 4 GB GDDR6 on a narrow bus, so **memory bandwidth — not FLOPs — is the binding
throughput constraint**. Tensor cores are present, so AMP is worthwhile. Planning number: **1.5–2.5 TFLOPS
sustained FP16** on a depthwise/elementwise-heavy workload. Every latency figure below is derived from that
assumption and is labelled as an estimate.

The bandwidth point matters for design: SPARC is full of depthwise convolutions, LayerNorms and elementwise
gates. These have excellent FLOP efficiency and *poor arithmetic intensity*. On a bandwidth-starved card
their wall-clock cost exceeds their FLOP share, which argues for fewer operations at high resolution rather
than narrower ones.

---

## 1. Module-by-module review

| Module | Why it exists | Phase 1 support | Expected gain | Cost (MAC / act) | Impl. difficulty | Verdict |
|---|---|---|---|---|---|---|
| **Per-image normalisation** | Means span 0.016–0.959 | Strong | +0.10–0.25 dB | ~0 / ~0 | Low | **CORE (simplified)** |
| **Noise-map conditioning** | `Var = σ_g²+σ_s²I²`, blind, σ varies 8.5× | **Very strong** | **+0.4–0.8 dB** | 0.013 G / 0.4 MB | Low–Med | **CORE** |
| **NAF blocks** | 84 % of headroom is denoising | Strong | baseline | dominant | Low | **CORE** |
| **Haar DWT/IDWT resampling** | Lossless at 7 dB SNR | Strong | +0.1–0.2 dB | ~0 / ~0 | Low | **CORE** |
| **DWT stem (new)** | 4× less compute *and* memory | Implied by 7 dB SNR | ~0 (neutral) | **−75 %** | Low | **CORE** |
| **U-Net hierarchy (3 levels)** | Receptive field, cheap context | Strong | baseline | structural | Low | **CORE** |
| **Gated skip fusion** | Noise level varies per image *and* spatially | Moderate | +0.05–0.15 dB | 0.038 G / 1.8 MB | Low | **CORE (simplified)** |
| **Sub-band reconstruction head** | 3.7 % of energy must be synthesised | Strong | +0.2–0.4 dB | 0.576 G / 48 MB | Low | **CORE (single path)** |
| **Global residual (bicubic ↑2)** | Network models only the correction | Strong | +0.2–0.5 dB (conditioning) | 0 | Trivial | **CORE** |
| **Output clamp [0,1]** | GT is exactly [0,1] | Exact | +0.02–0.05 dB | 0 | Trivial | **CORE** |
| **Global self-attention @32²/16²** | Test texture is repetitive, 1.16× HF | Moderate | +0.2–0.5 dB, best LPIPS | 0.51 G / 49 MB | **Medium** | **CORE, ablation-gated** |
| **LKA regional blocks** | Receptive-field gap at mid level | Weak (attention covers it) | +0.0–0.1 dB | 0.63 G / 20 MB | Low | **OPTIONAL** |
| **Cross-scale exchange** | Extra inter-level flow | **None** — U-Net skips already do this | **< 0.05 dB** | 0.24 G / 12 MB | Medium | **OPTIONAL** |
| **Median/MAD normalisation** | Heavy-tailed speckle | Overstated (see §7) | < 0.05 dB vs mean/std | 2 sorts of 16 k | Low | **OPTIONAL** |
| **Three sub-band branches** | Dedicated edge/texture paths | Weak (isotropy 0.939) | **< 0.1 dB** | +1.7 G / +90 MB | Medium | **EXPERIMENTAL** |
| **Soft data consistency** | Operator is known | Weak (see §7) | **0.0–0.15 dB** | 0.19 G at 256² | **High** | **EXPERIMENTAL** |
| **Content-adaptive routing** | ~30 % low-texture images | Weak | unknown | variable | High | **EXPERIMENTAL** |
| **LPIPS loss term** | LPIPS is scored | Moderate | LPIPS ↓, PSNR ↓ | training only | Low | **OPTIONAL (V1.1)** |

---

## 2. Classification

**CORE — build these, in this order.** Per-image normalisation · noise-map conditioning · NAF blocks ·
Haar DWT stem and resampling · 3-level U-Net · gated skip fusion · sub-band reconstruction head ·
global residual · output clamp · global attention at 32²/16² (last, behind a flag) ·
Charbonnier + MS-SSIM + wavelet + FFT + gradient loss.

**OPTIONAL — only after the CORE model has a validated number.** LKA blocks · cross-scale exchange ·
median/MAD normalisation · LPIPS term · self-ensemble TTA.

**EXPERIMENTAL — not in V1.** Soft data consistency · three separate sub-band branches ·
content-adaptive routing · native 256→512 mode · full unfolding.

---

## 3. Estimated cost/benefit

All gains are **hypotheses**, expressed relative to the CORE model without that module. Runtime is the
share of a 2.41 GMAC budget. These exist to be falsified by ablation, not cited as results.

| Module | ΔPSNR | ΔSSIM | ΔLPIPS | Runtime | VRAM (B=8) | Difficulty |
|---|---|---|---|---|---|---|
| Noise-map conditioning | **+0.4 … +0.8** | +0.005 | −0.005 | +0.5 % | +3 MB | ●●○ |
| Sub-band head vs. plain PixelShuffle | +0.2 … +0.4 | +0.004 | **−0.015** | +24 % | +390 MB | ●○○ |
| Global attention @32²/16² | +0.2 … +0.5 | +0.003 | **−0.020** | +21 % | +390 MB | ●●● |
| DWT resampling vs. strided conv | +0.1 … +0.2 | +0.002 | −0.003 | −2 % | −5 % | ●○○ |
| Per-image normalisation | +0.1 … +0.25 | +0.002 | −0.002 | +0.3 % | ~0 | ●○○ |
| Gated fusion vs. concat | +0.05 … +0.15 | +0.001 | −0.002 | +1.6 % | +14 MB | ●○○ |
| Output clamp | +0.02 … +0.05 | +0.001 | ~0 | 0 % | 0 | ●○○ |
| LKA blocks | +0.0 … +0.1 | ~0 | −0.002 | +26 % | +160 MB | ●○○ |
| Cross-scale exchange | **< 0.05** | ~0 | ~0 | +10 % | +96 MB | ●●○ |
| Data consistency | **0.0 … +0.15** | ~0 | ~0 | +8 % | +130 MB | ●●● |
| Three band branches | **< 0.1** | ~0 | −0.005 | +70 % | +720 MB | ●●○ |
| LPIPS term (0.03) | **−0.15 … −0.3** | −0.002 | **−0.03** | 0 % (infer) | +1.2 GB (train) | ●●○ |

The LPIPS row deserves emphasis: it *costs* PSNR. Whether it is worth adding depends entirely on the
competition's scoring formula, which should be confirmed before enabling it.

---

## 4–5. Three versions

### SPARC-Tiny — debugging

```
128²×1 →[norm]→[DWT stem]→ 64²×24 →1 NAF→ 32²×48 →1 NAF→ 16²×96 →2 NAF
                              ↓ decoder (concat skips, 1 NAF each) ↓
        64²×24 →conv→ IDWT→ 128²×16 →1 NAF→ conv→4 → IDWT → 256²×1  (+ bicubic residual)
```

| Params | MACs | Act/img | VRAM B=16 | Purpose |
|---|---|---|---|---|
| **0.295 M** | **0.236 G** | 21.5 MB | 0.34 GB | Overfit 8 images in < 2 min; pipeline debugging |

No attention, no noise head, no gated fusion. Every CORE interface is present so that upgrading to Base is
configuration, not rewriting.

### SPARC-Base — competition model (default)

```
 y (B,1,128,128) unclipped
 │
 ├─ per-image normalise (mean/std, invertible)                → ŷ
 ├─ noise head → (σ̂_g, σ̂_s);  σ̂ = √(σ̂_g² + σ̂_s²·box₅(y)²)/s → (B,1,128,128)
 ├─ concat[ŷ, σ̂] (B,2,128,128) ─ Haar DWT ─→ (B,8,64,64) ─ Conv3×3 ─→ (B,48,64,64)
 │
 │  L0  64²×48   4 × NAF ─────────────────────────────────── skip₀
 │      DWT ↓2 → (B,192,32,32) → 1×1 → 96
 │  L1  32²×96   4 × NAF + 2 × GSA (3 heads, d=48) ────────── skip₁
 │      DWT ↓2 → (B,384,16,16) → 1×1 → 160
 │  L2  16²×160  4 × NAF + 3 × GSA (5 heads, d=80)   ← bottleneck
 │      1×1 → 384 → IDWT ↑2 → (B,96,32,32)
 │  D1  32²×96   GatedFuse(skip₁) → 4 × NAF + 1 × GSA
 │      1×1 → 192 → IDWT ↑2 → (B,48,64,64)
 │  D0  64²×48   GatedFuse(skip₀) → 4 × NAF
 │
 ├─ HEAD  Conv3×3(48→128) → IDWT ↑2 → (B,32,128,128) → 3 × NAF
 │        → Conv3×3(32→4) → IDWT ↑2 → (B,1,256,256)
 ├─ + bicubic↑2(ŷ)            global residual
 ├─ × s + m                   de-normalise
 └─ clamp(0,1)                → (B,1,256,256)
```

| Params | MACs | GFLOPs | Act/img | VRAM B=8 | VRAM B=16 |
|---|---|---|---|---|---|
| **2.381 M** | **2.405 G** | 4.81 | 181 MB | **1.45 GB** | 2.89 GB |

**Recommended training point: batch 8, AMP fp16, no gradient checkpointing** — 1.49 GB including optimiser
state, leaving ~1.8 GB of headroom for fragmentation and the dataloader. Batch 16 fits at 2.94 GB but leaves
little margin; use it only if measurements confirm the headroom.

**Speed estimates** (from the 1.5–2.5 TFLOPS assumption; measure in Phase 4):
training ≈ 18.7 GFLOP/image ⇒ 60 TFLOP/epoch ⇒ **60–120 s/epoch** including data loading;
400 epochs ≈ **7–13 hours**. Inference ≈ 4.8 GFLOP/image ⇒ **~3–6 ms/image batched**, ~15–25 ms at batch 1
(launch-bound). The Phase 3 design would have been ~5× slower at a quarter of the batch size.

### SPARC-Large — research only

Phase 3 SPARC-B unchanged (4.90 M, 15.95 G, 881 MB/image) plus the EXPERIMENTAL modules: soft data
consistency, three sub-band branches, LKA blocks, cross-scale exchange. **Requires ≥ 12 GB to train at a
useful batch size. Not runnable on the target hardware.** It exists to test, on borrowed hardware, whether
any EXPERIMENTAL module justifies promotion.

---

## 6. SPARC-Base complete specification

Deferred to the standalone document — see **[FINAL_IMPLEMENTATION_SPEC.md](FINAL_IMPLEMENTATION_SPEC.md)**.

---

## 7. Is the architecture over-engineered? Yes, in four specific places.

**Are we solving problems the dataset does not have?**

Phase 3 already removed deblurring capacity (11 % MTF droop), motion-blur handling (absent), and
periodic-structure priors (natural images). Four remain:

1. **Soft data consistency solves a problem we do not have.** USRNet and DPIR earn their gains by
   *generalising across unknown operators*. Our operator is **fixed, known, and identical in train and
   test** — so a network trained on it has already internalised it. An explicit one-step correction is
   re-deriving something the weights encode. It also carries the highest debugging cost in the design
   (adjoint correctness, weight normalisation, λ dynamics, a second gradient path into the noise head).
   Phase 3 initialised λ = 0 precisely because the gain was uncertain. **Ship without it.**

2. **The three sub-band branches refute themselves.** Phase 3 Step 10 rejected dedicated edge/texture
   branches with the argument that *"the supervision, not the topology, is what matters"* — then built three
   branches anyway. The argument applies recursively: if the wavelet band loss supplies the band-specific
   gradient signal, a *single* head trained with that loss captures the same function. The branches cost
   +70 % head compute and +90 MB/image for a hypothesised < 0.1 dB. **Keep the loss, drop the branches.**

3. **Cross-scale exchange duplicates the U-Net skips.** A U-Net already routes information between scales;
   an extra 1×1 path between adjacent decoder levels adds a second, weaker copy of the same channel. No
   Phase 1 measurement motivates it.

4. **Median/MAD normalisation over-applies a correct argument.** The heavy-tail objection to mean/std is
   sound when *estimating noise* — and the noise head does exactly that. For *centring the input*, a 5 %
   error in the scale is irrelevant: the network sees the scale implicitly, and the global residual carries
   the DC term regardless. Two sorts of 16 384 elements per image is real bandwidth on a 96 GB/s-class card.
   **Use mean/std; keep median/MAD as an ablation.**

**Which modules make debugging significantly harder?** Data consistency (adjoint + λ + weighting, three
interacting failure modes), the three-branch head (band-energy balance), cross-scale wiring (silent shape
bugs), and relative-position-bias tables (the one attention detail that breaks on resolution change).

**Which are likely to yield < 0.1 dB?** Cross-scale exchange, data consistency, LKA given attention,
median/MAD versus mean/std, and separate band branches — five of the eighteen modules, together accounting
for roughly 40 % of Phase 3's compute.

---

## 8. Implementation roadmap

Ranked to minimise debugging time. The governing principle: **reach a trainable end-to-end model as early
as possible, then add one module at a time so every gain is attributable.**

| # | Step | Why here |
|---|---|---|
| 1 | Data layer, metrics, baselines | Nothing is measurable until bicubic reproduces **21.67 dB**. Catches split leakage and normalisation bugs before any model exists |
| 2 | Haar DWT/IDWT | Pure function, testable in isolation (`IDWT(DWT(x)) == x`). Every later stage depends on it |
| 3 | LayerNorm2d, SimpleGate, NAF block | Smallest unit with parameters; shape/gradient/param-count tests |
| 4 | **SPARC-Tiny end-to-end** (stem, 3 levels, concat skips, sub-band head, global residual, clamp) | **First trainable model.** Overfit 8 images to > 45 dB — proves capacity, gradient flow and the data path together |
| 5 | Charbonnier loss + training loop (AMP, EMA, cosine, checkpointing) | **First real PSNR number.** Establishes the reference every later module is measured against |
| 6 | Scale to SPARC-Base widths/depths | Config change only; confirms the memory model on real hardware |
| 7 | Per-image normalisation | Cheap, low-risk, isolated gain |
| 8 | **Noise head + auxiliary loss** | Highest expected gain. Deliberately placed *after* a validated baseline so the +0.4–0.8 dB is attributable |
| 9 | Gated skip fusion (replaces concat) | Small, isolated |
| 10 | Remaining loss terms, one at a time | Each is independently ablatable; adding them together makes regressions untraceable |
| 11 | **LR re-synthesis augmentation** | Phase 1 predicts the current augmentation is actively harmful; this is a data-side change with a potentially large effect |
| 12 | **Global attention at 32²/16²** | Highest implementation risk (rel-pos bias, SDPA, export). Added last so a failure here never blocks a submission |
| 13 | ONNX export + benchmark | Validates the deployment claim against measured latency |
| 14 | *(V2)* data consistency, band branches, LKA, cross-scale | Only if V1 is complete and the ablation budget allows |

**Why this order minimises debugging time.** Steps 1–5 produce a working, scoring model with roughly 15 % of
the eventual complexity. From step 6 onward every change is a single, reversible increment against a known
number, so any regression has exactly one candidate cause. The two highest-risk items (noise head, attention)
sit at 8 and 12, after the pipeline is trustworthy — the opposite of the Phase 3 ordering, which would have
had us debugging an adjoint operator before knowing whether the dataloader was correct.

---

## Summary of changes from Phase 3

| Change | Reason |
|---|---|
| **DWT stem: trunk starts at 64², not 128²** | 4× less compute and memory at the most expensive level; lossless, so no information is given up |
| **3 levels instead of 4** | 128² input with a DWT stem needs fewer; bottleneck already reaches 16² |
| **No convolution at 256²** | The head predicts sub-bands at 128² and upsamples once. Output-resolution work is pure bandwidth cost |
| **Single sub-band head, not three branches** | Phase 3's own "supervision not topology" argument |
| **Data consistency removed** | Fixed known operator ⇒ the network already encodes it |
| **Cross-scale exchange removed** | Duplicates U-Net skips |
| **LKA removed from Base** | Attention at 32² covers the same range |
| **mean/std instead of median/MAD** | Robustness argument applies to noise estimation, not centring |
| **Widths 48/96/160, params 4.90 M → 2.38 M** | 2749 effective scenes; smaller is also less overfitting surface |
| **MACs 15.95 G → 2.41 G, activations 881 → 181 MB/img** | Fits the 4 GB card at batch 8 with headroom |

**The architecture is not weaker for this.** Every removed module was hypothesised at < 0.15 dB, and the
resolution restructure is information-preserving. What was removed is complexity, not capability.
