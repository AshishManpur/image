# SPARC-Base V1.0 — Implementation Contract

**Status:** FROZEN. This document is the engineering contract for Phase 4.
No architectural decisions remain open. Any deviation requires a written amendment.

**Task:** `(B,1,128,128)` grayscale → `(B,1,256,256)` grayscale.
**Hardware:** NVIDIA RTX A400, 4 GB VRAM.
**Budget:** 2.346 M params · 2.449 GMAC · 140.7 MB activations/image (fp16, SDPA).
**Table source:** `reports/contract_table.py` — all figures below are computed, not estimated.

> **Delta from Phase 3.5:** 2.381 M → **2.346 M** params, 2.405 → **2.449** GMAC,
> 181 → **140.7** MB/image. Cause: LayerNorm and LayerScale parameters are now counted
> explicitly, and the activation count assumes fused SDPA (which this contract mandates).
> No architectural change.

---

## PART 1 — Architecture diagram

```
INPUT  y : (B,1,128,128) float32, UNCLIPPED, values may exceed [0,1]
  │
  ├─────────────────────────────────────────────────────────────────────┐
  │                                                                     │ (y retained)
  ▼                                                                     │
[0] ROBUST NORMALISATION (parameter-free, invertible)                    │
    m = mean(y, dim=(1,2,3), keepdim)                                    │
    s = clamp_min(std(y, dim=(1,2,3), keepdim), 0.02)                    │
    ŷ = (y - m) / s                            (B,1,128,128)             │
  │                                                                      │
  │   ┌──────────────────────────────────────────────────────────────────┘
  │   ▼
  │  [1] NOISE HEAD
  │      Î = box5(y)                                    (B,1,128,128)
  │      trunk: 4 × [Conv3×3 s2 → LN2d → SimpleGate]
  │             1→32→(SG)16 →48→(SG)24 →64→(SG)32 →64→(SG)32
  │             128² → 64² → 32² → 16² → 8²
  │      GAP → Linear(32→64) → SimpleGate → Linear(32→2) → softplus
  │                                                     ⇒ (σ̂_g, σ̂_s) : (B,2)
  │      σ̂ = sqrt(σ̂_g² + σ̂_s²·Î²) / s     clamp[1e-4,2.0]  (B,1,128,128)
  │   ┌──┘
  ▼   ▼
[2] STEM   concat[ŷ, σ̂]                                 (B,2,128,128)
           HaarDWT ↓2                                   (B,8,64,64)
           Conv3×3(8→48)                                (B,48,64,64)
  │
╔═╪══ ENCODER ═══════════════════════════════════════════════════════════════════╗
  ▼
[3a] L0  64×64×48   ── NAF ×4 ────────────────────────────────────────┐ skip₀
  │                                                                   │ (B,48,64,64)
  ├─ HaarDWT ↓2 (48→192) → Conv1×1(192→96)                            │
  ▼                                                                   │
[3b] L1  32×32×96   ── NAF ×4 → GSA ×2 (h=3, d=48, hd=16) ───────────┐│ skip₁
  │                                                                  ││ (B,96,32,32)
  ├─ HaarDWT ↓2 (96→384) → Conv1×1(384→160)                          ││
  ▼                                                                  ││
[3c] L2  16×16×160  ── NAF ×4 → GSA ×3 (h=5, d=80, hd=16)   BOTTLENECK││
╚══╪══════════════════════════════════════════════════════════════════╪╪═════════╝
   │                                                                  ││
╔══╪══ DECODER ════════════════════════════════════════════════════════╪╪═════════╗
   ├─ Conv1×1(160→384) → HaarIDWT ↑2                (B,96,32,32)      ││
   ▼                                                                  ││
  [4a] GatedFuse(96)  ◄──────────────────────────────────────────────┘│
   │      u = Conv1×1(192→96)(concat[skip₁, dec])                      │
   │      g = sigmoid(Conv1×1(24→96)(Conv1×1(96→24)(GAP(u))))          │
   │      out = g*skip₁ + (1-g)*dec                  (B,96,32,32)      │
   ├─ GSA ×1 (h=3, d=48, hd=16) → NAF ×4             (B,96,32,32)      │
   │                                                                   │
   ├─ Conv1×1(96→192) → HaarIDWT ↑2                  (B,48,64,64)      │
   ▼                                                                   │
  [4b] GatedFuse(48)  ◄───────────────────────────────────────────────┘
   │      out = g*skip₀ + (1-g)*dec                  (B,48,64,64)
   ├─ NAF ×4                                         (B,48,64,64)
╚══╪══════════════════════════════════════════════════════════════════════════════╝
   ▼
[5] RECONSTRUCTION HEAD   ── no convolution ever runs at 256×256 ──
    Conv3×3(48→128)                                  (B,128,64,64)
    HaarIDWT ↑2                                      (B,32,128,128)
    NAF ×3  (C=32)                                   (B,32,128,128)
    Conv3×3(32→4)   ⇒ [LL, LH, HL, HH] of the output (B,4,128,128)
    HaarIDWT ↑2                                      (B,1,256,256)
   │
[6] OUTPUT
    x = x + bicubic_up2(ŷ)          global residual  (B,1,256,256)
    x = x * s + m                   de-normalise
    x = clamp(x, 0.0, 1.0)
   ▼
OUTPUT : (B,1,256,256) float32 in [0,1]
```

**Skip connections (complete list, there are exactly 2 long skips):**

| ID | Source | Destination | Shape | Fusion |
|---|---|---|---|---|
| `skip₀` | end of Enc L0 | GatedFuse before Dec D0 | `(B,48,64,64)` | GatedFuse(48) |
| `skip₁` | end of Enc L1 | GatedFuse before Dec D1 | `(B,96,32,32)` | GatedFuse(96) |
| global | `ŷ` (normalised input) | after head IDWT | `(B,1,256,256)` | addition |

Every NAF and GSA block additionally contains two internal residual connections.
**There are no concatenations anywhere except inside `GatedFuse`.**

---

## PART 2 — Module specifications

### 2.1 `LayerNorm2d(C, eps=1e-6)`
| | |
|---|---|
| Purpose | Channel-wise normalisation. BatchNorm is **forbidden**. |
| Input / Output | `(B,C,H,W)` → `(B,C,H,W)` |
| Operation | `(x - mean_C) * rsqrt(var_C + eps) * w + b`, moments over dim=1 |
| Params | `2C` (`w`, `b` shape `(1,C,1,1)`) |
| MACs | `2·C·H·W` |
| Init | `w = 1`, `b = 0` |

### 2.2 `SimpleGate()`
| | |
|---|---|
| Purpose | Replaces all activation functions. |
| Input / Output | `(B,2C,H,W)` → `(B,C,H,W)` |
| Operation | `x1, x2 = chunk(x, 2, dim=1); return x1 * x2` |
| Params | 0 |

### 2.3 `LayerScale(C, init=1e-2)`
| | |
|---|---|
| Purpose | Scales every residual branch; near-identity at init. |
| Params | `C`, shape `(1,C,1,1)`, initialised to `1e-2` |

### 2.4 `HaarDWT` / `HaarIDWT`
| | |
|---|---|
| Purpose | Lossless resampling and the SR upsampler. |
| DWT | `(B,C,H,W)` → `(B,4C,H/2,W/2)`, channel order `[LL, LH, HL, HH]` |
| IDWT | `(B,4C,H,W)` → `(B,C,2H,2W)` |
| Forward | `LL=(a+b+c+d)/2` · `LH=(a+b-c-d)/2` · `HL=(a-b+c-d)/2` · `HH=(a-b-c+d)/2` |
| Inverse | `a=(LL+LH+HL+HH)/2` · `b=(LL+LH-HL-HH)/2` · `c=(LL-LH+HL-HH)/2` · `d=(LL-LH-HL+HH)/2` |
| Params | 0 |
| Constraint | Implement with `reshape`/`slice`/`add`/`mul` only. `H`, `W` must be even. |

### 2.5 `NAFBlock(C)`
```
t = LayerNorm2d(C)(x)
t = Conv1x1(C → 2C)(t)
t = Conv3x3(2C → 2C, groups=2C)(t)
t = SimpleGate(t)                                    # 2C → C
t = t * Conv1x1(C → C)(AdaptiveAvgPool2d(1)(t))      # SCA, no sigmoid
t = Conv1x1(C → C)(t)
x = x + LayerScale(C)(t)
u = LayerNorm2d(C)(x)
u = Conv1x1(C → 2C)(u); u = SimpleGate(u); u = Conv1x1(C → C)(u)
x = x + LayerScale(C)(u)
```
| | |
|---|---|
| Params | `7C² + 8C` |
| MACs | `(6C² + 20C)·H·W` |
| Activations | `15·C·H·W` elements |
| Bias | All convolutions have bias. |
| Input/output range | Unbounded (normalised feature space). |

### 2.6 `GSABlock(C, heads)` — exact global self-attention
```
d = C // 2 ; head_dim = d // heads          # head_dim MUST equal 16
t   = LayerNorm2d(C)(x)
qkv = Conv1x1(C → 3d)(t)
qkv = Conv3x3(3d → 3d, groups=3d)(qkv)
q, k, v = split(qkv, d, dim=1)              # each (B,d,H,W)
reshape to (B, heads, H*W, head_dim)
bias = rel_pos_table[rel_pos_index]         # (heads, N, N), N = H*W
t = SDPA(q, k, v, attn_mask=bias)           # training path
t = Conv1x1(d → C)(reshape back)
x = x + LayerScale(C)(t)
u = LayerNorm2d(C)(x)                       # GDFN
u = Conv1x1(C → 2C)(u); u = Conv3x3(2C → 2C, groups=2C)(u)
u = SimpleGate(u); u = Conv1x1(C → C)(u)
x = x + LayerScale(C)(u)
```
| | |
|---|---|
| Params | `≈5C² + 8C + heads·(2n-1)²`, `n = H` |
| MACs | `≈5C²·H·W + 2·(HW)²·d` |
| Activations | `11.5·C·H·W` **with SDPA**; `+ heads·(HW)²` without |
| Rel-pos table | `(heads, (2n-1)²)`, init `trunc_normal_(std=0.02)`, gathered by a registered non-trainable `(N,N)` int64 index buffer |
| Attention | **Exact and unrestricted.** No windows, no sparsity, no approximation. |
| Export path | Explicit `matmul → add bias → softmax → matmul`. Must match SDPA within 1e-4. |
| Constraint | **SDPA is mandatory in training** — the naive path costs +20.8 MB/image. |

Instantiated only at: Enc L1 (`C=96, heads=3, d=48`), Enc L2 (`C=160, heads=5, d=80`),
Dec D1 (`C=96, heads=3, d=48`). **Never at 64×64 or 128×128.**

### 2.7 `GatedFuse(C)`
```
u = Conv1x1(2C → C)(concat([skip, dec], dim=1))
g = sigmoid( Conv1x1(C//4 → C)( Conv1x1(C → C//4)( AdaptiveAvgPool2d(1)(u) ) ) )
return g * skip + (1 - g) * dec
```
| | |
|---|---|
| Params | `2C² + C + 2·(C·C/4) + C/4 + C` |
| MACs | `2C²·H·W + C²/2` |
| Activations | `3·C·H·W` |
| Output range | Convex combination of inputs. |

### 2.8 `RobustNormalizer`
| | |
|---|---|
| Purpose | Remove per-image exposure/contrast. |
| Operation | `m = mean(y)`, `s = clamp_min(std(y), 0.02)`, `ŷ = (y-m)/s` |
| Params | 0 |
| Inverse | `y = ŷ·s + m` — must be exact to 1e-6 |
| Note | Statistics computed on the **raw, unclamped** input. |

### 2.9 `NoiseHead`
| | |
|---|---|
| Purpose | Blind two-parameter noise estimation (Phase 1: `Var = σ_g² + σ_s²I²`). |
| Input / Output | `(B,1,128,128)` → `(B,2)` then σ-map `(B,1,128,128)` |
| Params | 42,050 |
| MACs | 52.3 M |
| Final layer init | weight `= 0`, bias `= (-3.718, -1.718)` so `softplus` yields exactly `(0.024, 0.165)` at init |
| Output range | `σ̂ ∈ [1e-4, 2.0]` after clamp |
| Auxiliary target | Per-image closed-form least squares of `r² = a + c·Î²` on `(D(GT), y)`; loss on `log σ̂` |

---

## PART 3 — Final network table

| # | Stage | Res | Operation | Out shape | Params | MMAC | Act MB |
|---|---|---|---|---|---|---|---|
| 0 | Norm | 128² | mean/std, invertible | `(B,1,128,128)` | 0 | 0.05 | 0.066 |
| 1 | Noise | 128²→8² | 4×[Conv3×3 s2+LN+SG] → GAP → MLP | `(B,1,128,128)` σ | 42,050 | 52.32 | 1.212 |
| 2 | Stem | 128²→64² | concat[ŷ,σ̂] → HaarDWT | `(B,8,64,64)` | 0 | 0.07 | 0.066 |
| 3 | Stem | 64² | Conv3×3(8→48) | `(B,48,64,64)` | 3,504 | 14.16 | 0.393 |
| 4 | Enc L0 | 64² | **NAF ×4** | `(B,48,64,64)` | 70,848 | 240.66 | 23.592 |
| 5 | Down0 | 64²→32² | HaarDWT | `(B,192,32,32)` | 0 | 0.20 | 0.393 |
| 6 | Down0 | 32² | Conv1×1(192→96) | `(B,96,32,32)` | 18,528 | 18.87 | 0.197 |
| 7 | Enc L1 | 32² | **NAF ×4** | `(B,96,32,32)` | 270,720 | 233.60 | 11.796 |
| 8 | Enc L1 | 32² | **GSA ×2** (h=3,d=48) | `(B,96,32,32)` | 124,902 | 301.90 | 4.522 |
| 9 | Down1 | 32²→16² | HaarDWT | `(B,384,16,16)` | 0 | 0.10 | 0.197 |
| 10 | Down1 | 16² | Conv1×1(384→160) | `(B,160,16,16)` | 61,600 | 15.73 | 0.082 |
| 11 | Enc L2 | 16² | **NAF ×4** | `(B,160,16,16)` | 737,920 | 160.32 | 4.916 |
| 12 | Enc L2 | 16² | **GSA ×3** (h=5,d=80) | `(B,160,16,16)` | 420,735 | 133.62 | 2.826 |
| 13 | Up1 | 16² | Conv1×1(160→384) | `(B,384,16,16)` | 61,824 | 15.73 | 0.197 |
| 14 | Up1 | 16²→32² | HaarIDWT | `(B,96,32,32)` | 0 | 0.39 | 0.197 |
| 15 | Dec D1 | 32² | **GatedFuse(96)** + skip₁ | `(B,96,32,32)` | 23,256 | 18.88 | 0.590 |
| 16 | Dec D1 | 32² | **GSA ×1** (h=3,d=48) | `(B,96,32,32)` | 62,451 | 150.95 | 2.261 |
| 17 | Dec D1 | 32² | **NAF ×4** | `(B,96,32,32)` | 270,720 | 233.60 | 11.796 |
| 18 | Up0 | 32² | Conv1×1(96→192) | `(B,192,32,32)` | 18,624 | 18.87 | 0.393 |
| 19 | Up0 | 32²→64² | HaarIDWT | `(B,48,64,64)` | 0 | 0.79 | 0.393 |
| 20 | Dec D0 | 64² | **GatedFuse(48)** + skip₀ | `(B,48,64,64)` | 5,868 | 18.88 | 1.180 |
| 21 | Dec D0 | 64² | **NAF ×4** | `(B,48,64,64)` | 70,848 | 240.66 | 23.592 |
| 22 | Head | 64² | Conv3×3(48→128) | `(B,128,64,64)` | 55,424 | 226.49 | 1.049 |
| 23 | Head | 64²→128² | HaarIDWT | `(B,32,128,128)` | 0 | 2.10 | 1.049 |
| 24 | Head | 128² | **NAF ×3** (C=32) | `(B,32,128,128)` | 24,672 | 330.30 | 47.186 |
| 25 | Head | 128² | Conv3×3(32→4) `[LL,LH,HL,HH]` | `(B,4,128,128)` | 1,156 | 18.87 | 0.131 |
| 26 | Head | 128²→256² | HaarIDWT | `(B,1,256,256)` | 0 | 0.07 | 0.131 |
| 27 | Out | 256² | `+ bicubic_up2(ŷ)` | `(B,1,256,256)` | 0 | 1.05 | 0.131 |
| 28 | Out | 256² | `*s + m`, `clamp(0,1)` | `(B,1,256,256)` | 0 | 0.13 | 0.131 |
| | **TOTAL** | | | | **2,345,650** | **2449.37** | **140.665** |

**Group summary**

| Group | Params | % | MMAC | % | Act MB | % |
|---|---|---|---|---|---|---|
| Noise head | 42,050 | 1.8 % | 52.3 | 2.1 % | 1.21 | 0.9 % |
| Stem | 3,504 | 0.1 % | 14.2 | 0.6 % | 0.46 | 0.3 % |
| Enc L0 | 70,848 | 3.0 % | 240.7 | 9.8 % | 23.59 | 16.8 % |
| Enc L1 | 395,622 | 16.9 % | 535.5 | 21.9 % | 16.32 | 11.6 % |
| Enc L2 (bottleneck) | 1,158,655 | **49.4 %** | 294.0 | 12.0 % | 7.74 | 5.5 % |
| Dec D1 | 356,427 | 15.2 % | 403.4 | 16.5 % | 14.65 | 10.4 % |
| Dec D0 | 76,716 | 3.3 % | 259.5 | 10.6 % | 24.77 | 17.6 % |
| Head | 81,252 | 3.5 % | 577.8 | **23.6 %** | 49.55 | **35.2 %** |
| Transitions + out | 160,576 | 6.8 % | 71.9 | 2.9 % | 2.42 | 1.7 % |

**Known risk (accepted, not fixed in V1):** the bottleneck holds 49.4 % of parameters on 2749 effective
scenes. Monitored by ablation A3; do not change it in V1.

---

## PART 4 — Tensor flow (one image, B=1, fp16 activations)

| Step | Tensor | Shape | Feature dim | Live memory |
|---|---|---|---|---|
| 0 | input `y` | `(1,1,128,128)` | 1 | 0.03 MB |
| 1 | `ŷ` normalised | `(1,1,128,128)` | 1 | 0.03 MB |
| 2 | `σ̂` noise map | `(1,1,128,128)` | 1 | 0.03 MB |
| 3 | `concat[ŷ,σ̂]` | `(1,2,128,128)` | 2 | 0.07 MB |
| 4 | after HaarDWT | `(1,8,64,64)` | 8 | 0.07 MB |
| 5 | after stem conv | `(1,48,64,64)` | 48 | 0.39 MB |
| 6 | **skip₀** (Enc L0 out) | `(1,48,64,64)` | 48 | 0.39 MB |
| 7 | after DWT+1×1 | `(1,96,32,32)` | 96 | 0.20 MB |
| 8 | **skip₁** (Enc L1 out) | `(1,96,32,32)` | 96 | 0.20 MB |
| 9 | after DWT+1×1 | `(1,160,16,16)` | 160 | 0.08 MB |
| 10 | bottleneck out | `(1,160,16,16)` | 160 | 0.08 MB |
| 11 | after 1×1+IDWT | `(1,96,32,32)` | 96 | 0.20 MB |
| 12 | after GatedFuse+D1 | `(1,96,32,32)` | 96 | 0.20 MB |
| 13 | after 1×1+IDWT | `(1,48,64,64)` | 48 | 0.39 MB |
| 14 | after GatedFuse+D0 | `(1,48,64,64)` | 48 | 0.39 MB |
| 15 | head projection | `(1,128,64,64)` | 128 | 1.05 MB |
| 16 | after IDWT | `(1,32,128,128)` | 32 | 1.05 MB |
| 17 | after NAF ×3 | `(1,32,128,128)` | 32 | 1.05 MB |
| 18 | sub-bands | `(1,4,128,128)` | 4 | 0.13 MB |
| 19 | after IDWT | `(1,1,256,256)` | 1 | 0.13 MB |
| 20 | + global residual | `(1,1,256,256)` | 1 | 0.13 MB |
| 21 | **output** | `(1,1,256,256)` | 1 | 0.13 MB |

Total retained-for-backward: **140.665 MB per image**.

---

## PART 5 — Frozen configuration

| Parameter | Value |
|---|---|
| `in_channels` / `out_channels` | 1 / 1 |
| `scale` | 2 |
| Base channels (stem) | 48 |
| Encoder widths | `(48, 96, 160)` |
| Decoder widths | `(96, 48)` |
| Encoder resolutions | `(64, 32, 16)` |
| Head width | 32 (at 128²) |
| Encoder NAF depths | `(4, 4, 4)` |
| Encoder GSA depths | `(0, 2, 3)` |
| Decoder NAF depths | `(4, 4)` for `(D1, D0)` |
| Decoder GSA depths | `(1, 0)` for `(D1, D0)` |
| Head NAF depth | 3 |
| Attention heads | `(–, 3, 5)` for `(L0, L1, L2)` |
| Attention dim ratio | 0.5 |
| Attention head dim | 16 (invariant) |
| NAF expansion | 2 |
| GDFN expansion | 2 |
| Fusion reduction | 4 |
| LayerScale init | 1e-2 |
| **Dropout** | **0.0 everywhere. No dropout, no stochastic depth.** |
| Normalisation | `LayerNorm2d`, eps 1e-6 |
| Activation | none (`SimpleGate` only) |
| Patch size | **128 (full image, no cropping)** |
| Batch size | **8** |
| Optimiser | AdamW, β = (0.9, 0.9), eps 1e-8 |
| Weight decay | 1e-4 (excluded on LayerNorm, LayerScale, bias, rel-pos tables) |
| Learning rate | 3e-4 |
| Scheduler | Cosine to 1e-6 over 400 epochs |
| Warmup | 5 epochs, linear from 1e-6 |
| Gradient clipping | 1.0 global norm |
| EMA | decay 0.999, updated every step, evaluated separately |
| AMP | fp16 + `GradScaler`, `init_scale=2**14` |
| Gradient checkpointing | **OFF** (not needed at batch 8) |
| `torch.compile` | **OFF in V1** (enable only after ONNX parity is verified) |
| Channels-last | **ON** (`memory_format=torch.channels_last`) |
| Seed | **1337** |
| Epochs | 400 |
| Dataloader | `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`, `drop_last=True` |

**Sanctioned batch-size override (the only permitted config deviation):** if acceptance test M9
measures peak VRAM at batch 8 below 1.6 GB, batch size may be raised to 16 and the learning rate to
4.2e-4 (`3e-4 · sqrt(2)`). No other change is permitted without an amendment.

---

## PART 6 — Training contract

**Loss**

$$\mathcal{L} = \mathcal{L}_{\text{Charb}} + 0.15\,\mathcal{L}_{\text{MS-SSIM}} + 0.10\,\mathcal{L}_{\text{wav}} + 0.05\,\mathcal{L}_{\text{FFT}} + 0.05\,\mathcal{L}_{\text{grad}} + 0.02\,\mathcal{L}_{\sigma}$$

| Term | Weight | Definition |
|---|---|---|
| Charbonnier | 1.00 | `mean(sqrt((x̂-x)² + 1e-6))` |
| MS-SSIM | 0.15 | `1 - MS_SSIM`, 5 scales, window 11, σ 1.5, `data_range=1.0` |
| Wavelet | 0.10 | 2-level Haar, L1 per band, band weights `LL=0.25, LH=HL=1.0, HH=1.5` |
| FFT | 0.05 | `mean(abs(|rfft2(x̂)| - |rfft2(x)|))`, amplitude only |
| Gradient | 0.05 | L1 on Sobel-x and Sobel-y |
| Noise aux | 0.02 | L1 on `log σ̂` vs. analytic σ from GT |

All losses computed **after de-normalisation and clamping**, against unmodified GT.
**LPIPS is NOT in V1.** Every term must be logged separately every step.

**Metrics:** PSNR (`data_range=1.0`), SSIM (window 11, σ 1.5, K=(0.01,0.03)), LPIPS (AlexNet,
grayscale replicated to 3 channels — evaluation only), latency, peak VRAM.
Report **mean and median**, plus σ-stratified and texture-stratified breakdowns.

| Item | Value |
|---|---|
| Validation frequency | every epoch |
| Checkpoint frequency | every epoch; keep `best_psnr`, `best_ema_psnr`, `last` |
| Early stopping | patience 40 epochs on EMA val PSNR, min delta 0.01 dB |
| Split | **group-aware**, contiguous 32-ID blocks, every 10th block to val (320 val / 2880 train) |
| Random seed | 1337 (`set_seed` covers python, numpy, torch, cudnn deterministic) |
| Expected training time | **60–120 s/epoch → 7–13 h for 400 epochs** (estimate; measure at step 5) |
| Expected GPU memory | **1.17 GB at batch 8** (analytic); acceptance limit 2.0 GB measured |
| Expected inference | **~3–6 ms/image batched**, 15–25 ms at batch 1 (estimate) |

**Augmentation (frozen):**
- Paired geometric only: h-flip (p=0.5), v-flip (p=0.5), rot90 (p=0.75, k∈{1,2,3}).
- **No input-only photometric jitter of any kind.**
- On-the-fly LR re-synthesis from GT (p=0.5, else use the supplied LR):
  `g_{σ~U(0.3,0.5)} → ·Γ(L~U(25,60))/L → +N(0,σ_g), σ_g~U(0,0.08) → bicubic↓2 antialias=False`, **no clipping**.

---

## PART 7 — Module status

| Module | Status | In V1? |
|---|---|---|
| RobustNormalizer (mean/std) | **CORE** | YES |
| NoiseHead + σ-map | **CORE** | YES |
| LayerNorm2d | **CORE** | YES |
| SimpleGate | **CORE** | YES |
| LayerScale | **CORE** | YES |
| HaarDWT / HaarIDWT | **CORE** | YES |
| NAFBlock | **CORE** | YES |
| GSABlock (exact MHSA, 32²/16² only) | **CORE** | YES |
| GatedFuse | **CORE** | YES |
| Sub-band reconstruction head (single path) | **CORE** | YES |
| Global residual (bicubic ×2) | **CORE** | YES |
| Output clamp | **CORE** | YES |
| Charbonnier / MS-SSIM / wavelet / FFT / gradient / noise-aux loss | **CORE** | YES |
| Median/MAD normalisation | OPTIONAL | **NO** |
| LKA regional blocks | OPTIONAL | **NO** |
| Cross-scale exchange | OPTIONAL | **NO** |
| LPIPS loss term | OPTIONAL | **NO** |
| Self-ensemble TTA | OPTIONAL | **NO** |
| `torch.compile` | OPTIONAL | **NO** |
| Soft data consistency | EXPERIMENTAL | **NO — must not appear in the codebase** |
| Separate edge/texture/structure branches | EXPERIMENTAL | **NO — must not appear in the codebase** |
| Content-adaptive routing | EXPERIMENTAL | **NO — must not appear in the codebase** |
| Native 256→512 mode | EXPERIMENTAL | **NO — must not appear in the codebase** |
| Mamba/SSM, window attention, GAN, diffusion, BatchNorm | FORBIDDEN | **NO** |

**Rule:** OPTIONAL modules are not implemented in V1. EXPERIMENTAL modules must not appear anywhere in
the source tree — not behind a flag, not commented out, not as dead code.

---

## PART 8 — Frozen implementation order

| Step | Files | Depends on | Tests | Acceptance |
|---|---|---|---|---|
| **1** | `utils/{logging_utils,init,complexity,profiling,seed,checkpoint}.py`, `configs/sparc_config.py` | — | T1 | Deterministic run reproduces to 1e-6; config validation rejects bad head-dim |
| **2** | `datasets/{packed_dataset,transforms,degradation,splits}.py`, `scripts/pack_dataset.py` | 1 | T2 | 3200/3200/400 counts verified; zero group overlap; re-synthesised LR σ-vs-I curve within 5 % of real |
| **3** | `evaluation/metrics.py`, `scripts/baselines.py` | 2 | T3 | Reproduces **bicubic 21.67 dB**, nearest 20.38 dB |
| **4** | `models/wavelet/haar.py` | 1 | T4 | `IDWT(DWT(x))` max abs err < 1e-6; orthonormality; ONNX export |
| **5** | `models/blocks/{layer_norm,simple_gate,layer_scale,naf_block}.py` | 1 | T5 | Shape, gradient, param count exact, identity-at-init, TorchScript, ONNX |
| **6** | `models/encoder.py`, `models/decoder/decoder.py`, `models/decoder/reconstruction_head.py`, `models/sparc_net.py` (concat skips, no noise head, no attention) | 4,5 | T6 | **SPARC-Tiny overfits 8 images to > 45 dB in < 2000 steps** |
| **7** | `losses/charbonnier.py`, `losses/composite.py`, `trainer/trainer.py`, `train.py` | 6 | T7 | 1 full epoch, no NaN, AMP stable, checkpoint resume bit-exact |
| **8** | scale config to Base widths | 7 | T8 | **Measured VRAM at batch 8 < 2.0 GB**; params 2.346 M ±2 %; MACs 2.449 G ±5 % |
| **9** | `models/normalization.py` | 8 | T9 | Invertibility 1e-6; ablation A1 recorded |
| **10** | `models/noise/{noise_head,noise_map}.py` | 9 | T10 | σ̂ correlates > 0.9 with analytic σ; ablation A2 recorded |
| **11** | `models/fusion/gated_fuse.py` | 10 | T11 | Gate ∈ (0,1); ablation A3 recorded |
| **12** | `losses/{ms_ssim,wavelet,fft,gradient,noise_aux}.py` | 11 | T12 | Each term added **one at a time**, each ablated (A4) |
| **13** | `models/attention/{gsa_block,rel_pos}.py` | 12 | T13 | SDPA vs. explicit path parity 1e-4; ablation A5 recorded |
| **14** | `evaluation/evaluate.py`, `scripts/benchmark.py` | 13 | T14 | Latency + VRAM measured vs. contract budgets |
| **15** | `scripts/export_onnx.py`, `scripts/export_trt.py` | 14 | T15 | ONNX parity < 1e-3, opset 17; TensorRT parity < 1e-2 |

**Why this order:** steps 1–7 produce a scoring model at ~15 % of final complexity. From step 8 every
change is a single reversible increment against a known number, so any regression has exactly one candidate
cause. The two highest-risk modules (noise head, attention) sit at 10 and 13, after the pipeline is
trustworthy. **No engineer may reorder these steps.**

---

## PART 9 — Unit-test contract

Every module must pass all applicable tests before integration. Tests live in `tests/`, one file per module.

| Test | Requirement |
|---|---|
| **Shape** | Output shape matches the contract for `B ∈ {1,2,8}` |
| **Gradient** | `loss.backward()` gives non-zero, finite grads for **every** parameter |
| **Numerical stability** | No NaN/Inf over 100 random batches, including inputs scaled ×1e-3 and ×1e3 |
| **Parameter count** | Matches Part 3 **exactly** (not within a tolerance — these are computable) |
| **MACs** | Within ±5 % of Part 3, measured with `FlopCounterMode` |
| **TorchScript** | `torch.jit.script` succeeds and matches eager to 1e-5 |
| **ONNX** | Export at opset 17 succeeds; `onnxruntime` matches eager to 1e-3 |
| **Memory** | Measured activation bytes within ±15 % of Part 3 |
| **Performance** | Latency recorded; no regression > 10 % vs. the previous commit |

| Module | Shape | Grad | Stab | Params | TorchScript | ONNX | Mem | Perf |
|---|---|---|---|---|---|---|---|---|
| `LayerNorm2d` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| `SimpleGate` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| `LayerScale` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| `HaarDWT/IDWT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| `NAFBlock` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GSABlock` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GatedFuse` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `RobustNormalizer` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| `NoiseHead` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `Encoder` / `Decoder` / `Head` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARCNet` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Each loss term | ✓ | ✓ | ✓ | – | ✓ | – | – | – |
| Dataset / DataLoader | ✓ | – | ✓ | – | – | – | ✓ | ✓ |

Additional invariants:
- `HaarIDWT(HaarDWT(x)) == x` to 1e-6.
- `RobustNormalizer.denormalize(forward(x)) == x` to 1e-6.
- `NAFBlock` with `LayerScale = 0` is exactly the identity.
- `GSABlock` SDPA path == explicit path to 1e-4.
- Full model on constant input produces a constant output (no checkerboard).

---

## PART 10 — Performance budget

| Budget | Limit | Contract value |
|---|---|---|
| Maximum parameters | **2.60 M** | 2.346 M |
| Maximum MACs (128²→256²) | **2.80 G** | 2.449 G |
| Maximum GFLOPs | **5.60** | 4.899 |
| Maximum activations/image (fp16) | **160 MB** | 140.7 MB |
| Maximum training VRAM @ batch 8 | **2.00 GB** | 1.17 GB analytic |
| Maximum inference VRAM @ batch 16 | **1.50 GB** | — measure |
| Maximum inference latency (batch 1) | **35 ms** | — measure |
| Maximum inference latency (batch 16) | **10 ms/image** | — measure |
| Maximum training time | **16 h** for 400 epochs | 7–13 h estimate |
| Maximum model size on disk | **12 MB** fp32 | 9.38 MB |

**No module may cause any budget to be exceeded.** If a change breaches a budget, it is reverted, not
accommodated.

---

## PART 11 — Ablation contract

Run in this exact order. Each ablation is a single change against the previous accepted model, 3 seeds,
identical schedule and split. Report mean and median PSNR/SSIM/LPIPS, latency, and peak VRAM.

| ID | Step | Ablation | Baseline for comparison |
|---|---|---|---|
| **A0** | after step 7 | **Baseline**: Tiny-config model, Charbonnier only, concat skips | bicubic 21.67 dB |
| **A1** | after step 9 | ± robust normalisation | A0 at Base widths |
| **A2** | after step 10 | ± noise head (and its aux loss) | A1 |
| **A3** | after step 11 | GatedFuse vs. concatenation | A2 |
| **A4** | after step 12 | ± each loss term individually (MS-SSIM, wavelet, FFT, gradient) | A3 |
| **A5** | after step 13 | ± all GSA blocks | A4 |
| **A6** | after step 13 | ± LR re-synthesis augmentation | A5 |
| **A7** | V1.1 only | OPTIONAL modules, one at a time, in the Part 7 order | final V1 |

No OPTIONAL module may be evaluated before A6 completes.

---

## PART 12 — Promotion rule

A module moves **OPTIONAL → CORE** only if **every** condition holds, on the leak-free validation split,
averaged over 3 seeds:

1. **ΔPSNR ≥ +0.15 dB** (mean) **and** ΔPSNR ≥ +0.10 dB (median)
2. **ΔSSIM ≥ +0.002**
3. **ΔLPIPS ≤ 0** (does not get worse)
4. **Runtime increase ≤ 10 %** (measured inference latency, batch 16)
5. **Memory increase ≤ 8 %** (measured peak training VRAM)
6. **Training stable**: no NaN, no loss spike > 3× running mean, no AMP scaler collapse
7. **No budget in Part 10 exceeded**
8. **Exports cleanly** to ONNX with < 1e-3 parity

Failing any single condition, the module **remains OPTIONAL and is removed from the branch**.
A module that is worse on PSNR but better on LPIPS is **not** promoted in V1 — PSNR and SSIM are the
primary scored metrics.

---

## PART 13 — Stopping rule

Architecture development **ends immediately** when any of the following occurs:

1. **Three consecutive** OPTIONAL-module trials each produce **< 0.05 dB** PSNR improvement; or
2. Any Part 10 budget is exceeded and cannot be recovered by reverting; or
3. Total ablation wall-clock exceeds **120 GPU-hours**; or
4. The V1 model has been unchanged for **2 consecutive ablation cycles**.

On stopping: freeze the architecture, tag `v1.0-frozen`, and move all remaining effort to training
schedule, augmentation tuning, EMA/TTA, and export optimisation. **No new modules after the stopping
condition fires.** This rule exists to prevent architecture creep and is not subject to negotiation.

---

## PART 14 — Frozen project structure

```
project/
├── configs/
│   ├── __init__.py
│   ├── default_config.py          # existing project config
│   └── sparc_config.py            # SPARC-Base V1 frozen config
├── datasets/
│   ├── __init__.py
│   ├── packed_dataset.py          # memmap-backed paired dataset
│   ├── transforms.py              # paired geometric augmentation ONLY
│   ├── degradation.py             # Phase 1 forward model, LR re-synthesis
│   └── splits.py                  # group-aware split
├── models/
│   ├── __init__.py
│   ├── sparc_net.py               # top-level assembly
│   ├── encoder.py
│   ├── normalization.py           # RobustNormalizer
│   ├── blocks/
│   │   ├── __init__.py
│   │   ├── layer_norm.py          # LayerNorm2d
│   │   ├── simple_gate.py         # SimpleGate, LayerScale
│   │   └── naf_block.py           # NAFBlock
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── rel_pos.py             # relative position bias table + index
│   │   └── gsa_block.py           # GSABlock
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── gated_fuse.py          # GatedFuse
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── noise_head.py          # NoiseHead
│   │   └── noise_map.py           # sigma-map assembly + analytic target
│   ├── wavelet/
│   │   ├── __init__.py
│   │   └── haar.py                # HaarDWT, HaarIDWT
│   └── decoder/
│       ├── __init__.py
│       ├── decoder.py
│       └── reconstruction_head.py
├── losses/
│   ├── __init__.py
│   ├── charbonnier.py
│   ├── ms_ssim.py
│   ├── wavelet.py
│   ├── fft.py
│   ├── gradient.py
│   ├── noise_aux.py
│   └── composite.py               # weighted sum, per-term logging
├── trainer/
│   ├── __init__.py
│   ├── trainer.py
│   └── ema.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # PSNR, SSIM, LPIPS
│   └── evaluate.py
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── init.py
│   ├── complexity.py
│   ├── profiling.py
│   ├── checkpoint.py
│   ├── seed.py
│   └── io.py
├── tests/
│   ├── test_haar.py
│   ├── test_blocks.py
│   ├── test_attention.py
│   ├── test_fusion.py
│   ├── test_noise.py
│   ├── test_normalization.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_export.py
├── scripts/
│   ├── pack_dataset.py
│   ├── baselines.py
│   ├── benchmark.py
│   ├── export_onnx.py
│   └── export_trt.py
├── checkpoints/
├── outputs/
├── reports/
├── requirements.txt
├── README.md
└── train.py
```

**Migration note (Phase 4, step 0):** files written during the aborted Phase 4 must be relocated —
`models/frequency/haar.py` → `models/wavelet/haar.py`; `models/normalization/layer_norm.py` →
`models/blocks/layer_norm.py`; `models/normalization/robust_norm.py` → `models/normalization.py`
(rewritten to mean/std per Part 2.8); delete `models/normalization/` and `models/frequency/` packages.
`models/restormer.py` and `models/layers.py` are legacy and must be deleted.

---

## PART 15 — Implementation checklist

```
□  0  Project structure created; legacy files deleted; Phase 4 files migrated
□  1  Dataset packed and verified (3200/3200/400, MD5, zero group leakage)
□  2  DataLoader verified (workers, pinning, throughput, deterministic order)
□  3  Baselines reproduced (bicubic 21.67 dB, nearest 20.38 dB)
□  4  Haar DWT/IDWT verified (invertibility 1e-6, orthonormality, ONNX)
□  5  LayerNorm2d / SimpleGate / LayerScale verified
□  6  NAF Block verified (shape, grad, params exact, identity-at-init)
□  7  Encoder verified (stage shapes printed and matched to Part 3)
□  8  Decoder verified (encoder/decoder symmetry, skip alignment)
□  9  Reconstruction head verified (no checkerboard on constant input)
□ 10  SPARC-Tiny overfits 8 images > 45 dB
□ 11  Charbonnier loss verified; training loop runs 1 epoch NaN-free
□ 12  Scaled to Base config; VRAM measured < 2.0 GB at batch 8
□ 13  Params 2.346 M ±2 % and MACs 2.449 G ±5 % measured
□ 14  Normalisation verified (invertibility, ablation A1)
□ 15  Noise Head verified (σ̂ correlation > 0.9, ablation A2)
□ 16  Fusion verified (gate range, ablation A3)
□ 17  All loss terms verified individually (ablation A4)
□ 18  Attention verified (SDPA/explicit parity 1e-4, ablation A5)
□ 19  Augmentation verified (ablation A6)
□ 20  Validation pipeline verified (stratified reporting)
□ 21  Benchmark verified against every Part 10 budget
□ 22  ONNX export verified (parity 1e-3, opset 17)
□ 23  TensorRT export verified (parity 1e-2)
□ 24  Evaluation script verified (submission assembly, ID order correct)
□ 25  README verified (setup, train, evaluate, export, reproduce)
□ 26  Seeded reproducibility verified (two runs match to 1e-4 @ 200 steps)
```

---

## PART 16 — Amendment procedure

This contract is frozen. To change it:

1. State the specific clause being amended.
2. State the measurement that justifies the change.
3. State the effect on every Part 10 budget.
4. Obtain approval **before** writing code.

Silent deviation from this contract is a defect, regardless of whether the result is better.
