# FINAL IMPLEMENTATION SPECIFICATION — SPARC-Base V1

Engineering specification only. Build exactly this. No architectural decisions are left open.

**Target:** 128×128×1 → 256×256×1 grayscale restoration. **Hardware:** RTX A400, 4 GB.
**Budget:** 2.381 M parameters · 2.405 GMAC · 181 MB activations/image (fp16) · 1.45 GB at batch 8.

---

## 1. Global constants

| Symbol | Value |
|---|---|
| Input | `(B, 1, 128, 128)` float32, **never clamped** |
| Output | `(B, 1, 256, 256)` float32, clamped to `[0, 1]` |
| Level widths `C0, C1, C2` | **48, 96, 160** |
| Level resolutions | **64², 32², 16²** |
| Head width `Ch` | **32** (at 128²) |
| Attention dim ratio | **0.5** (`d = C // 2`) |
| Attention head dim | **16** at every attention level |
| NAF expansion | **2** |
| GDFN expansion | **2** |
| LayerScale init | **1e-2** |
| Normalisation | `LayerNorm2d` (channel-wise). **BatchNorm is forbidden.** |
| Activation | **None.** `SimpleGate` only. |

---

## 2. Primitive operators

### 2.1 `LayerNorm2d(C, eps=1e-6)`
`y = (x - mean_C(x)) * rsqrt(var_C(x) + eps) * w + b`, moments over the channel dim, `w, b` shape `(1,C,1,1)`.

### 2.2 `SimpleGate()`
`x1, x2 = chunk(x, 2, dim=1); return x1 * x2`. Halves channels. No parameters.

### 2.3 `LayerScale(C, init=1e-2)`
Per-channel learnable vector shape `(1,C,1,1)`, multiplies every residual branch output.

### 2.4 `HaarDWT` / `HaarIDWT` — parameter-free, exactly invertible
For each 2×2 block `[[a,b],[c,d]]`, with output channel order `[LL(C), LH(C), HL(C), HH(C)]`:
```
LL = (a+b+c+d)/2      LH = (a+b-c-d)/2      HL = (a-b+c-d)/2      HH = (a-b-c+d)/2
```
Inverse (the matrix is orthonormal and symmetric, so it is the same operator):
```
a = (LL+LH+HL+HH)/2   b = (LL+LH-HL-HH)/2   c = (LL-LH+HL-HH)/2   d = (LL-LH-HL+HH)/2
```
`(B,C,H,W) → (B,4C,H/2,W/2)` and back. Implement with `reshape`/`slice`/`add`/`mul` only.

### 2.5 `NAFBlock(C)`
```
t = LayerNorm2d(x)
t = Conv1x1(C → 2C)
t = DWConv3x3(2C, groups=2C)
t = SimpleGate(t)                          # 2C → C
t = t * Conv1x1(C → C)(GlobalAvgPool(t))   # SCA, no sigmoid
t = Conv1x1(C → C)
x = x + LayerScale_1 * t
u = LayerNorm2d(x)
u = Conv1x1(C → 2C); u = SimpleGate(u); u = Conv1x1(C → C)
x = x + LayerScale_2 * u
```
Parameters ≈ `7C²`. **This is the only block used outside attention levels.**

### 2.6 `GSABlock(C, heads)` — exact global self-attention
`d = C // 2`; `head_dim = d // heads` must equal **16**.
```
t = LayerNorm2d(x)
qkv = Conv1x1(C → 3d)(t); qkv = DWConv3x3(3d, groups=3d)(qkv)
q, k, v = split(qkv, d, dim=1)                       # each (B,d,H,W)
reshape each to (B, heads, H*W, head_dim)
A = softmax(q @ kᵀ / sqrt(head_dim) + RelPosBias)    # (B, heads, N, N), N = H*W
t = (A @ v) → (B, d, H, W)
t = Conv1x1(d → C)(t)
x = x + LayerScale_1 * t

u = LayerNorm2d(x)
u = Conv1x1(C → 2C)(u); u = DWConv3x3(2C, groups=2C)(u)
u = SimpleGate(u); u = Conv1x1(C → C)(u)             # GDFN
x = x + LayerScale_2 * u
```
`RelPosBias`: learnable table `(heads, (2n-1)²)` gathered by a precomputed `(N, N)` index buffer, `n = H`.
Attention is **exact and unrestricted** — no windows, no sparsity.
Use `F.scaled_dot_product_attention` with the bias as `attn_mask` for training; use the explicit
`matmul + softmax + matmul` path for ONNX export. Both must be numerically equivalent to 1e-4.

### 2.7 `GatedFuse(C)` — skip fusion
```
u = Conv1x1(2C → C)(concat[skip, dec])
g = sigmoid( Conv1x1(C//4 → C)( Conv1x1(C → C//4)( GlobalAvgPool(u) ) ) )
return g * skip + (1 - g) * dec
```

---

## 3. Network definition

### Stage 0 — Normalisation (parameter-free, invertible)
```
m = mean(y, dim=(1,2,3), keepdim=True)
s = clamp_min(std(y, dim=(1,2,3), keepdim=True), 0.02)
ŷ = (y - m) / s
```
Retain `(m, s)`.

### Stage 1 — Noise map
```
Î  = box_filter_5x5(y)                                  # original units, reflect padding
trunk: 4 × [Conv3x3 stride2 → LayerNorm2d → SimpleGate]
       channels 1→32→(SG)16 →48→(SG)24 →64→(SG)32 →64→(SG)32
       resolutions 128→64→32→16→8
head:  GlobalAvgPool → Linear(32→64) → SimpleGate → Linear(32→2) → softplus
       ⇒ (σ̂_g, σ̂_s) per image
       final Linear weight init = 0, bias = (softplus⁻¹(0.024), softplus⁻¹(0.165))
                                = (-3.718, -1.718)
σ̂  = sqrt(σ̂_g² + σ̂_s²·Î²) / s            → (B,1,128,128)
```
Clamp `σ̂ ∈ [1e-4, 2.0]`. Parameters ≈ 0.042 M.

### Stage 2 — Stem
```
concat[ŷ, σ̂]      (B,2,128,128)
HaarDWT           (B,8,64,64)
Conv3x3(8 → 48)   (B,48,64,64)
```

### Stage 3 — Encoder
| Level | Resolution | Channels | Blocks | Output |
|---|---|---|---|---|
| L0 | 64×64 | 48 | **4 × NAF** | `skip₀` |
| ↓ | `HaarDWT` → `(B,192,32,32)` → `Conv1x1(192→96)` | | | |
| L1 | 32×32 | 96 | **4 × NAF, then 2 × GSA (heads=3, d=48)** | `skip₁` |
| ↓ | `HaarDWT` → `(B,384,16,16)` → `Conv1x1(384→160)` | | | |
| L2 | 16×16 | 160 | **4 × NAF, then 3 × GSA (heads=5, d=80)** | bottleneck |

### Stage 4 — Decoder
| Step | Operation | Output |
|---|---|---|
| 1 | `Conv1x1(160→384)` → `HaarIDWT` | `(B,96,32,32)` |
| 2 | `GatedFuse(96)(skip₁, ·)` | `(B,96,32,32)` |
| 3 | **1 × GSA (heads=3), then 4 × NAF** | `(B,96,32,32)` |
| 4 | `Conv1x1(96→192)` → `HaarIDWT` | `(B,48,64,64)` |
| 5 | `GatedFuse(48)(skip₀, ·)` | `(B,48,64,64)` |
| 6 | **4 × NAF** | `(B,48,64,64)` |

### Stage 5 — Reconstruction head
**No convolution runs at 256×256.**
| Step | Operation | Output |
|---|---|---|
| 1 | `Conv3x3(48 → 128)` | `(B,128,64,64)` |
| 2 | `HaarIDWT` | `(B,32,128,128)` |
| 3 | **3 × NAF (C=32)** | `(B,32,128,128)` |
| 4 | `Conv3x3(32 → 4)` — predicts `[LL,LH,HL,HH]` of the output | `(B,4,128,128)` |
| 5 | `HaarIDWT` | `(B,1,256,256)` |

### Stage 6 — Output
```
x = x + bicubic_upsample_2x(ŷ, align_corners=False)     # global residual
x = x * s + m                                            # de-normalise
x = clamp(x, 0.0, 1.0)
```

---

## 4. Tensor-shape trace

```
(B,1,128,128) → (B,2,128,128) → DWT → (B,8,64,64) → (B,48,64,64)
  → DWT/1x1 → (B,96,32,32) → DWT/1x1 → (B,160,16,16)
  → 1x1/IDWT → (B,96,32,32) → 1x1/IDWT → (B,48,64,64)
  → (B,128,64,64) → IDWT → (B,32,128,128) → (B,4,128,128) → IDWT → (B,1,256,256)
```

---

## 5. Parameter budget

| Stage | Params | MACs |
|---|---|---|
| Noise head | 0.042 M | 0.013 G |
| Stem | 0.004 M | 0.014 G |
| Encoder L0 (4 NAF) | 0.071 M | 0.241 G |
| DWT 0 → 1 | 0.019 M | 0.019 G |
| Encoder L1 (4 NAF + 2 GSA) | 0.419 M | 0.536 G |
| DWT 1 → 2 | 0.062 M | 0.016 G |
| Encoder L2 (4 NAF + 3 GSA) | 1.159 M | 0.294 G |
| IDWT 2 → 1 | 0.062 M | 0.016 G |
| GatedFuse L1 | 0.023 M | 0.019 G |
| Decoder D1 (1 GSA + 4 NAF) | 0.345 M | 0.385 G |
| IDWT 1 → 0 | 0.019 M | 0.019 G |
| GatedFuse L0 | 0.006 M | 0.019 G |
| Decoder D0 (4 NAF) | 0.071 M | 0.241 G |
| Reconstruction head | 0.081 M | 0.576 G |
| **Total** | **2.381 M** | **2.405 G** |

Acceptance: measured parameters within **±2 %**, measured MACs within **±5 %**.

---

## 6. Loss

$$\mathcal{L} = \mathcal{L}_{\text{Charb}} + 0.15\,\mathcal{L}_{\text{MS-SSIM}} + 0.10\,\mathcal{L}_{\text{wav}} + 0.05\,\mathcal{L}_{\text{FFT}} + 0.05\,\mathcal{L}_{\text{grad}} + 0.02\,\mathcal{L}_{\sigma}$$

| Term | Definition |
|---|---|
| Charbonnier | `mean(sqrt((x̂-x)² + 1e-6))` |
| MS-SSIM | `1 - MS_SSIM(x̂, x)`, 5 scales, window 11, σ 1.5, `data_range=1.0` |
| Wavelet | 2-level Haar; L1 per band with weights `LL=0.25, LH=HL=1.0, HH=1.5` |
| FFT | `mean(abs(|FFT(x̂)| - |FFT(x)|))`, amplitude only |
| Gradient | L1 on Sobel-x and Sobel-y responses |
| Noise aux | L1 on `log σ̂` vs. the analytic σ from GT (per-image closed-form fit of `r² = a + cÎ²`) |

All losses computed **after** de-normalisation and clamping, against the unmodified GT.
**LPIPS is not included in V1.**

---

## 7. Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, β=(0.9, 0.9), weight decay 1e-4 |
| LR | 3e-4, cosine to 1e-6 |
| Warmup | 5 epochs linear |
| Batch size | **8** |
| Epochs | 400 |
| AMP | fp16 with GradScaler |
| Gradient clipping | 1.0 (global norm) |
| EMA | decay 0.999, evaluated separately |
| Split | **Group-aware** by Phase 1 scene groups (contiguous 32-ID blocks) — never random |
| Augmentation | **Paired geometric only**: h-flip, v-flip, rot90. **No input-only photometric jitter.** |
| Degradation aug | On-the-fly LR re-synthesis from GT: `g_{σ~U(0.3,0.5)} → ·Γ(L~U(25,60))/L → +N(0,σ_g), σ_g~U(0,0.08) → bicubic↓2 no-AA`, **no clipping** |
| Checkpointing | Every epoch; keep best-by-val-PSNR and last |

---

## 8. Acceptance tests

| # | Test | Criterion |
|---|---|---|
| 1 | `IDWT(DWT(x)) == x` | max abs error < 1e-6 |
| 2 | Parameter count | 2.381 M ± 2 % |
| 3 | MAC count (`FlopCounterMode`) | 2.405 G ± 5 % |
| 4 | Forward shape | `(B,1,128,128)` → `(B,1,256,256)` |
| 5 | No NaN/Inf in forward or backward | 100 random batches |
| 6 | Gradient reaches every parameter | all `grad.abs().sum() > 0` |
| 7 | Overfit 8 images | > 45 dB PSNR within 2000 steps |
| 8 | Bicubic baseline reproduction | 21.67 dB on the Phase 1 sample |
| 9 | Peak VRAM at batch 8 | < 2.0 GB measured |
| 10 | AMP stability | 1 epoch with no scaler overflow loop |
| 11 | Seeded reproducibility | two runs identical to 1e-4 after 200 steps |
| 12 | ONNX export | parity with PyTorch < 1e-3, opset ≥ 17 |

---

## 9. Explicitly excluded from V1

Soft data consistency · separate edge/texture/structure branches · LKA blocks · cross-scale exchange ·
median/MAD normalisation · LPIPS loss · content-adaptive routing · native 256→512 mode · Mamba/SSM ·
window attention · GAN losses · diffusion · BatchNorm · any convolution at 256×256.
