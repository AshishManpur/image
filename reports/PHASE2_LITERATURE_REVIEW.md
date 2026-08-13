# Phase 2 — Literature Review and Building-Block Selection

**Scope:** research only. No architecture is proposed here; Phase 2 ends with a justified list of
components. Phase 1 conclusions are treated as fixed assumptions and are not re-derived.

**Fixed assumptions carried from Phase 1**

| Quantity | Measured value | Consequence for this review |
|---|---|---|
| Forward model | `GT → g_{σ=0.4} → ·Γ(L≈35)/L + N(0,σ_g²) → bicubic ↓2 noAA` | Operator is **known** ⇒ data-consistency methods become available |
| Median input SNR | **7.1 dB** (p10 0.7 dB) | Denoising is the dominant sub-problem |
| Noise law | `Var(r|I) = a + bI + cI²`, c=2.09e-2 dominant | Signal-dependent ⇒ per-pixel noise-map conditioning is applicable |
| Per-image σ | speckle 0.14–0.19, gauss 0–0.06, **unsignalled** | Must be **blind** |
| MTF droop | 1.00 → 0.89 at LR Nyquist | Deblurring capacity is nearly worthless |
| Energy above LR Nyquist | **3.7 %** | SR is a small sub-problem |
| Input resolution | **128×128** | Changes the attention cost calculus entirely (§3.1) |
| Effective training scenes | **2749** | Data-hungry architectures are penalised |
| Test vs train | HF energy ×1.16, KS D=0.215, p=8e-15 | Repetitive man-made texture ⇒ non-local self-similarity is valuable |
| Baselines | bicubic 21.67 dB · denoise-oracle 27.36 dB | Basis for capacity allocation (§6) |

---

## 1. Review methodology

Architectures are not ranked by their headline benchmark numbers, because those benchmarks
(DIV2K ×4 bicubic SR, SIDD/DND real denoising, GoPro deblurring) each stress a different sub-problem
from ours. Instead each method is scored against **eight axes derived from the measured degradation**,
and the review then extracts *mechanisms* rather than *networks*.

Two properties of this problem dominate everything and should be kept in view throughout:

1. **Input is 128×128.** Almost all efficiency arguments in the 2023–2026 literature are written for
   512×512–2048×2048 inputs. At 128×128 the asymptotic complexity class of the global-context operator is
   nearly irrelevant (§3.1). This invalidates the usual "quadratic attention is unaffordable, therefore
   Mamba" argument for this specific problem.
2. **The degradation operator is known and the noise is signal-dependent.** This unlocks two families of
   mechanism — noise-map conditioning and data-consistency/unfolding — that most general-purpose
   restoration backbones do not use, and which are far better matched to the measurements than any
   backbone choice.

> **Caveat on reported figures.** Parameter counts are from the original papers and are reliable.
> FLOPs/MACs are reported by different papers at different resolutions and channel counts (usually
> 3-channel, 256×256 or 1280×720). I state the measurement setting wherever I give a number, and mark
> derived figures as estimates. Treat all FLOPs as ±30 % and re-measure before committing to any design.

---

## 2. Literature review by family

### 2.1 CNN denoising lineage — DnCNN, FFDNet, RIDNet, DRUNet, SCUNet

**DnCNN** (TIP 2017; the ancestor of everything here). *Core idea:* a plain 17–20 layer conv stack that
predicts the **noise residual** rather than the clean image. *Structure:* Conv-BN-ReLU ×17, no downsampling.
*Complexity:* O(HW·C²·k²), fully local, receptive field ≈ 35 px. *Params:* 0.56 M. *FLOPs:* ~36 GMACs at
256²×3. *Memory:* very low (no skips to hold). *Speed:* fast. *Why it succeeds:* residual learning removes
the low-frequency component from the optimisation target, which conditions the loss landscape far better at
high noise — this is the single most transferable result in the denoising literature. *Why it fails:*
fixed-σ (a separate model per noise level), no global context, BN is harmful at test time under
distribution shift. *Failure cases:* smooth gradients get residual texture; blind noise.

**FFDNet** (TIP 2018). *Core idea:* **concatenate a noise-level map `M` as an extra input channel**, so a
single network covers a wide σ range and spatially-varying noise. Operates on a ×2 pixel-unshuffled input
for speed. *Params:* 0.49 M (gray). *Why it succeeds:* the noise level is an *observable nuisance
parameter*; supplying it converts a blind problem into a conditional one and removes the need for the
network to spend capacity estimating it internally. *Why it fails:* needs `M` at test time, which for real
noise must be estimated. *Relevance here — very high:* our noise law is **analytically known**,
`σ(I) = sqrt(a + bI + cI²)`, so a per-pixel map is computable from the input itself. This is the
best-matched single idea in the entire denoising literature to this dataset.

**RIDNet** (ICCV 2019). Single-stage blind real-image denoiser with feature attention (channel attention on
residual groups). 1.5 M params. Introduced the useful result that **channel attention is worth more than
depth** in denoisers. Weak on long-range structure.

**DRUNet** (TPAMI 2021, the DPIR prior). *Core idea:* U-Net + residual blocks + **FFDNet-style noise-map
conditioning**, used as a plug-and-play Gaussian denoiser prior inside a HQS solver. 32.6 M params.
*Why it matters here:* it is the strongest evidence that **noise-map conditioning + a U-Net with residual
blocks generalises across a very wide σ range from a single model** — precisely the blind, per-image-varying
condition we measured. *Weakness:* large for what it does; global context only via the U-Net bottleneck.

**SCUNet** (2023). Swin-Conv block inside a U-Net, plus a practical degradation-synthesis pipeline for real
noise. ~17 M. Chiefly relevant for its **degradation-synthesis** contribution rather than the backbone: it
demonstrates that matched synthetic degradation is worth more than architecture on OOD real noise. Directly
supports the Phase 1 recommendation to re-synthesise LR from GT rather than augment the supplied LR.

**Suitability summary (CNN denoisers):** best-in-class for Gaussian/speckle at low SNR, excellent parameter
efficiency, excellent stability, trivial export. Weak on high-frequency recovery and non-local
self-similarity.

---

### 2.2 CNN restoration backbones — HINet, MPRNet, NAFNet

**HINet** (CVPRW 2021). Two-stage UNet with **Half-Instance Normalisation** (IN on half the channels,
identity on the rest). 88.7 M params. *Why it succeeds:* IN provides contrast/appearance invariance where BN
fails; splitting channels retains the scale information that pure IN destroys. *Why it fails here:* very
heavy; two-stage design costs ~2× inference for modest gain.

**MPRNet** (CVPR 2021). Multi-stage progressive restoration with cross-stage feature fusion and supervised
attention. 20 M, ~760 GMACs at 256². *Idea worth keeping:* **supervised attention between stages** (produce
an intermediate image, supervise it, use it to gate features). *Cost:* multi-stage is expensive and largely
superseded by better single-stage designs.

**NAFNet** (ECCV 2022, and still the efficiency reference in 2026). *Core idea:* remove **all** nonlinear
activations. Replace GELU with **SimpleGate** (split channels, elementwise multiply: `x₁ ⊙ x₂`) and channel
attention with **Simplified Channel Attention** (global average pool → 1×1 → multiply, no sigmoid, no
hidden nonlinearity). Block = LayerNorm → 1×1 → 3×3 depthwise → SimpleGate → SCA → 1×1, plus a
LayerNorm → 1×1 → SimpleGate → 1×1 FFN, both residual with learnable scale.
*Structure:* 4-level UNet, block counts skewed to the bottleneck (e.g. [1,1,1,28]).
*Complexity:* O(HW·C²) from the 1×1s; depthwise 3×3 is negligible. Strictly linear in pixels.
*Params:* NAFNet-width32 **17.1 M**; width64 **67.9 M**. *FLOPs:* width32 ≈ **16 GMACs** at 256²×3;
width64 ≈ 63 GMACs. *Memory:* low — no attention maps, activations dominated by the widest level.
*Speed:* the fastest strong restoration backbone per dB; SimpleGate is a single elementwise multiply.
*Strengths:* superb accuracy/FLOP ratio, extremely stable training, trivially exportable to ONNX/TensorRT
(every op is a standard conv/mul/reduce), no attention kernels to fuse.
*Weaknesses:* receptive field is only as global as the UNet bottleneck; SCA gives *global channel* context
but **no spatial content-based matching** — it cannot exploit non-local self-similarity.
*Failure cases:* large repetitive structures where distant evidence would help; strong structured artefacts.
*Why it succeeds:* denoising is dominated by local low-level statistics, and removing nonlinearities both
reduces cost and removes the optimisation pathologies that come with them.
*Suitability here — very high* as the local workhorse.

---

### 2.3 Transformer family — IPT, SwinIR, Uformer, Restormer, HAT, DAT, GRL, ART, ATD, DRCT

**IPT** (CVPR 2021). ViT on image patches, 115 M params, requires ImageNet-scale pretraining
(1.1 M images). *Verdict here: rejected outright* — 2749 effective scenes is four orders of magnitude short
of its data requirement.

**SwinIR** (ICCVW 2021). Shifted-window self-attention (8×8 windows) in Residual Swin Transformer Blocks,
with a global residual and a PixelShuffle tail. Params: **11.8 M** (classical SR), 11.5 M (denoising),
**0.9 M** (lightweight). Complexity: O(HW·w²·C) — linear in pixels, quadratic only in window area.
*Strengths:* strong PSNR per parameter; window attention is genuine content-based spatial matching within
its window; well-understood, stable. *Weaknesses:* shifted windows are slow in practice (masking, rolls,
awkward padding), poor ONNX/TensorRT export, and receptive field grows only linearly with depth.
*Failure cases:* window-boundary artefacts; content whose self-similar match is farther than the effective
receptive field. *Suitability:* good but export-hostile.

**Uformer** (CVPR 2022). U-shaped transformer with **Locally-enhanced Window** attention (a depthwise conv
in the FFN) and modulators. Uformer-B ≈ **50.9 M**, ~89 GMACs at 256². *Idea worth keeping:* injecting a
depthwise conv into the FFN — the local inductive bias that pure attention lacks. *Weakness:* heavy;
the U-shape plus windows makes it neither the fastest nor the most accurate at its size.

**Restormer** (CVPR 2022). *Core idea:* **Multi-Dconv Head Transposed Attention (MDTA)** — compute the
attention matrix across the **channel** dimension instead of the spatial dimension:

$$\hat{A} = \mathrm{Softmax}\!\left(\frac{\hat{Q}\hat{K}^{\top}}{\alpha}\right) \in \mathbb{R}^{C\times C}, \qquad \text{cost } O(HW\,C^{2}) \text{ instead of } O(H^{2}W^{2}C)$$

plus a **Gated-Dconv FFN**. 4-level encoder–decoder. Params **26.1 M**; ~141 GMACs at 256²×3.
*Strengths:* linear in pixels, globally-informed *channel* statistics at every layer, strong on
denoising/deraining/deblurring, stable. Depthwise 3×3 projections give local bias.
*Critical limitation for our purpose:* MDTA is **not spatial attention**. It cannot match a patch to a
distant similar patch; it re-weights feature channels using globally pooled second-order statistics. For
non-local self-similarity (our texture-heavy test set) it provides much less than its name suggests.
*Failure cases:* fine repetitive texture where BM3D-style non-local averaging would win.

**HAT** (CVPR 2023). *Core idea:* "activate more pixels" — combine **channel attention + window
self-attention + an overlapping cross-attention block** to widen the range of pixels actually used.
HAT **20.8 M**, HAT-L **40.8 M**; SOTA PSNR on classical SR benchmarks, aided by ImageNet pretraining.
*Strengths:* highest PSNR of the window-transformer line; the hybrid channel+spatial attention idea is
sound and directly targets the receptive-field weakness of SwinIR. *Weaknesses:* very heavy, slow, benefits
materially from large-scale pretraining, export-hostile. *Verdict:* mechanism yes, network no.

**DAT** (ICCV 2023) alternates spatial-window and channel-group attention across blocks — 14.8 M,
a cheaper expression of HAT's insight. **GRL** (CVPR 2023) explicitly organises attention into
**anchored stripe (global), window (regional), channel (local)** — the cleanest existing statement of the
"different ranges need different operators" principle; GRL-B 20.2 M. **ART** (ICLR 2023) interleaves dense
and sparse (dilated) attention windows to enlarge receptive field cheaply — **sparse/dilated attention is a
cheap route to non-local matching**. **ATD** (CVPR 2024) introduces an **adaptive token dictionary**: a
learned global codebook that tokens attend to, giving global context at O(N·K) instead of O(N²), plus
similarity-based token grouping — a direct, efficient mechanism for non-local self-similarity.
**DRCT** (CVPRW 2024) adds dense residual connections inside SwinIR to stop information bottlenecking in
deep SR networks (27.6 M).

---

### 2.4 State Space Models — Vision Mamba, VMamba, LocalMamba, MambaIR, MambaIRv2

**Core idea.** A selective state-space model computes, along a scanned 1-D sequence,

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t$$

with input-dependent `(Δ, B, C)`. Cost is **O(N·d·d_state)** — linear in tokens with a global receptive
field along the scan.

**Vision Mamba / VMamba** adapt this to 2-D by scanning in multiple directions (VMamba's cross-scan uses 4
directions) to compensate for the fact that a 1-D scan destroys 2-D adjacency. **LocalMamba** adds
window-local scans because global scans lose local continuity. **MambaIR** (ECCV 2024) applies this to
restoration with a Residual State Space Block augmented by channel attention and a local conv, explicitly
because plain SSM under-uses local pixels; **MambaIRv2** (CVPR 2025) attacks the causality limitation with
an attentive state-space equation plus a semantic-guided neighbouring mechanism, so a single scan can attend
beyond its own prefix.

*Params:* MambaIR ≈ 20 M (SR), MambaIR-light ≈ 0.9 M. *Memory:* low. *Speed:* **theoretically** linear but
**practically** dependent on a fused associative-scan CUDA kernel; without it, throughput collapses.

*Strengths:* global receptive field at linear cost; excellent at very high resolution.
*Weaknesses, and they are decisive for this project:*
1. **The 1-D causal scan is a poor prior for 2-D images.** The entire MambaIR/LocalMamba/MambaIRv2 line
   consists of patches for this. Each patch adds cost and complexity.
2. **Deployment.** The selective scan is a custom CUDA kernel. There is no standard ONNX operator, and
   TensorRT support requires a hand-written plugin. Phase 1's brief explicitly requires ONNX/TensorRT
   export and fast inference. This is a hard blocker, not an inconvenience.
3. **The efficiency argument does not apply at 128×128** (§3.1). Mamba buys linear scaling we do not need,
   and pays for it in kernel complexity and 2-D-prior mismatch.

*Verdict:* **reject for this project.** The idea worth keeping is the *goal* — global context at
sub-quadratic cost — not the mechanism.

---

### 2.5 Frequency-domain methods

**Wavelet CNNs / MWCNN / WaveletSRNet.** Replace pooling with a **Discrete Wavelet Transform** (Haar) and
unpooling with the inverse DWT. Properties that matter: DWT is **invertible, lossless, non-redundant, and
free** (Haar is a fixed 2×2 orthogonal transform — 4 adds/subs per 2×2 block, zero parameters). Downsampling
by DWT loses no information, unlike strided conv or pooling, and separates LL/LH/HL/HH sub-bands so the
network can treat low- and high-frequency content differently.
*Why it succeeds:* enlarges receptive field per level without information loss; the sub-band split is a
genuinely useful inductive bias for restoration.
*Why it fails / when not to use:* Haar has poor frequency selectivity and introduces blocking if used
carelessly; **learned wavelet packets and deep multi-level wavelet trees add parameters and instability for
little measured gain**.

**FFT-based restoration.** FFC/DeepRFT/FSNet/SFNet-style blocks apply 1×1 convs in the Fourier domain,
giving an **image-wide receptive field in one layer** at O(HW log HW). *Strengths:* genuinely global and
cheap; excellent for periodic/global degradations. *Weaknesses:* real FFT layers assume circular boundary
conditions (edge artefacts), are awkward to export, are sensitive to resolution changes, and — critically
for us — **our noise is white and signal-dependent**, so it is not concentrated in any frequency band that a
Fourier gate could isolate. A Fourier *loss* is far better motivated than a Fourier *layer* here.

**Laplacian pyramid networks (LapSRN and successors).** Progressive band-by-band reconstruction. *Designed
for ×4/×8 SR.* At ×2 with only 3.7 % new energy the pyramid adds machinery for one level — no benefit.

**Frequency attention / FcaNet.** Channel attention using DCT bases rather than global average pooling
(GAP is exactly the lowest DCT coefficient, so GAP discards all frequency information). Cheap, small,
occasionally worth a point. Low priority.

---

### 2.6 Large-kernel CNNs — RepLKNet, SLaK, UniRepLKNet, VAN/LKA, ConvNeXt

**Large Kernel Attention** (VAN, 2022): factorise a large-kernel spatial attention into
depthwise (5×5) → dilated depthwise (7×7, d=3) → 1×1, approximating a 21×21 kernel at a fraction of the
cost, then use it multiplicatively as attention. **RepLKNet/UniRepLKNet** show 31×31–51×51 depthwise kernels
are trainable and give ViT-like effective receptive fields with pure convolution. **ConvNeXt** blocks
(7×7 depthwise → LN → 1×1 expand ×4 → GELU → 1×1) are the standard modernised residual block.

*Strengths:* CNN inductive bias (translation equivariance — valuable with only 2749 scenes), large effective
receptive field, **fully exportable**, no custom kernels, fast in FP16/TensorRT.
*Weaknesses:* a large kernel is a **fixed, content-independent** aggregation. It enlarges the receptive
field but does **not** perform content-based matching, so it is a weak substitute for attention when the
task genuinely requires finding a similar patch elsewhere. Depthwise large kernels are also
memory-bandwidth-bound rather than FLOP-bound, so their wall-clock cost exceeds their FLOP count.
*When to use:* as a cheap regional-context operator between local blocks and true global attention.
*When not to use:* as the sole global-context mechanism on self-similar texture.

---

### 2.7 Deep unfolding and data consistency — USRNet, DPIR, plug-and-play

**USRNet** (CVPR 2020) unfolds a MAP objective for SR with a **known** blur kernel and scale factor into
alternating data-consistency and denoising steps; **DPIR** plugs DRUNet into HQS. *Core idea:* when the
forward operator `A` is known, alternate
`z ← prox_data(x; y, A)` and `x ← Denoiser(z; σ)`, rather than learning the inverse map blindly.

*Why this matters uniquely here:* Phase 1 **recovered the operator** — blur σ=0.4, bicubic ↓2 no-AA. Almost
no general restoration backbone exploits a known operator; we can. A differentiable data-consistency step
enforces `A·x̂ ≈ y` and typically buys 0.3–1.0 dB essentially for free, and improves OOD behaviour because
it is a hard physical constraint rather than a learned correlation.

*Critical caveat:* at 7 dB SNR, a **hard** data-consistency projection re-injects the noise it is meant to
suppress. The correct form is a **soft, noise-weighted** consistency term (weight ∝ 1/σ²(I)), or a single
learned back-projection refinement rather than a full unfolded iteration. Full unfolding also multiplies
inference cost by the iteration count, which conflicts with the speed requirement.

---

### 2.8 Diffusion and generative restoration — DiffIR, ResShift, SinSR, StableSR

*Core idea:* model `p(x|y)` and sample, rather than regress the conditional mean.
*Strengths:* by far the best perceptual quality/LPIPS on severe SR; genuinely synthesises plausible detail.
*Weaknesses:* multi-step inference (ResShift 15 steps, SinSR 1 step but distilled from a large teacher);
**PSNR is systematically worse than regression** because sampling deliberately deviates from the posterior
mean; hallucination is unacceptable in an inspection context; training is expensive and unstable.
*Verdict:* rejected — the competition weights PSNR and SSIM, and inference speed is scored.

---

### 2.9 Efficient SR line — RLFN, SAFMN, ECBSR, CAMixerSR, SMFANet

Relevant as evidence on **how little is needed for ×2**. SAFMN (ICCV 2023) reaches competitive ×4 SR with
**0.24 M** params using spatially-adaptive feature modulation over a multi-scale pyramid; RLFN and ECBSR
show re-parameterisable plain convs matching much heavier designs at inference. CAMixerSR routes
content-adaptively between cheap conv and expensive attention **per token**, spending attention only where
the content is complex — a directly applicable idea given that ~30 % of our training images are
low-texture (§Phase 1 §5.1).

---

## 3. Analysis: three claims that determine the design

### 3.1 At 128×128, quadratic attention is affordable — the standard efficiency argument does not apply

Self-attention costs ≈ `2·N²·C` MACs for `QKᵀ` and `AV` combined, where `N = H·W`.

| Stage in a 4-level encoder | Tokens `N` | Channels `C` | Attention MACs |
|---|---|---|---|
| Level 1 — 128×128 | 16 384 | 48 | **25.8 G** — unaffordable |
| Level 2 — 64×64 | 4 096 | 96 | **3.2 G** — borderline |
| Level 3 — 32×32 | 1 024 | 192 | **0.40 G** — cheap |
| Level 4 — 16×16 | 256 | 384 | **0.05 G** — free |

For comparison, all of NAFNet-width32 is ~16 GMACs at 256²×3.

**Conclusion.** Full, unrestricted, content-based global self-attention at levels 3–4 costs **under 0.5
GMACs** — roughly 3 % of a small NAFNet. There is no need for Mamba, linear attention, RWKV or Hyena to
obtain global context on this input size. The efficiency literature's premise (long sequences) simply does
not hold at 128×128, and adopting an SSM here means paying its costs (1-D prior mismatch, custom CUDA
kernel, no ONNX path) to solve a problem we do not have.

This is the most consequential finding of Phase 2 and it contradicts the direction suggested in the brief.

### 3.2 Non-local self-similarity is worth real capacity — but it must be *spatial*, not channel

Phase 1 measured that the test split is dominated by repetitive man-made texture (brick, grid, tiling) with
1.16× the high-frequency energy of train. Repetitive content is exactly the regime where **non-local
means/BM3D-style aggregation** beats local filtering — averaging over `k` genuinely similar patches reduces
noise variance by ~`k` without blurring, which local smoothing cannot do at 7 dB SNR without destroying
detail.

This distinguishes the candidate global operators sharply:

| Operator | Global? | **Content-based spatial matching?** | Cost at 32×32 |
|---|---|---|---|
| NAFNet SCA (channel attention) | channel only | **No** | negligible |
| Restormer MDTA (transposed attention) | channel only | **No** | low |
| Large-kernel / LKA | regional, fixed | **No** | low |
| FFT block | global, fixed basis | **No** | low |
| Mamba / SSM | global along scan | partial (via selectivity) | low |
| **Full spatial self-attention** | **yes** | **Yes** | **0.4 GMAC** |
| Sparse/dilated or anchored-stripe attention | yes | Yes | lower |
| ATD-style token dictionary | yes | Yes (via codebook) | low |

Given §3.1, the operator that best delivers what the data actually needs is also affordable. Channel
attention (SCA/MDTA) remains valuable and near-free, but it should be understood as a *feature
recalibration* mechanism, not as the non-local mechanism.

### 3.3 The noise is signal-dependent and analytically characterised — this is the biggest free win

`σ(I) = sqrt(a + bI + cI²)` with `c` dominant. Two literature-backed mechanisms exploit this:

* **Noise-map conditioning (FFDNet/DRUNet).** Concatenate a computed per-pixel σ-map to the input. Evidence:
  a single conditioned network matches per-σ specialist networks across a wide range, and handles spatially
  varying noise. Our map is computable in closed form from the input plus one or two globally-estimated
  scalars. This converts our blind problem into a conditional one at the cost of one input channel.
* **Homomorphic / variance-stabilising transforms.** Classical SAR despeckling takes `log`, turning
  `y = x·n` into `log y = log x + log n`. **Evidence-based caveat from the despeckling literature:** after
  the log transform the speckle is stationary but **neither Gaussian nor zero-mean**, requiring an explicit
  bias correction; and our additive Gaussian floor `σ_g ≤ 0.06` breaks pure multiplicativity, while
  `log(I)` diverges as `I → 0` — and Phase 1 found genuinely near-zero pixels (5.6 % below 0.05, GT min
  exactly 0). A hard log transform is therefore **not** recommended as a core component; a *soft*,
  numerically-guarded variance-stabilising input branch is at most an ablation.

**Recommendation: noise-map conditioning as a core component; homomorphic transform as an optional ablation
only.**

---

## 4. Comparison tables

### 4.1 Cost and complexity

| Method | Year | Family | Params | Reported FLOPs (setting) | Global-context cost | Export |
|---|---|---|---|---|---|---|
| DnCNN | 2017 | CNN | 0.56 M | ~36 GMAC @256²×3 | none | trivial |
| FFDNet | 2018 | CNN + noise map | 0.49 M | ~2 GMAC @256²×1 | none | trivial |
| RIDNet | 2019 | CNN + CA | 1.5 M | ~49 GMAC @256² | channel only | trivial |
| DRUNet | 2021 | UNet + noise map | 32.6 M | ~72 GMAC @256² | bottleneck only | trivial |
| HINet | 2021 | CNN 2-stage | 88.7 M | ~171 GMAC @256² | bottleneck only | easy |
| MPRNet | 2021 | CNN multi-stage | 20.1 M | ~760 GMAC @256² | bottleneck only | easy |
| **NAFNet-32 / -64** | 2022 | CNN | **17.1 / 67.9 M** | **16 / 63 GMAC @256²** | channel only | **trivial** |
| SwinIR / -light | 2021 | Window transformer | 11.8 / 0.9 M | high (window-quadratic) | O(HW·w²·C) | poor |
| Uformer-B | 2022 | UNet transformer | 50.9 M | ~89 GMAC @256² | O(HW·w²·C) | poor |
| Restormer | 2022 | Transposed attention | 26.1 M | ~141 GMAC @256² | O(HW·C²) | moderate |
| HAT / HAT-L | 2023 | Hybrid attention | 20.8 / 40.8 M | very high | window + channel + OCA | poor |
| DAT | 2023 | Dual attention | 14.8 M | high | alternating | poor |
| GRL-B | 2023 | Multi-range attention | 20.2 M | high | stripe + window + channel | poor |
| ART | 2023 | Sparse attention | 16.6 M | high | dense + dilated windows | poor |
| ATD | 2024 | Token dictionary | ~20 M | moderate | O(N·K) | moderate |
| DRCT | 2024 | Dense SwinIR | 27.6 M | very high | window | poor |
| MambaIR / -light | 2024 | SSM | ~20 / 0.9 M | linear in N | O(N·d·d_state) | **blocked** |
| MambaIRv2 | 2025 | SSM + attention | ~20 M | linear in N | O(N·d) | **blocked** |
| SAFMN | 2023 | Efficient SR | 0.24 M | very low | multi-scale modulation | trivial |
| CAMixerSR | 2024 | Adaptive routing | ~1–4 M | content-dependent | routed attention | moderate |
| DiffIR / ResShift | 2023–24 | Diffusion | 25–120 M | ×N steps | — | poor |

FLOPs entries are collated from the original papers at the stated settings; they are **not** directly
comparable across rows and are given for order-of-magnitude only.

### 4.2 Suitability against the eight measured requirements

Scale: ●●● strong · ●● adequate · ● weak · ✗ unsuitable

| Method | Speckle | Gaussian | Blur | **Blind** | ×2 SR | Grayscale | **Low SNR** | **HF recovery** |
|---|---|---|---|---|---|---|---|---|
| DnCNN | ●● | ●●● | ● | ✗ | ✗ | ●●● | ●● | ● |
| **FFDNet** | ●● | ●●● | ● | **●●●** | ● | ●●● | ●●● | ● |
| **DRUNet** | ●●● | ●●● | ●● | **●●●** | ●● | ●●● | **●●●** | ●● |
| RIDNet | ●● | ●●● | ● | ●● | ✗ | ●●● | ●● | ● |
| SCUNet | ●●● | ●●● | ●● | ●●● | ●● | ●●● | ●●● | ●● |
| HINet | ●● | ●●● | ●●● | ●● | ●● | ●●● | ●● | ●● |
| **NAFNet** | ●●● | ●●● | ●●● | ●● | ●● | ●●● | **●●●** | ●● |
| SwinIR | ●● | ●●● | ●● | ●● | ●●● | ●●● | ●● | **●●●** |
| Uformer | ●● | ●●● | ●●● | ●● | ●● | ●●● | ●● | ●● |
| **Restormer** | ●●● | ●●● | ●●● | ●● | ●● | ●●● | **●●●** | ●● |
| HAT | ●● | ●●● | ●● | ● | **●●●** | ●●● | ●● | **●●●** |
| GRL / ART / ATD | ●● | ●●● | ●● | ●● | ●●● | ●●● | ●● | **●●●** |
| MambaIR(v2) | ●● | ●●● | ●● | ●● | ●●● | ●●● | ●● | ●●● |
| Wavelet CNN | ●● | ●●● | ●● | ●● | ●● | ●●● | ●●● | ●● |
| FFT networks | ● | ●● | ●●● | ● | ●● | ●●● | ● | ●● |
| Laplacian pyramid | ● | ● | ● | ● | ●● (×4+) | ●●● | ● | ●● |
| **USRNet / DPIR** | ●● | ●●● | ●●● | ● (needs op) | ●●● | ●●● | ●● | ●●● |
| Diffusion | ● | ●● | ●● | ●● | ●●● | ●● | ● | ●●● (hallucinated) |
| SAFMN / CAMixerSR | ● | ●● | ● | ● | ●● | ●●● | ● | ●● |

The two columns that carry the most weight for this dataset are **Blind** and **Low SNR** (7 dB median), and
the methods that lead those columns are the noise-map-conditioned CNNs — not the SR transformers that lead
the HF column.

### 4.3 Ranking on the ten requested criteria

Rank 1 = best. Rankings are for **this problem**, not in general.

| Criterion | 1 | 2 | 3 | 4 | 5 | Last |
|---|---|---|---|---|---|---|
| **PSNR potential** | Restormer | HAT | NAFNet-64 | DRUNet | GRL | Diffusion |
| **SSIM potential** | Restormer | NAFNet | DRUNet | HAT | SwinIR | Diffusion |
| **LPIPS potential** | Diffusion | HAT | GRL/ATD | SwinIR | Restormer | FFDNet |
| **Inference speed** | FFDNet | SAFMN | NAFNet-32 | DnCNN | DRUNet | Diffusion |
| **Memory** | FFDNet | SAFMN | NAFNet-32 | MambaIR-light | Restormer | HAT-L |
| **Generalisation (2749 scenes)** | FFDNet | DRUNet | NAFNet | Restormer | SCUNet | IPT |
| **OOD robustness** | DRUNet | FFDNet | SCUNet | NAFNet | Restormer | HAT-L |
| **Training stability** | NAFNet | DRUNet | FFDNet | Restormer | SwinIR | Diffusion/GAN |
| **Implementation complexity** | DnCNN | FFDNet | NAFNet | DRUNet | Restormer | MambaIRv2 |
| **Deployment complexity** | NAFNet | FFDNet | DRUNet | Restormer | SwinIR | MambaIR(v2) |

**Reading of the table.** No single existing network wins. The CNN-with-noise-map line
(FFDNet → DRUNet → SCUNet) dominates every axis that this dataset actually stresses — blind operation, low
SNR, OOD robustness, generalisation from little data, speed, exportability. The transformer line dominates
only high-frequency recovery, which Phase 1 measured to be worth ~16 % of the available headroom. **This
asymmetry is the entire design argument.**

---

## 5. Design trade-offs

| Trade-off | Evidence | Resolution for this project |
|---|---|---|
| PSNR ↔ LPIPS | 3.7 % of GT energy is genuinely absent from the input; recovering it perceptually requires synthesis that deviates from the posterior mean | Optimise PSNR/SSIM primarily; add a **small** perceptual term late in training only |
| Capacity ↔ overfitting | 2749 effective scenes; transformers historically need ImageNet-scale pretraining (IPT, HAT) | Target **5–15 M params**, CNN-dominant, with aggressive re-synthesis augmentation |
| Global context ↔ speed | §3.1: full attention at ≤32×32 costs <0.5 GMAC | Use **true spatial attention, but only at low resolution** |
| Local detail ↔ receptive field | Denoising at 7 dB needs aggregation; detail needs locality | Hierarchical encoder: local blocks at high res, attention at low res |
| Known operator ↔ noise amplification | Data consistency helps, but hard projection re-injects noise at 7 dB | **Soft, noise-weighted** consistency; not full unfolding |
| Depth ↔ trainability | NAFNet's removal of nonlinearities improves stability; deep transformers need warmup/careful init | Prefer NAFNet-style blocks |
| Accuracy ↔ export | Mamba scan and Swin windows are export-hostile; the brief requires ONNX/TensorRT | Restrict to ops with standard ONNX coverage |

---

## 6. Capacity allocation analysis

### 6.1 Deriving the split from measured dB headroom

| Milestone | PSNR | Δ |
|---|---|---|
| Bicubic ×2 of the noisy input | 21.67 dB | — |
| Perfect denoising, still bicubic upsampling | 27.36 dB | **+5.69 dB** |
| Realistic strong model (denoise + learned SR) | ~28.5–29 dB (est.) | +1.1–1.6 dB |

Denoising therefore accounts for roughly **5.69 / 6.8 ≈ 84 %** of the attainable improvement, and all
super-resolution/high-frequency synthesis for the remaining **≈16 %**.

A pure 84/16 parameter split would be wrong, though, for two reasons: (i) denoising exhibits strongly
diminishing returns in parameters — FFDNet reaches within ~0.3 dB of DRUNet with **66× fewer parameters** —
whereas HF synthesis is parameter-hungry; and (ii) LPIPS is scored, and LPIPS is driven almost entirely by
the high-frequency component. The allocation below therefore shifts capacity toward HF relative to the raw
dB share, deliberately and by a stated margin.

### 6.2 Recommended allocation

| Function | Share | Justification |
|---|---|---|
| **Noise removal (local restoration)** | **38 %** | Dominant sub-problem (84 % of dB headroom) but with steep diminishing returns per parameter; this is the largest single block, sized below its dB share on efficiency grounds |
| **Global context / non-local self-similarity** | **15 %** | Serves denoising primarily (§3.2 — non-local averaging is how you denoise repetitive texture without blurring it) and is cheap in FLOPs (§3.1); test split is texture-heavy, so this is where OOD robustness is bought |
| **Texture / high-frequency recovery** | **14 %** | Directly targets the 3.7 % missing band and dominates LPIPS; parameter-hungry, so over-weighted relative to its 16 % dB share |
| **Edge / structure preservation** | **12 %** | Not a separate degradation, but the principal *failure mode* of denoising at 7 dB — edges are where the denoiser destroys signal. Justified as protection of the 38 % block's output, not as an independent task |
| **Frequency-domain processing** | **8 %** | Wavelet (invertible, free) sub-band separation for lossless down/upsampling and band-selective processing; deliberately small because the noise is white and not band-isolable (§2.5) |
| **Super-resolution tail (upsampling)** | **7 %** | Only ×2, only 3.7 % new energy, and PixelShuffle is nearly free; more capacity here has been repeatedly shown to be wasted at low scale factors |
| **Deblurring** | **0 %** | MTF droop is 11 %; no dedicated capacity. Whatever inversion is needed is absorbed by the local blocks |

**Denoising-serving capacity in total** = 38 + 15 + 12 + part of 8 ≈ **68 %**, versus an 84 % dB share —
the 16-point gap is the deliberate, justified transfer to LPIPS-relevant high-frequency capacity.

---

## 7. Modules that should be used

Each entry states the mechanism, the literature evidence, and the measured fact that justifies it.

| Mechanism | Evidence | Measured justification |
|---|---|---|
| **Noise-map conditioning** | FFDNet, DRUNet/DPIR | `Var = a+bI+cI²` is known in closed form; per-image σ varies 0.022–0.188 and is unsignalled |
| **NAFNet-style local block** (SimpleGate + SCA + depthwise) | NAFNet | Best accuracy/FLOP for the 38 % noise block; export-trivial; most stable training |
| **Residual (noise-domain) learning** | DnCNN | Conditions the loss landscape at 7 dB SNR; standard in every strong restoration net |
| **Hierarchical encoder–decoder** | Restormer, NAFNet, Uformer, DRUNet | Needed to reach low resolution cheaply so global attention becomes affordable (§3.1) |
| **True spatial self-attention at low resolution only** | HAT, GRL, ART, ATD | §3.1 (cost <0.5 GMAC) + §3.2 (self-similar test texture) |
| **Channel attention (SCA / MDTA)** | NAFNet, Restormer, RIDNet | Near-free feature recalibration; complements but does not replace spatial attention |
| **Depthwise conv inside the FFN** | Uformer LeWin, Restormer GDFN | Restores local inductive bias that attention lacks; cheap |
| **Haar DWT down / IDWT up inside the encoder** | MWCNN, wavelet CNNs | Lossless, parameter-free downsampling; sub-band separation for the 8 % frequency block |
| **Learnable gating / adaptive fusion at skips** | SKNet, MPRNet SAM, AFF | Phase 1: noise level varies per image, so the optimal encoder/decoder mix is input-dependent — static concatenation cannot express this |
| **PixelShuffle tail with ICNR init** | ESPCN, EDSR, SwinIR | Fastest ×2 upsampler, no checkerboard with ICNR; §6 says the tail deserves ~7 % |
| **Global residual from a bicubic upsample of the input** | SwinIR, EDSR | The network then models only the 3.7 % + noise correction, not the whole image |
| **Soft, noise-weighted data-consistency step** | USRNet, DPIR | The operator is **known** (Phase 1 §7.3) — unavailable to any generic backbone; must be soft at 7 dB |
| **Output clamp to [0,1]** | — | GT is exactly [0,1] for all 3200 images |
| **Content-adaptive routing (optional)** | CAMixerSR | ~30 % of training images are low-texture; spending attention only where needed buys speed |
| **LayerNorm (not BatchNorm)** | EDSR, NAFNet | BN harms restoration and breaks under the per-image intensity spread (means 0.016–0.959) |
| **Degradation re-synthesis augmentation** | SCUNet, BSRGAN, Real-ESRGAN | Phase 1 recovered the exact operator; re-synthesising LR from GT is strictly better than double-degrading the supplied LR |

## 8. Modules that should NOT be used

| Rejected | Why — evidence and measurement |
|---|---|
| **State Space Models / Mamba** | Linear scaling buys nothing at 128×128 (§3.1: full attention at 32² costs 0.4 GMAC); 1-D causal scan is a poor 2-D prior (the entire MambaIR→LocalMamba→MambaIRv2 line exists to patch this); **no ONNX operator and no TensorRT path without a custom plugin**, which directly violates the deployment requirement |
| **RWKV / Hyena / linear attention** | Same reasoning — they solve long-sequence cost, which is not our constraint, while being strictly weaker than exact attention |
| **Full-resolution quadratic attention** | 25.8 GMAC at level 1 (§3.1) — this *is* genuinely unaffordable. The rule is *where*, not *whether* |
| **Shifted-window (Swin) attention** | Export-hostile (rolls, masks, padding), slow in practice; at our resolutions plain global attention at low res is both simpler and stronger |
| **Deep transformers (IPT, HAT-L, DRCT)** | 2749 effective scenes; IPT needs ImageNet-scale pretraining, HAT benefits materially from it. Guaranteed overfitting |
| **GAN / adversarial loss** | Directly reduces PSNR/SSIM (both scored); unstable; at 7 dB SNR the discriminator drives hallucination of structure that is not in the input — unacceptable for an inspection framing |
| **Diffusion / iterative sampling** | Multi-step inference conflicts with the speed criterion; systematically worse PSNR than regression |
| **Heavy LPIPS weighting** | LPIPS backbones are RGB-natural-image trained (grayscale must be replicated); with 3.7 % of energy genuinely absent, a large LPIPS weight buys perceptual score by fabricating detail and costs measurable PSNR/SSIM. Small weight, late in training, only |
| **Laplacian-pyramid progressive SR** | Designed for ×4/×8; at ×2 it is one level of machinery for no gain |
| **Multi-level learned wavelet packet branches** | Cost and instability without measured benefit; the noise is white, so there is no band to isolate. Fixed Haar for resampling only |
| **FFT/Fourier convolution layers** | Circular-boundary artefacts, resolution-fragile, export-awkward; and white signal-dependent noise is not frequency-localised. **A Fourier *loss* is well motivated; a Fourier *layer* is not** |
| **Hard homomorphic (log) transform** | Despeckling literature: post-log speckle is non-Gaussian and biased, needing explicit correction; our additive floor breaks multiplicativity; `log` diverges at the genuinely-zero pixels Phase 1 found (GT min is exactly 0). Ablation only |
| **Dedicated deblurring / large deconvolution kernels** | Measured MTF droop is 11 %; there is no PSF worth inverting, and **no motion blur exists in this data at all** |
| **Multi-stage / progressive restoration (MPRNet, HINet)** | ~2× inference cost for gains that single-stage designs have since matched |
| **BatchNorm** | Harmful in restoration (EDSR); incompatible with per-image means spanning 0.016–0.959 |
| **Aggressive multi-step SR decoders** | Only ×2 total; a single PixelShuffle is provably sufficient |
| **Input clamping to [0,1]** | Would destroy the 3.36 % of input pixels above 1.0 and 0.30 % below 0 |

---

## 9. Building Blocks for Phase 3

The components below — and only these — are approved for use in Phase 3. They are listed as independent
building blocks; they are **not** yet combined into an architecture.

**A. Input conditioning**
1. **Analytic noise-map generator** — per-pixel `σ̂(I) = sqrt(â + b̂·I + ĉ·I²)` computed from the input, using globally-fitted coefficients plus a small blind per-image scale estimate. Supplied as an additional input channel. *(FFDNet/DRUNet)*
2. **Per-image robust normalisation** — median/MAD or percentile-based, invertible, with the parameters retained for exact inversion at the output. *(Justified by per-image means spanning 0.016–0.959 and unclipped inputs.)*
3. **Pixel-unshuffle stem (optional)** — trade resolution for channels at the input to cut cost. *(FFDNet)*

**B. Local restoration operator**
4. **NAFNet block** — LayerNorm → 1×1 → 3×3 depthwise → SimpleGate → Simplified Channel Attention → 1×1, plus gated FFN, both residual with learnable per-block scale. *(Primary capacity consumer: 38 %.)*
5. **Large-kernel depthwise convolution (7×7–13×13, or LKA factorisation)** — used sparingly as a regional-context operator between local blocks. *(VAN/RepLKNet; cheap receptive-field expansion with CNN inductive bias.)*

**C. Multi-scale representation**
6. **Hierarchical encoder–decoder, 3–4 levels**, channel widths growing by ~2× per level, block counts skewed toward the low-resolution levels. *(NAFNet/Restormer/DRUNet.)*
7. **Haar DWT downsampling / IDWT upsampling** as the resampling operator inside the encoder — invertible, parameter-free, sub-band separating. *(MWCNN.)*

**D. Global context**
8. **Exact spatial multi-head self-attention, restricted to the ≤32×32 levels.** *(Justified by §3.1 cost analysis and §3.2 self-similarity requirement.)*
9. **Transposed / channel attention (MDTA- or SCA-style)** at higher-resolution levels where spatial attention is unaffordable. *(Restormer/NAFNet.)*
10. **Depthwise conv inside every attention FFN** for local bias. *(Uformer LeWin / Restormer GDFN.)*

**E. Fusion**
11. **Learned gated skip fusion** — predict per-channel (and optionally per-pixel) gates from the concatenated encoder/decoder pair rather than concatenating. *(SKNet/AFF; justified by per-image noise variation.)*
12. **Cross-scale feature exchange** between adjacent pyramid levels before decoding. *(HRNet/GRL principle.)*

**F. Reconstruction**
13. **Global residual connection from a fixed bicubic ×2 upsample of the input.** *(EDSR/SwinIR.)*
14. **PixelShuffle ×2 tail with ICNR initialisation.** *(ESPCN/EDSR.)*
15. **Soft, noise-weighted data-consistency refinement** using the Phase 1 operator `A = bicubic↓2 ∘ g_{σ=0.4}`, weighted by `1/σ̂²(I)`. Single step, not a full unfolded loop. *(USRNet/DPIR, with the low-SNR caveat of §2.7.)*
16. **Output de-normalisation followed by `clamp(0,1)`.** *(GT is exactly [0,1].)*

**G. Training-side blocks (not architecture, but selected in this phase)**
17. **On-the-fly LR re-synthesis from GT** using the Phase 1 forward model with randomised σ_speckle, σ_gauss, and a widened operator distribution. *(SCUNet/BSRGAN; replaces the harmful current augmentation.)*
18. **Paired geometric augmentation only** — flips and rot90. No input-only photometric jitter. *(Phase 1 measured gain 1.00 / offset 0.00.)*
19. **Group-aware (leak-free) validation split** from the Phase 1 scene groups.
20. **Weight EMA** for stability and a small free PSNR gain.

**Explicitly excluded from Phase 3:** Mamba/SSM, RWKV, Hyena, linear attention, shifted-window attention,
full-resolution attention, GAN losses, diffusion, Laplacian pyramids, learned wavelet packet trees, Fourier
convolution layers, hard log/homomorphic transforms, dedicated deblurring branches, multi-stage
architectures, and BatchNorm.

---

## 10. What Phase 3 must still decide

Deliberately left open, since these are architecture decisions rather than component selection:

1. Exact level count, widths, and per-level block allocation against the §6.2 percentages and a FLOP budget.
2. Whether the edge (12 %) and texture (14 %) functions are realised as **separate branches** or as
   band-specialised processing inside the wavelet sub-bands — the latter is cheaper and is the working
   hypothesis, but the brief specifies branches, so this needs an explicit decision.
3. Whether the noise-map is analytic-only or refined by a small learned estimator head.
4. Whether the data-consistency step (block 15) survives an ablation at 7 dB SNR, or is dropped.
5. The loss composition and the PSNR/LPIPS operating point.

**Phase 2 ends here. Awaiting approval before Phase 3 architecture design.**
