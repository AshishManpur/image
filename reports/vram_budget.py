"""Params / MACs / TRAINING ACTIVATION MEMORY for SPARC variants under a 4 GB budget.

Phase 3 counted ~8 feature tensors per LEVEL. That is wrong: autograd saves the
output of essentially every op inside every BLOCK. This recounts per-block.
"""
from collections import defaultdict
M, G = 1e6, 1e9
BYTES = 2  # fp16 activations

# ---- saved-for-backward tensor count, expressed in multiples of C*HW ----
# NAF: LN, conv1(2C), dw(2C), SG(C), SCA(C), conv3(C), add(C),
#      LN2, conv4(2C), SG2(C), conv5(C), add(C)  -> 1+2+2+1+1+1+1+1+2+1+1+1 = 15
NAF_ACT = 15.0
# GSA: LN, qkv(3d), dw(3d), attn_out(d), proj(C), add(C), LN2,
#      conv4(2C), dw(2C), SG(C), conv5(C), add(C)   with d = C/2
GSA_ACT = 1 + 1.5 + 1.5 + 0.5 + 1 + 1 + 1 + 2 + 2 + 1 + 1 + 1

def conv(ci, co, k, hw, g=1, b=True):
    return (k * k * ci * co // g + (co if b else 0), k * k * ci * co // g * hw)

def naf(C, hw):
    p = m = 0
    for f in [conv(C,2*C,1,hw), conv(2*C,2*C,3,hw,g=2*C), conv(C,C,1,hw),
              conv(C,2*C,1,hw), conv(C,C,1,hw)]:
        p += f[0]; m += f[1]
    p += C*C + C + 4*C + 2*C; m += C*C
    return p, m, NAF_ACT * C * hw

def gsa(C, hw, heads):
    d = C // 2; p = m = 0
    for f in [conv(C,3*d,1,hw), conv(3*d,3*d,3,hw,g=3*d), conv(d,C,1,hw),
              conv(C,2*C,1,hw), conv(2*C,2*C,3,hw,g=2*C), conv(C,C,1,hw)]:
        p += f[0]; m += f[1]
    m += 2 * hw * hw * d
    n = int(hw ** 0.5); p += heads * (2*n-1)**2 + 4*C + 2*C
    act = GSA_ACT * C * hw + heads * hw * hw      # + attention probs (no flash)
    return p, m, act

def fuse(C, hw):
    """Channel-gate skip fusion (simplified AFF: 1x1 merge + SE gate)."""
    p = m = 0
    for f in [conv(2*C,C,1,hw), conv(C,max(C//4,4),1,1), conv(max(C//4,4),C,1,1)]:
        p += f[0]; m += f[1]
    return p, m, 3.0 * C * hw

def build(name, widths, enc_naf, enc_gsa, dec_naf, dec_gsa, heads,
          head_width, head_blocks, stem_dwt=True, in_res=128):
    """Trunk starts at in_res/2 when stem_dwt (lossless Haar), else in_res."""
    rows = []
    base = in_res // 2 if stem_dwt else in_res
    res = [base // (2**i) for i in range(len(widths))]
    hw = [r*r for r in res]

    stem_in = 8 if stem_dwt else 2      # DWT of [y, sigma] -> 8ch ; else 2ch
    p, m = conv(stem_in, widths[0], 3, hw[0])
    rows.append(("stem", p, m, 1.0*widths[0]*hw[0]))

    for i, C in enumerate(widths):
        for n, fn, tag in [(enc_naf[i], naf, "naf"), (enc_gsa[i], gsa, "gsa")]:
            for _ in range(n):
                r = fn(C, hw[i], heads[i]) if fn is gsa else fn(C, hw[i])
                rows.append((f"enc{i}_{tag}", *r))
        if i + 1 < len(widths):
            p, m = conv(4*C, widths[i+1], 1, hw[i+1])
            rows.append((f"dwt{i}", p, m, 4.0*C*hw[i+1] + widths[i+1]*hw[i+1]))

    for j in range(len(widths) - 2, -1, -1):
        C = widths[j]
        p, m = conv(widths[j+1], 4*C, 1, hw[j+1])
        rows.append((f"idwt{j}", p, m, 4.0*C*hw[j+1] + C*hw[j]))
        rows.append((f"fuse{j}", *fuse(C, hw[j])))
        k = len(widths) - 2 - j
        for n, fn, tag in [(dec_naf[k], naf, "naf"), (dec_gsa[k], gsa, "gsa")]:
            for _ in range(n):
                r = fn(C, hw[j], heads[j]) if fn is gsa else fn(C, hw[j])
                rows.append((f"dec{j}_{tag}", *r))

    # reconstruction head: trunk_res -> 2x with blocks, then ONE sub-band conv -> 4x.
    # No convolution ever runs at 256^2.
    C = widths[0]; cur_hw = hw[0]
    p, m = conv(C, 4*head_width, 3, cur_hw)
    rows.append(("head_proj", p, m, 4.0*head_width*cur_hw))
    cur_hw *= 4
    for _ in range(head_blocks):
        rows.append(("head_naf", *naf(head_width, cur_hw)))
    p, m = conv(head_width, 4, 3, cur_hw)
    rows.append(("head_subband", p, m, 4.0*cur_hw))
    rows.append(("final_idwt", 0, 4*cur_hw, 1.0*cur_hw*4))

    # noise head
    p = m = 0
    for f in [conv(1,32,3,in_res**2//4), conv(16,48,3,in_res**2//16),
              conv(24,64,3,in_res**2//64), conv(32,64,3,in_res**2//256)]:
        p += f[0]; m += f[1]
    p += 32*64 + 64 + 32*2 + 2
    rows.append(("noise_head", p, m, 3.0*16*in_res**2//4))

    tp = sum(r[1] for r in rows); tm = sum(r[2] for r in rows)
    ta = sum(r[3] for r in rows)
    return name, rows, tp, tm, ta

CONFIGS = [
    ("SPARC-Tiny",  (24,48,96),  (1,1,2), (0,0,0), (1,1), (0,0), (0,0,0), 16, 1),
    ("SPARC-Base",  (48,96,160), (4,4,4), (0,2,3), (4,4), (1,0), (0,6,5), 32, 3),
    ("Base-noGSA",  (48,96,160), (4,4,6), (0,0,0), (4,4), (0,0), (0,0,0), 32, 3),
    ("Base-noattn", (48,96,160), (4,4,7), (0,0,0), (4,4), (0,0), (0,0,0), 32, 3),
    ("Base-wider",  (56,112,192),(4,4,4), (0,2,3), (4,4), (1,0), (0,7,6), 40, 3),
]
print(f"{'variant':18s} {'params':>9s} {'GMAC':>8s} {'act/img':>10s} "
      f"{'B=4':>8s} {'B=8':>8s} {'B=16':>9s}")
print("-" * 78)
results = {}
for cfg in CONFIGS:
    name, rows, tp, tm, ta = build(*cfg)
    results[name] = (rows, tp, tm, ta)
    a = ta * BYTES
    print(f"{name:18s} {tp/M:8.3f}M {tm/G:7.3f}G {a/1e6:9.1f}MB "
          f"{4*a/1e9:7.2f}GB {8*a/1e9:7.2f}GB {16*a/1e9:8.2f}GB")

print("\n--- Phase-3 SPARC-B (no DWT stem, trunk at 128^2) for comparison ---")
name, rows, tp, tm, ta = build(
    "P3-SPARC-B", (64,128,192,192), (4,4,3,0), (0,0,2,4), (2,4,4), (1,0,0),
    (0,0,6,6), 48, 4, stem_dwt=False)
a = ta * BYTES
print(f"{name:18s} {tp/M:8.3f}M {tm/G:7.3f}G {a/1e6:9.1f}MB "
      f"{4*a/1e9:7.2f}GB {8*a/1e9:7.2f}GB {16*a/1e9:8.2f}GB")

print("\n--- SPARC-Base per-stage breakdown ---")
rows, tp, tm, ta = results["SPARC-Base"]
agg = defaultdict(lambda: [0,0,0.0])
for n,p,m,act in rows:
    k = n.split("_")[0]
    agg[k][0]+=p; agg[k][1]+=m; agg[k][2]+=act
for k,(p,m,act) in agg.items():
    print(f"  {k:16s} {p/M:8.4f}M {m/G:7.4f}G {act*BYTES/1e6:8.2f}MB/img "
          f"({100*act/ta:4.1f}% act)")

print("\n--- training memory budget, SPARC-Base ---")
rows, tp, tm, ta = results["SPARC-Base"]
for B in (2,4,8,16):
    act = B*ta*BYTES/1e9
    params = tp*4/1e9            # fp32 master weights
    grads = tp*4/1e9
    adam = tp*8/1e9
    ema = tp*4/1e9
    total = act + params + grads + adam + ema
    ckpt = 0.35*act + params + grads + adam + ema
    print(f"  B={B:2d}: act {act:5.2f} + states {params+grads+adam+ema:.3f} "
          f"= {total:5.2f} GB   | with activation checkpointing: {ckpt:5.2f} GB")
