"""SPARC-Net budget calculator v3 (final config). Pure arithmetic; no network built.
Key change vs v2: low-rank attention (qkv at C/2, GDFN expansion 2) to stop parameters
concentrating in the bottleneck, and a widened Sub-band Reconstruction Module."""
from collections import defaultdict
M, G = 1e6, 1e9
rows = []
def add(n,f,p,m): rows.append((n,f,p,m))
def conv(ci,co,k,hw,g=1,b=True): return (k*k*ci*co//g+(co if b else 0), k*k*ci*co//g*hw)
def naf(C,hw):
    p=m=0
    for f in [conv(C,2*C,1,hw),conv(2*C,2*C,3,hw,g=2*C),conv(C,C,1,hw),conv(C,2*C,1,hw),conv(C,C,1,hw)]:
        p+=f[0]; m+=f[1]
    p+=C*C+C+4*C+2; m+=C*C
    return p,m
def lka(C,hw):
    p=m=0
    for f in [conv(C,C,5,hw,g=C),conv(C,C,7,hw,g=C),conv(C,C,1,hw),
              conv(C,2*C,1,hw),conv(2*C,2*C,3,hw,g=2*C),conv(C,C,1,hw)]:
        p+=f[0]; m+=f[1]
    p+=4*C+2; return p,m
def gsa(C,hw,heads):
    """Low-rank: qkv -> 3*(C/2); attention in C/2; out proj C/2 -> C; GDFN expansion 2."""
    d=C//2; p=m=0
    for f in [conv(C,3*d,1,hw),conv(3*d,3*d,3,hw,g=3*d),conv(d,C,1,hw),
              conv(C,2*C,1,hw),conv(2*C,2*C,3,hw,g=2*C),conv(C,C,1,hw)]:
        p+=f[0]; m+=f[1]
    m += 2*hw*hw*d
    n=int(hw**0.5); p += heads*(2*n-1)**2 + 4*C + 2
    return p,m
def fuse(C,hw):
    p=m=0
    for f in [conv(2*C,C,1,hw),conv(C,C//4,1,hw),conv(C//4,C,1,hw),conv(C,C//4,1,1),conv(C//4,C,1,1)]:
        p+=f[0]; m+=f[1]
    return p,m
def stack(fn,n,*a):
    p=m=0
    for _ in range(n): x,y=fn(*a); p+=x; m+=y
    return p,m

import sys
C1,C2,C3,C4 = [int(x) for x in sys.argv[1].split(",")]
NB = [int(x) for x in sys.argv[2].split(",")]
H1,H2,H3,H4 = 128**2,64**2,32**2,16**2
HO = 256**2

p=m=0
for f in [conv(1,16,3,H1),conv(16,24,3,H2),conv(24,32,3,H3),conv(32,32,3,H4)]:
    p+=f[0]; m+=f[1]
p+=32*32+32+64+2; m+=32*32+64+9*H1
add("Noise estimator (4 conv + global head)","noise",p,m)
add("Shallow stem 2->C1 3x3","local",*conv(2,C1,3,H1))

add("Enc L1: 4x NAF @128^2x64","local",*stack(naf,NB[0],C1,H1))
add("Enc L2: 4x NAF @64^2x128","local",*stack(naf,NB[1],C2,H2))
add("Enc L2: 2x LKA @64^2x128","context",*stack(lka,2,C2,H2))
add("Enc L3: 3x NAF @32^2x192","local",*stack(naf,NB[2],C3,H3))
add("Enc L3: 2x GSA @32^2x192 (h=6)","context",*stack(gsa,2,C3,H3,6))
add("Bottleneck L4: 4x GSA @16^2x192 (h=6)","context",*stack(gsa,4,C4,H4,6))
for (a,b,hw) in [(C1,C2,H2),(C2,C3,H3),(C3,C4,H4)]:
    add(f"DWT down + 1x1 {4*a}->{b}","freq",*conv(4*a,b,1,hw))
for (b,a,hw) in [(C4,C3,H4),(C3,C2,H3),(C2,C1,H2)]:
    add(f"1x1 {b}->{4*a} + IDWT up","freq",*conv(b,4*a,1,hw))
add("Dec L3: 1x GSA @32^2x192","context",*stack(gsa,1,C3,H3,6))
add("Dec L3: 2x NAF @32^2x192","local",*stack(naf,2,C3,H3))
add("Dec L2: 4x NAF @64^2x128","local",*stack(naf,NB[1],C2,H2))
add("Dec L1: 4x NAF @128^2x64","local",*stack(naf,NB[0],C1,H1))
p=m=0
for C,hw in [(C1,H1),(C2,H2),(C3,H3)]:
    x,y=fuse(C,hw); p+=x; m+=y
add("Gated skip fusion x3","fusion",p,m)
p=m=0
for (a,b,hw) in [(C2,C1,H1),(C3,C2,H2)]:
    x,y=conv(a,b,1,hw); p+=x; m+=y
add("Cross-scale exchange x2","fusion",p,m)

add("SRM shared trunk: 3x NAF @128^2x64","local",*stack(naf,NB[3],C1,H1))
for tag,name,w,nb in [("edge","LH+HL oriented path",NB[4],NB[5]),
                      ("texture","HH texture path",NB[4],NB[5]),
                      ("structure","LL structure path",32,2)]:
    p,m = conv(C1,w,1,H1); x,y = stack(naf,nb,w,H1); p+=x; m+=y
    add(f"SRM {name}: 1x1 + {nb}x NAF @128^2x{w}",tag,p,m)
p=m=0
for w,co in [(48,2),(48,1),(32,1)]:
    x,y=conv(w,co,3,H1); p+=x; m+=y
add("SRM sub-band heads (LH+HL / HH / LL)","sr",p,m)
add("Orthogonal IDWT x2 upsample (Haar, fixed)","sr",0,4*H1)
p=m=0
for ci,co in [(3,16),(16,16),(16,1)]:
    x,y=conv(ci,co,3,HO); p+=x; m+=y
p+=2; m+=2*(25*HO)+H1+HO
add("Soft data consistency (1 step)","dc",p,m)

tp=sum(r[2] for r in rows); tm=sum(r[3] for r in rows)
print(f"{'stage':48s} {'tag':10s} {'params':>11s} {'GMAC':>9s} {'%MAC':>6s}")
print("-"*93)
for n,f,p,m in rows:
    print(f"{n:48s} {f:10s} {p/M:10.3f}M {m/G:8.3f}G {100*m/tm:5.1f}%")
print("-"*93)
print(f"{'TOTAL':48s} {'':10s} {tp/M:10.3f}M {tm/G:8.3f}G     GFLOPs={2*tm/G:.1f}")
gp,gm=defaultdict(float),defaultdict(float)
for n,f,p,m in rows: gp[f]+=p; gm[f]+=m
tgt={"local":38,"context":15,"texture":14,"edge":12,"freq":8,"sr":7}
print(f"\n{'function':12s} {'params':>10s} {'par%':>6s} {'GMAC':>8s} {'MAC%':>6s} {'target':>7s}")
for f in ["local","context","edge","texture","structure","freq","sr","fusion","noise","dc"]:
    t=tgt.get(f); print(f"{f:12s} {gp[f]/M:9.3f}M {100*gp[f]/tp:5.1f}% {gm[f]/G:7.3f}G "
                        f"{100*gm[f]/tm:5.1f}% {(str(t)+'%') if t else '-':>7s}")
print(f"\nHF-reconstruction group (edge+texture+structure+sr): "
      f"{(gp['edge']+gp['texture']+gp['structure']+gp['sr'])/M:.3f}M params, "
      f"{100*(gm['edge']+gm['texture']+gm['structure']+gm['sr'])/tm:.1f}% MACs")
att=sum(2*hw*hw*(C//2)*n for C,hw,n in [(C3,H3,3),(C4,H4,4)])
print(f"Exact-MHSA matmul only: {att/G:.3f} GMAC = {100*att/tm:.1f}% of total")
for B in (1,8,16,32):
    a=B*2*(C1*H1*8+C2*H2*8+C3*H3*8+C4*H4*8+HO*6)
    print(f"batch {B:2d}: peak feature activations ~{a/1e6:7.1f} MB fp16")
