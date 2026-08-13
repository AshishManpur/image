"""Frozen SPARC-Base V1.0 per-module budget table for the implementation contract."""
M, G, MB = 1e6, 1e9, 1e6
B = 2  # fp16 bytes

rows = []  # (stage, op, res, cin, cout, params, macs, act_elems, out_shape)

def conv(ci, co, k, hw, g=1, bias=True):
    return (k*k*ci*co//g + (co if bias else 0), k*k*ci*co//g*hw)

def naf(C, hw):
    p = m = 0
    for f in [conv(C,2*C,1,hw), conv(2*C,2*C,3,hw,g=2*C), conv(C,C,1,hw),
              conv(C,2*C,1,hw), conv(C,C,1,hw)]:
        p += f[0]; m += f[1]
    p += C*C + C          # SCA 1x1
    p += 4*C              # 2 x LayerNorm2d (w,b)
    p += 2*C              # 2 x LayerScale
    m += C*C
    act = 15.0*C*hw
    return p, m, act

def gsa(C, hw, heads):
    d = C//2; p = m = 0
    for f in [conv(C,3*d,1,hw), conv(3*d,3*d,3,hw,g=3*d), conv(d,C,1,hw),
              conv(C,2*C,1,hw), conv(2*C,2*C,3,hw,g=2*C), conv(C,C,1,hw)]:
        p += f[0]; m += f[1]
    m += 2*hw*hw*d
    n = int(hw**0.5)
    p += heads*(2*n-1)**2 + 4*C + 2*C
    act_sdpa = 11.5*C*hw
    act_naive = act_sdpa + heads*hw*hw
    return p, m, act_sdpa, act_naive, heads*(2*n-1)**2

def fuse(C, hw):
    p = m = 0
    r = max(C//4, 4)
    for f in [conv(2*C,C,1,hw), conv(C,r,1,1), conv(r,C,1,1)]:
        p += f[0]; m += f[1]
    return p, m, 3.0*C*hw

C0,C1,C2,CH = 48,96,160,32
H0,H1,H2 = 64*64, 32*32, 16*16
HHEAD = 128*128
naive_extra = 0.0

def add(stage, op, res, cin, cout, p, m, act, shape):
    rows.append((stage,op,res,cin,cout,p,m,act,shape))

# noise head
p=m=0
for ci,co,hw in [(1,32,128*128),(16,48,64*64),(24,64,32*32),(32,64,16*16)]:
    f=conv(ci,co,3,hw); p+=f[0]; m+=f[1]
    p += 2*(co//2)   # LayerNorm2d on post-SG width
p += 32*64+64 + 32*2+2
m += 32*64 + 32*2 + 25*128*128
add("1 Noise","4x[Conv3x3 s2+LN+SG] -> GAP -> MLP -> softplus","128->8",1,2,p,m,3.0*16*64*64+25*16384,"(B,1,128,128) sigma")
add("0 Norm","per-image mean/std, invertible","128",1,1,0,3*16384,2.0*16384,"(B,1,128,128)")
add("2 Stem","concat[y,sigma] -> HaarDWT","128->64",2,8,0,4*16384,8.0*H0,"(B,8,64,64)")
add("2 Stem","Conv3x3(8->48)","64",8,C0,*conv(8,C0,3,H0),1.0*C0*H0,"(B,48,64,64)")

for i in range(4):
    add("3 Enc L0",f"NAFBlock #{i+1}","64",C0,C0,*naf(C0,H0),"(B,48,64,64)")
add("3 Down0","HaarDWT","64->32",C0,4*C0,0,4*C0*H1,4.0*C0*H1,"(B,192,32,32)")
add("3 Down0","Conv1x1(192->96)","32",4*C0,C1,*conv(4*C0,C1,1,H1),1.0*C1*H1,"(B,96,32,32)")
for i in range(4):
    add("3 Enc L1",f"NAFBlock #{i+1}","32",C1,C1,*naf(C1,H1),"(B,96,32,32)")
for i in range(2):
    p,m,a,an,rp = gsa(C1,H1,3); naive_extra += an-a
    add("3 Enc L1",f"GSABlock #{i+1} (h=3,d=48,hd=16, relpos {rp})","32",C1,C1,p,m,a,"(B,96,32,32)")
add("3 Down1","HaarDWT","32->16",C1,4*C1,0,4*C1*H2,4.0*C1*H2,"(B,384,16,16)")
add("3 Down1","Conv1x1(384->160)","16",4*C1,C2,*conv(4*C1,C2,1,H2),1.0*C2*H2,"(B,160,16,16)")
for i in range(4):
    add("3 Enc L2",f"NAFBlock #{i+1}","16",C2,C2,*naf(C2,H2),"(B,160,16,16)")
for i in range(3):
    p,m,a,an,rp = gsa(C2,H2,5); naive_extra += an-a
    add("3 Enc L2",f"GSABlock #{i+1} (h=5,d=80,hd=16, relpos {rp})","16",C2,C2,p,m,a,"(B,160,16,16)")

add("4 Up1","Conv1x1(160->384)","16",C2,4*C1,*conv(C2,4*C1,1,H2),4.0*C1*H2,"(B,384,16,16)")
add("4 Up1","HaarIDWT","16->32",4*C1,C1,0,4*C1*H1,1.0*C1*H1,"(B,96,32,32)")
add("4 Dec D1","GatedFuse(96)","32",2*C1,C1,*fuse(C1,H1),"(B,96,32,32)")
p,m,a,an,rp = gsa(C1,H1,3); naive_extra += an-a
add("4 Dec D1",f"GSABlock #1 (h=3,d=48,hd=16, relpos {rp})","32",C1,C1,p,m,a,"(B,96,32,32)")
for i in range(4):
    add("4 Dec D1",f"NAFBlock #{i+1}","32",C1,C1,*naf(C1,H1),"(B,96,32,32)")
add("4 Up0","Conv1x1(96->192)","32",C1,4*C0,*conv(C1,4*C0,1,H1),4.0*C0*H1,"(B,192,32,32)")
add("4 Up0","HaarIDWT","32->64",4*C0,C0,0,4*C0*H0,1.0*C0*H0,"(B,48,64,64)")
add("4 Dec D0","GatedFuse(48)","64",2*C0,C0,*fuse(C0,H0),"(B,48,64,64)")
for i in range(4):
    add("4 Dec D0",f"NAFBlock #{i+1}","64",C0,C0,*naf(C0,H0),"(B,48,64,64)")

add("5 Head","Conv3x3(48->128)","64",C0,4*CH,*conv(C0,4*CH,3,H0),4.0*CH*H0,"(B,128,64,64)")
add("5 Head","HaarIDWT","64->128",4*CH,CH,0,4*CH*HHEAD,1.0*CH*HHEAD,"(B,32,128,128)")
for i in range(3):
    add("5 Head",f"NAFBlock #{i+1}","128",CH,CH,*naf(CH,HHEAD),"(B,32,128,128)")
add("5 Head","Conv3x3(32->4)  [LL,LH,HL,HH]","128",CH,4,*conv(CH,4,3,HHEAD),4.0*HHEAD,"(B,4,128,128)")
add("5 Head","HaarIDWT","128->256",4,1,0,4*HHEAD,1.0*4*HHEAD,"(B,1,256,256)")
add("6 Out","+ bicubic_up2(y_hat)  (global residual)","256",1,1,0,16*4*HHEAD,1.0*4*HHEAD,"(B,1,256,256)")
add("6 Out","denormalise (*s + m), clamp(0,1)","256",1,1,0,2*4*HHEAD,1.0*4*HHEAD,"(B,1,256,256)")

tp = sum(r[5] for r in rows); tm = sum(r[6] for r in rows); ta = sum(r[7] for r in rows)
print(f"{'stage':10s} {'operation':50s} {'res':9s} {'params':>10s} {'MMAC':>9s} {'act MB':>8s}  shape")
print("-"*135)
for s,o,r,ci,co,p,m,a,sh in rows:
    print(f"{s:10s} {o:50s} {r:9s} {p:10,d} {m/1e6:8.2f} {a*B/MB:7.3f}  {sh}")
print("-"*135)
print(f"{'TOTAL':10s} {'':50s} {'':9s} {tp:10,d} {tm/1e6:8.2f} {ta*B/MB:7.3f}")
print(f"\nparams = {tp/M:.4f} M  |  MACs = {tm/G:.4f} G  |  GFLOPs = {2*tm/G:.3f}")
print(f"activations/img (SDPA)  = {ta*B/MB:.1f} MB")
print(f"activations/img (naive attention) = {(ta+naive_extra)*B/MB:.1f} MB  "
      f"(+{naive_extra*B/MB:.1f} MB -> SDPA is mandatory)")
print(f"model size: fp32 {tp*4/MB:.2f} MB | fp16 {tp*2/MB:.2f} MB")
opt = tp*(4+4+8+4)/1e9
for b in (4,8,12,16):
    print(f"  batch {b:2d}: activations {b*ta*B/1e9:.2f} GB + states {opt:.3f} GB = {b*ta*B/1e9+opt:.2f} GB")

by = {}
for s,o,r,ci,co,p,m,a,sh in rows:
    k = s.split(" ",1)[1] if " " in s else s
    d = by.setdefault(k, [0,0,0.0]); d[0]+=p; d[1]+=m; d[2]+=a
print(f"\n{'stage group':12s} {'params':>10s} {'%':>6s} {'MMAC':>9s} {'%':>6s} {'act MB':>8s} {'%':>6s}")
for k,(p,m,a) in by.items():
    print(f"{k:12s} {p:10,d} {100*p/tp:5.1f}% {m/1e6:8.2f} {100*m/tm:5.1f}% {a*B/MB:7.2f} {100*a/ta:5.1f}%")
