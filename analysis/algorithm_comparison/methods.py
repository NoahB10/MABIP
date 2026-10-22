"""Ten candidate air-event detectors. Each fn(r, **p) -> dict(spike=bool[n], dip=bool[n], delay=samples).
Arrays are indexed by DECISION sample: True at i means 'decided using samples <= i'."""
import numpy as np, pandas as pd, sys
from numpy.lib.stride_tricks import sliding_window_view as swv
sys.path.insert(0, '/home/rpi/Documents/MABIP')
CH = [0, 1, 2, 3]

def shift(b, D):
    """event at sample c -> decision at c+D."""
    out = np.zeros_like(b)
    if D == 0: return b.copy()
    out[D:] = b[:-D]; return out

def trail_mad(v, n=400, minp=20):
    s = pd.Series(np.abs(v - pd.Series(v).rolling(n, min_periods=minp).median().bfill().to_numpy()))
    return np.maximum(1.4826 * s.rolling(n, min_periods=minp).median().bfill().to_numpy(), 1e-9)

def centred_median(x, h):
    n = len(x); out = np.full(n, np.nan)
    if n > 2 * h: out[h:n - h] = np.median(swv(x, 2 * h + 1), axis=1)
    return out

# ---------------------------------------------------------------- 1 baseline
def m1_baseline(r, **p):
    from well_empty_detector import WellEmptyDetector, WellEmptyConfig
    det = WellEmptyDetector(WellEmptyConfig(lag_s=r.lag))
    n = len(r.t); spike = np.zeros(n, bool); dip = np.zeros(n, bool); empty = []
    k = 0; mv = r.moves
    for i in range(n):
        while k < len(mv) and mv[k]['t'] <= r.t[i]:
            det.note_move(mv[k]['t']); k += 1
        e = det.update(r.t[i], r.X[i])
        if e is None: continue
        if e.kind == 'EMPTY': empty.append((r.t[i], e.reason)); spike[i] = True
        elif 'dip' in e.reason: dip[i] = True
    return dict(spike=spike, dip=dip, delay=4, empty=empty)

# ---------------------------------------------------------------- 2 robust z of first differences
def m2_diffz(r, k=8, nmin=3, fall=True, D=2, span=1, cm=None, amin=0.0, amax=99.0, rec=0):
    """Robust z of first differences (MAD-scaled), multi-channel vote.
    span=2 also tests the 2-sample drop (dips that take 2 samples); cm = max/min abs-amplitude
    ratio gate across voters (air is common-mode in ABSOLUTE units); amin/amax = mean abs drop window;
    rec>0 = require the dip to stop falling / start recovering within rec samples (delay rec)."""
    Z = r.Zd[:, CH]
    rise = (Z > k).sum(1) >= nmin
    if fall:   # rise must be followed by a fall within D samples (a spike, not a step)
        Zm = np.median(Z, 1); f = np.zeros(len(Z), bool)
        for d in range(1, D + 1):
            f[:-d] |= Zm[d:] < -0.25 * Zm[:-d]
        sp = shift(rise & f, D)
    else:
        sp = rise
    X = r.X[:, CH]; n = len(X)
    drop = np.vstack([np.zeros((1, 4)), X[1:] - X[:-1]])
    if span >= 2:
        d2 = np.vstack([np.zeros((2, 4)), X[2:] - X[:-2]])
        drop = np.where(d2 < drop, d2, drop)
    zdrop = drop / r.sig[:, CH] / (1.0 if span < 2 else 1.4142)
    vote = zdrop < -k
    dp = vote.sum(1) >= nmin
    if cm is not None or amin > 0 or amax < 99:
        a = np.abs(np.where(vote, drop, np.nan))
        with np.errstate(all='ignore'):
            ratio = np.nanmax(a, 1) / np.maximum(np.nanmin(a, 1), 1e-4)
            mean = np.nanmean(a, 1)
        ok = np.ones(n, bool)
        if cm is not None: ok &= np.nan_to_num(ratio, nan=1e9) < cm
        ok &= (np.nan_to_num(mean) >= amin) & (np.nan_to_num(mean) <= amax)
        dp &= ok
    recent_rise = pd.Series(rise.astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    dp &= ~np.r_[False, recent_rise[:-1]]
    if rec > 0:
        # within rec samples the common-mode level must stop falling and recover >= 10 % of the drop
        cmx = X.mean(1); ok = np.zeros(n, bool)
        for i in np.where(dp)[0]:
            if i + rec >= n: continue
            pre = cmx[max(0, i - 3):i].mean() if i >= 3 else cmx[i]
            trough = cmx[i:i + rec + 1].min(); depth = pre - trough
            ok[i] = depth > 0 and cmx[i + rec] - trough >= 0.1 * depth
        dp = shift(ok, rec)
    return dict(spike=sp, dip=dp, delay=(D if fall else 0), dip_delay=rec)

# ---------------------------------------------------------------- 3 Hampel residual
def m3_hampel(r, h=3, k=6, nmin=3, scale='noise'):
    n = len(r.t); RS = np.zeros((n, 4))
    for j, c in enumerate(CH):
        x = r.X[:, c]; med = centred_median(x, h); res = x - med
        if scale == 'local':
            mad = np.full(n, np.nan)
            mad[h:n - h] = 1.4826 * np.median(np.abs(swv(x, 2 * h + 1) - med[h:n - h, None]), axis=1)
            s = np.maximum(mad, r.sig[:, c] * 2)
        else:
            s = r.sig[:, c]
        RS[:, j] = np.nan_to_num(res / s)
    sp = (RS > k).sum(1) >= nmin; dp = (RS < -k).sum(1) >= nmin
    return dict(spike=shift(sp, h), dip=shift(dp, h), delay=h, RS=RS)

# ---------------------------------------------------------------- 4 Savitzky-Golay derivative
def m4_sg(r, h=2, k=4, L=3):
    w = np.arange(-h, h + 1, dtype=float); coef = w / np.sum(w * w)   # poly-2 SG 1st derivative == local slope
    n = len(r.t); DZ = np.zeros((n, 4))
    for j, c in enumerate(CH):
        d = np.full(n, 0.0); d[h:n - h] = swv(r.X[:, c], 2 * h + 1) @ coef
        DZ[:, j] = d / trail_mad(d)
    cm = np.median(DZ, 1)
    rmax = pd.Series(cm).rolling(L, min_periods=1).max().shift(1).fillna(0).to_numpy()
    rmin = pd.Series(cm).rolling(L, min_periods=1).min().shift(1).fillna(0).to_numpy()
    sp = (rmax > k) & (cm < -0.5 * k)            # rise then fall
    dp = (rmin < -k) & (cm > -0.2 * k)           # fall then stop falling / recover
    dp &= ~(pd.Series(sp.astype(float)).rolling(6, min_periods=1).max().to_numpy() > 0)
    dp &= pd.Series(rmax).rolling(6, min_periods=1).max().to_numpy() < k   # not the tail of a spike
    return dict(spike=shift(sp, h), dip=shift(dp, h), delay=h)

# ---------------------------------------------------------------- 5 causal prominence / width peak search
def _causal_peaks(cm, prom, wmax, D, W=20):
    from scipy.signal import find_peaks
    n = len(cm); out = np.zeros(n, bool)
    pk, _ = find_peaks(cm)
    for p in pk:
        if p + D >= n or p < 2: continue
        left = cm[max(0, p - W):p]; right = cm[p + 1:p + D + 1]
        base = max(left.min(), right.min()); pr = cm[p] - base
        if pr < prom: continue
        half = cm[p] - pr / 2; seg = cm[max(0, p - W):p + D + 1]; q = p - max(0, p - W)
        a = q
        while a > 0 and seg[a - 1] > half: a -= 1
        b = q
        while b < len(seg) - 1 and seg[b + 1] > half: b += 1
        if b == len(seg) - 1: continue      # not back down within D: unknown width -> undecided
        if b - a + 1 <= wmax: out[p + D] = True
    return out

def m5_prominence(r, prom=10, wmax=4, D=4, dip_wmax=12):
    cm = np.cumsum(np.median(r.Zd[:, CH], 1))
    sp = _causal_peaks(cm, prom, wmax, D)
    dp = _causal_peaks(-cm, prom, dip_wmax, D) if False else _dip_drop(cm, prom, D)
    return dict(spike=sp, dip=dp, delay=D)

def _dip_drop(cm, prom, D):
    """dips recover too slowly for a width test within D: use drop from the trailing max over <=3 samples."""
    n = len(cm); dropped = cm - pd.Series(cm).rolling(4, min_periods=1).max().to_numpy()
    return dropped < -prom

# ---------------------------------------------------------------- 6 Ricker CWT
def ricker(a, half):
    t = np.arange(-half, half + 1, dtype=float); w = (1 - (t / a) ** 2) * np.exp(-t ** 2 / (2 * a * a))
    return w - w.mean()

def m6_cwt(r, scales=(1.0, 1.5, 2.0), k=6, nmin=3, nscale=2, D=5):
    n = len(r.t); vs = np.zeros(n, int); vd = np.zeros(n, int)
    for a in scales:
        half = min(D, int(np.ceil(4 * a))); kern = ricker(a, half)
        ups = np.zeros(n, int); dns = np.zeros(n, int)
        for c in CH:
            co = np.zeros(n); co[half:n - half] = swv(r.X[:, c], 2 * half + 1) @ kern
            z = co / trail_mad(co)
            ups += z > k; dns += z < -k
        vs += ups >= nmin; vd += dns >= nmin      # centre-indexed
    # 'ridge': event present at >= nscale scales; decided D samples after the centre
    return dict(spike=shift(vs >= nscale, D), dip=shift(vd >= nscale, D), delay=D)

def _sh(b, D):
    return shift(b, D)

# ---------------------------------------------------------------- 7 CUSUM / Page-Hinkley
def m7_cusum(r, nu=2.0, h=20.0, maxrun=4):
    u = np.median(r.Zd[:, CH], 1); n = len(u)
    sp = np.zeros(n, bool); dp = np.zeros(n, bool)
    gp = gm = 0.0; lp = lm = 0
    for i in range(n):
        gp = max(0.0, gp + u[i] - nu); gm = max(0.0, gm - u[i] - nu)
        lp = lp + 1 if gp > 0 else 0; lm = lm + 1 if gm > 0 else 0
        if gp > h:
            if lp <= maxrun: sp[i] = True       # fast accumulation only (spike), not a slow analyte rise
            gp = 0.0; lp = 0
        if gm > h:
            if lm <= maxrun: dp[i] = True
            gm = 0.0; lm = 0
    rr = pd.Series(sp.astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    dp &= ~np.r_[False, rr[:-1]]
    return dict(spike=sp, dip=dp, delay=0)

def ph_freeze(r, delta=0.1, lam=8.0):
    """Page-Hinkley on log|dx| (common-mode): detects a DROP in sample-to-sample movement (frozen signal)."""
    v = np.log(np.median(np.abs(r.dx[:, CH]) / r.sig[:, CH], 1) + 0.05); n = len(v)
    out = np.zeros(n, bool); mean = 0.0; m = 0.0; M = 0.0; cnt = 0
    for i in range(n):
        cnt += 1; mean += (v[i] - mean) / min(cnt, 200)
        m += mean - v[i] - delta      # positive drift when v falls below its mean
        M = min(M, m)
        if m - M > lam: out[i] = True; m = M = 0.0
    return out

# ---------------------------------------------------------------- 8 matched filter
TPL_PRE, TPL_POST = 3, 6
def templates(runs, exclude=None):
    acc = {'spike': [], 'dip': []}
    for r in runs:
        if r.key == exclude: continue
        for kind, idx in (('spike', r.spk_idx), ('dip', r.dip_idx)):
            for j in idx:
                if j - 8 < 0 or j + TPL_POST >= len(r.t): continue
                for c in CH:
                    x = r.X[j - TPL_PRE:j + TPL_POST + 1, c]; b = np.median(r.X[j - 8:j - 2, c])
                    acc[kind].append((x - b) / r.sig[j, c])
    return {k: np.median(np.array(v), 0) for k, v in acc.items()}

def m8_matched(r, tpl=None, rho=0.8, k=10, nmin=3):
    D = TPL_POST; L = TPL_PRE + TPL_POST + 1; n = len(r.t); out = {}
    for kind in ('spike', 'dip'):
        T = tpl[kind] - tpl[kind].mean(); Tn = T / np.linalg.norm(T)
        votes = np.zeros(n, int)
        for c in CH:
            W = swv(r.X[:, c] / r.sig[:, c], L)          # windows starting at s, centre s+PRE, decision s+L-1
            W0 = W - W.mean(1, keepdims=True)
            amp = W0 @ Tn; ncc = amp / (np.linalg.norm(W0, axis=1) + 1e-9)
            ok = (ncc > rho) & (amp > k)
            votes[L - 1:] += ok
        out[kind] = votes >= nmin
    return dict(spike=out['spike'], dip=out['dip'], delay=D)

# ---------------------------------------------------------------- 9 common-mode projection
def pcs(runs, exclude=None, h=3):
    acc = {'spike': [], 'dip': []}
    for r in runs:
        if r.key == exclude: continue
        RS = r.RS3
        for kind, idx in (('spike', r.spk_idx), ('dip', r.dip_idx)):
            for j in idx:
                if j < len(RS): acc[kind].append(RS[j])
    out = {}
    for kind, A in acc.items():
        A = np.array(A); u, s, vt = np.linalg.svd(A - 0 * A.mean(0), full_matrices=False)
        w = vt[0] * np.sign(vt[0].sum()); out[kind] = w / np.linalg.norm(w)
    return out

def m9_cmproj(r, pc=None, k=10, rho=0.9, mode='pca', h=3):
    RS = r.RS3; nrm = np.linalg.norm(RS, axis=1) + 1e-9
    if mode == 'pca':
        ws, wd = pc['spike'], pc['dip']
    else:
        ws = wd = np.ones(4) / 2.0
    ss = RS @ ws; sd = RS @ wd
    sp = (ss > k) & (ss / nrm > rho); dp = (sd < -k) & (-sd / nrm > rho)
    return dict(spike=shift(sp, h), dip=shift(dp, h), delay=h)

# ---------------------------------------------------------------- 10 learned classifier (features)
FD = 3
def features(r):
    """Feature row for event centre c, decided at c+FD. Returns F with row index = DECISION index."""
    n = len(r.t); Zd = np.clip(r.Zd, -200, 200); RS = np.clip(r.RS3, -200, 200)
    cols = []
    for off in (-2, -1, 0, 1, 2, 3):       # first-diff z around the centre (ch1-4 median/min/max, ch5/6)
        Zs = np.roll(Zd, -off, axis=0)
        cols += [np.median(Zs[:, CH], 1), Zs[:, CH].min(1), Zs[:, CH].max(1), Zs[:, 4], Zs[:, 5]]
    cols += [np.median(RS, 1), RS.min(1), RS.max(1), np.std(RS, 1)]
    F = np.column_stack(cols)
    F = np.sign(F) * np.log1p(np.abs(F))      # compress heavy tails
    out = np.zeros_like(F); out[FD:] = F[:-FD]   # row i holds centre i-FD
    out[:FD + 3] = 0.0                           # np.roll wrap-around at the very start
    return out

def m10_clf(r, model=None, theta=0.9):
    F = r.F10; ps = model['spike'](F); pd_ = model['dip'](F)
    return dict(spike=ps > theta, dip=pd_ > theta, delay=FD, ps=ps, pd=pd_)
