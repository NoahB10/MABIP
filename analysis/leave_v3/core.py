"""Shared: load cache, STEADY tracker (atlas definition), leave simulator equivalent to leave_eval2.leaves(), scoring."""
import pickle, numpy as np, os
LDIR = "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/leave"
import feats
REAL = {('261022_1627', 0): ('A1', 133.5), ('261022_1627', 1): ('A2', 120.2), ('261022_1627', 2): ('A2-re', 71.0),
        ('261022_1627', 3): ('A3', 142.5), ('261022_1627', 4): ('A4', 127.0), ('261022_1653', 0): ('A5', 189.4), ('261022_1653', 1): ('A6', 181.6)}
EARLY = lambda key: key[:7] <= '2026-05'

def load(with_feats=True):
    pk = LDIR + '/cache/runs_feats.pkl'
    if with_feats and os.path.exists(pk): return pickle.load(open(pk, 'rb'))
    R = pickle.load(open(LDIR + '/cache/runs.pkl', 'rb'))
    if with_feats:
        for r in R:
            r['F'] = feats.compute(r); r['steady'], r['steady_t'] = steady(r)
        pickle.dump(R, open(pk, 'wb'))
    return R

def steady(r, q=0.1, eps=0.05, H=30.0, nconf=2, arm_k=8.0, vote=2, blank_after=45.0):
    """Atlas STEADY: local-linear-trend Kalman per strong channel (ch3/5/6); steady when |slope|*30 s <= 5 % of the
    excursion + 2 sigma on 2 consecutive samples, on 2 of the armed channels. Returns bool[n] (steady reached in the
    current well by sample i; 'blank' wells count as steady at baseline after move+lag+45) and per-well first time."""
    t, X, F, lag = r['t'], r['X'], r['F'], r['lag']; n = len(t); out = np.zeros(n, bool); st = {}
    mv = F['mv']; sig = F['sig']
    for j, m in enumerate(mv):
        end = mv[j + 1] if j + 1 < len(mv) else np.inf
        idx = np.where((t >= m + max(10.0, lag - 2.0)) & (t < end))[0]
        if not len(idx): continue
        b0 = F['b0'][idx[0]]; s0 = sig[idx[0]]
        kf = {}; ss = set(); done = None
        for i in idx:
            armed_n = 0
            for c in feats.STRONG:
                x = X[i, c]
                if c not in kf:
                    kf[c] = [x, 0.0, [[20 * s0[c] ** 2, 0.0], [0.0, 20 * s0[c] ** 2]], t[i], 0]
                    continue
                lv, sl, P, tp, cnt = kf[c]; dt = t[i] - tp; R_ = 2 * s0[c] ** 2; Q_ = q * s0[c] ** 2
                lvp = lv + sl * dt
                p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1]; p01 = P[0][1] + dt * P[1][1]
                p10 = P[1][0] + dt * P[1][1]; p11 = P[1][1] + Q_ * dt
                S = p00 + R_; k0, k1 = p00 / S, p10 / S; e = x - lvp
                lv = lvp + k0 * e; sl = sl + k1 * e
                P = [[p00 - k0 * p00, p01 - k0 * p01], [p10 - k1 * p00, p11 - k1 * p01]]
                if abs(x - b0[c]) >= arm_k * s0[c]:
                    armed_n += 1
                    cnt = cnt + 1 if abs(sl) * H <= eps * abs(lv - b0[c]) + 2 * s0[c] else 0
                    if cnt >= nconf: ss.add(c)
                else:
                    cnt = 0
                kf[c] = [lv, sl, P, t[i], cnt]
            if done is None:
                if armed_n and len(ss) >= min(vote, armed_n): done = t[i]
                elif armed_n == 0 and t[i] >= m + lag + blank_after: done = t[i]
            if done is not None: out[i] = True
        st[j] = done
    return out, st

def simulate_fast(r, fire, tair, rule=True, gap=5, confirm_s=20.0, gate=None):
    """Same decision logic as leave_eval2.leaves(): fire[i] = detector says EMPTY at sample i (already armed / gated by
    the candidate); tair[i] = when the tip met air.  rule=True applies low_volume's plateau rule (plateau_before at
    min(t, t_air), else wait confirm_s unless a plateau appears).  Returns {k: (t_leave, 'good'|'flagged', i_fire)}."""
    t = r['t']; P = r['P5'] if gap == 5 else r['P2']; Q = r['Q5'] if gap == 5 else r['Q2']
    k = r['F']['k']; out = {}; fi = np.where(fire)[0]
    mv = r['F']['mv']
    for j in range(len(mv)):
        end = mv[j + 1] if j + 1 < len(mv) else np.inf
        lo = np.searchsorted(t, mv[j], 'left'); hi = np.searchsorted(t, end, 'left')
        if lo >= hi: continue
        fj = fi[(fi >= lo) & (fi < hi)]
        if not len(fj): continue
        if not rule: out[j] = (t[fj[0]], 'good', fj[0]); continue
        fset = set(fj.tolist()); pend = None
        for i in range(fj[0], hi):
            if pend is not None:
                if Q[i]: pend = None
                elif t[i] - pend >= confirm_s: out[j] = (t[i], 'flagged', i); break
            if i not in fset: continue
            tref = min(t[i], tair[i]) if np.isfinite(tair[i]) else t[i]
            ir = max(0, np.searchsorted(t, tref, 'right') - 1)
            if P[ir]: out[j] = (t[i], 'good', i); break
            if pend is None: pend = t[i]
    return out

def score(r, L, steady_arr=None):
    """Per-well outcome on gold plates (surrogate windows, as leave_eval2.score_plates)."""
    rows = []
    for w in r['windows']:
        if w['kind'] != 'surr': continue
        x = L.get(w['k']); lift = w['lift']
        if x is None or x[0] > lift + r['lag'] + 30: o = 'never'; lat = np.nan
        elif x[0] < lift - 5: o = 'early'; lat = x[0] - lift
        else: o = 'right'; lat = x[0] - lift
        st = r['steady_t'].get(w['k']) if 'steady_t' in r else None
        valid = (x is not None) and (st is not None) and (st <= x[0])
        rows.append(dict(plate=r['key'], k=w['k'], outcome=o, lat=lat, kind=(x[1] if x else ''), steady_before=valid))
    return rows

def real_check(r, L):
    out = []
    for (key, k), (name, ta) in REAL.items():
        if key != r['key']: continue
        x = L.get(k); m = r['moves'][k]['t']
        out.append(dict(well=name, left=None if x is None else round(x[0] - m - ta, 1), kind=None if x is None else x[1]))
    return out

def summarize(rows):
    import pandas as pd
    d = pd.DataFrame(rows); n = len(d)
    return dict(n=n, right=(d.outcome == 'right').mean() * 100, early=(d.outcome == 'early').mean() * 100,
                never=(d.outcome == 'never').mean() * 100, n_early=int((d.outcome == 'early').sum()),
                lat=float(np.nanmedian(d.lat[d.outcome == 'right'])) if (d.outcome == 'right').any() else np.nan,
                flagged_right=int(((d.outcome == 'right') & (d.kind == 'flagged')).sum()),
                no_steady=int(((d.outcome != 'never') & ~d.steady_before).sum()))

# ---- exact re-implementation of low_volume.plateau_before(t_ref) evaluated with the controller state at sample i_now
#      (history kept 60 s = fresh_window_s + 30; baseline 600 s of 5-sample medians; noise from the last 300 |dx|)
def _prep_ctl(r):
    if '_m5' in r: return
    import pandas as pd
    X = r['X']; r['_m5'] = pd.DataFrame(X).rolling(5, min_periods=1).median().to_numpy()
    r['_adx'] = np.vstack([np.zeros((1, 6)), np.abs(np.diff(X, axis=0))])
def plateau_at(r, i_now, t_ref, gap=5.0, window=15.0, k=8.0, keep=60.0, base_w=600.0):
    _prep_ctl(r); t = r['t']; X = r['X']; tn = t[i_now]
    lo, hi = t_ref - window - gap, t_ref - gap
    a = np.searchsorted(t, max(lo, tn - keep), 'left'); b = np.searchsorted(t, min(hi, tn), 'right')
    if b - a < 4: return False
    ba = np.searchsorted(t, tn - base_w, 'left'); bb = np.searchsorted(t, min(t_ref, tn), 'right')
    if bb - ba < 10: return False
    seg = X[a:b]; base = np.sort(r['_m5'][ba:bb], 0)
    v = np.sort(seg, 0); n = len(v); lvl = np.median(v, 0)
    rng = v[int(0.9 * (n - 1))] - v[int(0.1 * (n - 1))]
    bl = base[int(0.1 * (len(base) - 1))]
    d0 = max(1, i_now - 299); noise = np.maximum(np.median(r['_adx'][d0:i_now + 1], 0) * 1.4826 / 1.4142, 1e-4)
    up = lvl - bl
    return int(np.sum((up >= k * noise) & (up >= 0.05) & (rng <= 0.5 * up + 3 * noise))) >= 2

def simulate_exact(r, fire, tair, rule=True, gap=5.0, confirm_s=20.0):
    """As simulate(), but the plateau rule is evaluated with the controller's real (60 s) history at the call time."""
    t = r['t']; out = {}; fi = np.where(fire)[0]; mv = r['F']['mv']
    for j in range(len(mv)):
        end = mv[j + 1] if j + 1 < len(mv) else np.inf
        lo = np.searchsorted(t, mv[j], 'left'); hi = np.searchsorted(t, end, 'left')
        fj = fi[(fi >= lo) & (fi < hi)]
        if not len(fj): continue
        if not rule: out[j] = (t[fj[0]], 'good', fj[0]); continue
        fset = set(fj.tolist()); pend = None
        for i in range(fj[0], hi):
            if pend is not None:
                if plateau_at(r, i, t[i] + gap, gap): pend = None
                elif t[i] - pend >= confirm_s: out[j] = (t[i], 'flagged', i); break
            if i not in fset: continue
            tref = min(t[i], tair[i]) if np.isfinite(tair[i]) else t[i]
            if plateau_at(r, i, tref, gap): out[j] = (t[i], 'good', i); break
            if pend is None: pend = t[i]
    return out


def simulate(r, fire, tair, rule=True, gap=5, confirm_s=20.0):
    """Default = exact emulation of the installed controller (validated: installed 841/22, v3 1017/0 on 1376 wells)."""
    return simulate_exact(r, fire, tair, rule=rule, gap=float(gap), confirm_s=confirm_s)
