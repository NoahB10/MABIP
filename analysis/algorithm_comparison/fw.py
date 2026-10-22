"""Shared framework: preprocessing, labels, decision layer, scoring, causality check."""
import numpy as np, pandas as pd, data
GUARD = 20.0
TODAY = ('261022_1627', '261022_1653')     # real low-volume runs (wells pumped until dry)
# Real dry wells, labelled from plots: (run, move index) -> (name, t_air = first sample of the
# suction-break dip, s after the @P move, confidence)
REAL = {('261022_1627', 0): ('A1', 133.5, 'high'), ('261022_1627', 1): ('A2', 120.2, 'high'),
        ('261022_1627', 2): ('A2-re', 71.0, 'medium'), ('261022_1627', 3): ('A3', 142.5, 'medium-high'),
        ('261022_1627', 4): ('A4', 127.0, 'medium-high'), ('261022_1653', 0): ('A5', 189.4, 'medium-high'),
        ('261022_1653', 1): ('A6', 181.6, 'medium-high')}
LIFT_OFF = 3.0      # needle leaves liquid ~3 s after @P+dwell (dip timing, 16 runs)

def roll_med(v, n=400, minp=20):
    return pd.Series(v).rolling(n, min_periods=minp).median().bfill().to_numpy()

def prep(r):
    """Causal per-channel noise sigma (trailing robust std of first differences) and z of first differences."""
    X = r.X.astype(float); n = len(X)
    dx = np.vstack([np.zeros((1, 6)), np.diff(X, axis=0)])
    sig = np.zeros_like(X)
    for c in range(6):
        s = 1.4826 * roll_med(np.abs(dx[:, c])) / 1.4142
        # first-diff MAD can be 0 when data is quantised: floor at half the quantum
        sig[:, c] = np.maximum(s, 7e-4)
    r.dx = dx; r.sig = sig; r.Zd = dx / sig
    return r

def set_lifts(r):
    for m in r.moves:
        if m['how'] == 'dwell': m['lift'] = m['t'] + m['dwell'] + LIFT_OFF
    return r

def run_lag(r):
    if r.key in TODAY: return 47.0
    cm = np.median(r.Zd[:, :4], axis=1); d = []
    for m in r.moves:
        if m['dwell'] < 60: continue
        w = (r.t >= m['lift'] + 5) & (r.t <= min(m['lift'] + 110, m['next']))
        if w.sum() < 10: continue
        j = np.argmax(cm[w])
        if cm[w][j] > 6: d.append(r.t[w][j] - m['lift'])
    return float(np.median(d)) if len(d) >= 3 else 25.0

def label_events(r):
    """Sample indices of surrogate air spikes (arrival) and lift dips, for template/classifier learning."""
    cm = np.median(r.Zd[:, :4], axis=1); sp, dp = [], []
    for k, m in enumerate(r.moves):
        if r.key in TODAY or m['dwell'] < 60: continue
        w = np.where((r.t >= m['lift'] + 0.5 * r.lag) & (r.t <= m['lift'] + 1.6 * r.lag) & (r.t < m['next']))[0]
        if len(w) > 3:
            j = w[np.argmax(cm[w])]
            if cm[j] > 6: sp.append(j)
        w = np.where((r.t >= m['lift'] - 5) & (r.t <= m['lift'] + 8))[0]
        if len(w) > 3:
            j = w[np.argmin(cm[w])]
            if cm[j] < -4: dp.append(j)
    r.spk_idx = np.array(sp, int); r.dip_idx = np.array(dp, int)

def flat_signal(r, F=12, k=4.0, nmin=3):
    """Causal 'stale' test (as low_volume.signal_fresh, but ch1-4 common-mode): level now vs F samples ago."""
    X = r.X[:, :4]; n = len(X)
    med5 = pd.DataFrame(X).rolling(5, min_periods=1).median().to_numpy()
    ch = np.abs(med5 - np.vstack([np.full((F, 4), np.nan), med5[:-F]])) / r.sig[:, :4]
    fl = (np.nan_to_num(ch, nan=1e9) < k).sum(1) >= nmin
    return fl

def windows(r):
    """Per move: (arm, end, kind, t_air, lift). kind 'surr' or 'real'."""
    out = []
    for k, m in enumerate(r.moves):
        arm = m['t'] + r.lag + GUARD; end = min(m['next'], r.t[-1] + 1e-6)
        if r.key in TODAY:
            if (r.key, k) in REAL: out.append(dict(k=k, arm=arm, end=end, kind='real', tair=m['t'] + REAL[(r.key, k)][1], lift=m['lift'], name=REAL[(r.key, k)][0]))
            else: out.append(dict(k=k, arm=arm, end=end, kind='real?', tair=None, lift=m['lift'], name=f'm{k}'))
        else:
            if m['dwell'] < 60: continue
            if r.t[-1] < m['lift'] + r.lag + 30: continue
            out.append(dict(k=k, arm=arm, end=end, kind='surr', tair=m['lift'], lift=m['lift'], name=f'm{k}'))
    return out

def decide(r, spike, dip, rule, flat=None, train_s=30.0, stale_lo=20.0, stale_hi=120.0):
    """spike/dip: boolean arrays at DECISION index. Returns {move k: (t_call, reason)} (first EMPTY per move)."""
    t = r.t; calls = {}
    si = np.where(spike)[0]; di = np.where(dip)[0]
    tol = max(10.0, 0.15 * r.lag)
    for w in windows(r):
        S = t[si[(t[si] >= w['arm']) & (t[si] < w['end'])]]
        D = t[di[(t[di] >= w['arm']) & (t[di] < w['end'])]]
        cand = []
        if 'spike' in rule and len(S): cand.append((S[0], 'spike'))
        if 'pair' in rule:
            for j, s in enumerate(S):
                if np.any((D < s) & (s - D <= r.lag + tol)): cand.append((s, 'dip+spike')); break
                if j > 0 and s - S[j - 1] <= train_s: cand.append((s, 'spike train')); break
        if 'dipalone' in rule and len(D): cand.append((D[0], 'dip'))
        if 'stale' in rule and len(D) and flat is not None:
            for d in D:
                ok = np.where(flat & (t >= d + stale_lo) & (t <= d + stale_hi) & (t < w['end']))[0]
                if len(ok): cand.append((t[ok[0]], 'dip+stale')); break
        if cand: calls[w['k']] = min(cand)
    return calls

def score(runs, calls_by_run):
    """Returns dict of metrics + per-well detail for today's real wells."""
    hits = n = fa = 0; inwell_s = 0.0; lat = []; real = []; fa_real = 0; inwell_real = 0.0; nwell = 0
    for r in runs:
        C = calls_by_run.get(r.key, {})
        for w in windows(r):
            c = C.get(w['k'])
            if w['kind'] == 'surr':
                n += 1; nwell += 1; inwell_s += max(0.0, w['lift'] - 5 - w['arm'])
                if c is None: continue
                if c[0] < w['lift'] - 5: fa += 1
                elif c[0] <= w['lift'] + r.lag + 30: hits += 1; lat.append(c[0] - w['lift'])
            else:
                end_ok = w['lift'] + 5
                tair = w['tair'] if w['tair'] is not None else w['lift']
                inwell_real += max(0.0, tair - 5 - w['arm'])
                if c is not None and c[0] < tair - 5: fa_real += 1; real.append((w['name'], 'FA', c[0] - r.moves[w['k']]['t'], c[1])); continue
                if w['tair'] is None: continue
                if c is not None and c[0] <= end_ok: real.append((w['name'], 'hit', c[0] - r.moves[w['k']]['t'], c[1], c[0] - tair))
                else: real.append((w['name'], 'miss', None, None))
    rh = sum(1 for x in real if x[1] == 'hit'); rn = len(REAL)
    rlat = [x[4] for x in real if x[1] == 'hit']
    tot_fa = fa + fa_real; hrs = (inwell_s + inwell_real) / 3600
    return dict(surr_recall=hits / max(n, 1), surr_n=n, real_recall=rh / rn, real_hits=rh, real_n=rn,
                fa_per_h=tot_fa / max(hrs, 1e-9), fa_per_well=tot_fa / max(nwell + len(REAL), 1), fa_n=tot_fa, inwell_h=hrs,
                lat_surr_med=float(np.median(lat)) if lat else np.nan, lat_real_med=float(np.median(rlat)) if rlat else np.nan,
                lat_real_capped=float(np.mean([min(max(x[4], 0), 60) if x[1] == 'hit' else 60.0 for x in real if x[1] != 'FA'] + [60.0] * sum(1 for x in real if x[1] == 'FA' and x[0] in [v[0] for v in REAL.values()]))) if real else 60.0,
                real_detail=real)

def load_all(refresh=False):
    R = data.get_runs(refresh)
    out = []
    for r in R:
        set_lifts(r); prep(r); r.lag = run_lag(r); label_events(r); out.append(r)
    return out

def causality_check(method_fn, r, params, n_cut=None, delay=0):
    """Re-run on a truncated copy; decision arrays must match on the common prefix."""
    import copy
    full = method_fn(r, **params)
    n = n_cut or int(len(r.t) * 0.6)
    rr = copy.copy(r); rr.X = r.X[:n]; rr.t = r.t[:n]; prep(rr)
    rr.lag = r.lag; rr.spk_idx = r.spk_idx[r.spk_idx < n - 10]; rr.dip_idx = r.dip_idx[r.dip_idx < n - 10]
    part = method_fn(rr, **params)
    bad = 0
    for key in ('spike', 'dip'):
        a, b = full[key][:n], part[key]
        bad += int(np.sum(a != b))
    return bad
