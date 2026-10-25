"""Test the user's rule: only accept a bubble call if a steady, non-baseline plateau came right before it."""
import sys, itertools
sys.dont_write_bytecode = True
sys.argv = ["run_gold.py"]
exec(open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/run_gold.py").read().split("rows = []")[0])
dev = fw.load_all()
today = [r for r in dev if r.key in fw.TODAY]

def noise(r):
    d = np.abs(np.diff(r.X, axis=0)); return np.maximum(np.median(d, axis=0) * 1.4826 / 1.4142, 1e-4)

def _smooth_pct(r, lo_t, hi_t, q):
    m = (r.t >= lo_t) & (r.t <= hi_t)
    if m.sum() < 10: return None
    x = r.X[m]; k = 5
    sm = np.array([np.median(x[max(0, i - k + 1):i + 1], axis=0) for i in range(len(x))])
    return np.percentile(sm, q, axis=0)

def plateau_ok(r, sig, move_t, t_call, W, d, k, frac, nmin, base_mode="pre"):
    w = (r.t >= t_call - W - d) & (r.t <= t_call - d)
    if w.sum() < 4: return False
    seg = r.X[w]; lvl = np.median(seg, axis=0)
    if base_mode == "pre":
        b = (r.t >= move_t - 20) & (r.t <= move_t)
        if b.sum() < 3: return False
        up = np.abs(lvl - np.median(r.X[b], axis=0))
    else:
        lo = _smooth_pct(r, t_call - 600, t_call, 10)
        if lo is None: return False
        up = lvl - lo
        if base_mode == "both":
            hi = _smooth_pct(r, t_call - 600, t_call, 90)
            up = np.maximum(up, hi - lvl)
    rng = np.percentile(seg, 90, axis=0) - np.percentile(seg, 10, axis=0)
    good = (up >= k * sig) & (up >= 0.05) & (rng <= frac * up + 3 * sig)
    return int(good.sum()) >= nmin

def replay_gated(r, gate):
    sig = noise(r); det = installed.WellEmptyDetector(installed.WellEmptyConfig(lag_s=r.lag)); calls = {}; k = 0; mv = None
    for i in range(len(r.t)):
        while k < len(r.moves) and r.moves[k]['t'] <= r.t[i]:
            det.note_move(r.moves[k]['t']); mv = r.moves[k]['t']; k += 1
        e = det.update(r.t[i], r.X[i])
        if e is None or e.kind != 'EMPTY' or k == 0 or (k - 1) in calls: continue
        t_ref = min(r.t[i], getattr(e, 't_air', r.t[i]) or r.t[i])
        if gate is None or plateau_ok(r, sig, mv, t_ref, **gate):
            calls[k - 1] = (r.t[i], e.reason)
        else:                                   # blocked: keep listening in this well
            det._move_t = mv; det._armed = True
    return calls

def eligible(r, gate):
    """Wells that really had a non-baseline plateau before the lift-out (bubble expected)."""
    sig = noise(r); out = set()
    for w in fw.windows(r):
        if w['kind'] == 'surr' and plateau_ok(r, sig, r.moves[w['k']]['t'], w['lift'], **gate): out.add(w['k'])
    return out

def summarize(runs, gate, ref_gate):
    tot = dict(n=0, h=0, en=0, eh=0, fa=0, hrs=0.0, lat=[])
    for r in runs:
        calls = replay_gated(r, gate); s = fw.score([r], {r.key: calls}); el = eligible(r, ref_gate)
        for w in fw.windows(r):
            if w['kind'] != 'surr': continue
            c = calls.get(w['k']); hit = c is not None and w['lift'] - 5 <= c[0] <= w['lift'] + r.lag + 30
            tot['n'] += 1; tot['h'] += hit
            if w['k'] in el: tot['en'] += 1; tot['eh'] += hit
            if hit: tot['lat'].append(c[0] - w['lift'])
        tot['fa'] += s['fa_n']; tot['hrs'] += s['inwell_h']
    return tot

ref = dict(W=15, d=3, k=8, frac=0.5, nmin=2, base_mode="low")
GATES = [("installed, no plateau rule", None),
         ("plateau before air, 10-min baseline", dict(W=15, d=2, k=8, frac=0.5, nmin=2, base_mode="low")),
         ("same, k=5", dict(W=15, d=2, k=5, frac=0.5, nmin=2, base_mode="low"))]
print(f"{'rule':40s} {'all wells':>14s} {'plateau wells':>15s} {'false alarms':>13s} {'/h':>5s}")
for label, gate in GATES:
    t = summarize(runs, gate, ref)
    print(f"{label:40s} {t['h']:5d}/{t['n']:<5d}{t['h']/t['n']*100:3.0f}% {t['eh']:5d}/{t['en']:<5d}{t['eh']/max(t['en'],1)*100:3.0f}% {t['fa']:13d} {t['fa']/t['hrs']:5.2f}")
print("\n22 Oct real dry wells: which calls pass the plateau check (fail = leave anyway, flagged)")
for label, gate in GATES[1:]:
    sig_by = {r.key: noise(r) for r in today}; res = []
    for r in today:
        for (key, k), (name, tair, _) in fw.REAL.items():
            if key != r.key: continue
            mv = r.moves[k]['t']
            res.append(f"{name}:{'plateau' if plateau_ok(r, sig_by[key], mv, mv + tair, **gate) else 'NO plateau'}")
    print(f"  {label:40s} " + "; ".join(res))
