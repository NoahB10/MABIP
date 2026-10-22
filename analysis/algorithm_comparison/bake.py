import numpy as np, pandas as pd, itertools, pickle, time, sys, warnings
import fw, methods as M
warnings.filterwarnings('ignore')
R = fw.load_all(refresh='--refresh' in sys.argv)
for r in R:
    r.RS3 = M.m3_hampel(r, h=3)['RS']; r.F10 = M.features(r)
    r.flat = fw.flat_signal(r)
print('runs', len(R), 'samples', sum(len(r.t) for r in R))

# ---- LORO learned pieces
TPL = {r.key: M.templates(R, exclude=r.key) for r in R}
PCS = {r.key: M.pcs(R, exclude=r.key) for r in R}
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def train_rows(r):
    n = len(r.t); y = np.zeros(n, int); use = np.zeros(n, bool)
    for w in fw.windows(r):
        if w['kind'] != 'surr': continue
        use |= (r.t >= w['arm']) & (r.t < w['lift'] - 5)
    excl = np.zeros(n, bool)
    for j in np.r_[r.spk_idx, r.dip_idx]:
        excl[max(0, j - 3 + M.FD):j + 4 + M.FD] = True
    use &= ~excl
    for j in r.spk_idx:
        if j + M.FD < n: y[j + M.FD] = 1; use[j + M.FD] = True
    for j in r.dip_idx:
        if j + M.FD < n: y[j + M.FD] = 2; use[j + M.FD] = True
    return r.F10[use], y[use]
ROWS = {r.key: train_rows(r) for r in R}
MODELS = {}
t0 = time.time()
for r in R:
    Xs = np.vstack([ROWS[k][0] for k in ROWS if k != r.key]); ys = np.concatenate([ROWS[k][1] for k in ROWS if k != r.key])
    mdl = {}
    for cls, name in ((1, 'spike'), (2, 'dip')):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=300))
        clf.fit(Xs, (ys == cls).astype(int)); mdl[name] = (lambda c: (lambda F: c.predict_proba(F)[:, 1]))(clf)
    MODELS[r.key] = mdl
print('LORO logistic trained in %.0fs' % (time.time() - t0))

GRID = {
 'm1_baseline': (M.m1_baseline, [{}]),
 'm2_diffz': (M.m2_diffz, [dict(k=k, nmin=v, fall=f) for k in (5, 8, 12, 16) for v in (2, 3, 4) for f in (True, False)]
      + [dict(k=k, nmin=v, fall=False, span=sp, cm=cm, amin=am, rec=rc) for k in (5, 8, 12) for v in (2, 3, 4) for sp in (1, 2) for cm in (None, 2.5) for am in (0.0, 0.08) for rc in (0, 3)]),
 'm3_hampel': (M.m3_hampel, [dict(h=h, k=k, nmin=v, scale=s) for h in (2, 3, 5) for k in (5, 8, 12) for v in (2, 3, 4) for s in ('noise', 'local')]),
 'm4_sg': (M.m4_sg, [dict(h=h, k=k) for h in (2, 3) for k in (3, 5, 8, 12)]),
 'm5_prominence': (M.m5_prominence, [dict(prom=p, wmax=w, D=D) for p in (6, 10, 15, 25, 40) for w in (3, 5) for D in (3, 5)]),
 'm6_cwt': (M.m6_cwt, [dict(scales=s, k=k, nmin=v, nscale=ns) for s in ((1.0, 1.5), (1.0, 1.5, 2.0)) for k in (4, 6, 10) for v in (2, 3) for ns in (1, 2)]),
 'm7_cusum': (M.m7_cusum, [dict(nu=nu, h=h) for nu in (1.0, 2.0, 4.0) for h in (10, 20, 40, 80)]),
 'm8_matched': (M.m8_matched, [dict(rho=rho, k=k, nmin=v) for rho in (0.7, 0.8, 0.9) for k in (5, 10, 20) for v in (2, 3)]),
 'm9_cmproj': (M.m9_cmproj, [dict(k=k, rho=rho, mode=md) for k in (6, 10, 15, 25) for rho in (0.8, 0.9, 0.95) for md in ('pca', 'median')]),
 'm10_logreg': (M.m10_clf, [dict(theta=th) for th in (0.5, 0.8, 0.9, 0.97, 0.99, 0.995, 0.999)]),
}
RULES = ['spike', 'pair', 'dipalone', 'stale', 'spike+stale', 'pair+stale']
rows = []; DET = {}
only = [a for a in sys.argv[1:] if a.startswith('m')]
for name, (fn, plist) in GRID.items():
    if only and name not in only: continue
    t0 = time.time()
    for p in plist:
        outs = {}
        for r in R:
            kw = dict(p)
            if name == 'm8_matched': kw['tpl'] = TPL[r.key]
            if name == 'm9_cmproj': kw['pc'] = PCS[r.key]
            if name == 'm10_logreg': kw['model'] = MODELS[r.key]
            outs[r.key] = fn(r, **kw)
        rules = ['native', 'native+stale'] if name == 'm1_baseline' else RULES
        for rule in rules:
            calls = {}
            for r in R:
                o = outs[r.key]
                if name == 'm1_baseline':
                    c = {}
                    for te, why in o['empty']:
                        k = max([j for j, m in enumerate(r.moves) if m['t'] <= te], default=None)
                        if k is not None and k not in c: c[k] = (te, why)
                    if rule == 'native+stale':
                        c2 = fw.decide(r, np.zeros(len(r.t), bool), o['dip'], 'stale', flat=r.flat)
                        for k, v in c2.items():
                            if k not in c or v[0] < c[k][0]: c[k] = v
                    calls[r.key] = c
                else:
                    calls[r.key] = fw.decide(r, o['spike'], o['dip'], rule, flat=r.flat)
            s = fw.score(R, calls)
            s.update(method=name, params=repr(p), rule=rule, delay=outs[R[0].key]['delay'])
            rows.append(s); DET[(name, repr(p), rule)] = calls
    print(name, len(plist), 'params %.0fs' % (time.time() - t0), flush=True)
df = pd.DataFrame(rows)
tag = '_'.join(only) if only else 'all'
df.to_pickle(f'sweep_{tag}.pkl'); pickle.dump(DET, open(f'calls_{tag}.pkl', 'wb'))
