"""Causality test: every method re-run on a truncated copy must give identical decisions on the prefix."""
import copy, numpy as np, fw, methods as M
R = fw.load_all(); RD = {r.key: r for r in R}
def aux(r):
    r.RS3 = M.m3_hampel(r, h=3)['RS']; r.F10 = M.features(r); return r
for r in R: aux(r)
TPL = M.templates(R); PC = M.pcs(R)
mdl = {'spike': lambda F: 1 / (1 + np.exp(-F @ np.linspace(-1, 1, F.shape[1]))), 'dip': lambda F: 1 / (1 + np.exp(-F.sum(1)))}
tests = [('m2_diffz', M.m2_diffz, dict(k=8, nmin=3)), ('m2_diffz fastdip', M.m2_diffz, dict(k=8, nmin=3, fall=False, span=2, cm=2.5, amin=0.08, rec=3)),
         ('m3_hampel', M.m3_hampel, dict(h=3, k=6, scale='local')), ('m4_sg', M.m4_sg, dict(h=2, k=4)), ('m5_prominence', M.m5_prominence, dict(prom=10)),
         ('m6_cwt', M.m6_cwt, {}), ('m7_cusum', M.m7_cusum, {}), ('m8_matched', M.m8_matched, dict(tpl=TPL)), ('m9_cmproj', M.m9_cmproj, dict(pc=PC)),
         ('m10_logreg', M.m10_clf, dict(model=mdl, theta=0.6))]
for key in ('260508_1059', '261022_1627'):
    r = RD[key]
    for name, fn, p in tests:
        full = fn(r, **p); bad = 0
        for frac in (0.37, 0.61, 0.83):
            n = int(len(r.t) * frac)
            rr = copy.copy(r); rr.X = r.X[:n]; rr.t = r.t[:n]; fw.prep(rr); aux(rr)
            part = fn(rr, **p)
            for k in ('spike', 'dip'):
                bad += int(np.sum(full[k][:n] != part[k]))
        print(key, name, 'mismatches on prefix:', bad, '| events', int(full['spike'].sum()), int(full['dip'].sum()))
