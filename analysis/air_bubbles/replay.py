"""Replay WellEmptyDetector over every run with AMUZA move timing.
Surrogate ground truth: the scheduled lift-out (@P+dwell) is an unannounced liquid->air
event at the tip, i.e. exactly what a well running dry does. The detector is told only
about move-ins (note_move), never lift-outs.
  HIT  = EMPTY event with lift-3 <= t <= lift+lag+30
  FP   = EMPTY event with t < lift-3 (healthy plateau / transit leak)
"""
import sys, numpy as np, pandas as pd, load, feat
from well_empty_detector import WellEmptyDetector, WellEmptyConfig
def replay(a, mv, cfg=None, prior_lag=25.0, verbose=False):
    det = WellEmptyDetector(cfg or WellEmptyConfig(lag_s=prior_lag))
    t = a[:, 0]; evs = []; k = 0
    for i in range(len(t)):
        while k < len(mv) and mv[k][0] <= t[i]:
            det.note_move(mv[k][0]); k += 1
        e = det.update(t[i], a[i, 1:7])
        if e: evs.append((e, det.lag_s))
    return evs, det
PRIOR = {'261022_1421': 66.0, '261022_1047': 60.0, '261022_1316': 60.0, '261022_1339': 60.0}
if __name__ == "__main__":
    rows = []; allev = []
    for cr, f, a, mv, wl in feat.runs(3):
        key = cr.strftime('%y%m%d_%H%M')
        evs, det = replay(a, mv, prior_lag=PRIOR.get(key, 25.0))
        mt = np.array([m[0] for m in mv])
        for j, (m, dw, port) in enumerate(mv):
            if dw < 60: continue
            nxt = mv[j+1][0] if j+1 < len(mv) else a[-1, 0]
            lift = m + dw + 1.5
            E = [(e, L) for e, L in evs if m <= e.t < nxt]
            emp = [(e, L) for e, L in E if e.kind == 'EMPTY']
            sus_pl = [e for e, L in E if e.kind == 'SUSPECT' and e.t < lift - 3]
            hit = [e for e, L in emp if lift - 3 <= e.t <= lift + L + 30]
            fp = [e for e, L in emp if e.t < lift - 3]
            L = E[0][1] if E else det.lag_s
            if a[-1, 0] < lift + L + 30: continue   # run ended before lift air could arrive
            rows.append(dict(run=key, j=j, m=m, dw=dw, lag=L, hit=len(hit) > 0, fp=len(fp) > 0, sus_plateau=len(sus_pl) > 0,
                             lat=(hit[0].t - lift if hit else np.nan), fp_t=(fp[0].t - m if fp else np.nan),
                             reason=(hit[0].reason if hit else (fp[0].reason if fp else ''))))
            for e, L in E: allev.append(dict(run=key, j=j, kind=e.kind, t_rel_move=e.t - m, t_rel_lift=e.t - lift, reason=e.reason))
    df = pd.DataFrame(rows); df.to_csv('replay_cycles.csv', index=False); pd.DataFrame(allev).to_csv('replay_events.csv', index=False)
    g = df.groupby('run').agg(cycles=('j', 'size'), lag=('lag', 'median'), hit_rate=('hit', 'mean'), lat_med=('lat', 'median'),
                              fp_rate=('fp', 'mean'), suspect_plateau=('sus_plateau', 'mean'))
    pd.set_option('display.width', 200)
    print(g.round(2).to_string())
    CLEAN = ['260120_1304', '260218_1258', '260325_1214', '260329_1246', '260329_1650', '260507_1046', '260508_1059', '260510_1022']
    for name, sub in (('clean (8 runs)', df[df.run.isin(CLEAN)]), ('all', df)):
        print(f"{name}: cycles {len(sub)}  surrogate hit {sub.hit.mean():.3f}  EMPTY false-pos/cycle {sub.fp.mean():.3f}  "
              f"plateau SUSPECT/cycle {sub.sus_plateau.mean():.3f}  latency after lift med {sub.lat.median():.1f}s")
    print(df[df.fp].groupby('reason').size())
