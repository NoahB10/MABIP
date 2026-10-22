"""Replay well_empty_detector_v2 (and v1 for comparison) over every saved run and score it.
Run:  <scratch venv>/bin/python replay_v2.py      (v2 itself is pure Python)
Surrogate truth = scheduled lift-outs (needle leaves liquid at @P+dwell+3 s or @T+1 s);
real truth = the 7 dry wells of 22 Oct (fw.REAL). The detector is only told about move-ins."""
import sys, numpy as np, pandas as pd, fw
sys.path.insert(0, '/home/rpi/Documents/MABIP')
import well_empty_detector as v1
import well_empty_detector_v2 as v2

def replay(mod, r):
    det = mod.WellEmptyDetector(mod.WellEmptyConfig(lag_s=r.lag))
    calls, evs, k = {}, [], 0
    for i in range(len(r.t)):
        while k < len(r.moves) and r.moves[k]['t'] <= r.t[i]:
            det.note_move(r.moves[k]['t']); k += 1
        e = det.update(r.t[i], r.X[i])
        if e is None: continue
        evs.append((r.t[i], e.kind, e.reason))
        if e.kind == 'EMPTY' and k > 0 and (k - 1) not in calls:
            calls[k - 1] = (r.t[i], e.reason)
    return calls, evs

if __name__ == '__main__':
    R = fw.load_all()
    out = {}
    for name, mod in (('v1 (current)', v1), ('v2', v2)):
        C = {}; E = {}
        for r in R: C[r.key], E[r.key] = replay(mod, r)
        s = fw.score(R, C); out[name] = (s, C, E)
        print(f"\n== {name}: surrogate recall {s['surr_recall']:.3f} (n={s['surr_n']}, median latency {s['lat_surr_med']:.1f}s) | "
              f"real dry wells {s['real_hits']}/{s['real_n']} (median latency {s['lat_real_med']:.1f}s) | "
              f"false alarms {s['fa_n']} = {s['fa_per_h']:.2f}/in-well h, {s['fa_per_well']:.3f}/well")
        for d in s['real_detail']:
            print('   ', d[0], d[1], '' if d[2] is None else f"called {d[2]:.0f}s after move ({d[3]})" + (f", {d[4]:.1f}s after the dip" if len(d) > 4 else ''))
        rows = []
        for r in R:
            ws = [w for w in fw.windows(r) if w['kind'] == 'surr']
            if not ws: continue
            h = sum(1 for w in ws if C[r.key].get(w['k']) and w['lift'] - 5 <= C[r.key][w['k']][0] <= w['lift'] + r.lag + 30)
            fa = sum(1 for w in ws if C[r.key].get(w['k']) and C[r.key][w['k']][0] < w['lift'] - 5)
            rows.append(dict(run=r.key, cycles=len(ws), hit=h, fa=fa))
        print(pd.DataFrame(rows).set_index('run').T.to_string())
