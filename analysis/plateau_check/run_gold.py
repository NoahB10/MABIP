"""Score the INSTALLED bubble detector on all 15 gold plates (params frozen)."""
import sys
sys.dont_write_bytecode = True
ALG = "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/algos"
sys.path.insert(0, "/home/rpi/Documents/MABIP"); sys.path.insert(0, ALG)
import numpy as np, glob, os
import tdload, fw
import importlib.util
_spec = importlib.util.spec_from_file_location("installed_wed", "/home/rpi/Documents/MABIP/well_empty_detector.py")
installed = importlib.util.module_from_spec(_spec); sys.modules["installed_wed"] = installed; _spec.loader.exec_module(installed)
assert hasattr(installed.WellEmptyConfig(), "fast_dip"), "not the installed v2"
tdload.BASE = "/home/rpi/Documents/MABIP_datasets/TD_plate_pairs_gold"
OFF, OVERHEAD = 7.6, 68.0

def make_run(name, txt, csv):
    cr, a = tdload.load_sensor(txt); wl = tdload.load_welllog(csv)
    C = [w['t'] for w in wl]
    gaps = np.diff(C); gaps = gaps[(gaps > 60) & (gaps < 600)]
    dw = int(round(np.median(gaps) - OVERHEAD)) if len(gaps) else 110
    class Run: pass
    r = Run(); r.key = name; r.created = cr; r.a = a; r.t = a[:, 0]; r.X = a[:, 1:7]; r.flow = []
    P = [c - OFF - dw for c in C]
    r.moves = [dict(t=p, dwell=dw, port=w['well'], lift=p + dw + fw.LIFT_OFF, how='dwell',
                    next=(P[j + 1] if j + 1 < len(P) else np.inf), blocked=w['blocked'])
               for j, (p, w) in enumerate(zip(P, wl))]
    fw.prep(r); r.lag = fw.run_lag(r); fw.label_events(r); r.dwell = dw
    return r

def replay(r, **cfg):
    det = installed.WellEmptyDetector(installed.WellEmptyConfig(lag_s=r.lag, **cfg)); calls = {}; k = 0
    for i in range(len(r.t)):
        while k < len(r.moves) and r.moves[k]['t'] <= r.t[i]:
            det.note_move(r.moves[k]['t']); k += 1
        e = det.update(r.t[i], r.X[i])
        if e is not None and e.kind == 'EMPTY' and k > 0 and (k - 1) not in calls:
            calls[k - 1] = (r.t[i], e.reason)
    return calls

runs = [make_run(*f) for f in tdload.folders()]
rows = []
for label, cfg in (("installed (fast dip on)", {}), ("same, fast dip off (old v1)", {"fast_dip": False})):
    tot = dict(h=0, n=0, fa=0, hrs=0.0, lat=[])
    print(f"\n=== {label} ===")
    print(f"{'plate':34s} {'wells':>5s} {'dwell':>5s} {'lag':>5s} {'bubbles caught':>15s} {'median s':>9s} {'false alarms':>12s} {'per h':>6s}")
    for r in runs:
        s = fw.score([r], {r.key: replay(r, **cfg)})
        h = round(s['surr_recall'] * s['surr_n'])
        tot['h'] += h; tot['n'] += s['surr_n']; tot['fa'] += s['fa_n']; tot['hrs'] += s['inwell_h']
        print(f"{r.key:34s} {s['surr_n']:5d} {r.dwell:5d} {r.lag:5.0f} {h:6d} ({s['surr_recall']*100:3.0f} %) {s['lat_surr_med']:9.1f} {s['fa_n']:12d} {s['fa_per_h']:6.2f}")
        rows.append((label, r.key, s['surr_n'], h, s['surr_recall'], s['lat_surr_med'], s['fa_n'], s['fa_per_h']))
    print(f"{'ALL 15 PLATES':34s} {tot['n']:5d} {'':5s} {'':5s} {tot['h']:6d} ({tot['h']/tot['n']*100:3.0f} %) {'':9s} {tot['fa']:12d} {tot['fa']/tot['hrs']:6.2f}")
import csv
with open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/gold_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["detector", "plate", "wells", "caught", "recall", "median_latency_s", "false_alarms", "fa_per_inwell_h"]); w.writerows(rows)
