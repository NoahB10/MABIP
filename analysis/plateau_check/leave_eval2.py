"""When does the run LEAVE each well? Before vs after the plateau rule, using the real controller code."""
import sys
sys.dont_write_bytecode = True
sys.argv = ["x"]
exec(open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/run_gold.py").read().split("rows = []")[0])
sys.path.insert(0, "/home/rpi/Documents/MABIP")
import importlib.util
sp = importlib.util.spec_from_file_location("lv", "/home/rpi/Documents/MABIP/low_volume.py")
lv = importlib.util.module_from_spec(sp); sys.modules["lv"] = lv; sp.loader.exec_module(lv)
dev = fw.load_all(); today = [r for r in dev if r.key in fw.TODAY]

class Null:
    def update(self, t, c): return None
    def note_move(self, t): pass
    def note_lift(self, t): pass

def leaves(r, rule, gap=2.0, cancel="plateau"):
    """Return {well index: (leave_time, kind)} kind in good / flagged."""
    ctl = lv.LowVolumeController(lv.LowVolumeConfig(plateau_gap_s=gap), Null())
    det = installed.WellEmptyDetector(installed.WellEmptyConfig(lag_s=r.lag))
    out = {}; k = 0; mv = None; pend = None
    for i in range(len(r.t)):
        t = r.t[i]
        while k < len(r.moves) and r.moves[k]['t'] <= t:
            det.note_move(r.moves[k]['t']); mv = r.moves[k]['t']; k += 1; pend = None
        ctl._record(t, r.X[i])
        e = det.update(t, r.X[i])
        w = k - 1
        if k == 0 or w in out: continue
        if pend is not None:                                   # waiting after a no-plateau call
            if cancel == "plateau" and ctl.plateau_before(t + ctl.cfg.plateau_gap_s): pend = None
            elif cancel == "rising" and rising(ctl, pend_lvl, t): pend = None
            elif t - pend >= ctl.cfg.plateau_confirm_s: out[w] = (t, "flagged"); continue
        if e is None or e.kind != "EMPTY": continue
        if not rule: out[w] = (t, "good"); continue
        t_ref = min(t, float(getattr(e, "t_air", t) or t))
        if ctl.plateau_before(t_ref): out[w] = (t, "good"); pend = None
        else:
            if pend is None: pend = t; pend_lvl = level_now(ctl)
            det.rearm(mv)
    return out

def level_now(ctl):
    last = [c for _, c in list(ctl._hist)[-5:]]
    return [lv._median([c[i] for c in last]) for i in range(6)]

def rising(ctl, lvl0, t):
    now = level_now(ctl)
    return sum(1 for i in range(6) if now[i] - lvl0[i] >= 8 * ctl._noise(i) and now[i] - lvl0[i] >= 0.05) >= 2

def score_plates(runs, rule, **kw):
    c = dict(right=0, right_good=0, right_flag=0, early=0, early_flag=0, never=0, n=0, lat=[])
    for r in runs:
        L = leaves(r, rule, **kw)
        for w in fw.windows(r):
            if w['kind'] != 'surr': continue
            c['n'] += 1; x = L.get(w['k'])
            if x is None or x[0] > w['lift'] + r.lag + 30: c['never'] += 1
            elif x[0] < w['lift'] - 5: c['early'] += 1; c['early_flag'] += x[1] == "flagged"
            else:
                c['right'] += 1; c['lat'].append(x[0] - w['lift'])
                c['right_good' if x[1] == "good" else 'right_flag'] += 1
    return c

VARIANTS = [("before: leave at first call", False, {}),
            ("plateau rule as built (gap 2 s, cancel if plateau)", True, dict(gap=2.0, cancel="plateau")),
            ("gap 5 s, cancel if plateau", True, dict(gap=5.0, cancel="plateau")),
            ("gap 2 s, cancel only if still rising", True, dict(gap=2.0, cancel="rising")),
            ("gap 5 s, cancel only if still rising", True, dict(gap=5.0, cancel="rising"))]
print("ALL 15 LABELLED PLATES, 1376 wells: left at right time / left early / never left")
for label, rule, kw in VARIANTS:
    c = score_plates(runs, rule, **kw); n = c['n']
    print(f"  {label:52s} right {c['right']:4d} ({c['right']/n*100:4.1f}%) [flagged {c['right_flag']:3d}] med {np.median(c['lat']):4.1f}s | early {c['early']:3d} ({c['early']/n*100:3.1f}%) flagged {c['early_flag']:2d} | never {c['never']:3d}")
print("\n22 OCT REAL DRY WELLS (seconds after the tip met air)")
for label, rule, kw in VARIANTS:
    res = []
    for r in today:
        L = leaves(r, rule, **kw)
        for (key, k), (name, tair, _) in fw.REAL.items():
            if key != r.key: continue
            x = L.get(k); m = r.moves[k]['t']
            res.append(f"{name}:" + ("NEVER" if x is None else f"{'ok' if x[1]=='good' else 'flag'}+{x[0]-m-tair:.0f}"))
    print(f"  {label:52s} " + " ".join(res))
