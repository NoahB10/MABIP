"""When does the run LEAVE each well? Before vs after the plateau rule, using the real controller code."""
import sys
sys.dont_write_bytecode = True
sys.argv = ["x"]
exec(open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/run_gold.py").read().split("rows = []")[0].replace("assert hasattr(installed.WellEmptyConfig(), \"fast_dip\"), \"not the installed v2\"", "pass"))
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
    det = installed.WellEmptyDetector(installed.WellEmptyConfig(lag_s=r.lag, **CFG))
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


import json
from datetime import datetime
# 25 Oct low-volume run: moves from the AMUZA log, tip-met-air labels read off the plot
T0 = datetime(2026,10,25,15,43,48)
def ts(h): return (datetime.strptime("2026-10-25 "+h,"%Y-%m-%d %H:%M:%S")-T0).total_seconds()
OCT25 = [("A1","15:44:47","15:50:03",195),("A2","15:51:13","15:57:32",175),("A3","15:58:42","16:02:50",203),
         ("B1","16:04:01","16:10:49",155),("B2","16:11:58","16:18:20",283),("B3","16:19:30","16:30:38",370)]
cr, A25 = tdload.load_sensor("/home/rpi/Documents/MABIP/Sensor_Readings/MABIP_Run_25_10_26_15_43.txt")
def oct25():
    det = installed.WellEmptyDetector(flow_rate=130, **CFG); out=[]; k=0
    moves=[(w, ts(a), ts(b), lab) for w,a,b,lab in OCT25]; done=set()
    for i in range(len(A25)):
        t=A25[i,0]; x=A25[i,1:7]
        while k<len(moves) and moves[k][1]<=t:
            det.note_move(moves[k][1]); k+=1
        e=det.update(t,x)
        if e is not None and e.kind=="EMPTY" and k and (k-1) not in done:
            w,mt,ex,lab=moves[k-1]
            if t<=ex: done.add(k-1); out.append(f"{w}:{'EARLY' if t-mt<lab-5 else 'ok'}+{t-mt-lab:.0f}s")
    left={o.split(':')[0] for o in out}
    return out+[f"{w}:never" for w,*_ in OCT25 if w not in left]
CONFIGS = [("v3 as installed (settle path off)", {}),
           ("+ settle path, drop>=25%", dict(settle_path=True, settle_drop=0.25)),
           ("+ settle path, drop>=40%", dict(settle_path=True, settle_drop=0.40)),
           ("+ settle path, drop>=60%", dict(settle_path=True, settle_drop=0.60))]
for label, CFG in CONFIGS:
    c = score_plates(runs, False); n = c['n']
    res22=[]
    for r in today:
        L = leaves(r, False)
        for (key, k), (name, tair, _) in fw.REAL.items():
            if key != r.key: continue
            x = L.get(k); m = r.moves[k]['t']
            res22.append(f"{name}:" + ("never" if x is None else f"+{x[0]-m-tair:.0f}"))
    print(f"\n{label}\n  GOLD 1376: right {c['right']} ({c['right']/n*100:.1f}%) | EARLY {c['early']} | never {c['never']} | median {np.median(c['lat']):.0f}s")
    print("  22 Oct dry: " + " ".join(res22))
    print("  25 Oct dry: " + " ".join(oct25()), flush=True)
