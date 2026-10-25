"""Real-controller leave replay (leave_eval2.leaves logic, low_volume.plateau_before as installed, gap 5 s) for the
installed detector and v3, all 15 gold plates + 22 Oct. Run with python -I."""
import sys
sys.dont_write_bytecode = True
sys.argv = ["x"]
exec(open("/home/rpi/Documents/MABIP/analysis/plateau_check/leave_eval2.py").read().split("VARIANTS = [")[0])
v3 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v3", "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/leave/well_empty_detector_v3.py"))
sys.modules["v3"] = v3; v3.__spec__.loader.exec_module(v3)
import csv
_orig = installed.WellEmptyDetector; _origc = installed.WellEmptyConfig
out = []
for name, D, C in (("installed", _orig, _origc), ("v3", v3.WellEmptyDetector, v3.WellEmptyConfig)):
    installed.WellEmptyDetector, installed.WellEmptyConfig = D, C      # leaves() builds installed.WellEmptyDetector(...)
    for rule in (True, False):
        tot = dict(right=0, early=0, never=0, n=0, lat=[])
        for r in runs:
            c = score_plates([r], rule, gap=5.0, cancel="plateau")
            out.append([name, rule, r.key, c['n'], c['right'], c['early'], c['never'], float(np.median(c['lat'])) if c['lat'] else float('nan'), c['right_flag']])
            for k in ('right', 'early', 'never', 'n'): tot[k] += c[k]
            tot['lat'] += c['lat']
        n = tot['n']
        print(f"{name:10s} rule={rule!s:5s} right {tot['right']} ({tot['right']/n*100:.1f}%) early {tot['early']} ({tot['early']/n*100:.2f}%) never {tot['never']} median {np.median(tot['lat']):.1f}s", flush=True)
        res = []
        for r in today:
            L = leaves(r, rule, gap=5.0)
            for (key, k), (nm, tair, _) in fw.REAL.items():
                if key != r.key: continue
                x = L.get(k); m = r.moves[k]['t']
                res.append(f"{nm}:" + ("NEVER" if x is None else f"{'ok' if x[1]=='good' else 'flag'}+{x[0]-m-tair:.0f}"))
        print("    22 Oct:", " ".join(res), flush=True)
installed.WellEmptyDetector, installed.WellEmptyConfig = _orig, _origc
with open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/leave/results/harness_v3_vs_installed_per_plate.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["detector", "plateau_rule", "plate", "wells", "right", "early", "never", "median_latency_s", "right_flagged"]); w.writerows(out)
