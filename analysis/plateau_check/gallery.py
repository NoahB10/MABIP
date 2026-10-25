"""Galleries of missed and caught lift-out bubbles for the installed detector on the gold plates."""
import sys, random
sys.dont_write_bytecode = True
sys.argv = ["run_gold.py"]
exec(open("/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/run_gold.py").read().split("rows = []")[0])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/goldtest/"
SHORT = {k: k[5:10].replace("-", "/") + " " + k.split("_", 2)[2] for k in [r.key for r in runs]}
COLS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

cases = {"hit": [], "miss": [], "fa": []}
for r in runs:
    calls = replay(r)
    for w in fw.windows(r):
        if w["kind"] != "surr": continue
        c = calls.get(w["k"])
        if c is not None and c[0] < w["lift"] - 5: cases["fa"].append((r, w, c))
        elif c is not None and c[0] <= w["lift"] + r.lag + 30: cases["hit"].append((r, w, c))
        else: cases["miss"].append((r, w, None))

def pick(lst, per_plate, seed):
    rnd = random.Random(seed); by = {}
    for x in lst: by.setdefault(x[0].key, []).append(x)
    out = []
    for key in sorted(by):
        out += rnd.sample(by[key], min(per_plate, len(by[key])))
    return out

def panel(ax, r, w, c, title_extra=""):
    m = r.moves[w["k"]]
    t0, t1 = m["t"] - 15, w["lift"] + r.lag + 45
    i = (r.t >= t0) & (r.t <= t1)
    if i.sum() < 5: return False
    tt = r.t[i] - m["t"]
    for ch in range(4):
        y = r.X[i, ch]; y = y - np.median(r.X[(r.t >= m["t"] - 15) & (r.t <= m["t"]), ch]) if ((r.t >= m["t"] - 15) & (r.t <= m["t"])).any() else y - y[0]
        ax.plot(tt, y, color=COLS[ch], lw=1.0, label=f"ch{ch+1}")
    ax.axvline(0, color="#2a9d2a", lw=1)
    ax.axvline(w["lift"] - m["t"], color="#555", lw=1, ls="--")
    ax.axvspan(w["lift"] - m["t"] + r.lag - 10, w["lift"] - m["t"] + r.lag + 10, color="#f4a261", alpha=.25, lw=0)
    if c is not None:
        ax.axvline(c[0] - m["t"], color="#7b2cbf", lw=2)
    ax.set_title(f"{SHORT[r.key]}  well {m['port']}{title_extra}", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7); ax.grid(alpha=.2)
    return True

def gallery(items, fname, heading, ncol=5):
    n = len(items); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.35 * nrow), squeeze=False)
    for ax in axs.flat: ax.set_visible(False)
    for ax, (r, w, c) in zip(axs.flat, items):
        ax.set_visible(True)
        extra = "" if c is None else f"  call +{c[0]-w['lift']:.0f}s"
        panel(ax, r, w, c, extra)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(heading, x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, "x = seconds after move into the well. green = move in, grey dashed = needle lifts out, "
             "orange = where the lift-out air should reach the sensor (lag ±10 s), purple = detector's call. "
             "Each channel shifted to 0 at the move.", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(OUT + fname, dpi=80); plt.close(fig)

late = lambda r: r.key >= "2026-06"
miss_items = pick([x for x in cases["miss"] if late(x[0])], 4, 1) + pick([x for x in cases["miss"] if not late(x[0])], 2, 2)
hit_items = pick([x for x in cases["hit"] if not late(x[0])], 2, 3) + pick([x for x in cases["hit"] if late(x[0])], 3, 4)
gallery(miss_items, "missed.png", f"MISSED bubbles: {len(cases['miss'])} of {len(cases['miss'])+len(cases['hit'])} wells. Sample: 4 per June+ plate, 2 per earlier plate.")
gallery(hit_items, "caught.png", f"CAUGHT bubbles: {len(cases['hit'])} wells. Sample: 2 per earlier plate, 3 per June+ plate.")
gallery(pick(cases["fa"], 3, 5), "false_alarms.png", f"FALSE ALARMS: {len(cases['fa'])} wells called empty before the lift-out.")

# per-plate overview: the median cycle (hit vs miss) for one early and two late plates
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, key in zip(axs, ["2026-05-08_1059_gro3p1sup", "2026-06-30_1320_Batch11", "2026-08-17_1406_BAtch14"]):
    r = [x for x in runs if x.key == key][0]
    grid = np.arange(-15, 110 + r.lag + 45, 1.66); stack = []
    for w in fw.windows(r):
        if w["kind"] != "surr": continue
        m = r.moves[w["k"]]; i = (r.t >= m["t"] - 15) & (r.t <= m["t"] + grid[-1])
        if i.sum() < 20: continue
        base = np.median(r.X[(r.t >= m["t"] - 15) & (r.t <= m["t"]), 0]) if ((r.t >= m["t"] - 15) & (r.t <= m["t"])).any() else 0
        stack.append(np.interp(grid, r.t[i] - m["t"], r.X[i, 0] - base))
    S_ = np.array(stack)
    ax.fill_between(grid, np.percentile(S_, 10, 0), np.percentile(S_, 90, 0), color="#1f77b4", alpha=.2, lw=0)
    ax.plot(grid, np.median(S_, 0), color="#1f77b4", lw=1.5)
    lift = r.moves[0]["lift"] - r.moves[0]["t"]
    ax.axvline(0, color="#2a9d2a"); ax.axvline(lift, color="#555", ls="--")
    ax.axvspan(lift + r.lag - 10, lift + r.lag + 10, color="#f4a261", alpha=.25, lw=0)
    ax.set_title(f"{SHORT[key]}: ch1, all {len(S_)} cycles (median, 10–90 % band)", fontsize=9, loc="left")
    ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT + "typical_cycle_early_vs_late.png", dpi=80)
print({k: len(v) for k, v in cases.items()}, len(miss_items), len(hit_items))
