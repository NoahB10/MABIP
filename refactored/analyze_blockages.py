#!/usr/bin/env python3
"""Scan a saved Sensor_readings_*.txt for flow blockages, offline.

Replays the file through the same BlockageDetector the GUI runs live, so what it
reports here is exactly what would have been alarmed at the bench. Useful for
re-reading past experiments and for re-tuning against a run with known clogs.

    python analyze_blockages.py Sensor_Readings/Sensor_readings_14_07_26_15_00.txt
    python analyze_blockages.py <file> --cycle-min 2.95 --plot out.png

If a matching Well_Log_*.csv sits next to the sensor file, the true well cycle is
measured from it rather than assumed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from blockage_detector import BlockageDetector, DetectorConfig

CH = [f"#1ch{i}" for i in range(1, 7)]


def load(path: Path):
    """Return (t_seconds[], channels[][6], flow[] or None).

    The sensor file is a wide tab-separated table (~386 columns) with a 3-line
    header: 'Created:', the column names, then 'Start:'.
    """
    with open(path) as f:
        lines = f.readlines()
    header = lines[1].rstrip("\n").split("\t")
    try:
        idx = [header.index(c) for c in CH]
        t_i = header.index("t[min]")
    except ValueError:
        sys.exit(f"{path}: not a sensor-readings file (missing t[min]/#1chN columns)")
    flow_i = header.index("flow_uL_min") if "flow_uL_min" in header else None

    ts, chans, flow = [], [], []
    for ln in lines[3:]:
        p = ln.rstrip("\n").split("\t")
        if len(p) <= max(idx):
            continue
        try:
            ts.append(float(p[t_i]) * 60.0)
            chans.append([float(p[i]) for i in idx])
            if flow_i is not None and len(p) > flow_i:
                flow.append(float(p[flow_i]))
        except ValueError:
            continue
    return ts, chans, (flow or None)


def well_cycle_s(sensor_path: Path):
    """Median gap between well completions in the sibling Well_Log, or None."""
    m = re.search(r"Sensor_readings_(.+)\.txt$", sensor_path.name)
    if not m:
        return None
    wl = sensor_path.with_name(f"Well_Log_{m.group(1)}.csv")
    if not wl.exists():
        return None
    mins = []
    for ln in open(wl).readlines()[2:]:
        p = ln.split(",")
        if len(p) >= 3:
            try:
                mins.append(float(p[2]))
            except ValueError:
                pass
    if len(mins) < 3:
        return None
    gaps = sorted(b - a for a, b in zip(mins[:-1], mins[1:]) if 0 < b - a < 60)
    if not gaps:
        return None
    return gaps[len(gaps) // 2] * 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--cycle-min", type=float, default=None,
                    help="well cycle in minutes (default: from Well_Log, else 2.95)")
    ap.add_argument("--flat-frac", type=float, default=DetectorConfig.flat_frac)
    ap.add_argument("--confirm-s", type=float, default=DetectorConfig.confirm_s)
    ap.add_argument("--plot", type=Path, default=None, help="write a PNG overview")
    args = ap.parse_args()

    ts, chans, flow = load(args.file)
    if not ts:
        sys.exit(f"{args.file}: no data rows")

    if args.cycle_min:
        cycle_s, src = args.cycle_min * 60.0, "command line"
    else:
        measured = well_cycle_s(args.file)
        cycle_s, src = (measured, "Well_Log") if measured else (177.0, "default")

    det = BlockageDetector(config=DetectorConfig(
        cycle_s=cycle_s, flat_frac=args.flat_frac, confirm_s=args.confirm_s))

    print(f"{args.file.name}: {len(ts)} samples, {ts[-1]/60:.1f} min, "
          f"cycle {cycle_s/60:.2f} min (from {src})")

    blocked_flags, episodes, start = [], [], None
    for t, c in zip(ts, chans):
        ev = det.update(t, c)
        if ev and ev.blocked:
            start = t
            print(f"  [{t/60:7.2f} min] BLOCKED  ({', '.join(ev.channels)})")
        elif ev and not ev.blocked and start is not None:
            episodes.append((start, t))
            print(f"  [{t/60:7.2f} min] cleared  (lasted {(t-start)/60:.2f} min)")
            start = None
        blocked_flags.append(det.blocked)
    if start is not None:
        episodes.append((start, ts[-1]))
        print(f"  [{ts[-1]/60:7.2f} min] still blocked at end of run")

    total = sum(b - a for a, b in episodes) / 60.0
    pct = 100.0 * sum(blocked_flags) / len(blocked_flags)
    print(f"\n{len(episodes)} blockage(s), {total:.1f} min blocked "
          f"({pct:.2f}% of the run)")

    if args.plot:
        _plot(args.plot, ts, chans, flow, blocked_flags)
        print(f"wrote {args.plot}")


def _plot(out, ts, chans, flow, blocked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm = [t / 60 for t in ts]
    fig, ax = plt.subplots(figsize=(18, 5))
    for i, name in ((5, "ch6"), (4, "ch5")):
        ax.plot(tm, [c[i] for c in chans], lw=0.7, label=name)
    ax.fill_between(tm, 0, 1, where=blocked, transform=ax.get_xaxis_transform(),
                    color="red", alpha=0.25, label="detected blockage")
    if flow:
        ax2 = ax.twinx()
        ax2.plot(tm[:len(flow)], flow, lw=0.4, color="gray", alpha=0.5)
        ax2.set_ylabel("flow uL/min (gray, reference only)")
    ax.set_xlabel("t [min]")
    ax.set_ylabel("metabolite signal")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
