"""Loaders: sensor runs (deduped), AMUZA @P moves, well completions."""
import glob, os, re, datetime as dt
import numpy as np, pandas as pd
ROOT = "/home/rpi/Documents/MABIP"

def parse_created(line):
    m = re.match(r"Created:\s*(\d\d/\d\d/\d{4})\s+(\d\d:\d\d:\d\d [AP]M)", line.strip())
    return dt.datetime.strptime(m.group(1) + " " + m.group(2), "%m/%d/%Y %I:%M:%S %p") if m else None

def sensor_files():
    files = glob.glob(f"{ROOT}/Sensor_Readings/*.txt") + glob.glob(f"{ROOT}/*.txt") + glob.glob(f"{ROOT}/refactored/*.txt")
    best = {}
    for f in files:
        with open(f, errors="replace") as fh:
            c = parse_created(fh.readline())
        if c is None: continue
        n = sum(1 for _ in open(f, errors="replace"))
        # prefer Sensor_Readings copy, then longest
        score = (n, "Sensor_Readings" in f)
        if c not in best or score > best[c][0]:
            best[c] = (score, f)
    return sorted((c, v[1]) for c, v in best.items())

def load_sensor(path):
    rows, wells = [], []
    with open(path, errors="replace") as fh:
        created = parse_created(fh.readline())
        hdr = fh.readline().rstrip("\n").split("\t")
        ic = [hdr.index(f"#1ch{k}") for k in range(1, 7)]
        it = hdr.index("t[min]")
        for line in fh:
            if line.startswith("#WELL"):
                p = line.rstrip("\n").split("\t")
                try: wells.append((p[1], float(p[3]) * 60))
                except Exception: pass
                continue
            if line.startswith("#") or line.startswith("Start") or not line[:1].isdigit():
                # trailing completion log lines: well_id,completed_at,sensor_elapsed_min,...
                s = line.lstrip("# ").strip()
                m = re.match(r"([A-H]\d+),(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),([\d.]+)", s)
                if m: wells.append((m.group(1), float(m.group(3)) * 60))
                continue
            p = line.split("\t")
            try:
                rows.append([float(p[it]) * 60] + [float(p[i]) for i in ic])
            except Exception:
                continue
    a = np.array(rows) if rows else np.zeros((0, 7))
    if len(a):
        # drop non-monotonic
        keep = np.r_[True, np.diff(a[:, 0]) > 0]
        a = a[keep]
    return created, a, wells

_moves = None
def all_moves():
    """List of (datetime, dwell_s, port) for every @P TX."""
    global _moves
    if _moves is None:
        out = []
        for f in sorted(glob.glob(f"{ROOT}/Amuza_Logs/*.log")):
            for line in open(f, errors="replace"):
                if "[TX] @P," in line:
                    ts = dt.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
                    p = line.split("@P,")[1].split(",")
                    out.append((ts, int(p[1]), p[2].strip()))
        _moves = sorted(out)
    return _moves

def moves_for(created, tmax_s):
    return [((ts - created).total_seconds(), d, port) for ts, d, port in all_moves()
            if 0 <= (ts - created).total_seconds() <= tmax_s]

def well_log_for(created):
    out = []
    for f in glob.glob(f"{ROOT}/Sensor_Readings/Well_Log_*.csv"):
        h = open(f).readline()
        m = re.search(r"started: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", h)
        if not m: continue
        st = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if abs((st - created).total_seconds()) < 5:
            df = pd.read_csv(f, comment="#", header=None, names=["well", "at", "min", "seq"])
            df = df[df.well != "well_id"]
            out += [(w, float(x) * 60) for w, x in zip(df.well, df["min"])]
    return out

def flow_logs():
    out = {}
    for f in glob.glob(f"{ROOT}/Sensor_Readings/Flow_Log_*.csv"):
        out[f] = pd.read_csv(f, comment="#")
    return out
