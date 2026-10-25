"""Loader for the 13 REAL dry wells (22 Oct + 25 Oct 2026 low-volume runs).
Data files are parsed as text only. Times: seconds since the sensor file's 'Created' stamp."""
import sys, re, datetime as dt, numpy as np
sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study/tools')
import tdload
MAB = '/home/rpi/Documents/MABIP'
RUNS = {  # key: (sensor file, amuza log, gui log)
    '22a': (f'{MAB}/Sensor_Readings/MABIP_Run_22_10_26_16_27.txt', f'{MAB}/Amuza_Logs/AMUZA-2026-10-22_16-27-18.log',
            '/home/rpi/.mabip/logs/gui_20261022_162712.log'),
    '22b': (f'{MAB}/Sensor_Readings/MABIP_Run_22_10_26_16_53.txt', f'{MAB}/Amuza_Logs/AMUZA-2026-10-22_16-53-31.log',
            '/home/rpi/.mabip/logs/gui_20261022_165328.log'),
    '25':  (f'{MAB}/Sensor_Readings/MABIP_Run_25_10_26_15_43.txt', f'{MAB}/Amuza_Logs/AMUZA-2026-10-25_15-25-17.log',
            '/home/rpi/.mabip/logs/gui_20261025_152355.log'),
}
PORT2WELL = {f'{c}{r}': None for c in 'ABCDEFGH' for r in range(1, 13)}
def port_name(p):          # port 01..96, column-major A1=1, B1=2 ... (A2=9)
    p = int(p) - 1; return 'ABCDEFGH'[p % 8] + str(p // 8 + 1)

def _ts(line, fmt):
    return dt.datetime.strptime(line[:23], fmt)

def amuza(path):
    out = []
    for line in open(path, errors='replace'):
        if '[TX] @P,' in line or '[TX] @T' in line:
            ts = _ts(line, '%Y-%m-%d %H:%M:%S.%f')
            if '@P,' in line:
                p = line.split('@P,')[1].split(','); out.append((ts, 'P', int(p[1]), port_name(p[2])))
            else: out.append((ts, 'T', None, None))
    return out

PATS = [r"Flowing (\d+(?:\.\d+)?) µL/min", r"pump boost: (\d+(?:\.\d+)?) uL/min", r"back to: (\d+(?:\.\d+)?) uL/min",
        r"Pump (\d+(?:\.\d+)?) µL/min", r"Resumed (\d+(?:\.\d+)?)\."]
def flow(path, only_ctl=False):
    out = []
    for line in open(path, errors='replace'):
        if 'flow_tab' not in line: continue
        ts = None
        for p in (PATS[3:4] if only_ctl else PATS):
            m = re.search(p, line)
            if m: out.append((_ts(line, '%Y-%m-%d %H:%M:%S,%f'), float(m.group(1)))); break
        else:
            if 'Pump stopped' in line: out.append((_ts(line, '%Y-%m-%d %H:%M:%S,%f'), 0.0))
    return out

def operator_marks(path):
    return [_ts(l, '%Y-%m-%d %H:%M:%S,%f') for l in open(path, errors='replace') if 'operator marked the current well done' in l]

class Run: pass

def load_run(key):
    sf, af, gf = RUNS[key]
    cr, a = tdload.load_sensor(sf)
    r = Run(); r.key = key; r.created = cr; r.t = a[:, 0]; r.X = a[:, 1:7]
    rel = lambda ts: (ts - cr).total_seconds()
    ev = [(rel(ts), k, d, w) for ts, k, d, w in amuza(af)]
    r.flow = [(rel(ts), v) for ts, v in flow(gf)]
    r.ctl_flow = [(rel(ts), v) for ts, v in flow(gf, True) if v > 0]
    ops = [rel(x) for x in operator_marks(gf)]
    P = [e for e in ev if e[1] == 'P']; r.moves = []
    for j, (tm, _, dw, w) in enumerate(P):
        nxt = P[j + 1][0] if j + 1 < len(P) else np.inf
        T = [e[0] for e in ev if e[1] == 'T' and tm < e[0] < nxt]
        home = T[0] if T else tm + dw
        op = any(tm < o <= home + 1 for o in ops)
        r.moves.append(dict(t=tm, home=home, next=nxt, well=w, dwell=dw, operator=op))
    return r

def rate_at(r, t):
    v = 0.0
    for ts, x in r.flow:
        if ts <= t: v = x
        else: break
    return v

def rate_array(r, tt):
    return np.array([rate_at(r, x) for x in tt])

def volume(r, t0, t1, dt_=0.1):
    """µL drawn between t0 and t1 (integrated commanded combined-line rate)."""
    if t1 <= t0: return 0.0
    g = np.arange(t0, t1, dt_)
    return float(np.sum(rate_array(r, g)) * dt_ / 60.0)

def air_arrival(r, t0, dead=130.0, dt_=0.1):
    vol, t = 0.0, t0
    while t < t0 + 600:
        vol += rate_at(r, t) / 60.0 * dt_; t += dt_
        if vol >= dead: return t
    return None

def all_runs():
    return {k: load_run(k) for k in RUNS}

if __name__ == '__main__':
    for k, r in all_runs().items():
        print(k, r.created, len(r.t), 'dt=%.3f' % np.median(np.diff(r.t)), 't0=%.1f' % r.t[0], 'tend=%.0f' % r.t[-1])
        for m in r.moves:
            print('   ', m['well'], 'move %.1f home %.1f (%.0fs) op=%s' % (m['t'], m['home'], m['home'] - m['t'], m['operator']),
                  'vol to home %.0f uL' % volume(r, m['t'], m['home']))
