"""Parallel evaluation: gold plates (through the controller leave logic) + the 13 dry wells."""
import sys, numpy as np
sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study')
import h, sim, dw
from multiprocessing import Pool
LV = '/home/rpi/Documents/MABIP/low_volume.py'
_M = {}
def mods(det_path, lv_path):
    key = (det_path, lv_path)
    if key not in _M:
        _M[key] = (h.load_mod(det_path, 'det_%d' % len(_M)), h.load_mod(lv_path, 'lv_%d' % len(_M)))
    return _M[key]

def _gold_plate(args):
    pi, det_path, det_cfg, lv_path, ctl_cfg, scale, gate = args
    det, lv = mods(det_path, lv_path)
    r = h.gold()[pi]
    W = h.gold_windows(r); wells = []
    for w in W:
        m = r.moves[w['k']]
        wells.append((w['k'], m['t'], min(w['end'], w['lift'] + r.lag + 30)))
    ccfg = dict(ctl_cfg or {})
    rate_fn = None
    gates = {}
    if gate is not None:
        # simulate an entered well volume: the lift stands in for 'dry'; volume ~ time at constant flow.
        # gate = (frac, err): the volume gate opens at move + frac*err*(lift - move)
        for w in W:
            m = r.moves[w['k']]; gates[m['t']] = m['t'] + gate[0] * gate[1] * (w['lift'] - m['t'])
    out = sim.run_plate(lv, det, det_cfg, r, wells, 130.0, scale=scale, lag=r.lag, ctl_cfg=ccfg, gold_gates=gates)
    res = []
    for w in W:
        lt, why = out.get(w['k'], (None, None))
        if lt is None or lt > w['lift'] + r.lag + 30: cls = 'never'
        elif lt < w['lift'] - 5: cls = 'early'
        else: cls = 'right'
        res.append((r.key, w['k'], cls, None if lt is None else lt - w['lift'], why, None if lt is None else lt - r.moves[w['k']]['t']))
    return res

def gold(det_path, det_cfg=None, lv_path=LV, ctl_cfg=None, scale=1.0, plates=None, gate=None, procs=4):
    n = len(h.gold()); plates = range(n) if plates is None else plates
    with Pool(procs) as p:
        R = p.map(_gold_plate, [(i, det_path, det_cfg, lv_path, ctl_cfg, scale, gate) for i in plates])
    return [x for rr in R for x in rr]

def summarize_gold(res):
    n = len(res); e = sum(1 for x in res if x[2] == 'early'); rt = [x for x in res if x[2] == 'right']
    lat = [x[3] for x in rt]
    return dict(n=n, early=e, early_per1000=1000 * e / n, ub95=1000 * ub95(e, n), right=len(rt), never=n - e - len(rt),
                lat_med=float(np.median(lat)) if lat else None)

def ub95(k, n):
    """Exact (Clopper-Pearson) one-sided... two-sided 95 % upper bound."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(0.975, k + 1, n - k))

def lb95(k, n):
    from scipy.stats import beta
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))

def dry(det_path, det_cfg=None, lv_path=LV, ctl_cfg=None, vol=None):
    """vol: None | ('entered', {name: uL}) | ('learn', None).  Returns {name: (leave_rel_label, why)}."""
    det, lv = mods(det_path, lv_path)
    R = h.dry(); out = {}
    for key, r in R.items():
        names = {d[1]: d for d in h.DRY if d[0] == key}
        wells = [(j, m['t'], m['home'] + 1) for j, m in enumerate(r.moves) if j in names]
        wv = {}
        if vol is not None and vol[0] == 'entered':
            wv = {r.moves[j]['t']: vol[1][names[j][2]] for j in names}
        o = sim.run_plate(lv, det, det_cfg, r, wells, 130.0, flow_rate=130, rates=h.ctl_rates(r), ctl_cfg=ctl_cfg,
                          rate_fn=lambda t, _r=r: dw.rate_at(_r, t), well_vols=wv,
                          learn=(vol is not None and vol[0] == 'learn'))
        for j, d in names.items():
            lt, why = o.get(j, (None, None)); m = r.moves[j]
            out[d[2]] = (None if lt is None else round(lt - m['t'] - d[3], 1), why)
    return out
