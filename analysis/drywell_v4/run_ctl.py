"""Controller-level evaluation: gold (early/right/never) + 13 dry wells, for one configuration."""
import sys, json, time; sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study')
import ev, h
D = '/home/rpi/Documents/MABIP_datasets/drywell_study'
V3, V4 = '/home/rpi/Documents/MABIP/well_empty_detector.py', D + '/well_empty_detector_v4.py'
LV3, LV4 = '/home/rpi/Documents/MABIP/low_volume.py', D + '/low_volume_v4.py'
IMPLIED = {}   # loaded volume implied by the labels: V(t_dry) - 95 uL
def implied():
    import dw
    R = h.dry(); out = {}
    for d in h.DRY:
        r = R[d[0]]; m = r.moves[d[1]]; out[d[2]] = dw.volume(r, m['t'], m['t'] + d[3]) - 95.0
    return out
def one(label, det, lv, det_cfg=None, ctl_cfg=None, scale=1.0, gate=None, vol=None, plates=None):
    t0 = time.time()
    g = ev.gold(det, det_cfg, lv, ctl_cfg, scale=scale, gate=gate, plates=plates)
    sg = ev.summarize_gold(g)
    d = ev.dry(det, det_cfg, lv, ctl_cfg, vol=vol)
    caught = sum(1 for v in d.values() if v[0] is not None and v[0] >= -5)
    earlyd = [k for k, v in d.items() if v[0] is not None and v[0] < -5]
    res = dict(label=label, gold=sg, dry_caught=caught, dry_early=earlyd, dry=d,
               gold_early=[x for x in g if x[2] == 'early'], secs=round(time.time() - t0))
    return res, g
if __name__ == '__main__':
    pass
