import sys, pickle; sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study')
from run_ctl import *
import numpy as np, csv
out = {}
for label, det, lv, kw in (('v3 installed + controller as shipped', V3, LV3, {}), ('v4 final (no volume)', V4, LV4, {}),
                           ('v4 final + entered volume', V4, LV4, dict(vol=('entered', implied()), gate=(0.75, 1.0)))):
    res, g = one(label, det, lv, **kw)
    right = [x for x in g if x[2] == 'right' and x[4] != 'clog']
    clog_after = sum(1 for x in g if x[2] == 'right' and x[4] == 'clog')
    print(label, res['gold'], 'right excl clog', len(right), 'median', np.median([x[3] for x in right]), 'clog-after-lift', clog_after,
          '| dry', res['dry_caught'], res['dry_early'])
    out[label] = (res, g)
pickle.dump(out, open(D + '/results/final_check.pkl', 'wb'))
w = csv.writer(open(D + '/results/dry_wells_v3_vs_v4.csv', 'w'))
w.writerow(['well', 'run', 'label_s_after_move', 'confidence', 'notes', 'v3_leave_rel_s', 'v3_reason', 'v4_leave_rel_s', 'v4_reason'])
for d in h.DRY:
    a = out['v3 installed + controller as shipped'][0]['dry'][d[2]]; b = out['v4 final (no volume)'][0]['dry'][d[2]]
    w.writerow([d[2], d[0], d[3], d[4], d[5], a[0], a[1], b[0], b[1]])
