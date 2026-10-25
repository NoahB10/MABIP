import sys; sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study')
from run_ctl import *
import numpy as np
INST_DET, INST_LV = '/home/rpi/Documents/MABIP/well_empty_detector.py', '/home/rpi/Documents/MABIP/low_volume.py'
for label, kw in (('INSTALLED v4 (learn off, no volume)', {}),
                  ('INSTALLED v4 + entered volume', dict(vol=('entered', implied()), gate=(0.75, 1.0)))):
    res, g = one(label, INST_DET, INST_LV, **kw)
    right = [x for x in g if x[2] == 'right' and x[4] != 'clog']
    print(label, res['gold'], '| right excl clog', len(right), 'median %.0f' % np.median([x[3] for x in right]),
          '| dry caught', res['dry_caught'], 'dry early', res['dry_early'], flush=True)
