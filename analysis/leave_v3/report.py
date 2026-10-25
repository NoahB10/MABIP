"""Aggregate all candidate results -> results/*.csv and printed tables."""
import sys, pickle, os; sys.dont_write_bytecode = True
LDIR = "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/leave"; sys.path.insert(0, LDIR)
import numpy as np, pandas as pd, core
from scipy.stats import beta
TAGS = ['installed', 'evc_P1P2', 'evc_P1P2w', 'evc_P1', 'pair+S', 'pair', 'diponly+S', 'aironly+S', 'm8+S', 'm8', 'confirm+S', 'confirm', 'logreg+S', 'logreg', 'logreg2+S', 'gbm+S', 'gbm', 'gbm2+S']
DESC = {'installed': 'installed v2 (frozen)', 'evc_P1P2w': 'v3 with widened dry-path grid (grid widened after seeing 22 Oct)', 'evc_P1P2': 'v3: STEADY -> event -> (air+wash-out | dry suction break)', 'evc_P1': 'STEADY -> event -> air at sensor + wash-out',
        'pair+S': 'STEADY + tip dip -> sensor air 0.5-1.6 lag later', 'pair': 'tip dip -> sensor air (no STEADY)', 'diponly+S': 'STEADY + tip dip only (fast)',
        'aironly+S': 'STEADY + sensor air only (M6-style)', 'm8+S': 'atlas M8 (dip OR air) + STEADY', 'm8': 'atlas M8 unwindowed (dip OR air)',
        'confirm+S': 'event -> (air | wash | freeze), OR-confirm + STEADY', 'confirm': 'event -> OR-confirm', 'logreg+S': 'logistic regression + STEADY', 'logreg': 'logistic regression',
        'logreg2+S': 'logistic regr. + event-anchored feats + STEADY', 'gbm+S': 'gradient boosting + STEADY', 'gbm': 'gradient boosting', 'gbm2+S': 'gradient boosting + event-anchored feats + STEADY'}
LAB = pd.read_csv(LDIR + '/../atlas/results/well_labels.csv')
R = core.load(False); RK = {r['key']: r for r in R}
noflow = set()
for _, w in LAB[LAB.flow_label == 'noflow'].iterrows():
    r = RK[w.plate]; mv = np.array([m['t'] for m in r['moves']]); j = int(np.argmin(np.abs(mv - w.move)))
    if abs(mv[j] - w.move) < 15 and r['moves'][j]['port'] == w.well: noflow.add((w.plate, j))
def cp_hi(k, n): return beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
rows = []; real_rows = []; per_plate = []
for tag in TAGS:
    f = f'{LDIR}/cache/exp_{tag}.pkl'
    if not os.path.exists(f): continue
    o = pickle.load(open(f, 'rb'))
    for rule in (True, False):
        S = []
        for d in o['splits']:
            if d['rule'] != rule: continue
            s = core.summarize(d['rows']); S.append(s)
        lo = [x for d in o['lopo'] if d['rule'] == rule for x in d['rows']]; sl = core.summarize(lo)
        nf = [x for x in lo if (x['plate'], x['k']) in noflow]; ok = [x for x in lo if (x['plate'], x['k']) not in noflow]
        so = core.summarize(ok)
        a15 = o['all15'][rule]
        rows.append(dict(candidate=tag, description=DESC[tag], plateau_rule=rule,
                         split_right=np.mean([s['right'] for s in S]), split_right_sd=np.std([s['right'] for s in S]),
                         split_early=np.mean([s['early'] for s in S]), split_early_sd=np.std([s['early'] for s in S]),
                         split_early_n=np.mean([s['n_early'] for s in S]), split_never=np.mean([s['never'] for s in S]),
                         split_lat=np.mean([s['lat'] for s in S]), split_lat_sd=np.std([s['lat'] for s in S]),
                         split_no_steady=np.mean([s['no_steady'] for s in S]),
                         lopo_right=sl['right'], lopo_early_n=sl['n_early'], lopo_early_per1000=1000 * sl['n_early'] / sl['n'],
                         lopo_early_per1000_95hi=1000 * cp_hi(sl['n_early'], sl['n']), lopo_never=sl['never'], lopo_lat=sl['lat'],
                         lopo_no_steady=sl['no_steady'], lopo_flowing_right=so['right'], lopo_noflow_n=len(nf),
                         lopo_noflow_left=sum(x['outcome'] != 'never' for x in nf),
                         all15_right=core.summarize(a15['rows'])['right'], all15_early_n=core.summarize(a15['rows'])['n_early']))
        for x in a15['real']: real_rows.append(dict(candidate=tag, plateau_rule=rule, well=x['well'], left_s_after_air=x['left'], kind=x['kind']))
        # real wells in each split model
        if rule:
            for d in o['lopo']:
                if d['rule']: 
                    s = core.summarize(d['rows']); per_plate.append(dict(candidate=tag, plate=d['plate'], wells=s['n'], right=s['right'], early_n=s['n_early'], never=s['never'], lat=s['lat']))
            for i in sorted(set(d['split'] for d in o['splits'])):
                d = [d for d in o['splits'] if d['split'] == i and d['rule']][0]
                for x in d['real']: real_rows.append(dict(candidate=tag, plateau_rule='split%d' % i, well=x['well'], left_s_after_air=x['left'], kind=x['kind']))
T = pd.DataFrame(rows); T.to_csv(LDIR + '/results/ranked_candidates.csv', index=False)
pd.DataFrame(real_rows).to_csv(LDIR + '/results/real_dry_wells_22oct.csv', index=False)
P = pd.DataFrame(per_plate); P.to_csv(LDIR + '/results/per_plate_lopo.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
t = T[T.plateau_rule].sort_values(['split_early', 'split_right'], ascending=[True, False])
print(t[['candidate', 'split_right', 'split_right_sd', 'split_early', 'split_early_sd', 'split_never', 'split_lat', 'split_no_steady', 'lopo_right', 'lopo_early_n', 'lopo_early_per1000', 'lopo_early_per1000_95hi', 'lopo_lat', 'lopo_no_steady', 'lopo_noflow_n', 'lopo_noflow_left']].round(2).to_string(index=False))
print('\nplateau rule OFF')
t = T[~T.plateau_rule].sort_values(['split_early', 'split_right'], ascending=[True, False])
print(t[['candidate', 'split_right', 'split_right_sd', 'split_early', 'split_early_sd', 'split_never', 'split_lat', 'lopo_right', 'lopo_early_n', 'lopo_lat']].round(2).to_string(index=False))
RR = pd.DataFrame(real_rows); r1 = RR[RR.plateau_rule.astype(str) == 'True']
print('\n22 Oct (all-15 fit, rule on):'); print(r1.pivot(index='candidate', columns='well', values='left_s_after_air').to_string())
