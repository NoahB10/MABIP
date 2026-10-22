"""Re-score every (method, params, rule) from saved calls, pick each method's best, write results.csv."""
import glob, pickle, numpy as np, pandas as pd, fw
R = fw.load_all()
DET = {}
for f in sorted(glob.glob('calls_*.pkl')):
    if f == 'calls_all.pkl': continue
    DET.update(pickle.load(open(f, 'rb')))
DELAY = {}
for f in glob.glob('sweep_*.pkl'):
    d = pd.read_pickle(f)
    for _, x in d.iterrows(): DELAY[(x.method, x.params, x.rule)] = x.delay
rows = []
for key, calls in DET.items():
    s = fw.score(R, calls); s.update(method=key[0], params=key[1], rule=key[2], delay_samples=DELAY.get(key, np.nan)); rows.append(s)
df = pd.DataFrame(rows)
# ranking objective (user priority: speed + real dry-well recall; then false alarms; surrogate recall as a sanity check)
df['J'] = (0.35 * df.real_recall + 0.25 * (1 - df.lat_real_capped / 60) + 0.25 * (1 - np.minimum(df.fa_per_h, 5) / 5) + 0.15 * df.surr_recall)
df['delay_samples'] = np.where(df.rule.str.contains('v2|dipalone') & (df.method == 'm2_diffz'), df.delay_samples.fillna(2), df.delay_samples)
df.to_pickle('scored_all.pkl')
best = df[~((df.method == 'm1_baseline') & (df.rule != 'native'))].sort_values('J', ascending=False).groupby('method').head(1).sort_values('J', ascending=False)
best = best.assign(row='method best')
var = pd.concat([df[(df.method == 'm1_baseline') & (df.rule != 'native')],
                 df[(df.method == 'm2_diffz') & (df.rule == 'dipalone') & (df.params.str.contains("'k': 20, 'nmin': 3, 'fall': False, 'span': 2, 'cm': 2.0"))].head(1)]).assign(row='variant')
best = pd.concat([best, var])
# also each method's best 'dipalone' (dip-only) and best non-dip (peak-only) operating points
dip = df[df.rule == 'dipalone'].sort_values('J', ascending=False).groupby('method').head(1)
peak = df[df.rule.isin(['spike', 'pair', 'native'])].sort_values('J', ascending=False).groupby('method').head(1)
def fmt(x):
    out = []
    for d in x:
        if d[1] == 'hit': out.append(f"{d[0]}:hit@{d[2]:.0f}s({d[3]},+{d[4]:.0f}s)")
        elif d[1] == 'FA': out.append(f"{d[0]}:FALSE@{d[2]:.0f}s({d[3]})")
        else: out.append(f"{d[0]}:miss")
    return '; '.join(out)
cols = ['row', 'method', 'rule', 'params', 'J', 'real_recall', 'real_hits', 'lat_real_med', 'lat_real_capped', 'surr_recall', 'lat_surr_med', 'fa_per_h', 'fa_per_well', 'fa_n', 'delay_samples', 'wells']
for t, d in (('results.csv', best), ('results_dip_only.csv', dip.assign(row='dip only')), ('results_peak_only.csv', peak.assign(row='peak only'))):
    d = d.copy(); d['wells'] = d.real_detail.map(fmt); d['delay_s'] = d.delay_samples * 1.67
    d[cols + ['delay_s']].round(3).to_csv(t, index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
for t in ('results.csv', 'results_dip_only.csv', 'results_peak_only.csv'):
    print('==', t); print(pd.read_csv(t).drop(columns=['wells']).to_string())
print(pd.read_csv('results.csv')[['method', 'wells']].to_string())
