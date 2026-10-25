import sys, pickle; sys.dont_write_bytecode = True
LDIR = "/tmp/claude-1000/-home-rpi/39b37385-06a4-4bf4-8b8b-5524dd82bdb0/scratchpad/leave"; sys.path.insert(0, LDIR)
import numpy as np, pandas as pd, core, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B, O, INK, MUT = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'axes.edgecolor': MUT, 'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
# 1 per-plate bars (LOPO)
P = pd.read_csv(LDIR + '/results/per_plate_lopo.csv')
a = P[P.candidate == 'installed'].set_index('plate'); b = P[P.candidate == 'evc_P1P2'].set_index('plate')
pl = list(a.index); x = np.arange(len(pl)); lab = [p[5:10] + ' ' + p.split('_', 2)[2][:10] for p in pl]
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].bar(x - 0.2, a.loc[pl, 'right'], 0.38, color=O, label='installed v2'); ax[0].bar(x + 0.2, b.loc[pl, 'right'], 0.38, color=B, label='v3 (leave-one-plate-out)')
ax[0].set_ylabel('left at the right time, %'); ax[0].legend(frameon=False, loc='lower left'); ax[0].set_ylim(0, 100); ax[0].grid(axis='y', color='#e6e6e3')
ax[0].axvline(8.5, color=MUT, ls=':'); ax[0].text(8.6, 95, 'Jun-Aug', color=MUT)
ax[1].bar(x - 0.2, a.loc[pl, 'early_n'], 0.38, color=O); ax[1].bar(x + 0.2, b.loc[pl, 'early_n'], 0.38, color=B)
for i, p in enumerate(pl):
    ax[1].text(i + 0.2, b.loc[p, 'early_n'] + 0.1, str(int(b.loc[p, 'early_n'])), ha='center', color=INK, fontsize=8)
    ax[1].text(i - 0.2, a.loc[p, 'early_n'] + 0.1, str(int(a.loc[p, 'early_n'])), ha='center', color=INK, fontsize=8)
ax[1].set_ylabel('wells left EARLY (count)'); ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=40, ha='right'); ax[1].grid(axis='y', color='#e6e6e3')
fig.suptitle('Per plate: installed v2 vs v3, each plate scored with v3 tuned on the other 14 (plateau rule as installed)', color=INK)
plt.tight_layout(); plt.savefig(LDIR + '/plots/per_plate_v3_vs_installed.png', dpi=110); plt.close()
# 2 trade-off
Tt = pd.read_csv(LDIR + '/results/tradeoff_operating_points.csv')
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
lines = []
for i, r in Tt.reset_index(drop=True).iterrows():
    c = O if r.op.startswith('installed') else B
    ax[0].errorbar(r.early, r.right, xerr=r.early_sd, yerr=r.right_sd, fmt='o', color=c, ms=7, lw=1)
    ax[0].annotate(str(i + 1), (r.early, r.right), xytext=(5, 5), textcoords='offset points', fontsize=9, color=INK)
    nd = int(sum(0 <= r[w] <= 40 for w in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'] if pd.notna(r[w])))
    lines.append(f"{i+1}. {r.op}\n     early {r.early:.2f}%  right {r.right:.1f}%  latency {r.lat:.0f} s  22 Oct dry wells left <=40 s: {nd}/6")
ax[0].set_xlabel('left EARLY, % of test wells (mean +/- sd, 8 splits, trained on 3 plates)'); ax[0].set_ylabel('left at the right time, %'); ax[0].grid(color='#e6e6e3')
ax[0].set_title('Certainty vs recall (plateau rule as installed)', color=INK)
ax[1].axis('off'); ax[1].text(0, 1, "\n".join(lines), va='top', fontsize=8.5, color=INK, family='monospace')
plt.tight_layout(); plt.savefig(LDIR + '/plots/tradeoff_operating_points.png', dpi=110); plt.close()
# 3 gallery
if '--no-gallery' in sys.argv: sys.exit()
R = core.load(); EI = pickle.load(open(LDIR + '/cache/events_installed.pkl', 'rb')); EV = pickle.load(open(LDIR + '/cache/events_v3.pkl', 'rb'))
caught, worse = [], []
for r in R:
    if r['today']: continue
    Li = core.simulate(r, *EI[r['key']][:2]); Lv = core.simulate(r, *EV[r['key']][:2])
    si = {x['k']: x for x in core.score(r, Li)}; sv = {x['k']: x for x in core.score(r, Lv)}
    for k in sv:
        if si[k]['outcome'] != 'right' and sv[k]['outcome'] == 'right': caught.append((r, k, Li.get(k), Lv.get(k), si[k]['outcome']))
        if si[k]['outcome'] == 'right' and sv[k]['outcome'] != 'right' or sv[k]['outcome'] == 'early': worse.append((r, k, Li.get(k), Lv.get(k), sv[k]['outcome']))
rng = np.random.default_rng(3)
late = [c for c in caught if not core.EARLY(c[0]['key'])]; early_ = [c for c in caught if core.EARLY(c[0]['key'])]
sel = [late[i] for i in rng.choice(len(late), min(12, len(late)), replace=False)] + [early_[i] for i in rng.choice(len(early_), min(6, len(early_)), replace=False)]
selw = [worse[i] for i in rng.choice(len(worse), min(6, len(worse)), replace=False)]
items = [(c, 'caught') for c in sel] + [(c, 'worse') for c in selw]
nc = 6; nr = int(np.ceil(len(items) / nc)); fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.1 * nr)); axs = axs.ravel()
cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
for a_, ((r, k, li, lv, o), kind) in zip(axs, items):
    m = r['moves'][k]; t = r['t']; w = (t >= m['t'] - 10) & (t <= m['lift'] + r['lag'] + 40)
    for c in range(6):
        y = r['X'][w, c] - r['X'][w, c][0]; sc = np.max(np.abs(y)) or 1
        a_.plot(t[w] - m['t'], y / sc - (0 if c < 4 else 2.3 * (c - 3)), lw=0.9, color=cols[c], label=f'ch{c+1}')
    a_.axvline(m['lift'] - m['t'], color=MUT, ls='--', lw=1)
    st = r['steady_t'].get(k)
    if st: a_.axvline(st - m['t'], color='#008300', ls=':', lw=1)
    if lv: a_.axvline(lv[0] - m['t'], color=B, lw=2)
    if li: a_.axvline(li[0] - m['t'], color=O, lw=2, ls='-.')
    ttl = f"{r['key'][5:10]} {r['key'].split('_', 2)[2][:9]} {m['port']}: " + ('v2 ' + o + ', v3 right' if kind == 'caught' else 'v3 ' + o)
    a_.set_title(ttl, fontsize=8, color=INK if kind == 'caught' else '#e34948'); a_.set_yticks([])
for a_ in axs[len(items):]: a_.axis('off')
axs[0].legend(fontsize=6, ncol=2, frameon=False)
fig.suptitle(f'v3 vs installed (all-15 v3, plateau rule as installed). First {len(sel)}: wells v2 missed and v3 left at the right time ({len(caught)} in all); '
             f'last {len(selw)} (red titles): wells v3 gets wrong that v2 got right, or v3 early ({len(worse)} in all).\n'
             'grey dashed = lift-out, green dotted = STEADY, blue = v3 leave, orange dash-dot = v2 leave. Traces scaled per channel; ch5/ch6 offset below.', fontsize=9, color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(LDIR + '/plots/gallery_v3_vs_installed.png', dpi=80); plt.close()
print('caught', len(caught), 'worse', len(worse), pd.Series([w[4] for w in worse]).value_counts().to_dict())
