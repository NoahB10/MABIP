"""Honest selection over the cached grid: gold leave-one-plate-out, dry leave-one-run-out."""
import sys, glob, pickle, json, numpy as np; sys.path.insert(0, '/home/rpi/Documents/MABIP_datasets/drywell_study')
import ev
G = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob('/home/rpi/Documents/MABIP_datasets/drywell_study/results/grid/*.pkl'))]
G = [g for g in G if g['cfg'] != {'dd_path': False}]          # candidates = v4 variants
plates = sorted({x[0] for x in G[0]['gold']})
RUN = lambda name: name[:2]
def gstats(g, keep):
    rows = [x for x in g['gold'] if keep(x[0])]
    return sum(x[2] == 'early' for x in rows), sum(x[2] == 'right' for x in rows), len(rows)
def dstats(g, keep):
    d = {k: v for k, v in g['res']['dry'].items() if keep(k)}
    early = sum(1 for v in d.values() if v[0] is not None and v[0] < -5 and v[1] != 'clog')
    caught = sum(1 for v in d.values() if v[0] is not None and v[0] >= -5)
    lat = np.mean([min(max(v[0], 0), 300) if v[0] is not None and v[0] >= -5 else 300 for v in d.values()])
    return early, caught, lat
def key(g, gkeep, dkeep):
    ge, gr, _ = gstats(g, gkeep); de, dc, dl = dstats(g, dkeep)
    return (ge + de, -dc, -gr, dl)
# gold LOPO (selection on 14 plates + all dry wells)
tot = dict(early=0, right=0, n=0); chosen = []
for p in plates:
    best = min(G, key=lambda g: key(g, lambda k: k != p, lambda k: True))
    e, r, n = gstats(best, lambda k: k == p); tot['early'] += e; tot['right'] += r; tot['n'] += n; chosen.append(json.dumps(best['cfg']))
print('GOLD LOPO test: early', tot['early'], 'of', tot['n'], f"({1000*tot['early']/tot['n']:.2f}/1000, 95% UB {1000*ev.ub95(tot['early'], tot['n']):.2f}/1000)",
      'right', tot['right'], f"({100*tot['right']/tot['n']:.1f}%)")
from collections import Counter; print('   configs chosen', Counter(chosen).most_common(3))
# dry LORO (selection on the other run + all gold)
for H in ('22', '25'):
    best = min(G, key=lambda g: key(g, lambda k: True, lambda k: RUN(k) != H))
    e, c, l = dstats(best, lambda k: RUN(k) == H)
    d = {k: v for k, v in best['res']['dry'].items() if RUN(k) == H}
    print(f'DRY LORO test {H} Oct: tuned on the other run -> {best["cfg"]}: caught {c}/{len(d)} early {e}', {k: v[0] for k, v in d.items()})
# full table
print('\nall configs: gold early/right | dry caught/early(non-clog)')
for g in sorted(G, key=lambda g: key(g, lambda k: True, lambda k: True)):
    ge, gr, n = gstats(g, lambda k: True); de, dc, dl = dstats(g, lambda k: True)
    print(f"{json.dumps(g['cfg']):95s} {ge:2d} {gr:5d} | {dc:2d} {de:2d} lat {dl:5.1f}")
