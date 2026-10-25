"""Candidate E: event -> confirmation (non-learned), vectorised per candidate.
Event (armed: move+lag+20 s, and optionally STEADY reached): tip dip on ch1/2/4 or common-mode air on ch1-4.
Confirm P1 'air at sensor + wash-out': air seen (the event itself if air, else an air flag >= 0.5 lag after the dip),
          then within W s the strong channels (ch3/5/6) have washed back toward the move baseline by >= w of the
          pre-event excursion.
Confirm P2 'sustained break' (dry well, no air needed): event size >= big x learned air size, and from t_e+dmin to
          t_e+Tp at least 2 channels stay >= m excursions away from the pre-event plateau.
Candidate expires at t_e + 1.6 lag + W."""
import numpy as np, itertools
import cands

def events(r, p, need_steady=True):
    dip, air = cands.flags(r, p); A = cands.armed(r)
    if need_steady: A = A & r['steady']
    F = r['F']; sig = F['sig']
    R4 = F['R4'][:, [0, 1, 3]]; dip_a = np.where((R4 < 0).all(1), (-R4).min(1) / F['amp'][:, [0, 1, 3]].mean(1), 0)
    R6 = F['R6'][:, :4]; same = np.abs(np.sign(R6).sum(1)) == 4
    air_a = np.where(same, (np.abs(R6) / F['amp']).min(1), 0)
    return dip & A, air & A, np.maximum(dip_a, air_a), air

def candidates(r, p, need_steady=True, refr=5.0, held=None, Hs=20.0, wmode='sum'):
    """List of per-candidate arrays (over its horizon)."""
    t = r['t']; X = r['X']; F = r['F']; lag = r['lag']; m5 = F['m5']; b0 = F['b0']; sig = F['sig']; k = F['k']
    dipE, airE, size, airF_raw = events(r, p, need_steady)
    A = cands.armed(r)
    airF = airF_raw & A
    ev = np.where(dipE | airE)[0]; out = []; last = -1e9
    Hmax = 3.5 * lag + 40
    for e in ev:
        if t[e] - last < refr: continue
        last = t[e]
        lo = np.searchsorted(t, t[e] - 20, 'left'); hi = np.searchsorted(t, t[e] - 3, 'right')
        if hi - lo < 3: continue
        pre = np.median(X[lo:hi], 0); bb = b0[e]
        if not np.all(np.isfinite(bb)): continue
        if held is not None:                      # valid measurement: strong channels held a flat plateau before the event
            lo2 = np.searchsorted(t, t[e] - Hs, 'left'); seg = m5[lo2:hi][:, [2, 4, 5]]
            ex = np.abs(seg[-1] - bb[[2, 4, 5]]); s3 = sig[e, [2, 4, 5]]; arm_ = ex >= 8 * s3
            if not arm_.any() or not np.all((seg.max(0) - seg.min(0))[arm_] <= held * ex[arm_] + 4 * s3[arm_]): continue
        end = np.searchsorted(t, t[e] + Hmax, 'right')
        nxt = np.searchsorted(F['mv'], t[e], 'right'); tend = F['mv'][nxt] if nxt < len(F['mv']) else np.inf
        end = min(end, np.searchsorted(t, tend, 'left'))
        idx = np.arange(e, end)
        if len(idx) < 2: continue
        den = np.maximum(np.abs(pre - bb)[None, :], 8 * sig[idx])
        if wmode == 'max':
            wash = ((np.abs(pre - bb)[None, :] - np.abs(m5[idx] - bb)) / den)[:, [2, 4, 5]].max(1)
        else:                                      # noise-weighted total excursion of ch3/5/6 (dominated by the best-SNR channel)
            s3 = sig[e, [2, 4, 5]]; Epre = (np.abs(pre - bb)[[2, 4, 5]] / s3).sum()
            Enow = (np.abs(m5[idx][:, [2, 4, 5]] - bb[[2, 4, 5]]) / s3).sum(1)
            wash = (1 - Enow / Epre) if Epre >= 24 else np.full(len(idx), -9.0)
        dep = np.sort(np.abs(m5[idx] - pre[None, :]) / den, 1)[:, -2]
        lo3 = np.searchsorted(t, t[e] - 30, 'left'); jpre = np.maximum(np.median(np.abs(F['dx'][lo3:hi, :4]), 0), 7e-4)
        jr = np.median(np.abs(F['dx'][idx, :4]) / jpre, 1)
        out.append(dict(e=e, te=t[e], isdip=bool(dipE[e]), size=size[e], idx=idx, tt=t[idx] - t[e], wash=wash, dep=dep, air=airF[idx], jr=jr))
    return out

def confirm_times(r, C, q):
    """q: w, W, big, m, Tp, dmin, lo_air(=0.5 lag), p1, p2 -> fire bool[n], tair[n]"""
    n = len(r['t']); lag = r['lag']; fire = np.zeros(n, bool); tair = np.full(n, np.nan)
    for c in C:
        tt = c['tt']; i_conf = None
        if q['p1']:
            if c['isdip']:
                ai = np.where(c['air'] & (tt >= q['lo_air'] * lag) & (tt <= q.get('hi_air', 1.6) * lag))[0]
            else:
                ai = np.array([0])
            if len(ai) and not (q.get('strict') and not c['isdip']):
                ta = tt[ai[0]]
                if q.get('need_wash', True):
                    ok = np.where((tt >= ta) & (tt <= ta + q['W']) & (c['wash'] >= q['w']))[0]
                    if len(ok): i_conf = ok[0]
                elif c['isdip'] and c['size'] >= q.get('pair_big', 0):
                    i_conf = ai[0]
        if q['p2'] and c['size'] >= q['big']:
            seg = (tt >= q['dmin']) & (tt <= q['Tp'])
            if seg.any() and tt[-1] >= q['Tp'] and np.mean(c['jr'][seg] >= q.get('J', 3.0)) >= q['fd']:
                j = np.where(tt >= q['Tp'])[0][0]
                if i_conf is None or j < i_conf: i_conf = j
        if i_conf is not None:
            g = c['idx'][i_conf]; fire[g] = True
            if not np.isfinite(tair[g]): tair[g] = c['te'] if (c['isdip'] or not q['p1']) else c['te'] - lag
    return fire, tair

EV_GRID = [dict(dks=a, dfa=0.6, aks=b, afa=c) for a, b, c in itertools.product((12, 20), (8, 12), (0.5, 1.0))]
Q1_GRID = [dict(w=w, W=W, lo_air=0.5, p1=True, p2=False, big=99, m=9, Tp=99, dmin=8) for w in (0.3, 0.5, 0.7) for W in (15, 25)]
Q2_GRID = [dict(big=b, fd=f, Tp=T, dmin=8, p2=True, J=3.0) for b in (1.5, 2.0, 3.0) for f in (0.7, 0.85, 1.0) for T in (25, 35)]
