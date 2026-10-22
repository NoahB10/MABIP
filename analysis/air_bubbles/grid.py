import numpy as np, pandas as pd, feat, itertools
from replay import replay, PRIOR
from well_empty_detector import WellEmptyConfig
runs=[(cr.strftime('%y%m%d_%H%M'),a,mv) for cr,f,a,mv,wl in feat.runs(10)]
CLEAN=['260120_1304','260218_1258','260325_1214','260329_1246','260329_1650','260507_1046','260508_1059','260510_1022']
def evaluate(**kw):
    rows=[]
    for key,a,mv in runs:
        cfg=WellEmptyConfig(lag_s=PRIOR.get(key,25.0),**kw)
        evs,det=replay(a,mv,cfg)
        for j,(m,dw,port) in enumerate(mv):
            if dw<60: continue
            nxt=mv[j+1][0] if j+1<len(mv) else a[-1,0]; lift=m+dw+1.5
            E=[(e,L) for e,L in evs if m<=e.t<nxt]
            if a[-1,0]<lift+60: continue
            hit=any(e.kind=='EMPTY' and lift-3<=e.t<=lift+L+30 for e,L in E)
            fp=any(e.kind=='EMPTY' and e.t<lift-3 for e,L in E)
            sus=any(e.kind=='SUSPECT' and e.t<lift-3 for e,L in E)
            rows.append((key,hit,fp,sus))
    d=pd.DataFrame(rows,columns=['run','hit','fp','sus']); c=d[d.run.isin(CLEAN)]
    return c.hit.mean(),c.fp.mean(),d.hit.mean(),d.fp.mean(),d.sus.mean(),d.fp.sum()
print('k  tolfrac minv | clean hit fp | all hit fp(rate) suspect | fp count')
for k,tf,mv_,tr in itertools.product([6,8,10],[0.15,0.5],[3],[30]):
    r=evaluate(k_spike=k,k_dip=k,pair_tol_frac=tf,min_votes=mv_,train_s=tr)
    print(k,tf,mv_,tr,' | %.3f %.4f | %.3f %.4f %.3f | %d'%r)
