from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,accuracy_score
import nq_trend_detector_v8d_direction as d8

OUT=Path('trend_backtest_v8d2_results'); OUT.mkdir(exist_ok=True)


def nonoverlap(q,hours=12):
    q=q.sort_values('ts').copy()
    keep=[]; next_time=None
    for i,row in q.iterrows():
        if next_time is None or row.ts>=next_time:
            keep.append(i); next_time=row.ts+pd.Timedelta(hours=hours)
    return q.loc[keep].copy()


def period(d,gate,gi,start,end):
    allg,q=d8.gated(d,gate,gi,start,end)
    return allg,nonoverlap(q,12)


def choose_event_model(d,gate,gi,feats):
    tr=d[(d.ts>='2023-01-01')&(d.ts<'2024-01-01')&(d.trend_clean==1)].copy(); tr['y_long']=(tr.fwd_atr>0).astype(int); tr=nonoverlap(tr,12)
    _,va=period(d,gate,gi,'2024-01-01','2025-01-01')
    cand=[]
    for name,m in d8.models().items():
        m.fit(tr[feats],tr.y_long); p=m.predict_proba(va[feats])[:,1]; r=d8.metrics(va.y_long,p); cand.append(((r.get('balanced_accuracy',-1),r.get('auc',-1),-r.get('brier',99)),name,m,p,r))
    cand.sort(reverse=True,key=lambda z:z[0]); _,name,m,p,vr=cand[0]
    grid=[]
    for margin in [0,.05,.10,.15,.20,.25,.30]:
        r,_=d8.abstain(va.y_long,p,margin); r['eligible']=bool(r['n']>=12 and r['coverage']>=.35 and np.isfinite(r.get('balanced_accuracy',np.nan))); grid.append(r)
    ok=[r for r in grid if r['eligible']]; ok.sort(reverse=True,key=lambda r:(r['balanced_accuracy'],r['accuracy'],r['coverage'])); margin=float(ok[0]['margin']) if ok else 0.
    return name,m,vr,grid,margin,len(tr),len(va)


def single_feature_audit(q):
    y=q.y_long.to_numpy(int); rows=[]
    rules={
      'ema20_50':np.sign(q.ema20-q.ema50),
      'dmi':np.sign(q.pdi-q.mdi),
      'momentum6':np.sign(q.mom6_atr),
      'momentum12':np.sign(q.mom12_atr),
      'vwap_side':np.sign(q.dist_vwap_atr),
      'prev_high_distance':np.sign(q.dist_prev_high_atr),
      'prev_low_distance':np.sign(q.dist_prev_low_atr),
      'overnight_high_distance':np.sign(q.dist_on_high_atr),
      'overnight_low_distance':np.sign(q.dist_on_low_atr),
      'opening_high_distance':np.sign(q.dist_or_high_atr),
      'opening_low_distance':np.sign(q.dist_or_low_atr),
    }
    for name,s in rules.items():
        s=np.asarray(s,float); mask=np.isfinite(s)&(s!=0); pred=(s[mask]>0).astype(int)
        if mask.sum(): rows.append({'rule':name,'n':int(mask.sum()),'coverage':float(mask.mean()),'accuracy':float(accuracy_score(y[mask],pred)),'balanced_accuracy':float(balanced_accuracy_score(y[mask],pred)) if len(np.unique(y[mask]))==2 else np.nan})
    return rows


def main():
    d,quality=d8.build(); gi,gate=d8.fit_gate(d); results={}
    for fs,feats in d8.sets().items():
        print('RUN EVENT',fs,flush=True); mn,m,vr,grid,margin,ntr,nva=choose_event_model(d,gate,gi,feats)
        _,te=period(d,gate,gi,'2025-01-01','2025-12-01'); p=m.predict_proba(te[feats])[:,1]; raw=d8.metrics(te.y_long,p); ar,keep=d8.abstain(te.y_long,p,margin); pred=(p[keep]>=.5).astype(int); ar.update(d8.boot_ci(te.y_long.to_numpy()[keep],pred))
        results[fs]={'model':mn,'train_nonoverlap_clean_trends':ntr,'validation_nonoverlap_gated_clean_trends':nva,'selected_margin':margin,'validation_raw':vr,'validation_margin_grid':grid,'test_nonoverlap_gated_clean_trends':len(te),'test_raw':raw,'test_abstained':ar,'test_benchmarks':d8.benchmarks(te),'single_feature_audit':single_feature_audit(te)}
    ranking=[]
    for fs,r in results.items():
        v=r['validation_raw']; ranking.append({'feature_set':fs,'validation_balanced_accuracy':v['balanced_accuracy'],'validation_auc':v['auc'],'validation_brier':v['brier'],'margin':r['selected_margin']})
    ranking.sort(reverse=True,key=lambda x:(x['validation_balanced_accuracy'],x['validation_auc'],-x['validation_brier'])); winner=ranking[0]['feature_set']
    out={'method':{'purpose':'Correct V8d row-overlap inflation by enforcing >=12h spacing between clean-trend direction observations in train, validation and test','selection':'feature set/model/margin selected on 2024 non-overlapping events only','evaluation':'2025 Jan-Nov; chronological evaluation, not pristine globally because earlier exploratory 2025 summaries were viewed'},'data_quality':quality,'trend_gate':gi,'ranking_2024_event_level':ranking,'winner':winner,'winner_2025_event_level':results[winner],'all_feature_sets':results}
    (OUT/'v8d2_event_audit.json').write_text(json.dumps(out,indent=2,allow_nan=True)); pd.DataFrame(ranking).to_csv(OUT/'event_validation_ranking.csv',index=False)
    print('V8D2_COMPLETE'); print(json.dumps({'ranking':ranking,'winner':winner,'winner_2025':results[winner]},indent=2,allow_nan=True))

if __name__=='__main__': main()
