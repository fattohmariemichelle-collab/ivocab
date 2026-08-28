from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import nq_trend_detector_v8d_direction as d8
import nq_trend_detector_v8d2_event_audit as a8

OUT=Path('trend_backtest_v8d3_results'); OUT.mkdir(exist_ok=True)


def eval_rule(q,pred):
    y=(q.fwd_atr>0).astype(int).to_numpy(); pred=np.asarray(pred,int); mask=pred>=0
    if not mask.any(): return {'n':0}
    yy=y[mask]; pp=pred[mask]
    return {'n':int(mask.sum()),'coverage':float(mask.mean()),'accuracy':float(accuracy_score(yy,pp)),'balanced_accuracy':float(balanced_accuracy_score(yy,pp)) if len(np.unique(yy))==2 else np.nan,'long_rate':float(yy.mean())}


def counter_or(q):
    # Explicit causal contrarian opening-range rule, only when current close is outside completed OR30.
    pred=np.full(len(q),-1,int)
    valid=q.or30_high.notna().to_numpy() & q.or30_low.notna().to_numpy()
    above=valid & (q.close.to_numpy()>q.or30_high.to_numpy()); below=valid & (q.close.to_numpy()<q.or30_low.to_numpy())
    pred[above]=0; pred[below]=1
    return pred


def combined_all_gated(x,pred):
    pred=np.asarray(pred,int); valid=pred>=0; clean=x.trend_clean.to_numpy()==1; true=(x.fwd_atr.to_numpy()>0).astype(int)
    n=int(valid.sum()); correct=valid&clean&(pred==true); wrong=valid&clean&(pred!=true); no=valid&(~clean)
    return {'n_predictions':n,'coverage':float(valid.mean()),'correct_clean_direction_rate':float(correct.sum()/n) if n else np.nan,'wrong_clean_direction_rate':float(wrong.sum()/n) if n else np.nan,'no_clean_trend_rate':float(no.sum()/n) if n else np.nan,'conditional_direction_accuracy_when_clean':float(correct.sum()/(correct.sum()+wrong.sum())) if (correct.sum()+wrong.sum()) else np.nan,'clean_trend_rate_among_predictions':float((correct.sum()+wrong.sum())/n) if n else np.nan}


def main():
    d,_=d8.build(); gi,gate=d8.fit_gate(d); feats=d8.sets()['price_location_volume']; mn,model,_,_,margin=d8.choose(d,gate,gi,feats)
    out={'model':mn,'margin':margin,'periods':{}}
    for label,start,end in [('validation_2024','2024-01-01','2025-01-01'),('evaluation_2025','2025-01-01','2025-12-01')]:
        allg,q=d8.gated(d,gate,gi,start,end); q=a8.nonoverlap(q,12); allg=a8.nonoverlap(allg,12)
        pm=model.predict_proba(q[feats])[:,1]; predm=(pm>=.5).astype(int); keep=np.abs(pm-.5)>=margin; predma=np.where(keep,predm,-1)
        por=counter_or(q)
        # Predictions on all gated non-overlap rows, not conditioned on future trend label.
        pall=model.predict_proba(allg[feats])[:,1]; predall=np.where(np.abs(pall-.5)>=margin,(pall>=.5).astype(int),-1); porall=counter_or(allg)
        out['periods'][label]={
          'gated_nonoverlap_rows':len(allg),'clean_nonoverlap_rows':len(q),
          'direction_model_raw_on_clean':eval_rule(q,predm),
          'direction_model_abstained_on_clean':eval_rule(q,predma),
          'counter_opening_range_on_clean':eval_rule(q,por),
          'combined_model_on_all_gated':combined_all_gated(allg,predall),
          'combined_counter_or_on_all_gated':combined_all_gated(allg,porall),
        }
    (OUT/'v8d3_combined_audit.json').write_text(json.dumps(out,indent=2,allow_nan=True)); print('V8D3_COMPLETE'); print(json.dumps(out,indent=2,allow_nan=True))

if __name__=='__main__': main()
