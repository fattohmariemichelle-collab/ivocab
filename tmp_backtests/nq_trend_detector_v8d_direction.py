from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,log_loss,precision_score,recall_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import nq_trend_detector_v8_real_nq as v8
import nq_trend_detector_v8b_real_nq as v8b
import nq_trend_detector_v8c_ablation as v8c

OUT=Path('trend_backtest_v8d_results'); OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20260828)
v8.session_features=v8b.causal_session_features; v8.add_labels=v8b.gap_safe_labels


def build():
    m=v8.load(); quality={'rows_1m':len(m),'start':str(m.ts.min()),'end':str(m.ts.max()),'duplicates':int(m.ts.duplicated().sum())}
    m=v8.session_features(m); h=v8.make_h1(m); d=v8.make_context(m,h); d=v8.add_h4_d1(d,m); d=v8.add_labels(d,12)
    d['ema20_50_signed_atr']=(d.ema20-d.ema50)/d.atr.replace(0,np.nan)
    d['ema10_20_signed_atr']=(d.ema10-d.ema20)/d.atr.replace(0,np.nan)
    d['dmi_signed']=(d.pdi-d.mdi)/100
    d['close_ema20_signed_atr']=(d.close-d.ema20)/d.atr.replace(0,np.nan)
    d['h4_ema_signed']=np.sign(d.h4_ema20-d.h4_ema50); d['d1_ema_signed']=np.sign(d.d1_ema20-d.d1_ema50)
    return d,quality


def sets():
    p=['ema20_50_signed_atr','ema10_20_signed_atr','dmi_signed','close_ema20_signed_atr','mom6_atr','mom12_atr','adx','er10','er20','rv6','rv24','h4_ema_signed','h4_dir','h4_adx','h4_er10','d1_ema_signed','d1_dir','d1_adx','d1_er10']
    loc=['dist_vwap_atr','dist_prev_high_atr','dist_prev_low_atr','dist_on_high_atr','dist_on_low_atr','dist_or_high_atr','dist_or_low_atr','on_range_atr','or_range_atr']
    vol=['rel_vol24','on_vol_ratio','or_vol_ratio']; sess=['hour','dow','is_rth','is_opening_hour','is_power_hour','is_overnight']
    return {'price_direction':p,'price_plus_location':p+loc,'price_location_volume':p+loc+vol,'full_direction':p+loc+vol+sess}


def models():
    return {
      'logit':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=3000,C=.5,class_weight='balanced',random_state=42)),
      'hgb':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=180,max_depth=3,learning_rate=.035,l2_regularization=1.5,random_state=42))}


def metrics(y,p,pred=None):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=.5).astype(int) if pred is None else np.asarray(pred,int)
    if not len(y): return {'n':0}
    two=len(np.unique(y))==2; cm=confusion_matrix(y,pred,labels=[0,1])
    return {'n':len(y),'long_base_rate':float(y.mean()),'auc':float(roc_auc_score(y,p)) if two else np.nan,'ap_long':float(average_precision_score(y,p)) if two else np.nan,'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if two else np.nan,'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,np.c_[1-p,p],labels=[0,1])),'precision_long':float(precision_score(y,pred,pos_label=1,zero_division=0)),'recall_long':float(recall_score(y,pred,pos_label=1,zero_division=0)),'precision_short':float(precision_score(y,pred,pos_label=0,zero_division=0)),'recall_short':float(recall_score(y,pred,pos_label=0,zero_division=0)),'confusion_short_long':cm.tolist()}


def abstain(y,p,margin):
    y=np.asarray(y,int); p=np.asarray(p,float); keep=np.abs(p-.5)>=margin
    r=metrics(y[keep],p[keep]); r.update({'coverage':float(keep.mean()),'margin':float(margin)})
    return r,keep


def boot_ci(y,pred,reps=3000):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    if len(y)<20 or len(np.unique(y))<2: return {'accuracy_ci95':[np.nan,np.nan],'balanced_accuracy_ci95':[np.nan,np.nan]}
    a=[]; b=[]; n=len(y)
    for _ in range(reps):
        z=RNG.integers(0,n,n); yy=y[z]; pp=pred[z]
        if len(np.unique(yy))<2: continue
        a.append(accuracy_score(yy,pp)); b.append(balanced_accuracy_score(yy,pp))
    return {'accuracy_ci95':[float(np.quantile(a,.025)),float(np.quantile(a,.975))],'balanced_accuracy_ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}


def fit_gate(d):
    gf=v8c.feature_sets()[0]['price_plus_overnight_opening']
    tr=d[(d.ts>='2023-01-01')&(d.ts<'2024-01-01')].dropna(subset=['trend_clean']); va=d[(d.ts>='2024-01-01')&(d.ts<'2025-01-01')].dropna(subset=['trend_clean'])
    cand=[]
    for name,m in models().items():
        m.fit(tr[gf],tr.trend_clean.astype(int)); p=m.predict_proba(va[gf])[:,1]; cand.append((average_precision_score(va.trend_clean.astype(int),p),name,m,p))
    cand.sort(reverse=True,key=lambda z:z[0]); ap,name,m,p=cand[0]
    return {'features':gf,'model':name,'validation_ap':float(ap),'cut_q90_2024':float(np.quantile(p,.9))},m


def gated(d,gate,gi,start,end):
    x=d[(d.ts>=start)&(d.ts<end)].dropna(subset=['trend_clean']).copy(); x['trend_prob']=gate.predict_proba(x[gi['features']])[:,1]; x=x[x.trend_prob>=gi['cut_q90_2024']]; q=x[x.trend_clean==1].copy(); q['y_long']=(q.fwd_atr>0).astype(int); return x,q


def choose(d,gate,gi,feats):
    tr=d[(d.ts>='2023-01-01')&(d.ts<'2024-01-01')&(d.trend_clean==1)].copy(); tr['y_long']=(tr.fwd_atr>0).astype(int)
    _,va=gated(d,gate,gi,'2024-01-01','2025-01-01'); cand=[]
    for name,m in models().items():
        m.fit(tr[feats],tr.y_long); p=m.predict_proba(va[feats])[:,1]; r=metrics(va.y_long,p); cand.append(((r['balanced_accuracy'],r['auc'],-r['brier']),name,m,p,r))
    cand.sort(reverse=True,key=lambda z:z[0]); _,name,m,p,vr=cand[0]
    grid=[]
    for margin in [0,.05,.10,.15,.20,.25,.30]:
        r,_=abstain(va.y_long,p,margin); r['eligible']=bool(r['n']>=40 and r['coverage']>=.35 and np.isfinite(r['balanced_accuracy'])); grid.append(r)
    ok=[r for r in grid if r['eligible']]; ok.sort(reverse=True,key=lambda r:(r['balanced_accuracy'],r['accuracy'],r['coverage'])); margin=float(ok[0]['margin']) if ok else 0.
    return name,m,vr,grid,margin


def benchmarks(q):
    y=q.y_long.to_numpy(int); rows=[]
    def add(name,pred,mask):
        pred=np.asarray(pred,int); mask=np.asarray(mask,bool)
        if mask.any(): rows.append({'rule':name,'coverage':float(mask.mean()),**metrics(y[mask],pred[mask].astype(float),pred[mask])})
    add('always_long',np.ones(len(q),int),np.ones(len(q),bool))
    h1=np.sign(q.ema20-q.ema50).to_numpy(int); add('h1_ema',(h1>0).astype(int),h1!=0)
    h4=q.h4_dir.to_numpy(int); add('h1_h4_ema_aligned',(h1>0).astype(int),(h1==h4)&(h1!=0))
    dm=np.sign(q.pdi-q.mdi).to_numpy(int); add('dmi',(dm>0).astype(int),dm!=0)
    vw=np.sign(q.close-q.session_vwap).to_numpy(int); add('vwap_side',(vw>0).astype(int),vw!=0)
    return rows


def evaluate(d,gate,gi,name,feats):
    mn,m,vr,grid,margin=choose(d,gate,gi,feats)
    gate25,q=gated(d,gate,gi,'2025-01-01','2025-12-01'); p=m.predict_proba(q[feats])[:,1]; raw=metrics(q.y_long,p); ar,keep=abstain(q.y_long,p,margin); pred=(p[keep]>=.5).astype(int); ar.update(boot_ci(q.y_long.to_numpy()[keep],pred))
    gate_dec,qd=gated(d,gate,gi,'2025-12-01','2025-12-11'); pdx=m.predict_proba(qd[feats])[:,1] if len(qd) else np.array([]); dec=metrics(qd.y_long,pdx) if len(qd) else {'n':0}; deca,_=abstain(qd.y_long,pdx,margin) if len(qd) else ({'n':0},np.array([],bool))
    return {'feature_set':name,'n_features':len(feats),'selected_model_2024':mn,'selected_margin_2024':margin,'validation_raw':vr,'validation_margin_grid':grid,'evaluation_2025_jan_nov':{'gated_rows':len(gate25),'clean_trends':len(q),'raw':raw,'abstained':ar,'benchmarks':benchmarks(q)},'micro_holdout_2025_dec1_10':{'gated_rows':len(gate_dec),'clean_trends':len(qd),'raw':dec,'abstained':deca}}


def main():
    d,quality=build(); gi,gate=fit_gate(d); out={'method':{'purpose':'V8d direction conditional on V8c favorable trend environment','train_direction':'2023 clean trends','validation_model_and_abstention':'2024 gated clean trends only','evaluation':'2025-01-01 through 2025-11-30; chronological but previously exposed to earlier exploratory rule summaries','micro_holdout':'2025-12-01 through 2025-12-10; not previously used, too small for standalone conclusions','trend_gate':'V8c price + causal overnight/opening features, q90 probability cutoff learned on 2024','direction_label':'LONG if clean-trend 12H forward ATR > 0, SHORT if < 0','abstention':'validation-selected |P(LONG)-0.5| margin; minimum 35% coverage and 40 predictions'},'data_quality':quality,'trend_gate':gi,'feature_sets':{},'ranking_2024':[]}
    for name,feats in sets().items():
        print('RUN',name,flush=True); out['feature_sets'][name]=evaluate(d,gate,gi,name,feats)
    ranking=[]
    for name,r in out['feature_sets'].items():
        v=r['validation_raw']; ranking.append({'feature_set':name,'balanced_accuracy':v['balanced_accuracy'],'auc':v['auc'],'brier':v['brier'],'selected_margin':r['selected_margin_2024']})
    ranking.sort(reverse=True,key=lambda x:(x['balanced_accuracy'],x['auc'],-x['brier'])); out['ranking_2024']=ranking; winner=ranking[0]['feature_set']; out['winner_selected_without_2025']=winner; out['winner_2025']=out['feature_sets'][winner]['evaluation_2025_jan_nov']; out['winner_micro_holdout']=out['feature_sets'][winner]['micro_holdout_2025_dec1_10']
    (OUT/'v8d_direction_results.json').write_text(json.dumps(out,indent=2,allow_nan=True)); pd.DataFrame(ranking).to_csv(OUT/'validation_ranking.csv',index=False)
    print('V8D_COMPLETE'); print(json.dumps({'trend_gate':gi,'ranking_2024':ranking,'winner':winner,'winner_2025':out['winner_2025'],'winner_micro_holdout':out['winner_micro_holdout']},indent=2,allow_nan=True))

if __name__=='__main__': main()
