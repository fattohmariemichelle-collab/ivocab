from __future__ import annotations

import json, math, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(os.getenv('NQ_CSV','data_v8/Dataset_NQ_1min_2022_2025.csv'))
OUT = Path('trend_backtest_v8_results'); OUT.mkdir(exist_ok=True)


def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def er(s,n):
    path=s.diff().abs().rolling(n,min_periods=n).sum()
    return (s-s.shift(n)).abs()/path.replace(0,np.nan)

def add_adx(d,n=14):
    prev=d.close.shift(); tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    atr=rma(tr,n); up=d.high.diff(); down=-d.low.diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0.),index=d.index); mdm=pd.Series(np.where((down>up)&(down>0),down,0.),index=d.index)
    pdi=100*rma(pdm,n)/atr.replace(0,np.nan); mdi=100*rma(mdm,n)/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['atr']=atr; d['adx']=rma(dx,n); d['pdi']=pdi; d['mdi']=mdi
    return d

def load():
    d=pd.read_csv(DATA)
    d.columns=[c.strip().lower().replace(' ','_') for c in d.columns]
    tcol=next(c for c in d.columns if 'timestamp' in c or c in ('datetime','time'))
    d['ts']=pd.to_datetime(d[tcol],errors='coerce')
    rename={}
    for k in ['open','high','low','close','volume']:
        if k not in d.columns:
            alt=next((c for c in d.columns if c.endswith(k)),None)
            if alt: rename[alt]=k
    d=d.rename(columns=rename)
    need=['ts','open','high','low','close','volume']; d=d[need+[c for c in d.columns if c.startswith('vwap')]].copy()
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=need).sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    return d

def session_features(m):
    x=m.copy(); x['date']=x.ts.dt.date; x['minute']=x.ts.dt.hour*60+x.ts.dt.minute
    x['is_rth']=((x.minute>=570)&(x.minute<960)).astype(int)
    x['is_opening_hour']=((x.minute>=570)&(x.minute<630)).astype(int)
    x['is_power_hour']=((x.minute>=900)&(x.minute<960)).astype(int)
    x['is_overnight']=((x.minute<570)|(x.minute>=1080)).astype(int)
    # Futures session label: 18:00 ET belongs to next trade date.
    x['session_date']=pd.to_datetime(x.ts.dt.date)
    x.loc[x.minute>=1080,'session_date'] += pd.Timedelta(days=1)
    # cumulative session VWAP, causal
    pv=x.close*x.volume
    x['cum_vol']=x.groupby('session_date').volume.cumsum()
    x['cum_pv']=pv.groupby(x.session_date).cumsum()
    x['session_vwap']=x.cum_pv/x.cum_vol.replace(0,np.nan)
    x['dist_session_vwap_atr1m']=np.nan
    # daily/session refs using completed prior session only
    sess=x.groupby('session_date').agg(sess_high=('high','max'),sess_low=('low','min'),sess_close=('close','last'),sess_vol=('volume','sum'))
    prev=sess.shift(1).rename(columns=lambda c:'prev_'+c)
    x=x.merge(prev,left_on='session_date',right_index=True,how='left')
    # Overnight range known progressively; at/after RTH open use final overnight range.
    overnight=x[x.is_overnight.eq(1)].groupby('session_date').agg(on_high=('high','max'),on_low=('low','min'),on_vol=('volume','sum'))
    x=x.merge(overnight,left_on='session_date',right_index=True,how='left')
    # Opening range first 30m, only usable after 10:00.
    or30=x[(x.minute>=570)&(x.minute<600)].groupby('session_date').agg(or30_high=('high','max'),or30_low=('low','min'),or30_vol=('volume','sum'))
    x=x.merge(or30,left_on='session_date',right_index=True,how='left')
    x.loc[x.minute<600,['or30_high','or30_low','or30_vol']]=np.nan
    return x

def resample_ohlcv(m,rule):
    z=m.set_index('ts').resample(rule,label='right',closed='right').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna().reset_index()
    return z

def make_h1(m):
    h=resample_ohlcv(m,'1h'); h=add_adx(h); h['er10']=er(h.close,10); h['er20']=er(h.close,20)
    for n in [10,20,50,200]: h[f'ema{n}']=h.close.ewm(span=n,adjust=False,min_periods=n).mean()
    h['ema_sep_atr']=(h.ema20-h.ema50).abs()/h.atr
    h['extension_atr']=(h.close-h.ema20).abs()/h.atr
    h['mom6_atr']=(h.close-h.close.shift(6))/h.atr; h['mom12_atr']=(h.close-h.close.shift(12))/h.atr
    h['ret1']=h.close.pct_change(); h['rv6']=h.ret1.rolling(6).std(); h['rv24']=h.ret1.rolling(24).std()
    h['vol_ma24']=h.volume.rolling(24).mean(); h['rel_vol24']=h.volume/h.vol_ma24.replace(0,np.nan)
    return h

def make_context(m,h):
    # Hourly session/context features, using last 1m record in each hourly bucket.
    c=m.set_index('ts').resample('1h',label='right',closed='right').agg(
        is_rth=('is_rth','last'),is_opening_hour=('is_opening_hour','last'),is_power_hour=('is_power_hour','last'),is_overnight=('is_overnight','last'),
        session_vwap=('session_vwap','last'),prev_sess_high=('prev_sess_high','last'),prev_sess_low=('prev_sess_low','last'),prev_sess_close=('prev_sess_close','last'),prev_sess_vol=('prev_sess_vol','last'),
        on_high=('on_high','last'),on_low=('on_low','last'),on_vol=('on_vol','last'),or30_high=('or30_high','last'),or30_low=('or30_low','last'),or30_vol=('or30_vol','last')).reset_index()
    d=h.merge(c,on='ts',how='left')
    d['dist_vwap_atr']=(d.close-d.session_vwap)/d.atr
    d['dist_prev_high_atr']=(d.close-d.prev_sess_high)/d.atr; d['dist_prev_low_atr']=(d.close-d.prev_sess_low)/d.atr
    d['dist_on_high_atr']=(d.close-d.on_high)/d.atr; d['dist_on_low_atr']=(d.close-d.on_low)/d.atr
    d['dist_or_high_atr']=(d.close-d.or30_high)/d.atr; d['dist_or_low_atr']=(d.close-d.or30_low)/d.atr
    d['on_range_atr']=(d.on_high-d.on_low)/d.atr; d['or_range_atr']=(d.or30_high-d.or30_low)/d.atr
    d['on_vol_ratio']=d.on_vol/d.prev_sess_vol.replace(0,np.nan); d['or_vol_ratio']=d.or30_vol/d.prev_sess_vol.replace(0,np.nan)
    d['hour']=d.ts.dt.hour; d['dow']=d.ts.dt.dayofweek
    return d

def add_h4_d1(d,m):
    h4=add_adx(resample_ohlcv(m,'4h')); d1=add_adx(resample_ohlcv(m,'1D'))
    for q in [h4,d1]:
        q['er10']=er(q.close,10); q['ema20']=q.close.ewm(span=20,adjust=False,min_periods=20).mean(); q['ema50']=q.close.ewm(span=50,adjust=False,min_periods=50).mean()
    hc=h4[['ts','close','adx','er10','ema20','ema50','volume']].rename(columns={c:'h4_'+c for c in ['close','adx','er10','ema20','ema50','volume']})
    dc=d1[['ts','close','adx','er10','ema20','ema50','volume']].rename(columns={c:'d1_'+c for c in ['close','adx','er10','ema20','ema50','volume']})
    d=pd.merge_asof(d.sort_values('ts'),hc.sort_values('ts'),on='ts',direction='backward')
    d=pd.merge_asof(d.sort_values('ts'),dc.sort_values('ts'),on='ts',direction='backward')
    d['h4_dir']=np.sign(d.h4_ema20-d.h4_ema50); d['d1_dir']=np.sign(d.d1_ema20-d.d1_ema50)
    return d

def add_labels(d,hours=12):
    n=len(d); close=d.close.to_numpy(); high=d.high.to_numpy(); low=d.low.to_numpy(); atr=d.atr.to_numpy()
    sr=np.full(n,np.nan); eff=np.full(n,np.nan); mae_long=np.full(n,np.nan); mae_short=np.full(n,np.nan); max_up=np.full(n,np.nan); max_dn=np.full(n,np.nan)
    for i in range(n-hours):
        if not np.isfinite(atr[i]) or atr[i]<=0: continue
        c0=close[i]; segc=close[i:i+hours+1]; hs=high[i+1:i+hours+1]; ls=low[i+1:i+hours+1]
        sr[i]=(close[i+hours]-c0)/atr[i]; path=np.abs(np.diff(segc)).sum(); eff[i]=abs(close[i+hours]-c0)/path if path else 0
        max_up[i]=(hs.max()-c0)/atr[i]; max_dn[i]=(c0-ls.min())/atr[i]; mae_long[i]=max_dn[i]; mae_short[i]=max_up[i]
    d=d.copy(); d['fwd_atr']=sr; d['fwd_eff']=eff; d['max_up_atr']=max_up; d['max_dn_atr']=max_dn
    d['trend_clean']=((abs(d.fwd_atr)>=0.75)&(d.fwd_eff>=0.35)&(np.where(d.fwd_atr>=0,mae_long,mae_short)<=0.75)).astype(float)
    d.loc[d.fwd_atr.isna(),'trend_clean']=np.nan
    d['trend_dir']=np.sign(d.fwd_atr); d.loc[d.trend_clean.ne(1),'trend_dir']=0
    d['range_persist']=((abs(d.fwd_atr)<0.5)&(d.max_up_atr<1.0)&(d.max_dn_atr<1.0)&(d.fwd_eff<0.25)).astype(float); d.loc[d.fwd_atr.isna(),'range_persist']=np.nan
    return d

def metrics(y,p):
    mask=np.isfinite(y)&np.isfinite(p); y=np.asarray(y)[mask].astype(int); p=np.asarray(p)[mask]
    return {'n':int(len(y)),'base_rate':float(y.mean()),'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p))}

def top_quantile(y,p,q=.9):
    mask=np.isfinite(y)&np.isfinite(p); y=np.asarray(y)[mask].astype(int); p=np.asarray(p)[mask]; cut=np.quantile(p,q); sel=p>=cut
    return {'cut':float(cut),'n':int(sel.sum()),'rate':float(y[sel].mean()),'lift':float(y[sel].mean()/y.mean())}

def fit_eval(d,features,target):
    tr=d[(d.ts>='2023-01-01')&(d.ts<'2024-01-01')].dropna(subset=[target]); va=d[(d.ts>='2024-01-01')&(d.ts<'2025-01-01')].dropna(subset=[target]); te=d[(d.ts>='2025-01-01')&(d.ts<'2025-12-01')].dropna(subset=[target])
    Xtr,Xv,Xte=tr[features],va[features],te[features]; ytr,yv,yte=tr[target].astype(int),va[target].astype(int),te[target].astype(int)
    models={
      'logit':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=.5)),
      'hgb':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=160,max_depth=3,learning_rate=.04,l2_regularization=1.0,random_state=42))}
    cand=[]
    for name,mod in models.items():
        mod.fit(Xtr,ytr); pv=mod.predict_proba(Xv)[:,1]; cand.append((average_precision_score(yv,pv),name,mod,metrics(yv,pv),top_quantile(yv,pv)))
    cand.sort(reverse=True,key=lambda x:x[0]); _,name,mod,valm,valtop=cand[0]
    pt=mod.predict_proba(Xte)[:,1]
    return {'model':name,'validation':valm,'validation_top10':valtop,'test':metrics(yte,pt),'test_top10':top_quantile(yte,pt)},mod,te,pt

def dir_test(te,prob):
    # Only ask direction on top-decile trend-probability observations that actually became clean trends.
    cut=np.nanquantile(prob,.9); q=te[(prob>=cut)&(te.trend_clean==1)].copy();
    rules={
      'always_long':np.ones(len(q)),
      'h1_ema':np.sign(q.ema20-q.ema50),
      'h1_h4_ema':np.where(np.sign(q.ema20-q.ema50)==q.h4_dir,q.h4_dir,0),
      'dmi':np.sign(q.pdi-q.mdi),
      'vwap_side':np.sign(q.close-q.session_vwap)}
    rows=[]
    true=np.sign(q.fwd_atr).to_numpy()
    for name,pred in rules.items():
        pred=np.asarray(pred); m=pred!=0
        rows.append({'rule':name,'n':int(m.sum()),'accuracy':float((pred[m]==true[m]).mean()) if m.any() else np.nan,'coverage':float(m.mean())})
    return rows

def main():
    m=load(); quality={'rows':int(len(m)),'start':str(m.ts.min()),'end':str(m.ts.max()),'duplicates':int(m.ts.duplicated().sum()),'zero_volume':int((m.volume<=0).sum())}
    m=session_features(m); h=make_h1(m); d=make_context(m,h); d=add_h4_d1(d,m); d=add_labels(d,12)
    price=['adx','pdi','mdi','er10','er20','ema_sep_atr','extension_atr','mom6_atr','mom12_atr','rv6','rv24','h4_adx','h4_er10','h4_dir','d1_adx','d1_er10','d1_dir','hour','dow']
    enriched=price+['volume','rel_vol24','is_rth','is_opening_hour','is_power_hour','is_overnight','dist_vwap_atr','dist_prev_high_atr','dist_prev_low_atr','dist_on_high_atr','dist_on_low_atr','dist_or_high_atr','dist_or_low_atr','on_range_atr','or_range_atr','on_vol_ratio','or_vol_ratio']
    results={}
    for target in ['trend_clean','range_persist']:
        results[target]={}
        for label,features in [('price_only',price),('enriched',enriched)]:
            r,mod,te,p=fit_eval(d,features,target); results[target][label]=r
            if target=='trend_clean' and label=='enriched': results['direction_test']=dir_test(te,p)
    # session-specific prevalence / outcome, descriptive OOS 2025 only
    te=d[(d.ts>='2025-01-01')&(d.ts<'2025-12-01')].dropna(subset=['trend_clean'])
    sess=[]
    for nm,mask in [('RTH',te.is_rth==1),('ETH',te.is_rth==0),('OpeningHour',te.is_opening_hour==1),('PowerHour',te.is_power_hour==1),('Overnight',te.is_overnight==1)]:
        q=te[mask]; sess.append({'session':nm,'n':len(q),'trend_rate':float(q.trend_clean.mean()),'range_rate':float(q.range_persist.mean()),'mean_abs_fwd_atr':float(q.fwd_atr.abs().mean())})
    results['session_2025']=sess; results['data_quality']=quality; results['split']={'train':'2023','validation':'2024','test':'2025-01-01 to 2025-11-30','label_horizon':'12h'}
    (OUT/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=True))
    pd.DataFrame(results['direction_test']).to_csv(OUT/'direction_test.csv',index=False); pd.DataFrame(sess).to_csv(OUT/'session_2025.csv',index=False)
    cols=['ts','open','high','low','close','volume','trend_clean','range_persist','fwd_atr']+enriched
    d[cols].to_csv(OUT/'hourly_features_and_labels.csv',index=False)
    print('V8_COMPLETE'); print(json.dumps(results,indent=2,allow_nan=True))

if __name__=='__main__': main()
