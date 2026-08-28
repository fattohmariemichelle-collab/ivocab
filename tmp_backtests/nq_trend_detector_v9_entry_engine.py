from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import nq_trend_detector_v8d_direction as d8
import nq_trend_detector_v8d2_event_audit as a8

OUT = Path('trend_backtest_v9_results')
OUT.mkdir(exist_ok=True)

SCAN_HOURS = 6
HORIZON_HOURS = 12


def resample(m, rule):
    return (m.set_index('ts').resample(rule, label='right', closed='right')
            .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum'),
                 session_vwap=('session_vwap','last'), or30_high=('or30_high','last'), or30_low=('or30_low','last'))
            .dropna(subset=['open','high','low','close']).reset_index())


def enrich_tf(b, prefix):
    x=b.copy()
    for n in [9,20,50]: x[f'{prefix}_ema{n}']=x.close.ewm(span=n,adjust=False,min_periods=n).mean()
    x[f'{prefix}_atr14']=pd.concat([(x.high-x.low),(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x[f'{prefix}_ret1']=x.close.pct_change()
    x[f'{prefix}_vol_ma20']=x.volume.rolling(20,min_periods=20).mean()
    x[f'{prefix}_relvol']=x.volume/x[f'{prefix}_vol_ma20'].replace(0,np.nan)
    x[f'{prefix}_rh4']=x.high.shift(1).rolling(4,min_periods=4).max()
    x[f'{prefix}_rl4']=x.low.shift(1).rolling(4,min_periods=4).min()
    x[f'{prefix}_rh12']=x.high.shift(1).rolling(12,min_periods=12).max()
    x[f'{prefix}_rl12']=x.low.shift(1).rolling(12,min_periods=12).min()
    x[f'{prefix}_ema20_slope']=x[f'{prefix}_ema20'].diff()
    return x


def make_low_tf(m):
    return enrich_tf(resample(m,'5min'),'m5'), enrich_tf(resample(m,'15min'),'m15')


def direction_sign(prob): return 1 if prob >= .5 else -1


def cross_up(a0,a1,level0,level1):
    return np.isfinite([a0,a1,level0,level1]).all() and a0 <= level0 and a1 > level1


def cross_dn(a0,a1,level0,level1):
    return np.isfinite([a0,a1,level0,level1]).all() and a0 >= level0 and a1 < level1


def candidate_events_for_signal(sig, m5, m15):
    t0=sig.ts; t1=t0+pd.Timedelta(hours=SCAN_HOURS); side=int(sig.side)
    w5=m5[(m5.ts>t0)&(m5.ts<=t1)].copy(); w15=m15[(m15.ts>t0)&(m15.ts<=t1)].copy()
    out=[]

    def add(name,row,tf,score_parts=None):
        if row is None: return
        out.append({'setup':name,'entry_ts':row.ts,'entry':float(row.close),'tf':tf,'delay_min':float((row.ts-t0).total_seconds()/60),'side':side,
                    'relvol':float(row.get(f'{tf}_relvol',np.nan)) if isinstance(row,pd.Series) else np.nan,
                    'score_parts':score_parts or {}})

    # M15 directional breakout: close beyond previous 1h range, aligned with EMA20 slope.
    for _,r in w15.iterrows():
        if side>0 and np.isfinite(r.m15_rh4) and r.close>r.m15_rh4 and r.m15_ema20_slope>0:
            add('m15_breakout',r,'m15'); break
        if side<0 and np.isfinite(r.m15_rl4) and r.close<r.m15_rl4 and r.m15_ema20_slope<0:
            add('m15_breakout',r,'m15'); break

    # M15 pullback/reclaim of EMA20 in forecast direction.
    if len(w15)>=2:
        z=w15.reset_index(drop=True)
        for i in range(1,len(z)):
            p,r=z.iloc[i-1],z.iloc[i]
            if side>0 and cross_up(p.close,r.close,p.m15_ema20,r.m15_ema20) and r.close>r.open:
                add('m15_ema_reclaim',r,'m15'); break
            if side<0 and cross_dn(p.close,r.close,p.m15_ema20,r.m15_ema20) and r.close<r.open:
                add('m15_ema_reclaim',r,'m15'); break

    # M5 momentum trigger: break previous hour extreme with EMA20 slope and relative-volume confirmation.
    for _,r in w5.iterrows():
        rv=r.m5_relvol
        rvok=np.isfinite(rv) and rv>=1.0
        if side>0 and np.isfinite(r.m5_rh12) and r.close>r.m5_rh12 and r.m5_ema20_slope>0 and rvok:
            add('m5_momentum_relvol',r,'m5'); break
        if side<0 and np.isfinite(r.m5_rl12) and r.close<r.m5_rl12 and r.m5_ema20_slope<0 and rvok:
            add('m5_momentum_relvol',r,'m5'); break

    # Causal session VWAP reclaim.
    if len(w5)>=2:
        z=w5.reset_index(drop=True)
        for i in range(1,len(z)):
            p,r=z.iloc[i-1],z.iloc[i]
            if side>0 and cross_up(p.close,r.close,p.session_vwap,r.session_vwap) and r.m5_ema20_slope>0:
                add('m5_vwap_reclaim',r,'m5'); break
            if side<0 and cross_dn(p.close,r.close,p.session_vwap,r.session_vwap) and r.m5_ema20_slope<0:
                add('m5_vwap_reclaim',r,'m5'); break

    # Opening-range rejection/re-entry: especially relevant after V8d's OR-location finding.
    if len(w5)>=2:
        z=w5.reset_index(drop=True)
        for i in range(1,len(z)):
            p,r=z.iloc[i-1],z.iloc[i]
            if not np.isfinite([p.or30_high,p.or30_low,r.or30_high,r.or30_low]).all(): continue
            if side<0 and p.close>p.or30_high and r.close<r.or30_high:
                add('m5_or_rejection',r,'m5'); break
            if side>0 and p.close<p.or30_low and r.close>r.or30_low:
                add('m5_or_rejection',r,'m5'); break

    return out


def trade_outcome(ev,m):
    t=ev['entry_ts']; side=ev['side']; entry=ev['entry']
    f=m[(m.ts>t)&(m.ts<=t+pd.Timedelta(hours=HORIZON_HOURS))]
    if len(f)<30: return None
    if side>0:
        mfe=(f.high.max()-entry); mae=(entry-f.low.min()); end=f.close.iloc[-1]-entry
    else:
        mfe=(entry-f.low.min()); mae=(f.high.max()-entry); end=entry-f.close.iloc[-1]
    # ATR at signal is H1 ATR and is stored on event for scale comparability.
    atr=ev['signal_atr']
    return {'mfe_atr':float(mfe/atr),'mae_atr':float(mae/atr),'end_atr':float(end/atr),
            'hit_075_before_adverse_075':barrier_first(f,entry,side,.75*atr,.75*atr),
            'hit_100_before_adverse_075':barrier_first(f,entry,side,1.0*atr,.75*atr)}


def barrier_first(f,entry,side,target,stop):
    for _,r in f.iterrows():
        if side>0:
            hit_t=r.high>=entry+target; hit_s=r.low<=entry-stop
        else:
            hit_t=r.low<=entry-target; hit_s=r.high>=entry+stop
        if hit_t and hit_s: return np.nan  # intrabar order unknown
        if hit_t: return 1.0
        if hit_s: return 0.0
    return 0.0


def build_signals(d,gate,gi,model,feats,margin,start,end):
    allg,_=d8.gated(d,gate,gi,start,end)
    allg=a8.nonoverlap(allg,12).copy()
    p=model.predict_proba(allg[feats])[:,1]
    keep=np.abs(p-.5)>=margin
    s=allg[keep].copy(); s['dir_prob']=p[keep]; s['side']=np.where(s.dir_prob>=.5,1,-1)
    return s


def evaluate_period(signals,m,m5,m15):
    rows=[]
    for _,sig in signals.iterrows():
        evs=candidate_events_for_signal(sig,m5,m15)
        seen=set()
        for ev in evs:
            if ev['setup'] in seen: continue
            seen.add(ev['setup']); ev['signal_ts']=sig.ts; ev['signal_atr']=float(sig.atr); ev['dir_prob']=float(sig.dir_prob); ev['future_clean_trend']=int(sig.trend_clean==1); ev['future_dir_correct']=int(sig.trend_clean==1 and np.sign(sig.fwd_atr)==sig.side)
            o=trade_outcome(ev,m)
            if o: rows.append({**{k:v for k,v in ev.items() if k!='score_parts'},**o})
    return pd.DataFrame(rows)


def setup_summary(df):
    rows=[]
    for name,g in df.groupby('setup'):
        h075=g.hit_075_before_adverse_075.dropna(); h100=g.hit_100_before_adverse_075.dropna()
        rows.append({'setup':name,'n':len(g),'signal_coverage':g.signal_ts.nunique()/df.signal_ts.nunique() if len(df) else 0,
                     'median_delay_min':float(g.delay_min.median()),'clean_trend_rate':float(g.future_clean_trend.mean()),'correct_clean_direction_rate':float(g.future_dir_correct.mean()),
                     'median_mfe_atr':float(g.mfe_atr.median()),'median_mae_atr':float(g.mae_atr.median()),'median_end_atr':float(g.end_atr.median()),
                     'target075_stop075':float(h075.mean()) if len(h075) else np.nan,'target100_stop075':float(h100.mean()) if len(h100) else np.nan})
    return sorted(rows,key=lambda r:(r['target075_stop075'],r['correct_clean_direction_rate'],r['n']),reverse=True)


def select_setup(val_summary):
    # Validation-only choice. Require >=15 entries and prefer barrier success, then directional confirmation.
    eligible=[r for r in val_summary if r['n']>=15 and np.isfinite(r['target075_stop075'])]
    if not eligible: return val_summary[0]['setup'] if val_summary else None
    eligible.sort(key=lambda r:(r['target075_stop075'],r['correct_clean_direction_rate'],r['median_mfe_atr'],-r['median_mae_atr']),reverse=True)
    return eligible[0]['setup']


def baseline_signal_entry(signals,m):
    rows=[]
    for _,s in signals.iterrows():
        f=m[m.ts>s.ts]
        if f.empty: continue
        r=f.iloc[0]; ev={'entry_ts':r.ts,'entry':float(r.open),'side':int(s.side),'signal_atr':float(s.atr)}; o=trade_outcome(ev,m)
        if o: rows.append({'signal_ts':s.ts,'setup':'immediate_next_1m','delay_min':1.0,'future_clean_trend':int(s.trend_clean==1),'future_dir_correct':int(s.trend_clean==1 and np.sign(s.fwd_atr)==s.side),**o})
    return pd.DataFrame(rows)


def main():
    d,quality=d8.build(); m=d8.v8.load(); m=d8.v8.session_features(m); m5,m15=make_low_tf(m)
    gi,gate=d8.fit_gate(d); feats=d8.sets()['price_location_volume']; model_name,model,_,_,margin=d8.choose(d,gate,gi,feats)
    periods={}; frames={}; bases={}
    for label,start,end in [('validation_2024','2024-01-01','2025-01-01'),('evaluation_2025','2025-01-01','2025-12-01')]:
        sig=build_signals(d,gate,gi,model,feats,margin,start,end); df=evaluate_period(sig,m,m5,m15); base=baseline_signal_entry(sig,m)
        frames[label]=df; bases[label]=base; periods[label]={'signals':len(sig),'entries':len(df),'setup_summary':setup_summary(df),'immediate_entry_summary':setup_summary(base) if len(base) else []}
        df.to_csv(OUT/f'{label}_entries.csv',index=False)
    selected=select_setup(periods['validation_2024']['setup_summary'])
    val_sel=frames['validation_2024'][frames['validation_2024'].setup==selected] if selected else pd.DataFrame(); te_sel=frames['evaluation_2025'][frames['evaluation_2025'].setup==selected] if selected else pd.DataFrame()
    out={'method':{'purpose':'V9 entry timing after V8c environment + V8d direction','signal_scan_window_hours':SCAN_HOURS,'trade_horizon_hours':HORIZON_HOURS,'candidate_setups':['m15_breakout','m15_ema_reclaim','m5_momentum_relvol','m5_vwap_reclaim','m5_or_rejection'],'selection':'single setup selected on 2024 only; 2025 evaluated unchanged','anti_overlap':'upstream H1 signals separated by >=12h','barrier_note':'same-bar target+stop is treated unknown and excluded from barrier-rate denominator','caution':'2025 is chronological evaluation but not globally pristine because earlier exploratory 2025 research was viewed'},'data_quality':quality,'direction_model':model_name,'direction_margin':margin,'periods':periods,'selected_setup_2024':selected,
         'selected_validation':setup_summary(val_sel)[0] if len(val_sel) else None,'selected_evaluation_2025':setup_summary(te_sel)[0] if len(te_sel) else None}
    (OUT/'v9_entry_results.json').write_text(json.dumps(out,indent=2,allow_nan=True)); print('V9_COMPLETE'); print(json.dumps(out,indent=2,allow_nan=True))

if __name__=='__main__': main()
