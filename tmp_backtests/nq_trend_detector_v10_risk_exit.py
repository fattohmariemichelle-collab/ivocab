from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import nq_trend_detector_v8d_direction as d8
import nq_trend_detector_v9_entry_engine as v9

OUT = Path('trend_backtest_v10_results')
OUT.mkdir(exist_ok=True)
TICK_SIZE = 0.25
PRIMARY_COST_TICKS = 4.0
RNG = np.random.default_rng(20260828)


def profiles():
    out=[]
    for stop in [0.50,0.75,1.00,1.25]:
        for target in [1.00,1.50,2.00,2.50]:
            for hours in [6,12]:
                out.append({'family':'fixed','stop_type':'atr','stop_atr':stop,'target_r':target,'hours':hours,'be_trigger_r':None,'trail_trigger_r':None,'trail_distance_r':None,'partial_r':None,'partial_fraction':None,'name':f'fixed_s{stop:.2f}_t{target:.2f}_{hours}h'})
    for stop in [0.75,1.00]:
        for target in [1.50,2.00,2.50]:
            for hours in [6,12]:
                for be in [0.75,1.00]:
                    out.append({'family':'breakeven','stop_type':'atr','stop_atr':stop,'target_r':target,'hours':hours,'be_trigger_r':be,'trail_trigger_r':None,'trail_distance_r':None,'partial_r':None,'partial_fraction':None,'name':f'be_s{stop:.2f}_t{target:.2f}_be{be:.2f}_{hours}h'})
    for stop in [0.75,1.00]:
        for hours in [6,12]:
            for dist in [0.75,1.00,1.50]:
                out.append({'family':'trailing','stop_type':'atr','stop_atr':stop,'target_r':None,'hours':hours,'be_trigger_r':None,'trail_trigger_r':1.00,'trail_distance_r':dist,'partial_r':None,'partial_fraction':None,'name':f'trail_s{stop:.2f}_trig1.00_dist{dist:.2f}_{hours}h'})
    for stop in [0.75,1.00]:
        for final_target in [2.00,3.00]:
            for hours in [6,12]:
                out.append({'family':'partial','stop_type':'atr','stop_atr':stop,'target_r':final_target,'hours':hours,'be_trigger_r':None,'trail_trigger_r':None,'trail_distance_r':None,'partial_r':1.00,'partial_fraction':0.50,'name':f'partial_s{stop:.2f}_p1.00x50_final{final_target:.2f}_{hours}h'})
    for target in [1.00,1.50,2.00,2.50]:
        for hours in [6,12]:
            out.append({'family':'structure_fixed','stop_type':'structure','stop_atr':None,'target_r':target,'hours':hours,'be_trigger_r':None,'trail_trigger_r':None,'trail_distance_r':None,'partial_r':None,'partial_fraction':None,'name':f'structure_t{target:.2f}_{hours}h'})
    for target in [1.50,2.00,2.50]:
        for hours in [6,12]:
            out.append({'family':'structure_be','stop_type':'structure','stop_atr':None,'target_r':target,'hours':hours,'be_trigger_r':1.00,'trail_trigger_r':None,'trail_distance_r':None,'partial_r':None,'partial_fraction':None,'name':f'structure_t{target:.2f}_be1.00_{hours}h'})
    return out


def prepare_pipeline():
    d,quality=d8.build()
    m=d8.v8.load(); m=d8.v8.session_features(m)
    m5,m15=v9.make_low_tf(m)
    gi,gate=d8.fit_gate(d)
    feats=d8.sets()['price_location_volume']
    model_name,model,_,_,margin=d8.choose(d,gate,gi,feats)
    return d,m,m5,m15,quality,gi,gate,feats,model_name,model,margin


def build_events(d,m,m5,m15,gi,gate,feats,model,margin,start,end):
    signals=v9.build_signals(d,gate,gi,model,feats,margin,start,end)
    rows=[]
    for _,sig in signals.iterrows():
        candidates=v9.candidate_events_for_signal(sig,m5,m15)
        ev=next((x for x in candidates if x['setup']=='m15_breakout'),None)
        if ev is None: continue
        future=m[m.ts>ev['entry_ts']]
        if future.empty: continue
        entry_bar=future.iloc[0]
        gap_min=(entry_bar.ts-ev['entry_ts']).total_seconds()/60
        if gap_min>5: continue
        rr=m15[m15.ts==ev['entry_ts']]
        if rr.empty: continue
        r15=rr.iloc[0]
        side=int(sig.side); atr=float(sig.atr); entry=float(entry_bar.open)
        if not np.isfinite(atr) or atr<=0: continue
        if side>0 and np.isfinite(r15.m15_rl4):
            structure_stop=float(r15.m15_rl4-0.10*atr)
        elif side<0 and np.isfinite(r15.m15_rh4):
            structure_stop=float(r15.m15_rh4+0.10*atr)
        else:
            structure_stop=np.nan
        bars=m[(m.ts>=entry_bar.ts)&(m.ts<=entry_bar.ts+pd.Timedelta(hours=12))]
        if len(bars)<30: continue
        rows.append({'signal_ts':sig.ts,'entry_trigger_ts':ev['entry_ts'],'entry_ts':entry_bar.ts,'entry':entry,'side':side,'signal_atr':atr,'structure_stop':structure_stop,'delay_min':float((entry_bar.ts-sig.ts).total_seconds()/60),'dir_prob':float(sig.dir_prob),'future_clean_trend':int(sig.trend_clean==1),'future_dir_correct':int(sig.trend_clean==1 and np.sign(sig.fwd_atr)==side),'bars':bars[['ts','open','high','low','close']].copy()})
    return rows,len(signals)


def initial_risk(event,profile):
    if profile['stop_type']=='atr':
        risk=float(profile['stop_atr']*event['signal_atr'])
        stop=event['entry']-event['side']*risk
        return risk,stop
    stop=event['structure_stop']
    if not np.isfinite(stop): return None,None
    risk=event['side']*(event['entry']-stop)
    ratio=risk/event['signal_atr']
    if risk<=0 or ratio<0.35 or ratio>1.50: return None,None
    return float(risk),float(stop)


def stop_fill(side,stop,bar_open):
    return min(stop,bar_open) if side>0 else max(stop,bar_open)


def simulate_one(event,profile,cost_ticks=PRIMARY_COST_TICKS):
    risk,active_stop=initial_risk(event,profile)
    if risk is None: return None
    entry=event['entry']; side=event['side']; qty=1.0; realized_r=0.0
    partial_done=False; favorable_extreme=entry; exit_ts=None; exit_reason=None
    target_price=None if profile['target_r'] is None else entry+side*profile['target_r']*risk
    cutoff=event['entry_ts']+pd.Timedelta(hours=profile['hours'])
    bars=event['bars'][event['bars'].ts<=cutoff]
    if bars.empty: return None
    last_close=float(bars.iloc[-1].close)
    for bar in bars.itertuples(index=False):
        hit_stop=(bar.low<=active_stop) if side>0 else (bar.high>=active_stop)
        hit_target=False if target_price is None else ((bar.high>=target_price) if side>0 else (bar.low<=target_price))
        partial_price=None
        hit_partial=False
        if profile['family']=='partial' and not partial_done:
            partial_price=entry+side*profile['partial_r']*risk
            hit_partial=(bar.high>=partial_price) if side>0 else (bar.low<=partial_price)
        # Conservative intrabar rule: an active stop wins every same-bar collision.
        if hit_stop:
            fill=stop_fill(side,active_stop,float(bar.open))
            realized_r+=qty*(side*(fill-entry)/risk)
            qty=0.0; exit_ts=bar.ts; exit_reason='stop'; break
        if profile['family']=='partial' and hit_partial:
            frac=min(profile['partial_fraction'],qty)
            realized_r+=frac*profile['partial_r']; qty-=frac; partial_done=True
            # If final target is also inside this bar, conservatively defer it to a later bar.
        elif hit_target:
            realized_r+=qty*profile['target_r']; qty=0.0; exit_ts=bar.ts; exit_reason='target'; break
        if side>0: favorable_extreme=max(favorable_extreme,float(bar.high))
        else: favorable_extreme=min(favorable_extreme,float(bar.low))
        favorable_r=side*(favorable_extreme-entry)/risk
        # Stop changes take effect only from the next 1-minute bar.
        if profile['family']=='partial' and partial_done:
            active_stop=max(active_stop,entry) if side>0 else min(active_stop,entry)
        if profile['be_trigger_r'] is not None and favorable_r>=profile['be_trigger_r']:
            active_stop=max(active_stop,entry) if side>0 else min(active_stop,entry)
        if profile['trail_trigger_r'] is not None and favorable_r>=profile['trail_trigger_r']:
            candidate=favorable_extreme-side*profile['trail_distance_r']*risk
            active_stop=max(active_stop,candidate) if side>0 else min(active_stop,candidate)
    if qty>0:
        realized_r+=qty*(side*(last_close-entry)/risk); qty=0.0
        exit_ts=bars.iloc[-1].ts; exit_reason='time'
    cost_r=(cost_ticks*TICK_SIZE)/risk
    net_r=realized_r-cost_r
    return {'signal_ts':event['signal_ts'],'entry_ts':event['entry_ts'],'exit_ts':exit_ts,'side':side,'entry':entry,'risk_points':risk,'risk_atr':risk/event['signal_atr'],'delay_min':event['delay_min'],'gross_r':realized_r,'net_r':net_r,'cost_r':cost_r,'duration_min':float((exit_ts-event['entry_ts']).total_seconds()/60),'exit_reason':exit_reason,'future_clean_trend':event['future_clean_trend'],'future_dir_correct':event['future_dir_correct']}


def enforce_one_position(rows):
    rows=sorted(rows,key=lambda r:r['entry_ts']); accepted=[]; last_exit=None; skipped=0
    for r in rows:
        if last_exit is not None and r['entry_ts']<last_exit:
            skipped+=1; continue
        accepted.append(r); last_exit=r['exit_ts']
    return accepted,skipped


def longest_loss_streak(x):
    best=cur=0
    for v in x:
        if v<0: cur+=1; best=max(best,cur)
        else: cur=0
    return best


def max_drawdown(x):
    eq=np.cumsum(x); peak=np.maximum.accumulate(np.r_[0.0,eq]); dd=peak[1:]-eq
    return float(dd.max()) if len(dd) else 0.0


def bootstrap_mean(x,reps=1500):
    x=np.asarray(x,float)
    if len(x)<10: return [np.nan,np.nan,np.nan]
    samples=RNG.choice(x,size=(reps,len(x)),replace=True).mean(axis=1)
    return [float(np.quantile(samples,.05)),float(np.quantile(samples,.50)),float(np.quantile(samples,.95))]


def summarize(rows,signals_total,profile):
    if not rows: return {'profile':profile['name'],'family':profile['family'],'n':0}
    g=pd.DataFrame(rows).sort_values('entry_ts'); x=g.net_r.to_numpy(float)
    gp=x[x>0].sum(); gl=-x[x<0].sum(); mid=g.entry_ts.min()+(g.entry_ts.max()-g.entry_ts.min())/2
    first=g[g.entry_ts<=mid].net_r; second=g[g.entry_ts>mid].net_r
    longs=g[g.side==1].net_r; shorts=g[g.side==-1].net_r
    ci=bootstrap_mean(x)
    out={'profile':profile['name'],'family':profile['family'],'parameters':json.dumps(profile,sort_keys=True),'n':len(g),'signal_coverage':len(g)/signals_total if signals_total else 0.0,'win_rate':float((x>0).mean()),'loss_rate':float((x<0).mean()),'avg_net_r':float(x.mean()),'median_net_r':float(np.median(x)),'std_net_r':float(x.std(ddof=1)) if len(x)>1 else np.nan,'profit_factor':float(gp/gl) if gl>0 else np.inf,'max_drawdown_r':max_drawdown(x),'longest_loss_streak':longest_loss_streak(x),'avg_duration_min':float(g.duration_min.mean()),'median_duration_min':float(g.duration_min.median()),'first_half_avg_r':float(first.mean()) if len(first) else np.nan,'second_half_avg_r':float(second.mean()) if len(second) else np.nan,'long_n':len(longs),'long_avg_r':float(longs.mean()) if len(longs) else np.nan,'short_n':len(shorts),'short_avg_r':float(shorts.mean()) if len(shorts) else np.nan,'bootstrap_mean_r_p05':ci[0],'bootstrap_mean_r_median':ci[1],'bootstrap_mean_r_p95':ci[2]}
    for ticks in [0,2,4,6]:
        vals=g.gross_r-(ticks*TICK_SIZE/g.risk_points)
        out[f'avg_r_cost_{ticks}ticks']=float(vals.mean())
    for risk_pct in [0.25,0.50,1.00]: out[f'historical_max_dd_at_{risk_pct:.2f}pct_risk_pct']=out['max_drawdown_r']*risk_pct
    return out


def evaluate_profiles(events,signals_total,profile_list):
    summaries=[]; trades_by={}; skipped_by={}
    for p in profile_list:
        raw=[]
        for ev in events:
            r=simulate_one(ev,p)
            if r is not None: raw.append(r)
        accepted,skipped=enforce_one_position(raw)
        trades_by[p['name']]=accepted; skipped_by[p['name']]=skipped
        s=summarize(accepted,signals_total,p); s['overlap_trades_skipped']=skipped; summaries.append(s)
    return pd.DataFrame(summaries),trades_by


def select_profile(validation):
    x=validation.copy()
    eligible=x[(x.n>=50)&(x.avg_net_r>0)&(x.first_half_avg_r>0)&(x.second_half_avg_r>0)&(x.profit_factor>=1.10)]
    rule='n>=50, both chronological halves positive, PF>=1.10; maximize bootstrap 5th-percentile mean R'
    if eligible.empty:
        eligible=x[(x.n>=40)&(x.avg_net_r>0)]
        rule='fallback n>=40 and positive mean; maximize bootstrap 5th-percentile mean R'
    if eligible.empty:
        eligible=x[x.n>0]
        rule='fallback highest bootstrap 5th-percentile mean R'
    eligible=eligible.sort_values(['bootstrap_mean_r_p05','avg_net_r','profit_factor','max_drawdown_r'],ascending=[False,False,False,True])
    return str(eligible.iloc[0].profile),rule


def main():
    d,m,m5,m15,quality,gi,gate,feats,model_name,model,margin=prepare_pipeline()
    period_specs=[('validation_2024','2024-01-01','2025-01-01'),('evaluation_2025','2025-01-01','2025-12-01'),('micro_holdout_dec2025','2025-12-01','2025-12-11')]
    events={}; signal_counts={}
    for label,start,end in period_specs:
        events[label],signal_counts[label]=build_events(d,m,m5,m15,gi,gate,feats,model,margin,start,end)
    ps=profiles(); grids={}; trade_maps={}
    for label,_,_ in period_specs:
        grid,trades=evaluate_profiles(events[label],signal_counts[label],ps); grids[label]=grid; trade_maps[label]=trades; grid.to_csv(OUT/f'{label}_profile_grid.csv',index=False)
    selected,selection_rule=select_profile(grids['validation_2024'])
    selected_profile=next(p for p in ps if p['name']==selected)
    selected_results={}
    for label,_,_ in period_specs:
        row=grids[label][grids[label].profile==selected]
        selected_results[label]=row.iloc[0].replace({np.nan:None}).to_dict() if len(row) else None
        pd.DataFrame(trade_maps[label].get(selected,[])).to_csv(OUT/f'{label}_selected_trades.csv',index=False)
    top_validation=grids['validation_2024'].sort_values(['bootstrap_mean_r_p05','avg_net_r'],ascending=False).head(15).replace({np.nan:None}).to_dict(orient='records')
    baseline_name='fixed_s0.75_t1.00_12h'
    baseline={}
    for label,_,_ in period_specs:
        r=grids[label][grids[label].profile==baseline_name]
        baseline[label]=r.iloc[0].replace({np.nan:None}).to_dict() if len(r) else None
    out={'method':{'purpose':'V10 risk and exit optimization after V8c + V8d + V9 M15 breakout','entry_fill':'first 1-minute bar open after confirmed M15 breakout','primary_cost_assumption':'4 ticks all-in per round trip, applied as points/risk; sensitivity at 0/2/4/6 ticks included','same_bar_rule':'active stop wins every target/stop collision; stop changes become active next bar','position_rule':'one position at a time; overlapping later entries skipped','selection':'profile selected on 2024 only; 2025 unchanged chronological evaluation','candidate_families':['fixed ATR stop/target','breakeven','trailing','50% partial at 1R','M15 structure stop'],'risk_units':'all results normalized to initial R','caution':'public continuous NQ dataset; rollover handling, fees and true broker fills require independent replication'},'data_quality':quality,'direction_model':model_name,'direction_margin':margin,'signals':signal_counts,'entry_events':{k:len(v) for k,v in events.items()},'profiles_tested':len(ps),'selection_rule':selection_rule,'selected_profile':selected_profile,'selected_results':selected_results,'baseline_v9_style':baseline,'top_validation_profiles':top_validation}
    (OUT/'v10_risk_exit_results.json').write_text(json.dumps(out,indent=2,allow_nan=True),encoding='utf-8')
    print('V10_COMPLETE'); print(json.dumps({'profiles_tested':len(ps),'selection_rule':selection_rule,'selected_profile':selected_profile,'selected_results':selected_results,'baseline':baseline,'top_validation_profiles':top_validation[:5]},indent=2,allow_nan=True))

if __name__=='__main__': main()
