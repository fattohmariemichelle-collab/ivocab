from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nq_v13_fast_runner as fast

v13 = fast.v13
RNG = np.random.default_rng(20260829)


@dataclass(frozen=True)
class Variant:
    name: str
    range_min: float = 55.0
    range_max: float = 110.0
    gap_min: float = 20.0
    buffer: float = 4.0
    stop: float = 27.0
    rr: float = 3.0
    deadline: int = 630
    relvol: float = 1.0
    score_min: int = 3
    day_mode: str = "tue_thu"


def variants() -> list[Variant]:
    b = Variant("baseline")
    return [
        b,
        replace(b, name="range_min_45", range_min=45),
        replace(b, name="range_min_65", range_min=65),
        replace(b, name="range_max_100", range_max=100),
        replace(b, name="range_max_120", range_max=120),
        replace(b, name="gap_10", gap_min=10),
        replace(b, name="gap_30", gap_min=30),
        replace(b, name="buffer_2", buffer=2),
        replace(b, name="buffer_6", buffer=6),
        replace(b, name="stop_24", stop=24),
        replace(b, name="stop_30", stop=30),
        replace(b, name="rr_2_5", rr=2.5),
        replace(b, name="rr_3_5", rr=3.5),
        replace(b, name="deadline_1015", deadline=615),
        replace(b, name="deadline_1045", deadline=645),
        replace(b, name="relvol_0_8", relvol=.8),
        replace(b, name="relvol_1_2", relvol=1.2),
        replace(b, name="score_2", score_min=2),
        replace(b, name="score_4", score_min=4),
        replace(b, name="allow_friday", day_mode="no_mon"),
        replace(b, name="allow_monday", day_mode="no_fri"),
    ]


def day_ok(dow: int, mode: str) -> bool:
    if mode == "tue_thu": return dow in {1,2,3}
    if mode == "no_mon": return dow != 0
    if mode == "no_fri": return dow != 4
    return True


def signal(row: pd.Series, path: pd.DataFrame, v: Variant):
    if not day_ok(int(row.dow), v.day_mode): return None
    if not (np.isfinite(row.or15_range_points) and v.range_min <= row.or15_range_points <= v.range_max): return None
    g = path[(path.minute >= 586) & (path.minute <= v.deadline)]
    for bar in g.itertuples(index=False):
        med = getattr(bar, "volume_med20_prior", np.nan)
        relvol = float(bar.volume / med) if np.isfinite(med) and med > 0 else 1.0
        if relvol < v.relvol: continue
        side = 0
        if float(bar.close) > float(row.or15_high) + v.buffer: side = 1
        elif float(bar.close) < float(row.or15_low) - v.buffer: side = -1
        if side == 0: continue
        if side * float(row.gap15_points) <= v.gap_min: continue
        score = int(row.confidence_long if side == 1 else row.confidence_short)
        if score < v.score_min: continue
        return pd.Timestamp(bar.ts), side
    return None


def simulate(path: pd.DataFrame, row: pd.Series, sig_ts: pd.Timestamp, side: int, v: Variant, cost_ticks: float):
    future = path[path.ts > sig_ts]
    if future.empty: return None
    entry_bar = future.iloc[0]
    if (pd.Timestamp(entry_bar.ts)-sig_ts).total_seconds() > 300: return None
    entry=float(entry_bar.open); stop=entry-side*v.stop; target=entry+side*v.stop*v.rr
    exit_price=float(future.iloc[-1].close); reason="time"; exit_ts=pd.Timestamp(future.iloc[-1].ts)
    for bar in future.itertuples(index=False):
        if int(bar.minute) >= 955:
            exit_price=float(bar.close); exit_ts=pd.Timestamp(bar.ts); reason="time"; break
        hs=float(bar.low)<=stop if side==1 else float(bar.high)>=stop
        ht=float(bar.high)>=target if side==1 else float(bar.low)<=target
        if hs: exit_price=stop; exit_ts=pd.Timestamp(bar.ts); reason="stop"; break
        if ht: exit_price=target; exit_ts=pd.Timestamp(bar.ts); reason="target"; break
    net_r=(side*(exit_price-entry)-cost_ticks*.25)/v.stop
    return {"date":str(pd.Timestamp(row.session_date).date()),"variant":v.name,"side":side,"entry_ts":pd.Timestamp(entry_bar.ts),"exit_ts":exit_ts,"net_r":net_r,"reason":reason}


def summarize(rows: list[dict[str,Any]]):
    if not rows: return {"n":0}
    r=np.array([x["net_r"] for x in rows],float); gains=r[r>0].sum(); losses=-r[r<0].sum(); eq=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq
    return {"n":len(r),"win_rate":float((r>0).mean()),"expectancy_r":float(r.mean()),"profit_factor":float(gains/losses) if losses>0 else None,"max_drawdown_r":float(dd.max()),"total_r":float(r.sum()),"long_n":sum(x["side"]==1 for x in rows),"short_n":sum(x["side"]==-1 for x in rows)}


def evaluate(obs: pd.DataFrame, paths: dict[pd.Timestamp,pd.DataFrame], vs: list[Variant], cost: float):
    metrics=[]; trades=[]
    for v in vs:
        rr=[]
        for row in obs.itertuples(index=False):
            s=pd.Series(row._asdict()); path=paths.get(pd.Timestamp(s.session_date))
            if path is None: continue
            ev=signal(s,path,v)
            if ev is None: continue
            x=simulate(path,s,ev[0],ev[1],v,cost)
            if x: rr.append(x); trades.append(x)
        metrics.append({"variant":v.name,"cost_ticks":cost,**summarize(rr)})
    return pd.DataFrame(metrics),pd.DataFrame(trades)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary",type=Path,required=True); ap.add_argument("--ev2024",type=Path,required=True); ap.add_argument("--ev2025",type=Path,required=True); ap.add_argument("--ev2026",type=Path,required=True); ap.add_argument("--topstep-nq",type=Path,required=True); ap.add_argument("--topstep-mnq",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("v13c_results"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); vs=variants(); sources={}; audits={}
    minute=v13.base.load_ohlcv(a.primary,"primary_et"); o,p,au=v13.add_day_context(minute,"primary"); audits["primary"]=au
    for y in [2023,2024,2025]: sources[f"primary_{y}"]=(o[pd.to_datetime(o.session_date).dt.year==y],{k:v for k,v in p.items() if pd.Timestamp(k).year==y})
    for y,path in [(2024,a.ev2024),(2025,a.ev2025),(2026,a.ev2026)]:
        m,la=v13.load_ev(path); oo,pp,aa=v13.add_day_context(m,f"ev_{y}"); audits[f"ev_{y}"]={"load":la,"features":aa}
        if y==2026:
            sources["ev_2026_jan_apr"]=(oo[pd.to_datetime(oo.session_date)<pd.Timestamp("2026-04-16")],{k:v for k,v in pp.items() if k<pd.Timestamp("2026-04-16")})
            sources["ev_2026_apr_jul"]=(oo[pd.to_datetime(oo.session_date)>=pd.Timestamp("2026-04-16")],{k:v for k,v in pp.items() if k>=pd.Timestamp("2026-04-16")})
        else: sources[f"ev_{y}"]=(oo,pp)
    for label,path in [("topstep_nq",a.topstep_nq),("topstep_mnq",a.topstep_mnq)]:
        m=v13.load_topstep(path); oo,pp,aa=v13.add_day_context(m,label); audits[label]=aa; sources[label]=(oo,pp)
    frames=[]; trade_frames=[]
    for src,(oo,pp) in sources.items():
        for cost in [4.,12.]:
            mm,tt=evaluate(oo,pp,vs,cost); mm["source"]=src; tt["source"]=src; tt["cost_ticks"]=cost; frames.append(mm); trade_frames.append(tt)
    metrics=pd.concat(frames,ignore_index=True); trades=pd.concat(trade_frames,ignore_index=True); metrics.to_csv(a.out/"neighborhood_metrics.csv",index=False); trades.to_csv(a.out/"neighborhood_trades.csv",index=False)
    base4=metrics[(metrics.variant=="baseline")&(metrics.cost_ticks==4)]; base12=metrics[(metrics.variant=="baseline")&(metrics.cost_ticks==12)]
    rows=[]
    for name,g in metrics[metrics.cost_ticks==4].groupby("variant"):
        large=g[g.n>=12]; stress=metrics[(metrics.variant==name)&(metrics.cost_ticks==12)&(metrics.n>=12)]
        rows.append({"variant":name,"large_sources":len(large),"positive_large":int((large.expectancy_r>0).sum()),"min_expectancy":float(large.expectancy_r.min()) if len(large) else None,"weighted_expectancy":float((large.expectancy_r*large.n).sum()/large.n.sum()) if large.n.sum() else None,"positive_stress":int((stress.expectancy_r>0).sum()),"stress_sources":len(stress)})
    ranking=pd.DataFrame(rows).sort_values(["positive_large","min_expectancy","weighted_expectancy"],ascending=[False,False,False]); ranking.to_csv(a.out/"neighborhood_ranking.csv",index=False)
    result={"version":"V13c-ORB-neighborhood","baseline":base4.to_dict("records"),"baseline_12tick":base12.to_dict("records"),"ranking":ranking.to_dict("records"),"audits":audits}
    (a.out/"summary.json").write_text(json.dumps(v13.json_safe(result),indent=2),encoding="utf-8")
    print("V13C_COMPLETE"); print(ranking.to_string(index=False)); print("\nBASELINE\n",base4.to_string(index=False)); print("\nBASELINE 12 TICKS\n",base12.to_string(index=False))

if __name__=="__main__": main()
