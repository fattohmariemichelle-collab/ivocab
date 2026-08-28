from __future__ import annotations
import numpy as np
import pandas as pd
import nq_trend_detector_v11_robustness as v11


def build_events_m15_only(signals, m, m15):
    rows=[]
    for _,sig in signals.iterrows():
        t0=sig.ts; t1=t0+pd.Timedelta(hours=6); side=int(sig.side)
        w=m15[(m15.ts>t0)&(m15.ts<=t1)]
        trigger=None
        for _,r in w.iterrows():
            if side>0 and np.isfinite(r.m15_rh4) and r.close>r.m15_rh4 and r.m15_ema20_slope>0:
                trigger=r; break
            if side<0 and np.isfinite(r.m15_rl4) and r.close<r.m15_rl4 and r.m15_ema20_slope<0:
                trigger=r; break
        if trigger is None:
            continue
        future=m[m.ts>trigger.ts]
        if future.empty:
            continue
        entry_bar=future.iloc[0]
        if (entry_bar.ts-trigger.ts).total_seconds()/60>5:
            continue
        bars=m[(m.ts>=entry_bar.ts)&(m.ts<=entry_bar.ts+pd.Timedelta(hours=12))]
        if len(bars)<30 or not np.isfinite(sig.atr) or sig.atr<=0:
            continue
        rows.append({
            'signal_ts':sig.ts,'entry_trigger_ts':trigger.ts,'entry_ts':entry_bar.ts,
            'entry':float(entry_bar.open),'side':side,'signal_atr':float(sig.atr),
            'structure_stop':np.nan,'delay_min':float((entry_bar.ts-sig.ts).total_seconds()/60),
            'dir_prob':float(sig.dir_prob),'future_clean_trend':int(sig.trend_clean==1),
            'future_dir_correct':int(sig.trend_clean==1 and np.sign(sig.fwd_atr)==side),
            'bars':bars[['ts','open','high','low','close']].copy(),
        })
    return rows

v11.build_events_from_signals=build_events_m15_only
v11.main()
