from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import nq_trend_detector_v8_real_nq as v8

# Separate output so the original exploratory V8 remains auditable.
v8.OUT = Path('trend_backtest_v8b_results')
v8.OUT.mkdir(exist_ok=True)


def causal_session_features(m: pd.DataFrame) -> pd.DataFrame:
    x = m.copy()
    x['date'] = x.ts.dt.date
    x['minute'] = x.ts.dt.hour * 60 + x.ts.dt.minute
    x['is_rth'] = ((x.minute >= 570) & (x.minute < 960)).astype(int)
    x['is_opening_hour'] = ((x.minute >= 570) & (x.minute < 630)).astype(int)
    x['is_power_hour'] = ((x.minute >= 900) & (x.minute < 960)).astype(int)
    x['is_overnight'] = ((x.minute < 570) | (x.minute >= 1080)).astype(int)

    # CME-style session label approximation in ET: 18:00 belongs to next trade date.
    x['session_date'] = pd.to_datetime(x.ts.dt.date)
    x.loc[x.minute >= 1080, 'session_date'] += pd.Timedelta(days=1)

    # Causal session VWAP.
    pv = x.close * x.volume
    x['cum_vol'] = x.groupby('session_date').volume.cumsum()
    x['cum_pv'] = pv.groupby(x.session_date).cumsum()
    x['session_vwap'] = x.cum_pv / x.cum_vol.replace(0, np.nan)
    x['dist_session_vwap_atr1m'] = np.nan

    # Previous COMPLETED session references only.
    sess = x.groupby('session_date').agg(
        sess_high=('high', 'max'), sess_low=('low', 'min'),
        sess_close=('close', 'last'), sess_vol=('volume', 'sum')
    )
    prev = sess.shift(1).rename(columns=lambda c: 'prev_' + c)
    x = x.merge(prev, left_on='session_date', right_index=True, how='left')

    # Overnight high/low/volume are cumulative while overnight is unfolding.
    # After 09:30 ET they remain frozen at the final overnight values.
    x['on_high'] = np.nan
    x['on_low'] = np.nan
    x['on_vol'] = np.nan
    for _, idx in x.groupby('session_date').groups.items():
        idx = np.asarray(list(idx), dtype=int)
        g = x.loc[idx]
        is_on = g['is_overnight'].to_numpy(dtype=bool)
        if not is_on.any():
            continue
        highs = g['high'].to_numpy(float)
        lows = g['low'].to_numpy(float)
        vols = g['volume'].to_numpy(float)
        running_high = np.nan
        running_low = np.nan
        running_vol = 0.0
        final_high = np.nan
        final_low = np.nan
        final_vol = np.nan
        values_high = np.full(len(g), np.nan)
        values_low = np.full(len(g), np.nan)
        values_vol = np.full(len(g), np.nan)
        # First pass: cumulative only on overnight records.
        for j in range(len(g)):
            if is_on[j]:
                running_high = highs[j] if not np.isfinite(running_high) else max(running_high, highs[j])
                running_low = lows[j] if not np.isfinite(running_low) else min(running_low, lows[j])
                running_vol += vols[j]
                values_high[j] = running_high
                values_low[j] = running_low
                values_vol[j] = running_vol
        # Final values become available only at/after RTH open.
        on_positions = np.flatnonzero(is_on)
        final_high = values_high[on_positions[-1]]
        final_low = values_low[on_positions[-1]]
        final_vol = values_vol[on_positions[-1]]
        for j in range(len(g)):
            minute = int(g.iloc[j]['minute'])
            if 570 <= minute < 1080:  # after RTH open, before next session begins
                values_high[j] = final_high
                values_low[j] = final_low
                values_vol[j] = final_vol
            elif not is_on[j] and minute < 570:
                # Defensive: no future-filled values.
                values_high[j] = running_high if np.isfinite(running_high) else np.nan
                values_low[j] = running_low if np.isfinite(running_low) else np.nan
                values_vol[j] = running_vol if running_vol else np.nan
        x.loc[idx, 'on_high'] = values_high
        x.loc[idx, 'on_low'] = values_low
        x.loc[idx, 'on_vol'] = values_vol

    # Opening range: final 09:30-10:00 range is exposed only at/after 10:00 ET.
    or30 = x[(x.minute >= 570) & (x.minute < 600)].groupby('session_date').agg(
        or30_high=('high', 'max'), or30_low=('low', 'min'), or30_vol=('volume', 'sum')
    )
    x = x.merge(or30, left_on='session_date', right_index=True, how='left')
    x.loc[x.minute < 600, ['or30_high', 'or30_low', 'or30_vol']] = np.nan
    return x


def gap_safe_labels(d: pd.DataFrame, hours: int = 12) -> pd.DataFrame:
    n = len(d)
    close = d.close.to_numpy(float)
    high = d.high.to_numpy(float)
    low = d.low.to_numpy(float)
    atr = d.atr.to_numpy(float)
    ts = d.ts.to_numpy(dtype='datetime64[s]').astype(np.int64)
    sr = np.full(n, np.nan); eff = np.full(n, np.nan)
    mae_long = np.full(n, np.nan); mae_short = np.full(n, np.nan)
    max_up = np.full(n, np.nan); max_dn = np.full(n, np.nan)
    rejected_gap_windows = 0

    # Allow the normal daily maintenance gap but reject weekends / large data gaps.
    MAX_GAP_SECONDS = 3 * 3600
    for i in range(n - hours):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if np.max(np.diff(ts[i:i + hours + 1])) > MAX_GAP_SECONDS:
            rejected_gap_windows += 1
            continue
        c0 = close[i]
        segc = close[i:i + hours + 1]
        hs = high[i + 1:i + hours + 1]
        ls = low[i + 1:i + hours + 1]
        sr[i] = (close[i + hours] - c0) / atr[i]
        path = np.abs(np.diff(segc)).sum()
        eff[i] = abs(close[i + hours] - c0) / path if path else 0.0
        max_up[i] = (hs.max() - c0) / atr[i]
        max_dn[i] = (c0 - ls.min()) / atr[i]
        mae_long[i] = max_dn[i]
        mae_short[i] = max_up[i]

    out = d.copy()
    out['fwd_atr'] = sr; out['fwd_eff'] = eff
    out['max_up_atr'] = max_up; out['max_dn_atr'] = max_dn
    adverse = np.where(out.fwd_atr >= 0, mae_long, mae_short)
    out['trend_clean'] = ((abs(out.fwd_atr) >= 0.75) & (out.fwd_eff >= 0.35) & (adverse <= 0.75)).astype(float)
    out.loc[out.fwd_atr.isna(), 'trend_clean'] = np.nan
    out['trend_dir'] = np.sign(out.fwd_atr)
    out.loc[out.trend_clean.ne(1), 'trend_dir'] = 0
    out['range_persist'] = ((abs(out.fwd_atr) < 0.5) & (out.max_up_atr < 1.0) & (out.max_dn_atr < 1.0) & (out.fwd_eff < 0.25)).astype(float)
    out.loc[out.fwd_atr.isna(), 'range_persist'] = np.nan
    out.attrs['rejected_gap_windows'] = rejected_gap_windows
    return out


v8.session_features = causal_session_features
v8.add_labels = gap_safe_labels

if __name__ == '__main__':
    v8.main()
    # Record explicit methodological fixes alongside the normal summary.
    p = v8.OUT / 'methodology_fixes.json'
    p.write_text(json.dumps({
        'overnight_features': 'causal cumulative before 09:30 ET; frozen final values only after 09:30',
        'opening_range': 'final 30m range only exposed at/after 10:00 ET',
        'forward_label_gap_filter': 'reject 12-bar windows containing any timestamp gap > 3h',
        'purpose': 'remove overnight look-ahead and weekend/large-gap leakage from exploratory V8'
    }, indent=2), encoding='utf-8')
