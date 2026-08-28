from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import nq_trend_detector_v8_real_nq as v8

v8.OUT = Path('trend_backtest_v8b_results')
v8.OUT.mkdir(exist_ok=True)


def causal_session_features(m: pd.DataFrame) -> pd.DataFrame:
    x = m.copy()
    x['date'] = x.ts.dt.date
    x['minute'] = x.ts.dt.hour * 60 + x.ts.dt.minute
    x['is_rth'] = ((x.minute >= 570) & (x.minute < 960)).astype(int)
    x['is_opening_hour'] = ((x.minute >= 570) & (x.minute < 630)).astype(int)
    x['is_power_hour'] = ((x.minute >= 900) & (x.minute < 960)).astype(int)
    # Overnight definition used here: 18:00 ET through 09:29 ET.
    x['is_overnight'] = ((x.minute < 570) | (x.minute >= 1080)).astype(int)

    # Approximate CME trade date in ET: 18:00 belongs to the next trade date.
    x['session_date'] = pd.to_datetime(x.ts.dt.date)
    x.loc[x.minute >= 1080, 'session_date'] += pd.Timedelta(days=1)

    # Causal session VWAP, recomputed from raw OHLCV rather than trusting supplied VWAP columns.
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

    # Causal overnight high/low/volume. Before 09:30, only information observed so far is used.
    # During RTH these series simply carry forward the completed overnight values.
    overnight = x['is_overnight'].eq(1)
    gh = x['high'].where(overnight).groupby(x['session_date']).cummax()
    gl = x['low'].where(overnight).groupby(x['session_date']).cummin()
    gv = x['volume'].where(overnight, 0.0).groupby(x['session_date']).cumsum()
    x['on_high'] = gh.groupby(x['session_date']).ffill()
    x['on_low'] = gl.groupby(x['session_date']).ffill()
    x['on_vol'] = gv
    # Do not expose overnight aggregate before the first overnight observation for a trade date.
    has_on = overnight.groupby(x['session_date']).cummax().astype(bool)
    x.loc[~has_on, ['on_high', 'on_low', 'on_vol']] = np.nan

    # Opening range 09:30-10:00 ET, only visible once it is completed.
    or30 = x[(x.minute >= 570) & (x.minute < 600)].groupby('session_date').agg(
        or30_high=('high', 'max'), or30_low=('low', 'min'), or30_vol=('volume', 'sum')
    )
    x = x.merge(or30, left_on='session_date', right_index=True, how='left')
    x.loc[x.minute < 600, ['or30_high', 'or30_low', 'or30_vol']] = np.nan
    return x


def gap_safe_labels(d: pd.DataFrame, hours: int = 12) -> pd.DataFrame:
    n = len(d)
    close = d.close.to_numpy(float); high = d.high.to_numpy(float); low = d.low.to_numpy(float); atr = d.atr.to_numpy(float)
    ts = d.ts.to_numpy(dtype='datetime64[s]').astype(np.int64)
    sr = np.full(n, np.nan); eff = np.full(n, np.nan)
    mae_long = np.full(n, np.nan); mae_short = np.full(n, np.nan)
    max_up = np.full(n, np.nan); max_dn = np.full(n, np.nan)
    MAX_GAP_SECONDS = 3 * 3600
    for i in range(n - hours):
        if not np.isfinite(atr[i]) or atr[i] <= 0: continue
        if np.max(np.diff(ts[i:i + hours + 1])) > MAX_GAP_SECONDS: continue
        c0 = close[i]; segc = close[i:i + hours + 1]; hs = high[i + 1:i + hours + 1]; ls = low[i + 1:i + hours + 1]
        sr[i] = (close[i + hours] - c0) / atr[i]
        path = np.abs(np.diff(segc)).sum(); eff[i] = abs(close[i + hours] - c0) / path if path else 0.0
        max_up[i] = (hs.max() - c0) / atr[i]; max_dn[i] = (c0 - ls.min()) / atr[i]
        mae_long[i] = max_dn[i]; mae_short[i] = max_up[i]
    out = d.copy(); out['fwd_atr'] = sr; out['fwd_eff'] = eff; out['max_up_atr'] = max_up; out['max_dn_atr'] = max_dn
    adverse = np.where(out.fwd_atr >= 0, mae_long, mae_short)
    out['trend_clean'] = ((abs(out.fwd_atr) >= 0.75) & (out.fwd_eff >= 0.35) & (adverse <= 0.75)).astype(float)
    out.loc[out.fwd_atr.isna(), 'trend_clean'] = np.nan
    out['trend_dir'] = np.sign(out.fwd_atr); out.loc[out.trend_clean.ne(1), 'trend_dir'] = 0
    out['range_persist'] = ((abs(out.fwd_atr) < 0.5) & (out.max_up_atr < 1.0) & (out.max_dn_atr < 1.0) & (out.fwd_eff < 0.25)).astype(float)
    out.loc[out.fwd_atr.isna(), 'range_persist'] = np.nan
    return out


v8.session_features = causal_session_features
v8.add_labels = gap_safe_labels

if __name__ == '__main__':
    v8.main()
    (v8.OUT / 'methodology_fixes.json').write_text(json.dumps({
        'overnight_features': 'causal cumulative 18:00-09:29 ET; no final overnight high/low visible early',
        'opening_range': 'final 09:30-10:00 ET range only exposed at/after 10:00 ET',
        'prior_session': 'shifted by one completed trade-date session',
        'session_vwap': 'causal cumulative recomputation from OHLCV',
        'forward_label_gap_filter': 'reject 12-H1-bar windows containing any timestamp gap >3h',
        'purpose': 'remove overnight look-ahead and weekend/large-gap leakage from exploratory V8'
    }, indent=2), encoding='utf-8')
