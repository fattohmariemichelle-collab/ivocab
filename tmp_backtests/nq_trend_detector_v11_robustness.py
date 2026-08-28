from __future__ import annotations

import calendar
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

import nq_trend_detector_v8_real_nq as v8
import nq_trend_detector_v8b_real_nq as v8b
import nq_trend_detector_v8c_ablation as v8c
import nq_trend_detector_v8d_direction as d8
import nq_trend_detector_v8d2_event_audit as a8
import nq_trend_detector_v9_entry_engine as v9
import nq_trend_detector_v10_risk_exit as v10

OUT = Path('trend_backtest_v11_results')
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)
TICK_SIZE = 0.25
PRIMARY_COST_TICKS = 4.0

SELECTED_PROFILE = {
    'family': 'trailing', 'stop_type': 'atr', 'stop_atr': 1.0,
    'target_r': None, 'hours': 12, 'be_trigger_r': None,
    'trail_trigger_r': 1.0, 'trail_distance_r': 1.5,
    'partial_r': None, 'partial_fraction': None,
    'name': 'trail_s1.00_trig1.00_dist1.50_12h',
}


def strict_causal_session_features(m: pd.DataFrame) -> pd.DataFrame:
    """V8b session features with a corrected Opening Range availability timestamp.

    The completed 09:30-10:00 ET range for trade date D may not be exposed during
    the preceding evening session (18:00-23:59 on D-1) or before 10:00 on D.
    """
    x = m.copy()
    x['date'] = x.ts.dt.date
    x['minute'] = x.ts.dt.hour * 60 + x.ts.dt.minute
    x['is_rth'] = ((x.minute >= 570) & (x.minute < 960)).astype(int)
    x['is_opening_hour'] = ((x.minute >= 570) & (x.minute < 630)).astype(int)
    x['is_power_hour'] = ((x.minute >= 900) & (x.minute < 960)).astype(int)
    x['is_overnight'] = ((x.minute < 570) | (x.minute >= 1080)).astype(int)

    x['session_date'] = pd.to_datetime(x.ts.dt.date)
    x.loc[x.minute >= 1080, 'session_date'] += pd.Timedelta(days=1)

    pv = x.close * x.volume
    x['cum_vol'] = x.groupby('session_date').volume.cumsum()
    x['cum_pv'] = pv.groupby(x.session_date).cumsum()
    x['session_vwap'] = x.cum_pv / x.cum_vol.replace(0, np.nan)
    x['dist_session_vwap_atr1m'] = np.nan

    sess = x.groupby('session_date').agg(
        sess_high=('high', 'max'), sess_low=('low', 'min'),
        sess_close=('close', 'last'), sess_vol=('volume', 'sum'),
    )
    prev = sess.shift(1).rename(columns=lambda c: 'prev_' + c)
    x = x.merge(prev, left_on='session_date', right_index=True, how='left')

    overnight = x['is_overnight'].eq(1)
    gh = x['high'].where(overnight).groupby(x['session_date']).cummax()
    gl = x['low'].where(overnight).groupby(x['session_date']).cummin()
    gv = x['volume'].where(overnight, 0.0).groupby(x['session_date']).cumsum()
    x['on_high'] = gh.groupby(x['session_date']).ffill()
    x['on_low'] = gl.groupby(x['session_date']).ffill()
    x['on_vol'] = gv
    has_on = overnight.groupby(x['session_date']).cummax().astype(bool)
    x.loc[~has_on, ['on_high', 'on_low', 'on_vol']] = np.nan

    or30 = x[(x.minute >= 570) & (x.minute < 600)].groupby('session_date').agg(
        or30_high=('high', 'max'), or30_low=('low', 'min'), or30_vol=('volume', 'sum'),
    )
    x = x.merge(or30, left_on='session_date', right_index=True, how='left')
    or_available_at = x['session_date'] + pd.Timedelta(hours=10)
    x.loc[x.ts < or_available_at, ['or30_high', 'or30_low', 'or30_vol']] = np.nan
    x['or_available_at'] = or_available_at
    return x


def add_signed_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x['ema20_50_signed_atr'] = (x.ema20 - x.ema50) / x.atr.replace(0, np.nan)
    x['ema10_20_signed_atr'] = (x.ema10 - x.ema20) / x.atr.replace(0, np.nan)
    x['dmi_signed'] = (x.pdi - x.mdi) / 100.0
    x['close_ema20_signed_atr'] = (x.close - x.ema20) / x.atr.replace(0, np.nan)
    x['h4_ema_signed'] = np.sign(x.h4_ema20 - x.h4_ema50)
    x['d1_ema_signed'] = np.sign(x.d1_ema20 - x.d1_ema50)
    return x


def build_dataset(session_fn=strict_causal_session_features):
    v8.session_features = session_fn
    v8.add_labels = v8b.gap_safe_labels
    raw = v8.load()
    m = session_fn(raw)
    h = v8.make_h1(m)
    d = v8.make_context(m, h)
    d = v8.add_h4_d1(d, m)
    d = add_signed_features(v8.add_labels(d, 12))
    quality = {
        'rows_1m': int(len(raw)), 'start': str(raw.ts.min()), 'end': str(raw.ts.max()),
        'duplicates': int(raw.ts.duplicated().sum()), 'zero_volume': int((raw.volume <= 0).sum()),
    }
    return raw, m, d, quality


def hgb_model():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        HistGradientBoostingClassifier(
            max_iter=180, max_depth=3, learning_rate=.035,
            l2_regularization=1.5, random_state=42,
        ),
    )


def spaced_rows(x: pd.DataFrame, hours: int = 12) -> pd.DataFrame:
    if x.empty:
        return x.copy()
    return a8.nonoverlap(x.sort_values('ts'), hours).copy()


def binary_metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {'n': 0}
    two = len(np.unique(y)) == 2
    cut = float(np.quantile(p, .90)) if len(p) else np.nan
    top = p >= cut if len(p) else np.array([], dtype=bool)
    base = float(y.mean())
    return {
        'n': int(len(y)), 'base_rate': base,
        'auc': float(roc_auc_score(y, p)) if two else np.nan,
        'ap': float(average_precision_score(y, p)) if two else np.nan,
        'top10_cut': cut, 'top10_n': int(top.sum()),
        'top10_rate': float(y[top].mean()) if top.any() else np.nan,
        'top10_lift': float(y[top].mean() / base) if top.any() and base > 0 else np.nan,
    }


def fit_strict_models(d: pd.DataFrame, purged: bool = True,
                      train_start='2023-01-01', train_end='2024-01-01',
                      val_start='2024-01-01', val_end='2025-01-01'):
    gate_features = v8c.feature_sets()[0]['price_plus_overnight_opening']
    direction_features = d8.sets()['price_location_volume']

    tr = d[(d.ts >= train_start) & (d.ts < train_end)].dropna(subset=['trend_clean']).copy()
    va = d[(d.ts >= val_start) & (d.ts < val_end)].dropna(subset=['trend_clean']).copy()
    tr_fit = spaced_rows(tr, 12) if purged else tr
    va_eval = spaced_rows(va, 12) if purged else va

    gate = hgb_model()
    gate.fit(tr_fit[gate_features], tr_fit.trend_clean.astype(int))
    pv = gate.predict_proba(va_eval[gate_features])[:, 1]
    gate_cut = float(np.quantile(pv, .90))
    gate_info = {
        'features': gate_features, 'model': 'hgb_fixed', 'purged_12h': purged,
        'train_n': int(len(tr_fit)), 'validation_n': int(len(va_eval)),
        'validation': binary_metrics(va_eval.trend_clean.astype(int), pv),
        'cut_q90_validation': gate_cut,
    }

    dir_train = tr[tr.trend_clean == 1].copy()
    if purged:
        dir_train = spaced_rows(dir_train, 12)
    va_all = va.copy()
    va_all['trend_prob'] = gate.predict_proba(va_all[gate_features])[:, 1]
    dir_val = va_all[(va_all.trend_prob >= gate_cut) & (va_all.trend_clean == 1)].copy()
    if purged:
        dir_val = spaced_rows(dir_val, 12)
    dir_train['y_long'] = (dir_train.fwd_atr > 0).astype(int)
    dir_val['y_long'] = (dir_val.fwd_atr > 0).astype(int)

    direction = hgb_model()
    direction.fit(dir_train[direction_features], dir_train.y_long)
    pdv = direction.predict_proba(dir_val[direction_features])[:, 1] if len(dir_val) else np.array([])
    dir_info = {
        'features': direction_features, 'model': 'hgb_fixed', 'margin_fixed': .20,
        'train_n': int(len(dir_train)), 'validation_n': int(len(dir_val)),
        'validation': d8.metrics(dir_val.y_long, pdv) if len(dir_val) else {'n': 0},
    }
    return gate, gate_info, direction, dir_info


def make_signals(d, gate, gate_info, direction, dir_info, start, end):
    x = d[(d.ts >= start) & (d.ts < end)].dropna(subset=['trend_clean']).copy()
    x['trend_prob'] = gate.predict_proba(x[gate_info['features']])[:, 1]
    x = x[x.trend_prob >= gate_info['cut_q90_validation']].copy()
    if x.empty:
        return x
    x['dir_prob'] = direction.predict_proba(x[dir_info['features']])[:, 1]
    x = x[np.abs(x.dir_prob - .5) >= dir_info['margin_fixed']].copy()
    x['side'] = np.where(x.dir_prob >= .5, 1, -1)
    return spaced_rows(x, 12)


def build_events_from_signals(signals, m, m15):
    rows = []
    for _, sig in signals.iterrows():
        candidates = v9.candidate_events_for_signal(sig, pd.DataFrame(), m15)
        ev = next((z for z in candidates if z['setup'] == 'm15_breakout'), None)
        if ev is None:
            continue
        future = m[m.ts > ev['entry_ts']]
        if future.empty:
            continue
        entry_bar = future.iloc[0]
        if (entry_bar.ts - ev['entry_ts']).total_seconds() / 60 > 5:
            continue
        bars = m[(m.ts >= entry_bar.ts) & (m.ts <= entry_bar.ts + pd.Timedelta(hours=12))]
        if len(bars) < 30:
            continue
        rows.append({
            'signal_ts': sig.ts, 'entry_trigger_ts': ev['entry_ts'],
            'entry_ts': entry_bar.ts, 'entry': float(entry_bar.open),
            'side': int(sig.side), 'signal_atr': float(sig.atr),
            'structure_stop': np.nan,
            'delay_min': float((entry_bar.ts - sig.ts).total_seconds() / 60),
            'dir_prob': float(sig.dir_prob),
            'future_clean_trend': int(sig.trend_clean == 1),
            'future_dir_correct': int(sig.trend_clean == 1 and np.sign(sig.fwd_atr) == sig.side),
            'bars': bars[['ts', 'open', 'high', 'low', 'close']].copy(),
        })
    return rows


def simulate_events(events, signals_total, profile=SELECTED_PROFILE, cost_ticks=PRIMARY_COST_TICKS):
    raw = []
    for ev in events:
        r = v10.simulate_one(ev, profile, cost_ticks=cost_ticks)
        if r is not None:
            raw.append(r)
    accepted, skipped = v10.enforce_one_position(raw)
    summary = v10.summarize(accepted, signals_total, profile)
    summary['overlap_trades_skipped'] = skipped
    return accepted, summary


def run_protocol(session_fn, purged=True):
    raw, m, d, quality = build_dataset(session_fn)
    _, m15 = v9.make_low_tf(m)
    gate, gi, direction, di = fit_strict_models(d, purged=purged)
    periods = {}
    events_by = {}; trades_by = {}; signals_by = {}
    for label, start, end in [
        ('validation_2024', '2024-01-01', '2025-01-01'),
        ('evaluation_2025', '2025-01-01', '2025-12-01'),
        ('micro_dec2025', '2025-12-01', '2025-12-11'),
    ]:
        s = make_signals(d, gate, gi, direction, di, start, end)
        e = build_events_from_signals(s, m, m15)
        t, summary = simulate_events(e, len(s))
        signals_by[label] = s; events_by[label] = e; trades_by[label] = t
        periods[label] = {'signals': int(len(s)), 'events': int(len(e)), 'trade_summary': summary}
    return {
        'raw': raw, 'm': m, 'd': d, 'quality': quality,
        'gate': gate, 'gate_info': gi, 'direction': direction, 'direction_info': di,
        'signals': signals_by, 'events': events_by, 'trades': trades_by, 'periods': periods,
    }


def leak_audit(raw):
    old = v8b.causal_session_features(raw)
    strict = strict_causal_session_features(raw)
    avail = old.session_date + pd.Timedelta(hours=10)
    leak = old.or30_high.notna() & (old.ts < avail)
    evening = old.minute >= 1080
    return {
        'leaked_1m_rows': int(leak.sum()),
        'leaked_1m_pct': float(leak.mean()),
        'leaked_evening_rows': int((leak & evening).sum()),
        'old_or_nonnull_before_availability_rate': float(leak.sum() / max(1, old.or30_high.notna().sum())),
        'strict_remaining_leaked_rows': int((strict.or30_high.notna() & (strict.ts < strict.or_available_at)).sum()),
        'first_examples': old.loc[leak, ['ts', 'session_date', 'or30_high', 'or30_low']].head(10).astype(str).to_dict(orient='records'),
    }


def trades_frame(trades):
    return pd.DataFrame(trades).sort_values('entry_ts') if trades else pd.DataFrame()


def subset_summary(g, name):
    if g.empty:
        return {'segment': name, 'n': 0}
    x = g.net_r.to_numpy(float); gp = x[x > 0].sum(); gl = -x[x < 0].sum()
    return {
        'segment': name, 'n': int(len(g)), 'avg_r': float(x.mean()),
        'median_r': float(np.median(x)), 'win_rate': float((x > 0).mean()),
        'profit_factor': float(gp / gl) if gl > 0 else np.inf,
        'max_drawdown_r': v10.max_drawdown(x),
    }


def segment_audit(trades):
    g = trades_frame(trades)
    if g.empty:
        return []
    g['entry_ts'] = pd.to_datetime(g.entry_ts)
    h = g.entry_ts.dt.hour
    g['session'] = np.select(
        [(h >= 18) | (h < 8), (h >= 8) & (h < 10), (h >= 10) & (h < 12),
         (h >= 12) & (h < 15), (h >= 15) & (h < 18)],
        ['overnight', 'premarket_open', 'morning', 'midday', 'power_post'], default='other')
    rows = []
    for side, z in g.groupby('side'):
        rows.append(subset_summary(z, 'LONG' if side == 1 else 'SHORT'))
    for sess, z in g.groupby('session'):
        rows.append(subset_summary(z, f'session_{sess}'))
    for q, z in g.groupby(g.entry_ts.dt.to_period('Q').astype(str)):
        rows.append(subset_summary(z, f'quarter_{q}'))
    return rows


def third_friday(year, month):
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def rollover_mask(ts):
    t = pd.to_datetime(ts)
    mask = np.zeros(len(t), dtype=bool)
    for year in sorted(set(t.dt.year)):
        for month in [3, 6, 9, 12]:
            exp = third_friday(year, month)
            mask |= (t >= exp - pd.Timedelta(days=7)) & (t <= exp + pd.Timedelta(days=2))
    return mask


def robustness_from_trades(trades):
    g = trades_frame(trades)
    if g.empty:
        return {}
    g['entry_ts'] = pd.to_datetime(g.entry_ts)
    base = subset_summary(g, 'all')
    no_roll = g[~rollover_mask(g.entry_ts)]
    sorted_best = g.sort_values('net_r', ascending=False)
    removals = {}
    for k in [1, 3, 5]:
        z = sorted_best.iloc[k:]
        removals[f'remove_top_{k}'] = subset_summary(z, f'remove_top_{k}')
    costs = {}
    for ticks in [0, 4, 8, 12, 20]:
        vals = g.gross_r - ticks * TICK_SIZE / g.risk_points
        costs[str(ticks)] = {
            'avg_r': float(vals.mean()), 'win_rate': float((vals > 0).mean()),
            'max_drawdown_r': v10.max_drawdown(vals.to_numpy(float)),
        }
    monthly = g.assign(month=g.entry_ts.dt.to_period('M').astype(str)).groupby('month').net_r.agg(['count', 'mean', 'sum']).reset_index()
    return {
        'base': base, 'exclude_roll_window': subset_summary(no_roll, 'exclude_roll_window'),
        'top_trade_removals': removals, 'cost_stress_ticks_roundtrip': costs,
        'monthly': monthly.to_dict(orient='records'), 'segments': segment_audit(trades),
    }


def neighborhood_profiles():
    rows = []
    for stop in [.75, 1.0, 1.25]:
        for trig in [.75, 1.0, 1.25]:
            for dist in [1.0, 1.5, 2.0]:
                for hours in [6, 9, 12]:
                    rows.append({
                        'family': 'trailing', 'stop_type': 'atr', 'stop_atr': stop,
                        'target_r': None, 'hours': hours, 'be_trigger_r': None,
                        'trail_trigger_r': trig, 'trail_distance_r': dist,
                        'partial_r': None, 'partial_fraction': None,
                        'name': f'nb_s{stop:.2f}_tr{trig:.2f}_d{dist:.2f}_{hours}h',
                    })
    return rows


def parameter_neighborhood(events_2024, nsignals_2024, events_2025, nsignals_2025):
    rows = []
    for p in neighborhood_profiles():
        _, s24 = simulate_events(events_2024, nsignals_2024, p)
        _, s25 = simulate_events(events_2025, nsignals_2025, p)
        rows.append({
            'profile': p['name'], 'stop': p['stop_atr'], 'trigger': p['trail_trigger_r'],
            'distance': p['trail_distance_r'], 'hours': p['hours'],
            'n_2024': s24.get('n', 0), 'avg_r_2024': s24.get('avg_net_r', np.nan),
            'pf_2024': s24.get('profit_factor', np.nan), 'dd_2024': s24.get('max_drawdown_r', np.nan),
            'n_2025': s25.get('n', 0), 'avg_r_2025': s25.get('avg_net_r', np.nan),
            'pf_2025': s25.get('profit_factor', np.nan), 'dd_2025': s25.get('max_drawdown_r', np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'strict_parameter_neighborhood.csv', index=False)
    both = df[(df.avg_r_2024 > 0) & (df.avg_r_2025 > 0)]
    return {
        'profiles': int(len(df)), 'positive_both': int(len(both)),
        'positive_both_rate': float(len(both) / len(df)),
        'median_avg_r_2024': float(df.avg_r_2024.median()),
        'median_avg_r_2025': float(df.avg_r_2025.median()),
        'worst_avg_r_2024': float(df.avg_r_2024.min()),
        'worst_avg_r_2025': float(df.avg_r_2025.min()),
        'selected_profile_row': df[(df.stop == 1.0) & (df.trigger == 1.0) & (df.distance == 1.5) & (df.hours == 12)].to_dict(orient='records'),
    }


def direction_controls(signals, m, m15):
    if signals.empty:
        return {}
    modes = {}
    variants = {
        'model': signals.copy(),
        'inverted': signals.assign(side=-signals.side),
        'always_long': signals.assign(side=1),
        'h1_ema': signals.assign(side=np.where(signals.ema20 >= signals.ema50, 1, -1)),
    }
    for name, s in variants.items():
        e = build_events_from_signals(s, m, m15)
        t, summary = simulate_events(e, len(s))
        modes[name] = {'signals': int(len(s)), 'events': int(len(e)), 'summary': summary}
    return modes


def walk_forward(d, m, m15):
    rows = []; all_trades = []
    quarters = pd.period_range('2024Q2', '2025Q4', freq='Q')
    for q in quarters:
        test_start = q.start_time; test_end = q.end_time + pd.Timedelta(nanoseconds=1)
        val_q = q - 1; val_start = val_q.start_time; val_end = val_q.end_time + pd.Timedelta(nanoseconds=1)
        train_start = pd.Timestamp('2023-01-01'); train_end = val_start
        try:
            gate, gi, direction, di = fit_strict_models(
                d, purged=True, train_start=train_start, train_end=train_end,
                val_start=val_start, val_end=val_end,
            )
            s = make_signals(d, gate, gi, direction, di, test_start, test_end)
            e = build_events_from_signals(s, m, m15)
            t, summary = simulate_events(e, len(s))
            all_trades.extend(t)
            rows.append({
                'quarter': str(q), 'train_end': str(train_end), 'validation_quarter': str(val_q),
                'signals': int(len(s)), 'events': int(len(e)),
                'n': summary.get('n', 0), 'avg_r': summary.get('avg_net_r', np.nan),
                'win_rate': summary.get('win_rate', np.nan), 'profit_factor': summary.get('profit_factor', np.nan),
                'max_drawdown_r': summary.get('max_drawdown_r', np.nan),
                'gate_validation_ap': gi['validation'].get('ap', np.nan),
                'direction_validation_balanced_accuracy': di['validation'].get('balanced_accuracy', np.nan),
            })
        except Exception as exc:
            rows.append({'quarter': str(q), 'error': type(exc).__name__ + ': ' + str(exc)})
    wf = pd.DataFrame(rows)
    wf.to_csv(OUT / 'walk_forward_quarters.csv', index=False)
    accepted, skipped = v10.enforce_one_position(all_trades)
    agg = v10.summarize(accepted, sum(r.get('signals', 0) for r in rows), SELECTED_PROFILE) if accepted else {'n': 0}
    agg['overlap_skipped_across_quarters'] = skipped
    return {'quarters': rows, 'aggregate': agg}


def secondary_dataset_audit(primary_raw):
    path = Path('data_v11_secondary/NQ_in_1_minute.csv')
    if not path.exists():
        return {'status': 'not_downloaded'}
    s = pd.read_csv(path)
    s.columns = [c.lower().strip() for c in s.columns]
    s['ts_utc'] = pd.to_datetime(s['datetime'], errors='coerce', utc=True)
    s['ts'] = s.ts_utc.dt.tz_convert('America/New_York').dt.tz_localize(None)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
    s = s.dropna(subset=['ts', 'open', 'high', 'low', 'close']).sort_values('ts')
    p = primary_raw[['ts', 'open', 'high', 'low', 'close', 'volume']]
    z = p.merge(s[['ts', 'open', 'high', 'low', 'close', 'volume']], on='ts', suffixes=('_primary', '_secondary'))
    if z.empty:
        return {'status': 'no_timestamp_overlap', 'secondary_start': str(s.ts.min()), 'secondary_end': str(s.ts.max()), 'secondary_rows': int(len(s))}
    exact = np.ones(len(z), dtype=bool)
    diffs = {}
    for c in ['open', 'high', 'low', 'close']:
        diff = (z[f'{c}_primary'] - z[f'{c}_secondary']).abs()
        diffs[c] = {'median_abs_diff': float(diff.median()), 'p95_abs_diff': float(diff.quantile(.95)), 'exact_rate': float((diff < 1e-9).mean())}
        exact &= diff < 1e-9
    volcorr = z.volume_primary.corr(z.volume_secondary)
    exact_rate = float(exact.mean())
    status = 'not_independent_near_duplicate' if exact_rate >= .95 else 'different_feed_but_short_overlap'
    return {
        'status': status, 'secondary_start': str(s.ts.min()), 'secondary_end': str(s.ts.max()),
        'secondary_rows': int(len(s)), 'overlap_rows': int(len(z)), 'all_ohlc_exact_rate': exact_rate,
        'volume_correlation': float(volcorr) if np.isfinite(volcorr) else np.nan, 'differences': diffs,
        'replication_note': 'The secondary 1-minute file is too short for a fresh multi-year train/validation/test replication.'
    }


def main():
    primary_raw = v8.load()
    leak = leak_audit(primary_raw)

    # Re-run the complete system with the corrected OR clock. Purged protocol is the V11 reference.
    strict_purged = run_protocol(strict_causal_session_features, purged=True)
    strict_unpurged = run_protocol(strict_causal_session_features, purged=False)

    # Determine how many old selected-signal timestamps sat inside the OR leak window.
    old = run_protocol(v8b.causal_session_features, purged=False)
    old_signal_leak = {}
    old_m = v8b.causal_session_features(primary_raw)
    old_h = v8.make_context(old_m, v8.make_h1(old_m))
    old_h['or_available_at'] = pd.to_datetime(old_h.ts.dt.date) + pd.Timedelta(hours=10)
    for label, s in old['signals'].items():
        if s.empty:
            old_signal_leak[label] = {'signals': 0, 'leaked': 0}
            continue
        q = s[['ts']].merge(old_h[['ts', 'or30_high', 'or_available_at']], on='ts', how='left')
        leaked = q.or30_high.notna() & (q.ts < q.or_available_at)
        old_signal_leak[label] = {'signals': int(len(q)), 'leaked': int(leaked.sum()), 'leaked_rate': float(leaked.mean())}

    strict_events_24 = strict_purged['events']['validation_2024']
    strict_events_25 = strict_purged['events']['evaluation_2025']
    neigh = parameter_neighborhood(
        strict_events_24, len(strict_purged['signals']['validation_2024']),
        strict_events_25, len(strict_purged['signals']['evaluation_2025']),
    )

    strict_robustness = {
        label: robustness_from_trades(strict_purged['trades'][label])
        for label in ['validation_2024', 'evaluation_2025', 'micro_dec2025']
    }
    _, strict_m15 = v9.make_low_tf(strict_purged['m'])
    controls = {
        label: direction_controls(strict_purged['signals'][label], strict_purged['m'], strict_m15)
        for label in ['validation_2024', 'evaluation_2025']
    }
    wf = walk_forward(strict_purged['d'], strict_purged['m'], strict_m15)
    secondary = secondary_dataset_audit(primary_raw)

    for protocol_name, protocol in [('strict_purged', strict_purged), ('strict_unpurged', strict_unpurged), ('old_leaky', old)]:
        for label, trades in protocol['trades'].items():
            trades_frame(trades).to_csv(OUT / f'{protocol_name}_{label}_trades.csv', index=False)

    result = {
        'status': 'V11_COMPLETE',
        'method': {
            'purpose': 'Falsification-oriented audit of V8c-V10, followed by a full causal rebuild.',
            'critical_fix': 'Opening Range for trade date D is hidden until 10:00 ET on D, including the preceding 18:00-23:59 session.',
            'purging': 'Reference protocol uses observations spaced by at least 12 hours for gate validation and direction training/validation; emitted signals are also 12h-spaced.',
            'fixed_components': 'HGB models, V8c feature block, V8d 0.20 abstention margin, M15 breakout entry, and V10 trailing profile are not optimized on 2025.',
            'cost': 'Primary results use four ticks round-trip and conservative stop-first intrabar handling inherited from V10.',
        },
        'data_quality': strict_purged['quality'],
        'opening_range_leak_audit': leak,
        'old_protocol_signal_leak': old_signal_leak,
        'protocol_comparison': {
            'old_leaky': old['periods'],
            'strict_unpurged': strict_unpurged['periods'],
            'strict_purged_reference': strict_purged['periods'],
        },
        'strict_gate': strict_purged['gate_info'],
        'strict_direction': strict_purged['direction_info'],
        'strict_robustness': strict_robustness,
        'direction_controls': controls,
        'parameter_neighborhood': neigh,
        'walk_forward': wf,
        'secondary_dataset_audit': secondary,
        'verdict_rules': {
            'survives': 'Positive mean R in 2024 and 2025 after strict correction, positive majority of parameter neighborhood, and no collapse in walk-forward.',
            'fails': 'Non-positive expectancy after strict correction, direction no better than controls, or edge concentrated in leaked/pre-roll/top-trade subsets.',
        },
    }
    (OUT / 'v11_robustness_results.json').write_text(json.dumps(result, indent=2, allow_nan=True), encoding='utf-8')
    pd.DataFrame([
        {'protocol': p, 'period': period, **data['trade_summary']}
        for p, prot in [('old_leaky', old), ('strict_unpurged', strict_unpurged), ('strict_purged', strict_purged)]
        for period, data in prot['periods'].items()
    ]).to_csv(OUT / 'protocol_comparison.csv', index=False)
    print('V11_COMPLETE')
    print(json.dumps({
        'leak_audit': leak,
        'old_signal_leak': old_signal_leak,
        'protocol_comparison': result['protocol_comparison'],
        'strict_gate': result['strict_gate'],
        'strict_direction': result['strict_direction'],
        'parameter_neighborhood': neigh,
        'walk_forward_aggregate': wf['aggregate'],
        'secondary': secondary,
    }, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
