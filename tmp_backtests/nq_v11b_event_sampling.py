from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import nq_trend_detector_v11_robustness as v11

OUT = Path('trend_backtest_v11b_results')
OUT.mkdir(exist_ok=True)


def build_events_m15_only(signals: pd.DataFrame, m: pd.DataFrame, m15: pd.DataFrame):
    rows = []
    for _, sig in signals.iterrows():
        t0 = sig.ts
        t1 = t0 + pd.Timedelta(hours=6)
        side = int(sig.side)
        w = m15[(m15.ts > t0) & (m15.ts <= t1)]
        trigger = None
        for _, r in w.iterrows():
            if side > 0 and np.isfinite(r.m15_rh4) and r.close > r.m15_rh4 and r.m15_ema20_slope > 0:
                trigger = r
                break
            if side < 0 and np.isfinite(r.m15_rl4) and r.close < r.m15_rl4 and r.m15_ema20_slope < 0:
                trigger = r
                break
        if trigger is None:
            continue
        future = m[m.ts > trigger.ts]
        if future.empty:
            continue
        entry_bar = future.iloc[0]
        if (entry_bar.ts - trigger.ts).total_seconds() / 60 > 5:
            continue
        bars = m[(m.ts >= entry_bar.ts) & (m.ts <= entry_bar.ts + pd.Timedelta(hours=12))]
        if len(bars) < 30 or not np.isfinite(sig.atr) or sig.atr <= 0:
            continue
        rows.append({
            'signal_ts': sig.ts,
            'entry_trigger_ts': trigger.ts,
            'entry_ts': entry_bar.ts,
            'entry': float(entry_bar.open),
            'side': side,
            'signal_atr': float(sig.atr),
            'structure_stop': np.nan,
            'delay_min': float((entry_bar.ts - sig.ts).total_seconds() / 60),
            'dir_prob': float(sig.dir_prob),
            'future_clean_trend': int(sig.trend_clean == 1),
            'future_dir_correct': int(sig.trend_clean == 1 and np.sign(sig.fwd_atr) == side),
            'bars': bars[['ts', 'open', 'high', 'low', 'close']].copy(),
        })
    return rows


def independent_trend_onsets(x: pd.DataFrame, min_gap_hours: int = 12) -> pd.DataFrame:
    """Return the first causal observation of each clean-trend episode.

    An episode begins when trend_clean turns on or when its direction changes.
    A 12-hour embargo then prevents overlapping forward labels from being counted
    as independent direction examples.
    """
    z = x.sort_values('ts').copy()
    clean = z.trend_clean.eq(1)
    direction = np.sign(z.fwd_atr.fillna(0)).astype(int)
    prev_clean = clean.shift(1, fill_value=False)
    prev_direction = direction.shift(1, fill_value=0)
    onset = clean & ((~prev_clean) | (direction != prev_direction))
    q = z[onset].copy()
    if q.empty:
        return q
    selected = []
    next_allowed = None
    for idx, row in q.iterrows():
        if next_allowed is None or row.ts >= next_allowed:
            selected.append(idx)
            next_allowed = row.ts + pd.Timedelta(hours=min_gap_hours)
    return q.loc[selected].copy()


def fit_models_event_sampled(
    d: pd.DataFrame,
    purged: bool = True,
    train_start='2023-01-01',
    train_end='2024-01-01',
    val_start='2024-01-01',
    val_end='2025-01-01',
):
    gate_features = v11.v8c.feature_sets()[0]['price_plus_overnight_opening']
    direction_features = v11.d8.sets()['price_location_volume']

    tr = d[(d.ts >= train_start) & (d.ts < train_end)].dropna(subset=['trend_clean']).copy()
    va = d[(d.ts >= val_start) & (d.ts < val_end)].dropna(subset=['trend_clean']).copy()

    # Gate: use all strictly causal rows for fitting so every session phase is represented.
    # Report an independent 12h-spaced validation view in addition to the full validation set.
    gate = v11.hgb_model()
    gate.fit(tr[gate_features], tr.trend_clean.astype(int))
    pv_full = gate.predict_proba(va[gate_features])[:, 1]
    gate_cut = float(np.quantile(pv_full, .90))
    va_ind = v11.spaced_rows(va, 12)
    pv_ind = gate.predict_proba(va_ind[gate_features])[:, 1]
    gate_info = {
        'features': gate_features,
        'model': 'hgb_fixed',
        'sampling': 'all causal training rows; independent 12h-spaced validation reporting',
        'train_n': int(len(tr)),
        'validation_full_n': int(len(va)),
        'validation_independent_n': int(len(va_ind)),
        'validation_full': v11.binary_metrics(va.trend_clean.astype(int), pv_full),
        'validation_independent': v11.binary_metrics(va_ind.trend_clean.astype(int), pv_ind),
        'cut_q90_validation': gate_cut,
    }

    # Direction: train and validate only on independent clean-trend onsets.
    dir_train = independent_trend_onsets(tr, 12)
    va_scored = va.copy()
    va_scored['trend_prob'] = pv_full
    dir_val = independent_trend_onsets(va_scored, 12)
    dir_val = dir_val[dir_val.trend_prob >= gate_cut].copy()
    dir_train['y_long'] = (dir_train.fwd_atr > 0).astype(int)
    dir_val['y_long'] = (dir_val.fwd_atr > 0).astype(int)

    direction = v11.hgb_model()
    direction.fit(dir_train[direction_features], dir_train.y_long)
    pdv = direction.predict_proba(dir_val[direction_features])[:, 1] if len(dir_val) else np.array([])
    dir_info = {
        'features': direction_features,
        'model': 'hgb_fixed',
        'margin_fixed': .20,
        'sampling': 'independent clean-trend onsets with 12h embargo',
        'train_n': int(len(dir_train)),
        'validation_n': int(len(dir_val)),
        'validation': v11.d8.metrics(dir_val.y_long, pdv) if len(dir_val) else {'n': 0},
    }
    return gate, gate_info, direction, dir_info


def run():
    v11.build_events_from_signals = build_events_m15_only
    v11.fit_strict_models = fit_models_event_sampled

    strict = v11.run_protocol(v11.strict_causal_session_features, purged=True)
    _, m15 = v11.v9.make_low_tf(strict['m'])

    neighborhood = v11.parameter_neighborhood(
        strict['events']['validation_2024'], len(strict['signals']['validation_2024']),
        strict['events']['evaluation_2025'], len(strict['signals']['evaluation_2025']),
    )
    robustness = {
        label: v11.robustness_from_trades(strict['trades'][label])
        for label in ['validation_2024', 'evaluation_2025', 'micro_dec2025']
    }
    controls = {
        label: v11.direction_controls(strict['signals'][label], strict['m'], m15)
        for label in ['validation_2024', 'evaluation_2025']
    }
    walk_forward = v11.walk_forward(strict['d'], strict['m'], m15)

    for label, trades in strict['trades'].items():
        v11.trades_frame(trades).to_csv(OUT / f'v11b_{label}_trades.csv', index=False)

    result = {
        'status': 'V11B_COMPLETE',
        'method': {
            'purpose': 'Correct fixed-phase purging from V11 while preserving the strict Opening Range clock fix.',
            'gate_sampling': 'all strictly causal training rows; independent 12h-spaced validation reporting',
            'direction_sampling': 'first observation of each clean-trend episode, 12h embargo',
            'signals': '12h-spaced after gate and direction abstention',
            'fixed_strategy': 'V8c feature block, HGB models, V8d margin 0.20, M15 breakout, V10 trailing stop 1 ATR / activation 1R / distance 1.5R / 12h',
        },
        'data_quality': strict['quality'],
        'gate': strict['gate_info'],
        'direction': strict['direction_info'],
        'periods': strict['periods'],
        'robustness': robustness,
        'direction_controls': controls,
        'parameter_neighborhood': neighborhood,
        'walk_forward': walk_forward,
    }
    (OUT / 'v11b_event_sampling_results.json').write_text(json.dumps(result, indent=2, allow_nan=True), encoding='utf-8')
    pd.DataFrame([
        {'period': label, **data['trade_summary']}
        for label, data in strict['periods'].items()
    ]).to_csv(OUT / 'v11b_period_summary.csv', index=False)

    print('V11B_COMPLETE')
    print(json.dumps({
        'gate': result['gate'],
        'direction': result['direction'],
        'periods': result['periods'],
        'parameter_neighborhood': neighborhood,
        'walk_forward_aggregate': walk_forward['aggregate'],
        'direction_controls': controls,
    }, indent=2, allow_nan=True))


if __name__ == '__main__':
    run()
