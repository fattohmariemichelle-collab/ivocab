from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import nq_trend_detector_v8_real_nq as v8
import nq_trend_detector_v8b_real_nq as v8b

OUT = Path('trend_backtest_v8c_results')
OUT.mkdir(exist_ok=True)

# Preserve V8b causal corrections.
v8.session_features = v8b.causal_session_features
v8.add_labels = v8b.gap_safe_labels


def build_dataset():
    m = v8.load()
    quality = {
        'rows': int(len(m)),
        'start': str(m.ts.min()),
        'end': str(m.ts.max()),
        'duplicates': int(m.ts.duplicated().sum()),
        'zero_volume': int((m.volume <= 0).sum()),
    }
    m = v8.session_features(m)
    h = v8.make_h1(m)
    d = v8.make_context(m, h)
    d = v8.add_h4_d1(d, m)
    d = v8.add_labels(d, 12)
    return d, quality


def feature_sets():
    price = [
        'adx','pdi','mdi','er10','er20','ema_sep_atr','extension_atr',
        'mom6_atr','mom12_atr','rv6','rv24',
        'h4_adx','h4_er10','h4_dir','d1_adx','d1_er10','d1_dir',
    ]
    time_session = ['hour','dow','is_rth','is_opening_hour','is_power_hour','is_overnight']
    volume = ['volume','rel_vol24']
    vwap_prev = ['dist_vwap_atr','dist_prev_high_atr','dist_prev_low_atr']
    overnight_or = [
        'dist_on_high_atr','dist_on_low_atr','dist_or_high_atr','dist_or_low_atr',
        'on_range_atr','or_range_atr','on_vol_ratio','or_vol_ratio',
    ]

    # Standalone additions isolate where the edge comes from.
    sets = {
        'price_technical': price,
        'price_plus_time_session': price + time_session,
        'price_plus_volume': price + volume,
        'price_plus_vwap_prevlevels': price + vwap_prev,
        'price_plus_overnight_opening': price + overnight_or,
        'full': price + time_session + volume + vwap_prev + overnight_or,
    }

    # Cumulative sequence quantifies marginal gain when blocks are added in a fixed order.
    sets.update({
        'cumulative_1_price': price,
        'cumulative_2_time_session': price + time_session,
        'cumulative_3_volume': price + time_session + volume,
        'cumulative_4_vwap_prevlevels': price + time_session + volume + vwap_prev,
        'cumulative_5_overnight_opening_full': price + time_session + volume + vwap_prev + overnight_or,
    })
    groups = {
        'price': price,
        'time_session': time_session,
        'volume': volume,
        'vwap_prevlevels': vwap_prev,
        'overnight_opening': overnight_or,
    }
    return sets, groups


def bootstrap_top10_lift(y, p, n_boot=1000, seed=42):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        base = yy.mean()
        if base <= 0: continue
        cut = np.quantile(pp, .9)
        sel = pp >= cut
        if sel.any(): vals.append(yy[sel].mean() / base)
    if not vals:
        return {'lo': np.nan, 'median': np.nan, 'hi': np.nan}
    vals = np.asarray(vals)
    return {
        'lo': float(np.quantile(vals, .025)),
        'median': float(np.quantile(vals, .5)),
        'hi': float(np.quantile(vals, .975)),
    }


def run_one(d, features, target):
    result, model, te, p = v8.fit_eval(d, features, target)
    result['n_features'] = len(features)
    result['features'] = features
    result['test_top10_lift_bootstrap95'] = bootstrap_top10_lift(te[target].astype(int).to_numpy(), p)
    return result


def main():
    d, quality = build_dataset()
    sets, groups = feature_sets()
    results = {
        'method': {
            'purpose': 'V8c feature-block ablation on the corrected causal V8b pipeline',
            'train': '2023',
            'validation': '2024 model selection by validation average precision',
            'test': '2025-01-01 through 2025-11-30 untouched holdout',
            'label_horizon': '12 H1 bars',
            'trend_clean': 'abs forward move >=0.75 ATR, efficiency >=0.35, adverse excursion <=0.75 ATR',
            'range_persist': 'abs forward move <0.5 ATR, max excursions <1 ATR, efficiency <0.25',
            'causal_fixes': 'V8b overnight, opening range, prior session, session VWAP and >3h gap filter preserved',
        },
        'data_quality': quality,
        'feature_groups': groups,
        'trend_clean': {},
        'range_persist': {},
    }

    for target in ['trend_clean', 'range_persist']:
        for name, feats in sets.items():
            print(f'RUN {target} {name} n_features={len(feats)}', flush=True)
            results[target][name] = run_one(d, feats, target)

    # Compact ranking by untouched 2025 metrics.
    rankings = {}
    for target in ['trend_clean', 'range_persist']:
        rows = []
        for name, r in results[target].items():
            rows.append({
                'feature_set': name,
                'n_features': r['n_features'],
                'test_auc': r['test']['auc'],
                'test_ap': r['test']['ap'],
                'test_top10_rate': r['test_top10']['rate'],
                'test_top10_lift': r['test_top10']['lift'],
                'validation_ap': r['validation']['ap'],
            })
        rankings[target] = sorted(rows, key=lambda z: (z['test_ap'], z['test_auc']), reverse=True)
    results['rankings'] = rankings

    # Marginal deltas along the prespecified cumulative path, not cherry-picked ordering.
    cumulative = ['cumulative_1_price','cumulative_2_time_session','cumulative_3_volume','cumulative_4_vwap_prevlevels','cumulative_5_overnight_opening_full']
    deltas = {}
    for target in ['trend_clean','range_persist']:
        out = []
        prev = None
        for name in cumulative:
            r = results[target][name]
            row = {
                'feature_set': name,
                'test_auc': r['test']['auc'],
                'test_ap': r['test']['ap'],
                'test_top10_lift': r['test_top10']['lift'],
            }
            if prev is not None:
                row['delta_auc'] = row['test_auc'] - prev['test_auc']
                row['delta_ap'] = row['test_ap'] - prev['test_ap']
                row['delta_top10_lift'] = row['test_top10_lift'] - prev['test_top10_lift']
            out.append(row)
            prev = row
        deltas[target] = out
    results['cumulative_deltas'] = deltas

    (OUT / 'v8c_ablation_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame(rankings['trend_clean']).to_csv(OUT / 'trend_ablation_ranking.csv', index=False)
    pd.DataFrame(rankings['range_persist']).to_csv(OUT / 'range_ablation_ranking.csv', index=False)
    pd.DataFrame(deltas['trend_clean']).to_csv(OUT / 'trend_cumulative_deltas.csv', index=False)
    pd.DataFrame(deltas['range_persist']).to_csv(OUT / 'range_cumulative_deltas.csv', index=False)
    print('V8C_COMPLETE')
    print(json.dumps({'rankings': rankings, 'cumulative_deltas': deltas}, indent=2))


if __name__ == '__main__':
    main()
