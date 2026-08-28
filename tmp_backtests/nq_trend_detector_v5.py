from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nq_trend_detector_v3 import (
    D1_URL,
    H1_URL,
    H4_URL,
    compute_market_outcomes,
    load_data,
    merge_context,
    onset_mask,
    periodic_mask,
    prepare,
)

OUT = Path("trend_backtest_v5_results")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)
HORIZONS = [6, 12, 24]
PERIODS = {
    "train_2013_2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-01-01")),
    "validation_2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
    "test_2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
}
STATE_NAMES = {
    0: "transition",
    1: "healthy_up",
    -1: "healthy_down",
    2: "mature_up",
    -2: "mature_down",
    3: "exhausted_up",
    -3: "exhausted_down",
    9: "range",
}


def build_data() -> pd.DataFrame:
    h1 = prepare(load_data(H1_URL), 60, reg_n=24, mom_n=12)
    h4 = prepare(load_data(H4_URL), 240, reg_n=12, mom_n=6)
    d1 = prepare(load_data(D1_URL), 1440, reg_n=10, mom_n=10)
    cols = [
        "close", "low", "high", "atr", "adx", "plus_di", "minus_di", "er10", "er20",
        "ema10", "ema20", "ema50", "ema200", "reg_slope", "reg_r2", "reg_move_atr",
        "momentum", "momentum_atr", "ema_sep_atr", "extension_atr",
    ]
    d = merge_context(h1, h4, "h4_", cols)
    d = merge_context(d, d1, "d1_", cols)
    d["prior_low_10"] = d["low"].shift(1).rolling(10, min_periods=10).min()
    d["prior_low_20"] = d["low"].shift(1).rolling(20, min_periods=20).min()
    d["prior_low_40"] = d["low"].shift(1).rolling(40, min_periods=40).min()
    d["atr_ratio_20"] = d["atr"] / d["atr"].rolling(20, min_periods=20).median().replace(0, np.nan)
    d["year"] = d["time"].dt.year
    d["hour"] = d["time"].dt.hour
    d["atr_bin"] = pd.qcut(d["atr_ratio_20"].rank(method="first"), 5, labels=False, duplicates="drop")
    d["adx_bin"] = pd.cut(d["adx"], bins=[-np.inf, 18, 25, 35, np.inf], labels=False)
    return d.reset_index(drop=True)


def classify_states(d: pd.DataFrame) -> pd.Series:
    h1_dir = np.sign((d["ema20"] - d["ema50"]).fillna(0).to_numpy(float)).astype(int)
    h4_dir = np.sign((d["h4_ema20"] - d["h4_ema50"]).fillna(0).to_numpy(float)).astype(int)
    aligned = np.where((h1_dir == h4_dir) & (h1_dir != 0), h1_dir, 0)
    recent_move = aligned * (d["close"] - d["close"].shift(5)).to_numpy(float) / d["atr"].to_numpy(float)
    positive_recent_move = np.maximum(np.nan_to_num(recent_move, nan=0.0), 0.0)
    maturity = np.maximum(d["extension_atr"].fillna(np.inf).to_numpy(float), positive_recent_move)

    strong = (
        (aligned != 0)
        & (d["er10"].to_numpy(float) >= 0.40)
        & (d["adx"].to_numpy(float) >= 20.0)
        & (d["ema_sep_atr"].to_numpy(float) >= 0.25)
    )
    range_state = (
        (d["er10"].to_numpy(float) < 0.25)
        & (d["adx"].to_numpy(float) < 18.0)
        & (d["ema_sep_atr"].to_numpy(float) < 0.50)
        & (d["h4_adx"].to_numpy(float) < 22.0)
        & (d["h4_ema_sep_atr"].to_numpy(float) < 0.80)
    )

    state = np.zeros(len(d), dtype=int)
    state[range_state] = 9
    healthy = strong & (maturity <= 2.0)
    mature = strong & (maturity > 2.0) & (maturity <= 3.0)
    exhausted = strong & (maturity > 3.0)
    state[healthy] = aligned[healthy]
    state[mature] = 2 * aligned[mature]
    state[exhausted] = 3 * aligned[exhausted]
    return pd.Series(state, index=d.index, name="state")


def state_episode_mask(state: np.ndarray, target: int, horizon: int) -> np.ndarray:
    active = np.asarray(state, dtype=int) == int(target)
    chosen = np.zeros(len(active), dtype=bool)
    next_allowed = 0
    previous = False
    for i, current in enumerate(active):
        if current and not previous and i >= next_allowed:
            chosen[i] = True
            next_allowed = i + horizon + 1
        previous = bool(current)
    return chosen


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


def directional_event_metrics(out: dict[str, np.ndarray], idx: np.ndarray, direction: int) -> dict:
    idx = np.asarray(idx, dtype=int)
    idx = idx[out["valid"][idx]] if len(idx) else idx
    if len(idx) == 0:
        return {"n": 0}
    signed = out["long_ret"][idx] if direction == 1 else -out["long_ret"][idx]
    mfe = out["long_mfe"][idx] if direction == 1 else out["short_mfe"][idx]
    mae = out["long_mae"][idx] if direction == 1 else out["short_mae"][idx]
    first = out["long_first"][idx] if direction == 1 else out["short_first"][idx]
    hit = signed > 0
    material = signed >= 0.5
    clean = material & (out["efficiency"][idx] >= 0.30) & (mae <= 0.75)
    reversal = signed <= -0.5
    lo, hi = wilson(int(hit.sum()), len(idx))
    return {
        "n": int(len(idx)),
        "hit_rate": float(hit.mean()),
        "hit_ci_low": lo,
        "hit_ci_high": hi,
        "material_rate": float(material.mean()),
        "clean_rate": float(clean.mean()),
        "reversal_0_5_rate": float(reversal.mean()),
        "first_pass_rate": float(first.mean()),
        "mean_signed_atr": float(np.mean(signed)),
        "median_signed_atr": float(np.median(signed)),
        "median_mfe_atr": float(np.median(mfe)),
        "median_mae_atr": float(np.median(mae)),
        "median_efficiency": float(np.median(out["efficiency"][idx])),
    }


def range_event_metrics(out: dict[str, np.ndarray], idx: np.ndarray) -> dict:
    idx = np.asarray(idx, dtype=int)
    idx = idx[out["valid"][idx]] if len(idx) else idx
    if len(idx) == 0:
        return {"n": 0}
    endpoint = np.abs(out["long_ret"][idx])
    up = out["long_mfe"][idx]
    down = out["short_mfe"][idx]
    contained_close = endpoint < 0.5
    contained_band = (up < 1.0) & (down < 1.0)
    breakout = (up >= 1.0) | (down >= 1.0)
    return {
        "n": int(len(idx)),
        "median_abs_endpoint_atr": float(np.median(endpoint)),
        "mean_abs_endpoint_atr": float(np.mean(endpoint)),
        "close_within_0_5_rate": float(contained_close.mean()),
        "contained_plusminus_1_rate": float(contained_band.mean()),
        "any_1atr_breakout_rate": float(breakout.mean()),
        "median_two_sided_excursion_atr": float(np.median(np.maximum(up, down))),
        "median_efficiency": float(np.median(out["efficiency"][idx])),
    }


def state_duration_table(d: pd.DataFrame) -> pd.DataFrame:
    state = d["state"].to_numpy(int)
    rows = []
    i = 0
    while i < len(state):
        j = i + 1
        while j < len(state) and state[j] == state[i]:
            j += 1
        rows.append({
            "state_code": int(state[i]),
            "state": STATE_NAMES[int(state[i])],
            "start": d.loc[i, "time"],
            "end": d.loc[j - 1, "time"],
            "duration_bars": int(j - i),
            "duration_hours": float(j - i),
        })
        i = j
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["state_code", "state"], as_index=False).agg(
        episodes=("duration_bars", "size"),
        median_duration_hours=("duration_hours", "median"),
        p25_duration_hours=("duration_hours", lambda s: float(s.quantile(0.25))),
        p75_duration_hours=("duration_hours", lambda s: float(s.quantile(0.75))),
        max_duration_hours=("duration_hours", "max"),
    )
    runs.to_csv(OUT / "state_runs.csv", index=False)
    return summary


def evaluate_states(d: pd.DataFrame, outcomes: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict] = []
    state = d["state"].to_numpy(int)
    for horizon, out in outcomes.items():
        for code, name in STATE_NAMES.items():
            if code == 0:
                continue
            episodes = state_episode_mask(state, code, horizon)
            idx_all = np.flatnonzero(episodes)
            for period, (start, end) in PERIODS.items():
                p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
                idx = idx_all[p_mask[idx_all]]
                base = {
                    "state_code": code,
                    "state": name,
                    "horizon_hours": horizon,
                    "period": period,
                }
                if code == 9:
                    rows.append({**base, **range_event_metrics(out, idx)})
                else:
                    rows.append({**base, **directional_event_metrics(out, idx, 1 if code > 0 else -1)})
    return pd.DataFrame(rows)


def evaluate_short_signal(
    d: pd.DataFrame,
    out: dict[str, np.ndarray],
    signal: np.ndarray,
    horizon: int = 12,
) -> dict[str, dict]:
    episodes = onset_mask(signal, horizon)
    idx_all = np.flatnonzero(episodes & out["valid"] & (signal == -1))
    result = {}
    for period, (start, end) in PERIODS.items():
        p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
        idx = idx_all[p_mask[idx_all]]
        result[period] = directional_event_metrics(out, idx, -1)
    return result


def short_candidate_grid(d: pd.DataFrame, out12: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    h1_bear = (d["ema20"] < d["ema50"]).to_numpy()
    h4_bear = (d["h4_ema20"] < d["h4_ema50"]).to_numpy()
    base = h1_bear & h4_bear
    d1_filters = {
        "none": np.ones(len(d), dtype=bool),
        "d1_close_below_ema20": (d["d1_close"] < d["d1_ema20"]).to_numpy(),
        "d1_ema_bear": (d["d1_ema20"] < d["d1_ema50"]).to_numpy(),
        "d1_momentum_negative": (d["d1_momentum"] < 0).to_numpy(),
    }
    confirmations = {
        "ema_only": np.ones(len(d), dtype=bool),
        "h4_momentum": (d["h4_momentum"] < 0).to_numpy(),
        "regression": (
            (d["reg_slope"] < 0)
            & (d["h4_reg_slope"] < 0)
            & (d["reg_r2"] >= 0.15)
        ).to_numpy(),
        "breakdown10": (d["close"] < d["prior_low_10"]).to_numpy(),
        "breakdown20": (d["close"] < d["prior_low_20"]).to_numpy(),
        "dmi_alignment": (
            (d["minus_di"] > d["plus_di"])
            & (d["h4_minus_di"] > d["h4_plus_di"])
        ).to_numpy(),
    }
    rows = []
    signals: dict[str, np.ndarray] = {}
    for er in [0.20, 0.30, 0.40, 0.50]:
        for adx in [18, 20, 25, 30]:
            for ext in [1.5, 2.0, 2.5, 3.0]:
                quality = (
                    (d["er10"].to_numpy(float) >= er)
                    & (d["adx"].to_numpy(float) >= adx)
                    & (d["extension_atr"].to_numpy(float) <= ext)
                )
                for d1_name, d1_filter in d1_filters.items():
                    for confirmation_name, confirmation in confirmations.items():
                        active = base & quality & d1_filter & confirmation
                        signal = np.where(active, -1, 0).astype(int)
                        name = f"short_er{er:.2f}_adx{adx}_ext{ext:.1f}_{d1_name}_{confirmation_name}"
                        result = evaluate_short_signal(d, out12, signal, 12)
                        train = result["train_2013_2018"]
                        val = result["validation_2019_2020"]
                        test = result["test_2021_2023"]
                        min_mean = min(train.get("mean_signed_atr", -np.inf), val.get("mean_signed_atr", -np.inf))
                        min_hit = min(train.get("hit_rate", -np.inf), val.get("hit_rate", -np.inf))
                        min_first = min(train.get("first_pass_rate", -np.inf), val.get("first_pass_rate", -np.inf))
                        eligible = (
                            train.get("n", 0) >= 70
                            and val.get("n", 0) >= 20
                            and train.get("mean_signed_atr", -np.inf) > 0
                            and val.get("mean_signed_atr", -np.inf) > 0
                        )
                        score = min_mean + 0.50 * (min_hit - 0.50) + 0.25 * (min_first - 1 / 3)
                        row = {
                            "model": name,
                            "er": er,
                            "adx": adx,
                            "extension_cap": ext,
                            "d1_filter": d1_name,
                            "confirmation": confirmation_name,
                            "eligible_pretest": bool(eligible),
                            "selection_score_pretest": float(score),
                        }
                        for prefix, metrics in [("train", train), ("validation", val), ("test", test)]:
                            for key in ["n", "hit_rate", "mean_signed_atr", "median_signed_atr", "material_rate", "clean_rate", "reversal_0_5_rate", "first_pass_rate"]:
                                row[f"{prefix}_{key}"] = metrics.get(key, math.nan)
                        rows.append(row)
                        signals[name] = signal
    grid = pd.DataFrame(rows).sort_values(
        ["eligible_pretest", "selection_score_pretest"], ascending=[False, False]
    )
    return grid, signals


def champion_horizon_metrics(
    d: pd.DataFrame,
    outcomes: dict[int, dict[str, np.ndarray]],
    signal: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for horizon, out in outcomes.items():
        episodes = onset_mask(signal, horizon)
        idx_all = np.flatnonzero(episodes & out["valid"] & (signal == -1))
        for period, (start, end) in PERIODS.items():
            p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
            idx = idx_all[p_mask[idx_all]]
            rows.append({
                "horizon_hours": horizon,
                "period": period,
                **directional_event_metrics(out, idx, -1),
            })
    return pd.DataFrame(rows)


def month_cluster_bootstrap(events: pd.DataFrame, reps: int = 5000) -> dict:
    if len(events) < 20:
        return {"mean_ci_low": math.nan, "mean_ci_high": math.nan, "hit_ci_low": math.nan, "hit_ci_high": math.nan, "p_mean_le_zero": math.nan}
    g = events.groupby("month").agg(sum_return=("signed_return_atr", "sum"), hits=("hit", "sum"), n=("hit", "size"))
    sums = g["sum_return"].to_numpy(float)
    hits = g["hits"].to_numpy(float)
    counts = g["n"].to_numpy(float)
    k = len(g)
    means = np.empty(reps)
    hit_rates = np.empty(reps)
    for b in range(reps):
        draw = RNG.integers(0, k, size=k)
        n = counts[draw].sum()
        means[b] = sums[draw].sum() / n
        hit_rates[b] = hits[draw].sum() / n
    return {
        "mean_ci_low": float(np.quantile(means, 0.025)),
        "mean_ci_high": float(np.quantile(means, 0.975)),
        "hit_ci_low": float(np.quantile(hit_rates, 0.025)),
        "hit_ci_high": float(np.quantile(hit_rates, 0.975)),
        "p_mean_le_zero": float((np.sum(means <= 0) + 1) / (reps + 1)),
    }


def make_direction_events(
    d: pd.DataFrame,
    out: dict[str, np.ndarray],
    idx: np.ndarray,
    direction: int,
) -> pd.DataFrame:
    idx = np.asarray(idx, dtype=int)
    idx = idx[out["valid"][idx]] if len(idx) else idx
    signed = out["long_ret"][idx] if direction == 1 else -out["long_ret"][idx]
    return pd.DataFrame({
        "index": idx,
        "time": d.loc[idx, "time"].to_numpy(),
        "year": d.loc[idx, "year"].to_numpy(),
        "hour": d.loc[idx, "hour"].to_numpy(),
        "atr_bin": d.loc[idx, "atr_bin"].fillna(-1).astype(int).to_numpy(),
        "adx_bin": d.loc[idx, "adx_bin"].fillna(-1).astype(int).to_numpy(),
        "direction": direction,
        "signed_return_atr": signed,
        "hit": signed > 0,
        "month": d.loc[idx, "time"].dt.to_period("M").astype(str).to_numpy(),
    })


def matched_regime_null(
    d: pd.DataFrame,
    out: dict[str, np.ndarray],
    events: pd.DataFrame,
    horizon: int,
    reps: int = 3000,
) -> dict:
    if len(events) < 20:
        return {"observed_mean": math.nan, "null_mean": math.nan, "null_ci_low": math.nan, "null_ci_high": math.nan, "p_selected_not_better": math.nan}
    candidate_mask = periodic_mask(out["valid"], horizon)
    candidate_idx = np.flatnonzero(candidate_mask)
    meta = pd.DataFrame({
        "idx": candidate_idx,
        "year": d.loc[candidate_idx, "year"].to_numpy(),
        "hour": d.loc[candidate_idx, "hour"].to_numpy(),
        "atr_bin": d.loc[candidate_idx, "atr_bin"].fillna(-1).astype(int).to_numpy(),
        "adx_bin": d.loc[candidate_idx, "adx_bin"].fillna(-1).astype(int).to_numpy(),
    })
    pools_exact: dict[tuple, np.ndarray] = {
        key: group["idx"].to_numpy(int)
        for key, group in meta.groupby(["year", "hour", "atr_bin", "adx_bin"])
    }
    pools_year_hour: dict[tuple, np.ndarray] = {
        key: group["idx"].to_numpy(int)
        for key, group in meta.groupby(["year", "hour"])
    }
    pools_year: dict[int, np.ndarray] = {
        int(key): group["idx"].to_numpy(int)
        for key, group in meta.groupby("year")
    }
    grouped = events.groupby(["year", "hour", "atr_bin", "adx_bin", "direction"]).size().reset_index(name="count")
    null_means = np.empty(reps)
    for b in range(reps):
        total = 0.0
        count_total = 0
        for row in grouped.itertuples(index=False):
            key = (int(row.year), int(row.hour), int(row.atr_bin), int(row.adx_bin))
            pool = pools_exact.get(key)
            if pool is None or len(pool) < 3:
                pool = pools_year_hour.get((int(row.year), int(row.hour)))
            if pool is None or len(pool) < 3:
                pool = pools_year.get(int(row.year))
            if pool is None or len(pool) == 0:
                continue
            draw = RNG.choice(pool, size=int(row.count), replace=len(pool) < int(row.count))
            values = out["long_ret"][draw] if int(row.direction) == 1 else -out["long_ret"][draw]
            total += float(np.nansum(values))
            count_total += int(np.isfinite(values).sum())
        null_means[b] = total / count_total if count_total else np.nan
    null_means = null_means[np.isfinite(null_means)]
    observed = float(events["signed_return_atr"].mean())
    return {
        "observed_mean": observed,
        "null_mean": float(np.mean(null_means)),
        "null_ci_low": float(np.quantile(null_means, 0.025)),
        "null_ci_high": float(np.quantile(null_means, 0.975)),
        "p_selected_not_better": float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1)),
    }


def baseline_tests(
    d: pd.DataFrame,
    out12: dict[str, np.ndarray],
    champion_signal: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = d["state"].to_numpy(int)
    healthy_up_idx = np.flatnonzero(state_episode_mask(state, 1, 12))
    healthy_down_idx = np.flatnonzero(state_episode_mask(state, -1, 12))
    champion_idx = np.flatnonzero(onset_mask(champion_signal, 12) & (champion_signal == -1))
    event_sets = {
        "healthy_up": make_direction_events(d, out12, healthy_up_idx, 1),
        "healthy_down": make_direction_events(d, out12, healthy_down_idx, -1),
        "short_champion": make_direction_events(d, out12, champion_idx, -1),
    }
    boot_rows = []
    null_rows = []
    for name, events in event_sets.items():
        for period, (start, end) in PERIODS.items():
            x = events[(events["time"] >= start) & (events["time"] < end)].copy()
            if period not in ["full", "test_2021_2023"]:
                continue
            boot_rows.append({"model": name, "period": period, "n": len(x), **month_cluster_bootstrap(x)})
            null_rows.append({"model": name, "period": period, "n": len(x), **matched_regime_null(d, out12, x, 12)})
    return pd.DataFrame(boot_rows), pd.DataFrame(null_rows)


def state_transition_matrix(d: pd.DataFrame, ahead: int = 12) -> pd.DataFrame:
    state = d["state"].to_numpy(int)
    rows = []
    for code, name in STATE_NAMES.items():
        idx = np.flatnonzero(state[:-ahead] == code)
        if len(idx) == 0:
            continue
        future = state[idx + ahead]
        same = future == code
        same_direction = np.sign(future) == np.sign(code) if code not in [0, 9] else np.zeros(len(idx), dtype=bool)
        rows.append({
            "state_code": code,
            "state": name,
            "ahead_hours": ahead,
            "n": len(idx),
            "same_state_rate": float(same.mean()),
            "same_directional_family_rate": float(same_direction.mean()) if code not in [0, 9] else math.nan,
            "to_transition_rate": float((future == 0).mean()),
            "to_range_rate": float((future == 9).mean()),
            "to_opposite_direction_rate": float((np.sign(future) == -np.sign(code)).mean()) if code not in [0, 9] else math.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = build_data()
    d["state"] = classify_states(d)
    outcomes = {h: compute_market_outcomes(d, h, 3 * 3600) for h in HORIZONS}

    state_metrics = evaluate_states(d, outcomes)
    state_metrics.to_csv(OUT / "state_metrics.csv", index=False)
    duration_summary = state_duration_table(d)
    duration_summary.to_csv(OUT / "state_duration_summary.csv", index=False)
    transitions = pd.concat([state_transition_matrix(d, 6), state_transition_matrix(d, 12), state_transition_matrix(d, 24)], ignore_index=True)
    transitions.to_csv(OUT / "state_transitions.csv", index=False)
    d[["time", "close", "atr", "adx", "er10", "ema_sep_atr", "extension_atr", "state"]].to_csv(OUT / "classified_h1.csv", index=False)

    short_grid, short_signals = short_candidate_grid(d, outcomes[12])
    short_grid.to_csv(OUT / "short_candidate_grid.csv", index=False)
    eligible = short_grid[short_grid["eligible_pretest"]].copy()
    if len(eligible):
        champion_name = str(eligible.iloc[0]["model"])
    else:
        champion_name = str(short_grid.iloc[0]["model"])
    champion_signal = short_signals[champion_name]
    champion_metrics = champion_horizon_metrics(d, outcomes, champion_signal)
    champion_metrics.to_csv(OUT / "short_champion_horizons.csv", index=False)
    short_grid.head(30).to_csv(OUT / "short_top30_pretest.csv", index=False)

    boot, null = baseline_tests(d, outcomes[12], champion_signal)
    boot.to_csv(OUT / "block_bootstrap.csv", index=False)
    null.to_csv(OUT / "matched_regime_null.csv", index=False)

    key_state = state_metrics[
        (state_metrics["horizon_hours"] == 12)
        & (state_metrics["period"].isin(["train_2013_2018", "validation_2019_2020", "test_2021_2023", "full"]))
    ].copy()
    key_state.to_csv(OUT / "key_state_metrics_12h.csv", index=False)

    champion_row = json.loads(short_grid[short_grid["model"] == champion_name].iloc[0].to_json())
    summary = {
        "status": "BACKTEST_V5_COMPLETE",
        "data": {
            "rows": int(len(d)),
            "start": str(d["time"].min()),
            "end": str(d["time"].max()),
            "instrument": "USATECHIDXUSD Nasdaq-100 proxy; not CME NQ",
        },
        "state_definition": {
            "trend_strength": "H1/H4 EMA20-EMA50 aligned, ER10>=0.40, ADX>=20, EMA separation>=0.25 ATR",
            "healthy": "max(H1 extension from EMA20, positive 5h directional move)<=2 ATR",
            "mature": "maturity >2 and <=3 ATR",
            "exhausted": "maturity >3 ATR",
            "range": "ER10<0.25, ADX<18, H1 EMA separation<0.50 ATR, H4 ADX<22, H4 EMA separation<0.80 ATR",
        },
        "short_selection": {
            "rule": "Selected on 2013-2018 train and 2019-2020 validation only; 2021-2023 test not used in ranking.",
            "eligible_candidates": int(len(eligible)),
            "champion": champion_row,
        },
        "baseline_matching": "Year + hour + ATR-regime quintile + ADX bin, with progressively broader fallback pools.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print("BACKTEST_V5_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("\nSTATE METRICS — 12H")
    print(key_state.to_string(index=False))
    print("\nSHORT CHAMPION — ALL HORIZONS")
    print(champion_metrics.to_string(index=False))
    print("\nBOOTSTRAP")
    print(boot.to_string(index=False))
    print("\nMATCHED REGIME NULL")
    print(null.to_string(index=False))


if __name__ == "__main__":
    main()
