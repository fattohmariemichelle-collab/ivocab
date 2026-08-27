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
    aligned_direction,
    compute_market_outcomes,
    load_data,
    merge_context,
    onset_mask,
    periodic_mask,
    prepare,
    signal_stats,
    summarize,
)

OUT = Path("trend_backtest_v4_results")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260827)
HORIZONS = [6, 12, 24]
PERIODS = {
    "train_2013_2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-01-01")),
    "validation_2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
    "test_2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
}


def build_data() -> pd.DataFrame:
    h1 = prepare(load_data(H1_URL), 60, reg_n=24, mom_n=12)
    h4 = prepare(load_data(H4_URL), 240, reg_n=12, mom_n=6)
    d1 = prepare(load_data(D1_URL), 1440, reg_n=10, mom_n=10)
    cols = [
        "close", "atr", "adx", "plus_di", "minus_di", "er10", "er20", "ema10", "ema20",
        "ema50", "ema200", "reg_slope", "reg_r2", "reg_move_atr", "momentum", "momentum_atr",
        "ema_sep_atr", "extension_atr",
    ]
    d = merge_context(h1, h4, "h4_", cols)
    d = merge_context(d, d1, "d1_", cols)
    return d.reset_index(drop=True)


def raw_signal(d: pd.DataFrame, er: float, adx: float, extension: float, use_d1: bool) -> np.ndarray:
    h1_dir = d["ema20"] - d["ema50"]
    h4_dir = d["h4_ema20"] - d["h4_ema50"]
    d1_dir = d["d1_ema20"] - d["d1_ema50"]
    direction = aligned_direction(h1_dir, h4_dir, d1_dir if use_d1 else None)
    quality = (
        (d["er10"].to_numpy(float) >= er)
        & (d["adx"].to_numpy(float) >= adx)
        & (d["extension_atr"].to_numpy(float) <= extension)
    )
    return np.where(quality, direction, 0).astype(int)


def confirmed_signal(signal: np.ndarray, bars: int = 2) -> np.ndarray:
    s = np.asarray(signal, dtype=int)
    out = np.zeros(len(s), dtype=int)
    for i in range(bars - 1, len(s)):
        window = s[i - bars + 1 : i + 1]
        if window[-1] != 0 and np.all(window == window[-1]):
            out[i] = window[-1]
    return out


def hysteresis_signal(
    d: pd.DataFrame,
    use_d1: bool,
    confirm_bars: int = 1,
    enter_er: float = 0.40,
    enter_adx: float = 20.0,
    enter_ext: float = 2.0,
    stay_er: float = 0.25,
    stay_adx: float = 17.0,
    stay_ext: float = 3.0,
    exit_fail_bars: int = 2,
) -> np.ndarray:
    h1_dir = np.sign((d["ema20"] - d["ema50"]).fillna(0).to_numpy(float)).astype(int)
    h4_dir = np.sign((d["h4_ema20"] - d["h4_ema50"]).fillna(0).to_numpy(float)).astype(int)
    d1_dir = np.sign((d["d1_ema20"] - d["d1_ema50"]).fillna(0).to_numpy(float)).astype(int)
    aligned = np.where((h1_dir == h4_dir) & (h1_dir != 0), h1_dir, 0)
    if use_d1:
        aligned = np.where((aligned == d1_dir) & (aligned != 0), aligned, 0)
    er = d["er10"].to_numpy(float)
    adx = d["adx"].to_numpy(float)
    ext = d["extension_atr"].to_numpy(float)
    enter_ok = (er >= enter_er) & (adx >= enter_adx) & (ext <= enter_ext) & (aligned != 0)
    stay_ok = (er >= stay_er) & (adx >= stay_adx) & (ext <= stay_ext) & (aligned != 0)
    out = np.zeros(len(d), dtype=int)
    state = 0
    enter_count = 0
    fail_count = 0
    candidate = 0
    for i in range(len(d)):
        if state == 0:
            if enter_ok[i]:
                if aligned[i] == candidate:
                    enter_count += 1
                else:
                    candidate = int(aligned[i])
                    enter_count = 1
                if enter_count >= confirm_bars:
                    state = candidate
                    fail_count = 0
            else:
                candidate = 0
                enter_count = 0
        else:
            same_direction = aligned[i] == state
            if same_direction and stay_ok[i]:
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= exit_fail_bars:
                    state = 0
                    fail_count = 0
                    candidate = 0
                    enter_count = 0
        out[i] = state
    return out


def state_noise(signal: np.ndarray) -> dict:
    s = np.asarray(signal, dtype=int)
    episodes = []
    reversals_6 = 0
    reentries_6 = 0
    i = 0
    while i < len(s):
        if s[i] == 0:
            i += 1
            continue
        direction = s[i]
        j = i + 1
        while j < len(s) and s[j] == direction:
            j += 1
        episodes.append((i, j, direction))
        future = s[j : min(len(s), j + 6)]
        if np.any(future == -direction):
            reversals_6 += 1
        if np.any(future == direction):
            reentries_6 += 1
        i = j
    durations = np.array([j - i for i, j, _ in episodes], dtype=float)
    return {
        "episode_count": int(len(episodes)),
        "median_duration_h": float(np.median(durations)) if len(durations) else math.nan,
        "p25_duration_h": float(np.quantile(durations, 0.25)) if len(durations) else math.nan,
        "p75_duration_h": float(np.quantile(durations, 0.75)) if len(durations) else math.nan,
        "one_bar_episode_rate": float(np.mean(durations == 1)) if len(durations) else math.nan,
        "reverse_within_6h_rate": float(reversals_6 / len(episodes)) if episodes else math.nan,
        "same_direction_reentry_6h_rate": float(reentries_6 / len(episodes)) if episodes else math.nan,
    }


def event_frame(d: pd.DataFrame, out: dict[str, np.ndarray], signal: np.ndarray, horizon: int) -> pd.DataFrame:
    episode = onset_mask(signal, horizon)
    idx = np.flatnonzero(episode & out["valid"] & (signal != 0))
    direction = signal[idx]
    signed = np.where(direction == 1, out["long_ret"][idx], -out["long_ret"][idx])
    mfe = np.where(direction == 1, out["long_mfe"][idx], out["short_mfe"][idx])
    mae = np.where(direction == 1, out["long_mae"][idx], out["short_mae"][idx])
    first = np.where(direction == 1, out["long_first"][idx], out["short_first"][idx])
    f = pd.DataFrame({
        "index": idx,
        "time": d.loc[idx, "time"].to_numpy(),
        "direction": direction,
        "signed_return_atr": signed,
        "mfe_atr": mfe,
        "mae_atr": mae,
        "efficiency": out["efficiency"][idx],
        "first_pass": first,
    })
    f["hit"] = f["signed_return_atr"] > 0
    f["material"] = f["signed_return_atr"] >= 0.5
    f["clean"] = f["material"] & (f["efficiency"] >= 0.30) & (f["mae_atr"] <= 0.75)
    f["year"] = pd.to_datetime(f["time"]).dt.year
    f["month"] = pd.to_datetime(f["time"]).dt.to_period("M").astype(str)
    return f


def cluster_bootstrap(events: pd.DataFrame, reps: int = 5000) -> dict:
    if len(events) < 20:
        return {"mean_ci_low": math.nan, "mean_ci_high": math.nan, "hit_ci_low": math.nan, "hit_ci_high": math.nan, "p_mean_le_zero": math.nan}
    grouped = events.groupby("month").agg(
        sum_return=("signed_return_atr", "sum"),
        hits=("hit", "sum"),
        n=("hit", "size"),
    ).reset_index(drop=True)
    k = len(grouped)
    boot_mean = np.empty(reps)
    boot_hit = np.empty(reps)
    sums = grouped["sum_return"].to_numpy(float)
    hits = grouped["hits"].to_numpy(float)
    counts = grouped["n"].to_numpy(float)
    for b in range(reps):
        draw = RNG.integers(0, k, size=k)
        total_n = counts[draw].sum()
        boot_mean[b] = sums[draw].sum() / total_n
        boot_hit[b] = hits[draw].sum() / total_n
    return {
        "mean_ci_low": float(np.quantile(boot_mean, 0.025)),
        "mean_ci_high": float(np.quantile(boot_mean, 0.975)),
        "hit_ci_low": float(np.quantile(boot_hit, 0.025)),
        "hit_ci_high": float(np.quantile(boot_hit, 0.975)),
        "p_mean_le_zero": float((np.sum(boot_mean <= 0) + 1) / (reps + 1)),
    }


def matched_null(
    d: pd.DataFrame,
    out: dict[str, np.ndarray],
    events: pd.DataFrame,
    horizon: int,
    reps: int = 3000,
) -> dict:
    if len(events) < 20:
        return {"null_mean": math.nan, "null_ci_low": math.nan, "null_ci_high": math.nan, "p_selected_not_better": math.nan, "observed_mean": math.nan}
    candidate_mask = periodic_mask(out["valid"], horizon)
    candidate_idx = np.flatnonzero(candidate_mask)
    candidate_year = pd.to_datetime(d.loc[candidate_idx, "time"]).dt.year.to_numpy()
    long_ret = out["long_ret"]
    observed = float(events["signed_return_atr"].mean())
    null_means = np.empty(reps)
    counts = events.groupby(["year", "direction"]).size().to_dict()
    for b in range(reps):
        vals = []
        for (year, direction), count in counts.items():
            pool = candidate_idx[candidate_year == year]
            if len(pool) == 0:
                continue
            draw = RNG.choice(pool, size=int(count), replace=len(pool) < count)
            vals.extend((long_ret[draw] if direction == 1 else -long_ret[draw]).tolist())
        null_means[b] = np.nanmean(vals) if vals else np.nan
    null_means = null_means[np.isfinite(null_means)]
    return {
        "null_mean": float(np.mean(null_means)),
        "null_ci_low": float(np.quantile(null_means, 0.025)),
        "null_ci_high": float(np.quantile(null_means, 0.975)),
        "p_selected_not_better": float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1)),
        "observed_mean": observed,
    }


def main() -> None:
    d = build_data()
    outcomes = {h: compute_market_outcomes(d, h, 3 * 3600) for h in HORIZONS}

    models: dict[str, tuple[np.ndarray, dict]] = {}
    for use_d1 in [False, True]:
        suffix = "_d1" if use_d1 else ""
        models[f"raw_er040_adx20_ext2{suffix}"] = (
            raw_signal(d, 0.40, 20, 2.0, use_d1),
            {"type": "raw", "er": 0.40, "adx": 20, "extension": 2.0, "d1": use_d1},
        )
        models[f"confirm2_er040_adx20_ext2{suffix}"] = (
            confirmed_signal(raw_signal(d, 0.40, 20, 2.0, use_d1), 2),
            {"type": "confirm2", "er": 0.40, "adx": 20, "extension": 2.0, "d1": use_d1},
        )
        models[f"hysteresis_er040_adx20_ext2{suffix}"] = (
            hysteresis_signal(d, use_d1, confirm_bars=1),
            {"type": "hysteresis", "d1": use_d1, "enter": [0.40, 20, 2.0], "stay": [0.25, 17, 3.0], "exit_fail_bars": 2},
        )
        models[f"hysteresis_confirm2_er040_adx20_ext2{suffix}"] = (
            hysteresis_signal(d, use_d1, confirm_bars=2),
            {"type": "hysteresis_confirm2", "d1": use_d1, "enter": [0.40, 20, 2.0], "stay": [0.25, 17, 3.0], "exit_fail_bars": 2},
        )

    # Parameter neighbourhood: fixed before reading V4 results, used only as robustness map.
    for use_d1 in [False, True]:
        for er in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for adx in [18, 20, 22, 25]:
                for ext in [1.5, 2.0, 2.5]:
                    name = f"grid_er{er:.2f}_adx{adx}_ext{ext:.1f}{'_d1' if use_d1 else ''}"
                    models[name] = (
                        raw_signal(d, er, adx, ext, use_d1),
                        {"type": "grid_raw", "er": er, "adx": adx, "extension": ext, "d1": use_d1},
                    )

    metrics_rows = []
    year_rows = []
    bootstrap_rows = []
    null_rows = []
    noise_rows = []
    selected_events = []
    total_years = max((d["time"].max() - d["time"].min()).days / 365.25, 0.01)

    for name, (signal, params) in models.items():
        noise_rows.append({"model": name, "params": json.dumps(params, sort_keys=True), **state_noise(signal), **signal_stats(signal, 60, total_years)})
        for horizon, out in outcomes.items():
            events = event_frame(d, out, signal, horizon)
            if name.startswith(("raw_er040", "confirm2_er040", "hysteresis_er040", "hysteresis_confirm2_er040")):
                e = events.copy()
                e["model"] = name
                e["horizon_hours"] = horizon
                selected_events.append(e)
            for period, (start, end) in PERIODS.items():
                p = events[(events["time"] >= start) & (events["time"] < end)]
                for direction_label, direction in [("combined", 0), ("long", 1), ("short", -1)]:
                    q = p if direction == 0 else p[p["direction"] == direction]
                    if len(q):
                        hit_low, hit_high = _wilson(int(q["hit"].sum()), len(q))
                        row = {
                            "model": name,
                            "params": json.dumps(params, sort_keys=True),
                            "horizon_hours": horizon,
                            "period": period,
                            "direction": direction_label,
                            "n": int(len(q)),
                            "hit_rate": float(q["hit"].mean()),
                            "hit_ci_low": hit_low,
                            "hit_ci_high": hit_high,
                            "material_rate": float(q["material"].mean()),
                            "clean_rate": float(q["clean"].mean()),
                            "first_pass_rate": float(q["first_pass"].mean()),
                            "mean_signed_atr": float(q["signed_return_atr"].mean()),
                            "median_signed_atr": float(q["signed_return_atr"].median()),
                            "median_mfe_atr": float(q["mfe_atr"].median()),
                            "median_mae_atr": float(q["mae_atr"].median()),
                            "median_efficiency": float(q["efficiency"].median()),
                        }
                    else:
                        row = {"model": name, "params": json.dumps(params, sort_keys=True), "horizon_hours": horizon, "period": period, "direction": direction_label, "n": 0}
                    metrics_rows.append(row)
                    if name.startswith(("raw_er040", "confirm2_er040", "hysteresis_er040", "hysteresis_confirm2_er040")) and period in ["full", "test_2021_2023"]:
                        bootstrap_rows.append({
                            "model": name,
                            "horizon_hours": horizon,
                            "period": period,
                            "direction": direction_label,
                            "n": int(len(q)),
                            **cluster_bootstrap(q),
                        })
                        null_rows.append({
                            "model": name,
                            "horizon_hours": horizon,
                            "period": period,
                            "direction": direction_label,
                            "n": int(len(q)),
                            **matched_null(d, out, q, horizon),
                        })
            if name in ["raw_er040_adx20_ext2", "raw_er040_adx20_ext2_d1", "hysteresis_confirm2_er040_adx20_ext2", "hysteresis_confirm2_er040_adx20_ext2_d1"]:
                for year, q in events.groupby("year"):
                    for direction_label, direction in [("combined", 0), ("long", 1), ("short", -1)]:
                        z = q if direction == 0 else q[q["direction"] == direction]
                        if not len(z):
                            continue
                        year_rows.append({
                            "model": name,
                            "horizon_hours": horizon,
                            "year": int(year),
                            "direction": direction_label,
                            "n": int(len(z)),
                            "hit_rate": float(z["hit"].mean()),
                            "mean_signed_atr": float(z["signed_return_atr"].mean()),
                            "median_signed_atr": float(z["signed_return_atr"].median()),
                            "first_pass_rate": float(z["first_pass"].mean()),
                        })

    metrics = pd.DataFrame(metrics_rows)
    noise = pd.DataFrame(noise_rows)
    boot = pd.DataFrame(bootstrap_rows)
    null = pd.DataFrame(null_rows)
    years = pd.DataFrame(year_rows)
    events_out = pd.concat(selected_events, ignore_index=True) if selected_events else pd.DataFrame()

    metrics.to_csv(OUT / "episode_metrics.csv", index=False)
    noise.to_csv(OUT / "state_noise.csv", index=False)
    boot.to_csv(OUT / "cluster_bootstrap.csv", index=False)
    null.to_csv(OUT / "matched_null_tests.csv", index=False)
    years.to_csv(OUT / "year_by_year.csv", index=False)
    events_out.to_csv(OUT / "selected_model_events.csv", index=False)

    # Robustness map for 12h test, with no post-test optimisation claim.
    test_grid = metrics[
        metrics["model"].str.startswith("grid_")
        & (metrics["horizon_hours"] == 12)
        & (metrics["period"] == "test_2021_2023")
    ].copy()
    test_grid.to_csv(OUT / "test_parameter_grid_12h.csv", index=False)
    robustness = []
    for d1_flag in [False, True]:
        names = [n for n, (_, p) in models.items() if p.get("type") == "grid_raw" and p.get("d1") == d1_flag]
        subset = metrics[
            metrics["model"].isin(names)
            & (metrics["horizon_hours"] == 12)
            & (metrics["direction"] == "combined")
            & (metrics["period"].isin(["train_2013_2018", "validation_2019_2020", "test_2021_2023"]))
        ]
        pivot = subset.pivot(index="model", columns="period", values=["mean_signed_atr", "hit_rate", "n"])
        positive_all = 0
        test_positive = 0
        total = len(pivot)
        for _, r in pivot.iterrows():
            vals = [r.get(("mean_signed_atr", p), np.nan) for p in ["train_2013_2018", "validation_2019_2020", "test_2021_2023"]]
            if np.isfinite(vals).all() and all(v > 0 for v in vals):
                positive_all += 1
            if np.isfinite(vals[-1]) and vals[-1] > 0:
                test_positive += 1
        robustness.append({
            "d1_required": d1_flag,
            "grid_models": total,
            "positive_mean_all_three_periods": positive_all,
            "positive_mean_all_three_rate": positive_all / total if total else math.nan,
            "positive_mean_test": test_positive,
            "positive_mean_test_rate": test_positive / total if total else math.nan,
        })
    pd.DataFrame(robustness).to_csv(OUT / "grid_robustness_summary.csv", index=False)

    key_names = [
        "raw_er040_adx20_ext2",
        "raw_er040_adx20_ext2_d1",
        "confirm2_er040_adx20_ext2",
        "confirm2_er040_adx20_ext2_d1",
        "hysteresis_er040_adx20_ext2",
        "hysteresis_er040_adx20_ext2_d1",
        "hysteresis_confirm2_er040_adx20_ext2",
        "hysteresis_confirm2_er040_adx20_ext2_d1",
    ]
    key = metrics[
        metrics["model"].isin(key_names)
        & (metrics["horizon_hours"].isin([6, 12, 24]))
        & (metrics["period"].isin(["train_2013_2018", "validation_2019_2020", "test_2021_2023", "full"]))
    ].copy()
    key.to_csv(OUT / "key_model_metrics.csv", index=False)

    summary = {
        "status": "BACKTEST_V4_COMPLETE",
        "data_start": str(d["time"].min()),
        "data_end": str(d["time"].max()),
        "rows": int(len(d)),
        "horizons_hours": HORIZONS,
        "key_models": key_names,
        "robustness": robustness,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("BACKTEST_V4_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("\nKEY 12H TEST METRICS")
    print(key[(key.horizon_hours == 12) & (key.period == "test_2021_2023")].to_string(index=False))
    print("\nSTATE NOISE")
    print(noise[noise.model.isin(key_names)].to_string(index=False))
    print("\nBOOTSTRAP 12H TEST")
    print(boot[(boot.horizon_hours == 12) & (boot.period == "test_2021_2023")].to_string(index=False))
    print("\nMATCHED NULL 12H TEST")
    print(null[(null.horizon_hours == 12) & (null.period == "test_2021_2023")].to_string(index=False))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


if __name__ == "__main__":
    main()
