from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

M15_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_M15.csv"
H1_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_H1.csv"
H4_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_H4.csv"
D1_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_D1.csv"

OUT = Path("trend_backtest_v3_results")
OUT.mkdir(exist_ok=True)

PERIODS = {
    "train_2013_2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-01-01")),
    "validation_2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
    "test_2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
}


def load_data(url: str) -> pd.DataFrame:
    d = pd.read_csv(url, sep="\t", parse_dates=["Time"])
    d.columns = [str(c).lower() for c in d.columns]
    needed = ["time", "open", "high", "low", "close", "volume"]
    d = d[needed].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return (
        d.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rolling_regression(close: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling log-price OLS slope and R², vectorised and causal."""
    y = np.log(close.to_numpy(float))
    m = len(y)
    slope = np.full(m, np.nan)
    r2 = np.full(m, np.nan)
    if m < n:
        return pd.Series(slope, index=close.index), pd.Series(r2, index=close.index)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    ones = np.ones(n, dtype=float)
    sum_y = np.convolve(y, ones, mode="valid")
    sum_y2 = np.convolve(y * y, ones, mode="valid")
    sum_xy = np.convolve(y, x[::-1], mode="valid")
    sxy = sum_xy - x_mean * sum_y
    syy = np.maximum(sum_y2 - (sum_y * sum_y) / n, 0.0)
    sl = sxy / sxx
    rr = np.divide(sxy * sxy, sxx * syy, out=np.zeros_like(sxy), where=syy > 1e-18)
    slope[n - 1 :] = sl
    r2[n - 1 :] = np.clip(rr, 0.0, 1.0)
    return pd.Series(slope, index=close.index), pd.Series(r2, index=close.index)


def prepare(d: pd.DataFrame, minutes: int, reg_n: int, mom_n: int) -> pd.DataFrame:
    x = d.copy()
    prev = x["close"].shift(1)
    tr = pd.concat(
        [x["high"] - x["low"], (x["high"] - prev).abs(), (x["low"] - prev).abs()], axis=1
    ).max(axis=1)
    x["atr"] = rma(tr, 14)
    up = x["high"].diff()
    down = -x["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atr_di = rma(tr, 14).replace(0, np.nan)
    x["plus_di"] = 100.0 * rma(plus_dm, 14) / atr_di
    x["minus_di"] = 100.0 * rma(minus_dm, 14) / atr_di
    denom = (x["plus_di"] + x["minus_di"]).replace(0, np.nan)
    dx = 100.0 * (x["plus_di"] - x["minus_di"]).abs() / denom
    x["adx"] = rma(dx, 14)
    for n in [10, 20]:
        path = x["close"].diff().abs().rolling(n, min_periods=n).sum()
        x[f"er{n}"] = (x["close"] - x["close"].shift(n)).abs() / path.replace(0, np.nan)
    for n in [10, 20, 50, 200]:
        x[f"ema{n}"] = x["close"].ewm(span=n, adjust=False, min_periods=n).mean()
    x["reg_slope"], x["reg_r2"] = rolling_regression(x["close"], reg_n)
    x["reg_move_atr"] = (x["reg_slope"].abs() * (reg_n - 1) * x["close"]) / x["atr"].replace(0, np.nan)
    x["momentum"] = x["close"] - x["close"].shift(mom_n)
    x["momentum_atr"] = x["momentum"].abs() / x["atr"].replace(0, np.nan)
    x["ema_sep_atr"] = (x["ema20"] - x["ema50"]).abs() / x["atr"].replace(0, np.nan)
    x["extension_atr"] = (x["close"] - x["ema20"]).abs() / x["atr"].replace(0, np.nan)
    x["event_time"] = x["time"] + pd.Timedelta(minutes=minutes)
    return x


def merge_context(base: pd.DataFrame, ctx: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
    right = ctx[["event_time"] + cols].copy().rename(columns={c: f"{prefix}{c}" for c in cols})
    return pd.merge_asof(
        base.sort_values("event_time"), right.sort_values("event_time"), on="event_time", direction="backward"
    )


def compute_market_outcomes(d: pd.DataFrame, horizon: int, max_gap_seconds: int) -> dict[str, np.ndarray]:
    n = len(d)
    close = d["close"].to_numpy(float)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    times = d["event_time"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    valid = np.zeros(n, dtype=bool)
    long_ret = np.full(n, np.nan)
    long_mfe = np.full(n, np.nan)
    long_mae = np.full(n, np.nan)
    short_mfe = np.full(n, np.nan)
    short_mae = np.full(n, np.nan)
    efficiency = np.full(n, np.nan)
    long_first = np.zeros(n, dtype=bool)
    short_first = np.zeros(n, dtype=bool)
    for i in range(n - horizon):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if np.max(np.diff(times[i : i + horizon + 1])) > max_gap_seconds:
            continue
        c0, a = close[i], atr[i]
        hs = high[i + 1 : i + horizon + 1]
        ls = low[i + 1 : i + horizon + 1]
        long_ret[i] = (close[i + horizon] - c0) / a
        long_mfe[i] = (np.max(hs) - c0) / a
        long_mae[i] = (c0 - np.min(ls)) / a
        short_mfe[i] = (c0 - np.min(ls)) / a
        short_mae[i] = (np.max(hs) - c0) / a
        path = np.sum(np.abs(np.diff(close[i : i + horizon + 1])))
        efficiency[i] = abs(close[i + horizon] - c0) / path if path > 0 else 0.0
        for direction in (1, -1):
            passed = False
            for j in range(i + 1, i + horizon + 1):
                if direction == 1:
                    fav = high[j] >= c0 + a
                    adv = low[j] <= c0 - 0.5 * a
                else:
                    fav = low[j] <= c0 - a
                    adv = high[j] >= c0 + 0.5 * a
                if adv:
                    passed = False
                    break
                if fav:
                    passed = True
                    break
            if direction == 1:
                long_first[i] = passed
            else:
                short_first[i] = passed
        valid[i] = True
    return {
        "valid": valid,
        "long_ret": long_ret,
        "long_mfe": long_mfe,
        "long_mae": long_mae,
        "short_mfe": short_mfe,
        "short_mae": short_mae,
        "efficiency": efficiency,
        "long_first": long_first,
        "short_first": short_first,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def summarize(out: dict[str, np.ndarray], signal: np.ndarray, mask: np.ndarray | None = None) -> dict:
    sig = np.asarray(signal, dtype=int)
    valid = out["valid"] & (sig != 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    signed_ret = np.where(sig == 1, out["long_ret"], -out["long_ret"])
    mfe = np.where(sig == 1, out["long_mfe"], out["short_mfe"])
    mae = np.where(sig == 1, out["long_mae"], out["short_mae"])
    first = np.where(sig == 1, out["long_first"], out["short_first"])
    x = signed_ret[valid]
    n = len(x)
    if n == 0:
        return {k: math.nan for k in [
            "n", "hit_rate", "material_rate", "clean_rate", "hit_ci_low", "hit_ci_high",
            "first_pass_rate", "mean_signed_atr", "median_signed_atr", "median_mfe_atr",
            "median_mae_atr", "median_efficiency"
        ]}
    hit = x > 0
    material = x >= 0.5
    clean = material & (out["efficiency"][valid] >= 0.30) & (mae[valid] <= 0.75)
    lo, hi = wilson(int(hit.sum()), n)
    return {
        "n": int(n),
        "hit_rate": float(hit.mean()),
        "material_rate": float(material.mean()),
        "clean_rate": float(clean.mean()),
        "hit_ci_low": float(lo),
        "hit_ci_high": float(hi),
        "first_pass_rate": float(first[valid].mean()),
        "mean_signed_atr": float(np.mean(x)),
        "median_signed_atr": float(np.median(x)),
        "median_mfe_atr": float(np.median(mfe[valid])),
        "median_mae_atr": float(np.median(mae[valid])),
        "median_efficiency": float(np.median(out["efficiency"][valid])),
    }


def onset_mask(signal: np.ndarray, horizon: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=int)
    chosen = np.zeros(len(sig), dtype=bool)
    previous = 0
    next_allowed = 0
    for i, s in enumerate(sig):
        if s != 0 and s != previous and i >= next_allowed:
            chosen[i] = True
            next_allowed = i + horizon + 1
        previous = s
    return chosen


def periodic_mask(valid: np.ndarray, horizon: int) -> np.ndarray:
    selected = np.zeros(len(valid), dtype=bool)
    next_allowed = 0
    for i, ok in enumerate(valid):
        if ok and i >= next_allowed:
            selected[i] = True
            next_allowed = i + horizon + 1
    return selected


def signal_stats(signal: np.ndarray, minutes: int, total_years: float) -> dict:
    sig = np.asarray(signal, dtype=int)
    nonzero = sig != 0
    starts = np.zeros(len(sig), dtype=bool)
    starts[0] = nonzero[0]
    starts[1:] = nonzero[1:] & (sig[1:] != sig[:-1])
    durations = []
    i = 0
    while i < len(sig):
        if sig[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(sig) and sig[j] == sig[i]:
            j += 1
        durations.append((j - i) * minutes / 60.0)
        i = j
    return {
        "signal_coverage": float(nonzero.mean()),
        "episodes": int(starts.sum()),
        "episodes_per_year": float(starts.sum() / max(total_years, 1e-9)),
        "median_duration_hours": float(np.median(durations)) if durations else math.nan,
    }


def aligned_direction(a: pd.Series, b: pd.Series, extra: pd.Series | None = None) -> np.ndarray:
    aa = np.sign(a.fillna(0).to_numpy(float)).astype(int)
    bb = np.sign(b.fillna(0).to_numpy(float)).astype(int)
    signal = np.where((aa == bb) & (aa != 0), aa, 0)
    if extra is not None:
        ee = np.sign(extra.fillna(0).to_numpy(float)).astype(int)
        signal = np.where((signal == ee) & (signal != 0), signal, 0)
    return signal.astype(int)


def apply_condition(signal: np.ndarray, condition: pd.Series | np.ndarray) -> np.ndarray:
    return np.where(np.asarray(condition, dtype=bool), signal, 0).astype(int)


def evaluate_signal(
    rows: list[dict],
    name: str,
    family: str,
    d: pd.DataFrame,
    signal: np.ndarray,
    outcomes: dict[int, dict[str, np.ndarray]],
    horizons: list[int],
    minutes: int,
    params: dict,
    experiment: str,
) -> None:
    total_years = max((d["time"].max() - d["time"].min()).days / 365.25, 0.01)
    stats = signal_stats(signal, minutes, total_years)
    past_bars = max(1, round(5 * 60 / minutes))
    past_move = signal * (d["close"] - d["close"].shift(past_bars)).to_numpy(float) / d["atr"].to_numpy(float)
    for horizon in horizons:
        out = outcomes[horizon]
        episode = onset_mask(signal, horizon)
        for period, (start, end) in PERIODS.items():
            p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
            for sample, sample_mask in [("bars", p_mask), ("episodes", p_mask & episode)]:
                result = summarize(out, signal, sample_mask)
                valid_sample = out["valid"] & (signal != 0) & sample_mask
                row = {
                    "experiment": experiment,
                    "model": name,
                    "family": family,
                    "params": json.dumps(params, sort_keys=True),
                    "timeframe_minutes": minutes,
                    "horizon_bars": horizon,
                    "horizon_hours": horizon * minutes / 60.0,
                    "period": period,
                    "sample": sample,
                    **stats,
                    "median_past5_signed_atr": float(np.nanmedian(past_move[valid_sample])) if valid_sample.any() else math.nan,
                    **result,
                }
                rows.append(row)
        for direction_name, direction_value in [("long", 1), ("short", -1)]:
            dmask = signal == direction_value
            result = summarize(out, signal, dmask)
            rows.append({
                "experiment": experiment,
                "model": name,
                "family": family,
                "params": json.dumps(params, sort_keys=True),
                "timeframe_minutes": minutes,
                "horizon_bars": horizon,
                "horizon_hours": horizon * minutes / 60.0,
                "period": f"full_{direction_name}",
                "sample": "bars",
                **stats,
                "median_past5_signed_atr": float(np.nanmedian(past_move[out["valid"] & dmask])) if (out["valid"] & dmask).any() else math.nan,
                **result,
            })


def timeframe_experiment() -> pd.DataFrame:
    specs = {
        "M15": (M15_URL, 15, 96, [48, 96], 45 * 60),
        "H1": (H1_URL, 60, 24, [12, 24], 3 * 3600),
        "H4": (H4_URL, 240, 6, [3, 6], 12 * 3600),
    }
    rows: list[dict] = []
    for tf, (url, minutes, reg_n, horizons, gap) in specs.items():
        d = prepare(load_data(url), minutes, reg_n=reg_n, mom_n=reg_n)
        outcomes = {h: compute_market_outcomes(d, h, gap) for h in horizons}
        direction = np.sign(d["reg_slope"].fillna(0).to_numpy(float)).astype(int)
        for r2_threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
            for move_threshold in [0.0, 0.5, 1.0]:
                signal = np.where(
                    (d["reg_r2"].to_numpy(float) >= r2_threshold)
                    & (d["reg_move_atr"].to_numpy(float) >= move_threshold),
                    direction,
                    0,
                )
                evaluate_signal(
                    rows,
                    f"{tf}_reg24h_r2_{r2_threshold:.2f}_move_{move_threshold:.1f}",
                    "rolling_regression_24h",
                    d,
                    signal,
                    outcomes,
                    horizons,
                    minutes,
                    {"r2": r2_threshold, "move_atr": move_threshold, "lookback_hours": 24},
                    "timeframe_comparison",
                )
        # Baseline long, sampled periodically for episodes.
        long_signal = np.ones(len(d), dtype=int)
        total_years = max((d["time"].max() - d["time"].min()).days / 365.25, 0.01)
        stats = signal_stats(long_signal, minutes, total_years)
        for horizon in horizons:
            out = outcomes[horizon]
            periodic = periodic_mask(out["valid"], horizon)
            for period, (start, end) in PERIODS.items():
                p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
                for sample, sample_mask in [("bars", p_mask), ("episodes", p_mask & periodic)]:
                    result = summarize(out, long_signal, sample_mask)
                    rows.append({
                        "experiment": "timeframe_comparison",
                        "model": f"{tf}_always_long",
                        "family": "baseline_long",
                        "params": "{}",
                        "timeframe_minutes": minutes,
                        "horizon_bars": horizon,
                        "horizon_hours": horizon * minutes / 60.0,
                        "period": period,
                        "sample": sample,
                        **stats,
                        "median_past5_signed_atr": math.nan,
                        **result,
                    })
    return pd.DataFrame(rows)


def practical_h1_experiment() -> pd.DataFrame:
    h1 = prepare(load_data(H1_URL), 60, reg_n=24, mom_n=12)
    h4 = prepare(load_data(H4_URL), 240, reg_n=12, mom_n=6)
    d1 = prepare(load_data(D1_URL), 1440, reg_n=10, mom_n=10)
    ctx_cols = [
        "close", "atr", "adx", "plus_di", "minus_di", "er10", "er20", "ema10", "ema20",
        "ema50", "ema200", "reg_slope", "reg_r2", "reg_move_atr", "momentum", "momentum_atr",
        "ema_sep_atr", "extension_atr",
    ]
    d = merge_context(h1, h4, "h4_", ctx_cols)
    d = merge_context(d, d1, "d1_", ctx_cols)
    outcomes = {12: compute_market_outcomes(d, 12, 3 * 3600), 24: compute_market_outcomes(d, 24, 3 * 3600)}
    rows: list[dict] = []
    configs: list[tuple[str, str, dict, np.ndarray]] = []

    ema_h1 = d["ema20"] - d["ema50"]
    ema_h4 = d["h4_ema20"] - d["h4_ema50"]
    ema_d1 = d["d1_ema20"] - d["d1_ema50"]
    reg_h1 = d["reg_slope"]
    reg_h4 = d["h4_reg_slope"]
    reg_d1 = d["d1_reg_slope"]
    dmi_h1 = d["plus_di"] - d["minus_di"]
    dmi_h4 = d["h4_plus_di"] - d["h4_minus_di"]
    mom_h1 = d["momentum"]
    mom_h4 = d["h4_momentum"]

    base_ema = aligned_direction(ema_h1, ema_h4)
    triple_ema = aligned_direction(ema_h1, ema_h4, ema_d1)
    configs.append(("ema20_50_h1_h4", "ema_alignment", {}, base_ema))
    configs.append(("ema20_50_h1_h4_d1", "ema_alignment", {"d1": True}, triple_ema))

    for er in [0.20, 0.30, 0.40]:
        for adx in [15, 20, 25]:
            for extension_cap in [2.0, 3.0]:
                cond = (d["er10"] >= er) & (d["adx"] >= adx) & (d["extension_atr"] <= extension_cap)
                signal = apply_condition(base_ema, cond)
                params = {"er10": er, "adx": adx, "extension_cap": extension_cap, "d1": False}
                configs.append((f"ema_quality_er{er:.2f}_adx{adx}_x{extension_cap:.1f}", "ema_quality", params, signal))
                signal_d1 = apply_condition(triple_ema, cond)
                params_d1 = {**params, "d1": True}
                configs.append((f"ema_quality_d1_er{er:.2f}_adx{adx}_x{extension_cap:.1f}", "ema_quality", params_d1, signal_d1))

    for r2_h1 in [0.15, 0.25, 0.35, 0.45]:
        for r2_h4 in [0.10, 0.20, 0.30]:
            reg_signal = aligned_direction(reg_h1, reg_h4)
            cond = (d["reg_r2"] >= r2_h1) & (d["h4_reg_r2"] >= r2_h4) & (d["reg_move_atr"] >= 0.5)
            signal = apply_condition(reg_signal, cond)
            params = {"h1_r2": r2_h1, "h4_r2": r2_h4, "move_atr": 0.5, "d1": False}
            configs.append((f"reg_h1{r2_h1:.2f}_h4{r2_h4:.2f}", "regression_alignment", params, signal))
            reg_signal_d1 = aligned_direction(reg_h1, reg_h4, reg_d1)
            signal_d1 = apply_condition(reg_signal_d1, cond)
            params_d1 = {**params, "d1": True}
            configs.append((f"reg_d1_h1{r2_h1:.2f}_h4{r2_h4:.2f}", "regression_alignment", params_d1, signal_d1))

    for er in [0.20, 0.30, 0.40]:
        momentum_signal = aligned_direction(mom_h1, mom_h4)
        cond = (d["er10"] >= er) & (d["momentum_atr"] >= 0.5)
        signal = apply_condition(momentum_signal, cond)
        configs.append((f"momentum_er{er:.2f}", "momentum_alignment", {"er10": er, "d1": False}, signal))
        momentum_d1 = aligned_direction(mom_h1, mom_h4, ema_d1)
        signal_d1 = apply_condition(momentum_d1, cond)
        configs.append((f"momentum_d1_er{er:.2f}", "momentum_alignment", {"er10": er, "d1": True}, signal_d1))

    for use_d1 in [False, True]:
        ema_signal = triple_ema if use_d1 else base_ema
        reg_signal = aligned_direction(reg_h1, reg_h4, reg_d1 if use_d1 else None)
        dmi_signal = aligned_direction(dmi_h1, dmi_h4)
        signal = np.where((ema_signal == reg_signal) & (ema_signal == dmi_signal) & (ema_signal != 0), ema_signal, 0)
        cond = (d["adx"] >= 18) & (d["er10"] >= 0.20) & (d["extension_atr"] <= 2.5)
        signal = apply_condition(signal, cond)
        configs.append((
            f"triple_confirmation{'_d1' if use_d1 else ''}",
            "ema_regression_dmi",
            {"d1": use_d1, "adx": 18, "er10": 0.20, "extension_cap": 2.5},
            signal,
        ))

    for name, family, params, signal in configs:
        evaluate_signal(rows, name, family, d, signal, outcomes, [12, 24], 60, params, "h1_practical_models")

    # H1 always-long baseline.
    long_signal = np.ones(len(d), dtype=int)
    total_years = max((d["time"].max() - d["time"].min()).days / 365.25, 0.01)
    stats = signal_stats(long_signal, 60, total_years)
    for horizon, out in outcomes.items():
        periodic = periodic_mask(out["valid"], horizon)
        for period, (start, end) in PERIODS.items():
            p_mask = ((d["time"] >= start) & (d["time"] < end)).to_numpy()
            for sample, sample_mask in [("bars", p_mask), ("episodes", p_mask & periodic)]:
                result = summarize(out, long_signal, sample_mask)
                rows.append({
                    "experiment": "h1_practical_models",
                    "model": "always_long",
                    "family": "baseline_long",
                    "params": "{}",
                    "timeframe_minutes": 60,
                    "horizon_bars": horizon,
                    "horizon_hours": horizon,
                    "period": period,
                    "sample": sample,
                    **stats,
                    "median_past5_signed_atr": math.nan,
                    **result,
                })
    return pd.DataFrame(rows)


def select_configs(results: pd.DataFrame, experiment: str) -> pd.DataFrame:
    x = results[
        (results["experiment"] == experiment)
        & (results["horizon_hours"] == 12)
        & (results["sample"] == "episodes")
        & (results["period"].isin(["train_2013_2018", "validation_2019_2020", "test_2021_2023"]))
        & (results["family"] != "baseline_long")
    ].copy()
    piv = x.pivot_table(
        index=["model", "family", "params", "timeframe_minutes"],
        columns="period",
        values=["n", "hit_rate", "mean_signed_atr", "first_pass_rate", "median_past5_signed_atr"],
        aggfunc="first",
    )
    piv.columns = [f"{metric}__{period}" for metric, period in piv.columns]
    piv = piv.reset_index()
    required = [
        "n__validation_2019_2020", "mean_signed_atr__train_2013_2018",
        "mean_signed_atr__validation_2019_2020", "hit_rate__validation_2019_2020",
        "first_pass_rate__validation_2019_2020",
    ]
    for c in required:
        if c not in piv:
            piv[c] = np.nan
    piv["eligible_pretest"] = (
        (piv["n__validation_2019_2020"] >= 40)
        & (piv["mean_signed_atr__train_2013_2018"] > 0)
        & (piv["mean_signed_atr__validation_2019_2020"] > 0)
    )
    piv["selection_score_pretest"] = (
        piv["mean_signed_atr__validation_2019_2020"]
        + 0.50 * (piv["hit_rate__validation_2019_2020"] - 0.50)
        + 0.25 * (piv["first_pass_rate__validation_2019_2020"] - 1 / 3)
    )
    return piv.sort_values(["eligible_pretest", "selection_score_pretest"], ascending=[False, False])


def main() -> None:
    tf_results = timeframe_experiment()
    h1_results = practical_h1_experiment()
    all_results = pd.concat([tf_results, h1_results], ignore_index=True)
    all_results.to_csv(OUT / "all_model_metrics.csv", index=False)
    tf_selected = select_configs(all_results, "timeframe_comparison")
    h1_selected = select_configs(all_results, "h1_practical_models")
    tf_selected.to_csv(OUT / "timeframe_selection.csv", index=False)
    h1_selected.to_csv(OUT / "h1_model_selection.csv", index=False)

    # Compact test table for configurations chosen without test data.
    chosen = pd.concat([
        tf_selected[tf_selected["eligible_pretest"]].head(10).assign(selection_group="timeframe"),
        h1_selected[h1_selected["eligible_pretest"]].head(15).assign(selection_group="h1_models"),
    ], ignore_index=True)
    chosen.to_csv(OUT / "pretest_selected_configs.csv", index=False)

    data_quality = {}
    for label, url in [("M15", M15_URL), ("H1", H1_URL), ("H4", H4_URL), ("D1", D1_URL)]:
        d = load_data(url)
        data_quality[label] = {
            "rows": int(len(d)),
            "start": str(d["time"].min()),
            "end": str(d["time"].max()),
            "duplicate_timestamps": int(d["time"].duplicated().sum()),
            "invalid_ohlc": int(((d["high"] < d[["open", "close"]].max(axis=1)) | (d["low"] > d[["open", "close"]].min(axis=1))).sum()),
        }
    summary = {
        "status": "BACKTEST_V3_COMPLETE",
        "data_quality": data_quality,
        "selection_rule": "Models selected using train 2013-2018 and validation 2019-2020 only; test 2021-2023 was not used for selection.",
        "trend_outcomes": {
            "hit": "forward close in signalled direction",
            "material": "forward signed return >= 0.5 ATR",
            "clean": "material + forward efficiency >= 0.30 + MAE <= 0.75 ATR",
            "first_pass": "+1 ATR reached before -0.5 ATR; same-bar collision is failure",
        },
        "top_timeframe_pretest": tf_selected.head(5).replace({np.nan: None}).to_dict(orient="records"),
        "top_h1_pretest": h1_selected.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("BACKTEST_V3_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("\nTOP TIMEFRAME CONFIGS (PRE-TEST SELECTION)")
    print(tf_selected.head(15).to_string(index=False))
    print("\nTOP H1 CONFIGS (PRE-TEST SELECTION)")
    print(h1_selected.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
