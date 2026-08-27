from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

H1_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_H1.csv"
H4_URL = "https://raw.githubusercontent.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/main/indices/nasdaq100/USATECHIDXUSD_H4.csv"
OUT = Path("trend_backtest_results")
OUT.mkdir(exist_ok=True)
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85]
HORIZONS = [3, 6, 12, 24]
PERIODS = {
    "train_2013_2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-01-01")),
    "validation_2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
    "test_2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
}


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep="\t", parse_dates=["Time"])
    df.columns = [str(c).lower() for c in df.columns]
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required].dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame, er_len: int = 10, adx_len: int = 14, atr_len: int = 14) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = rma(tr, atr_len)
    up = out["high"].diff()
    down = -out["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=out.index)
    atr_for_di = rma(tr, adx_len)
    out["plus_di"] = 100.0 * rma(plus_dm, adx_len) / atr_for_di.replace(0, np.nan)
    out["minus_di"] = 100.0 * rma(minus_dm, adx_len) / atr_for_di.replace(0, np.nan)
    denom = (out["plus_di"] + out["minus_di"]).replace(0, np.nan)
    dx = 100.0 * (out["plus_di"] - out["minus_di"]).abs() / denom
    out["adx"] = rma(dx, adx_len)
    path = out["close"].diff().abs().rolling(er_len, min_periods=er_len).sum()
    out["er"] = (out["close"] - out["close"].shift(er_len)).abs() / path.replace(0, np.nan)
    out["move_atr"] = (out["close"] - out["close"].shift(5)).abs() / out["atr"].replace(0, np.nan)
    return out


def add_confirmed_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Pivots become available only after `right` complete bars; no look-ahead."""
    out = df.copy()
    highs = out["high"].to_numpy(float)
    lows = out["low"].to_numpy(float)
    n = len(out)
    ph_event = np.full(n, np.nan)
    pl_event = np.full(n, np.nan)
    for center in range(left, n - right):
        h = highs[center]
        l = lows[center]
        if h > np.max(highs[center-left:center]) and h >= np.max(highs[center+1:center+right+1]):
            ph_event[center + right] = h
        if l < np.min(lows[center-left:center]) and l <= np.min(lows[center+1:center+right+1]):
            pl_event[center + right] = l
    last_ph = np.full(n, np.nan)
    prev_ph = np.full(n, np.nan)
    last_pl = np.full(n, np.nan)
    prev_pl = np.full(n, np.nan)
    lp_h = pp_h = lp_l = pp_l = np.nan
    for i in range(n):
        if not np.isnan(ph_event[i]):
            pp_h, lp_h = lp_h, ph_event[i]
        if not np.isnan(pl_event[i]):
            pp_l, lp_l = lp_l, pl_event[i]
        last_ph[i], prev_ph[i] = lp_h, pp_h
        last_pl[i], prev_pl[i] = lp_l, pp_l
    out["last_ph"], out["prev_ph"] = last_ph, prev_ph
    out["last_pl"], out["prev_pl"] = last_pl, prev_pl
    out["bull_structure"] = (out["last_ph"] > out["prev_ph"]) & (out["last_pl"] > out["prev_pl"])
    out["bear_structure"] = (out["last_ph"] < out["prev_ph"]) & (out["last_pl"] < out["prev_pl"])
    return out


def bucket_er(x: pd.Series) -> pd.Series:
    return pd.Series(np.select([x < .20, x < .30, x < .40, x < .55], [0, 5, 10, 15], default=20), index=x.index)


def bucket_move(x: pd.Series) -> pd.Series:
    return pd.Series(np.select([x < .5, x < 1, x < 1.5, x < 2], [0, 4, 8, 12], default=15), index=x.index)


def bucket_break(x: pd.Series) -> pd.Series:
    return pd.Series(np.select([x <= 0, x < .10, x < .25, x < .50], [0, 2, 5, 8], default=10), index=x.index)


def build_score(h1: pd.DataFrame, h4: pd.DataFrame) -> pd.DataFrame:
    h1 = add_confirmed_pivots(add_indicators(h1), 3, 3)
    h4 = add_confirmed_pivots(add_indicators(h4), 3, 3)
    h1["event_time"] = h1["time"] + pd.Timedelta(hours=1)
    h4["event_time"] = h4["time"] + pd.Timedelta(hours=4)
    ctx = h4[["event_time", "bull_structure", "bear_structure", "last_ph", "prev_ph", "last_pl", "prev_pl"]].copy()
    ctx = ctx.rename(columns={c: f"h4_{c}" for c in ctx.columns if c != "event_time"})
    d = pd.merge_asof(h1.sort_values("event_time"), ctx.sort_values("event_time"), on="event_time", direction="backward")
    d["long_structure_points"] = (
        np.where(d["last_ph"] > d["prev_ph"], 10, 0)
        + np.where(d["last_pl"] > d["prev_pl"], 10, 0)
        + np.where(d["close"] > d["last_ph"], 5, 0)
    )
    d["short_structure_points"] = (
        np.where(d["last_ph"] < d["prev_ph"], 10, 0)
        + np.where(d["last_pl"] < d["prev_pl"], 10, 0)
        + np.where(d["close"] < d["last_pl"], 5, 0)
    )
    d["er_points"] = bucket_er(d["er"].fillna(-1))
    adx_base = np.select([d["adx"] < 18, d["adx"] < 22, d["adx"] < 25, d["adx"] < 30], [0, 3, 6, 10], default=12)
    d["adx_points"] = np.minimum(15, adx_base + np.where(d["adx"] > d["adx"].shift(3), 3, 0))
    d.loc[d["adx"].isna(), "adx_points"] = 0
    d["move_points"] = bucket_move(d["move_atr"].fillna(-1))
    d["long_align_points"] = np.select(
        [d["bull_structure"] & d["h4_bull_structure"], d["bull_structure"] & ~d["h4_bear_structure"]], [15, 8], default=0)
    d["short_align_points"] = np.select(
        [d["bear_structure"] & d["h4_bear_structure"], d["bear_structure"] & ~d["h4_bull_structure"]], [15, 8], default=0)
    long_break = ((d["close"] - d["last_ph"]) / d["atr"].replace(0, np.nan)).clip(lower=0)
    short_break = ((d["last_pl"] - d["close"]) / d["atr"].replace(0, np.nan)).clip(lower=0)
    d["long_break_points"] = bucket_break(long_break.fillna(-1))
    d["short_break_points"] = bucket_break(short_break.fillna(-1))
    d["long_score"] = d["long_structure_points"] + d["er_points"] + d["adx_points"] + d["move_points"] + d["long_align_points"] + d["long_break_points"]
    d["short_score"] = d["short_structure_points"] + d["er_points"] + d["adx_points"] + d["move_points"] + d["short_align_points"] + d["short_break_points"]
    d["score"] = d[["long_score", "short_score"]].max(axis=1)
    d["direction"] = np.where(d["long_score"] > d["short_score"], 1, np.where(d["short_score"] > d["long_score"], -1, 0))
    d["strict_aligned"] = np.where(d["direction"] == 1, d["bull_structure"] & d["h4_bull_structure"], np.where(d["direction"] == -1, d["bear_structure"] & d["h4_bear_structure"], False))
    d["past5_signed_atr"] = d["direction"] * (d["close"] - d["close"].shift(5)) / d["atr"].replace(0, np.nan)
    return d.reset_index(drop=True)


def future_metrics(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    n = len(df)
    result = pd.DataFrame(index=df.index)
    for c, value in {
        "valid": False, "signed_return_atr": np.nan, "mfe_atr": np.nan, "mae_atr": np.nan,
        "future_efficiency": np.nan, "hit_direction": False, "material_0_5": False,
        "clean_trend": False, "plus1_before_minus0_5": False,
    }.items():
        result[c] = value
    times = df["event_time"].to_numpy(dtype="datetime64[ns]")
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    direction = df["direction"].to_numpy(int)
    for i in range(n - horizon):
        if direction[i] == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        tseg = times[i:i+horizon+1].astype("datetime64[s]").astype(np.int64)
        if np.max(np.diff(tseg)) > 3 * 3600:
            continue
        d, c0 = direction[i], close[i]
        end_signed = d * (close[i+horizon] - c0) / atr[i]
        highs, lows = high[i+1:i+horizon+1], low[i+1:i+horizon+1]
        if d == 1:
            mfe, mae = (np.max(highs)-c0)/atr[i], (c0-np.min(lows))/atr[i]
        else:
            mfe, mae = (c0-np.min(lows))/atr[i], (np.max(highs)-c0)/atr[i]
        path = np.sum(np.abs(np.diff(close[i:i+horizon+1])))
        eff = abs(close[i+horizon]-c0)/path if path > 0 else 0.0
        first_pass = False
        for j in range(i+1, i+horizon+1):
            if d == 1:
                fav, adv = high[j] >= c0 + atr[i], low[j] <= c0 - .5*atr[i]
            else:
                fav, adv = low[j] <= c0 - atr[i], high[j] >= c0 + .5*atr[i]
            if fav and adv:
                first_pass = False; break
            if adv:
                first_pass = False; break
            if fav:
                first_pass = True; break
        result.at[i, "valid"] = True
        result.at[i, "signed_return_atr"] = end_signed
        result.at[i, "mfe_atr"] = mfe
        result.at[i, "mae_atr"] = mae
        result.at[i, "future_efficiency"] = eff
        result.at[i, "hit_direction"] = end_signed > 0
        result.at[i, "material_0_5"] = end_signed >= .5
        result.at[i, "clean_trend"] = end_signed >= .5 and eff >= .30 and mae <= .75
        result.at[i, "plus1_before_minus0_5"] = first_pass
    return result


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k/n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half


def summarize(fut: pd.DataFrame, mask: pd.Series) -> dict:
    m = mask.fillna(False) & fut["valid"]
    x = fut.loc[m]
    n = len(x)
    k = int(x["clean_trend"].sum()) if n else 0
    lo, hi = wilson(k, n)
    return {
        "n": int(n),
        "direction_hit_rate": float(x["hit_direction"].mean()) if n else math.nan,
        "material_0_5_rate": float(x["material_0_5"].mean()) if n else math.nan,
        "clean_trend_rate": float(x["clean_trend"].mean()) if n else math.nan,
        "clean_ci_low": float(lo), "clean_ci_high": float(hi),
        "plus1_before_minus0_5_rate": float(x["plus1_before_minus0_5"].mean()) if n else math.nan,
        "median_signed_return_atr": float(x["signed_return_atr"].median()) if n else math.nan,
        "mean_signed_return_atr": float(x["signed_return_atr"].mean()) if n else math.nan,
        "median_mfe_atr": float(x["mfe_atr"].median()) if n else math.nan,
        "median_mae_atr": float(x["mae_atr"].median()) if n else math.nan,
        "median_future_efficiency": float(x["future_efficiency"].median()) if n else math.nan,
    }


def onset_mask(signal: pd.Series, horizon: int) -> pd.Series:
    sig = signal.fillna(0).astype(int).to_numpy()
    chosen = np.zeros(len(sig), dtype=bool)
    next_allowed, prev = 0, 0
    for i, s in enumerate(sig):
        if s != 0 and s != prev and i >= next_allowed:
            chosen[i] = True
            next_allowed = i + horizon + 1
        prev = s
    return pd.Series(chosen, index=signal.index)


def auc_rank(scores: pd.Series, labels: pd.Series) -> float:
    mask = scores.notna() & labels.notna()
    s, y = scores[mask], labels[mask].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = s.rank(method="average")
    return float((ranks[y].sum() - n1*(n1+1)/2)/(n1*n0))


def main() -> None:
    h1, h4 = load_data(H1_URL), load_data(H4_URL)
    quality = {
        "h1_rows": len(h1), "h4_rows": len(h4),
        "h1_start": str(h1.time.min()), "h1_end": str(h1.time.max()),
        "h4_start": str(h4.time.min()), "h4_end": str(h4.time.max()),
        "h1_duplicate_timestamps": int(h1.time.duplicated().sum()),
        "h1_invalid_ohlc": int(((h1.high < h1[["open", "close"]].max(axis=1)) | (h1.low > h1[["open", "close"]].min(axis=1))).sum()),
    }
    d = build_score(h1, h4)
    future = {h: future_metrics(d, h) for h in HORIZONS}
    threshold_rows, episode_rows, period_rows, bin_rows = [], [], [], []
    for h in HORIZONS:
        fut = future[h]
        valid_dir = d.direction != 0
        cats = pd.cut(d.score, bins=[0,40,50,60,70,80,101], labels=["0-39","40-49","50-59","60-69","70-79","80-100"], right=False)
        for label in cats.cat.categories:
            mask = valid_dir & (cats == label)
            row = {"horizon_bars": h, "score_bin": str(label)}
            row.update(summarize(fut, mask))
            row["median_past5_signed_atr"] = float(d.loc[mask & fut.valid, "past5_signed_atr"].median()) if (mask & fut.valid).any() else math.nan
            bin_rows.append(row)
        for strict in [False, True]:
            variant = "score_only" if not strict else "strict_h1_h4_alignment"
            for threshold in THRESHOLDS:
                signal = pd.Series(np.where((d.score >= threshold) & valid_dir, d.direction, 0), index=d.index)
                if strict:
                    signal = signal.where(d.strict_aligned, 0)
                mask = signal != 0
                row = {"variant": variant, "threshold": threshold, "horizon_bars": h}
                row.update(summarize(fut, mask))
                row["coverage_pct"] = float((mask & fut.valid).sum()/max(1,(valid_dir & fut.valid).sum()))
                row["median_past5_signed_atr"] = float(d.loc[mask & fut.valid, "past5_signed_atr"].median()) if (mask & fut.valid).any() else math.nan
                threshold_rows.append(row)
                episode = onset_mask(signal, h)
                erow = {"variant": variant, "threshold": threshold, "horizon_bars": h}
                erow.update(summarize(fut, episode))
                erow["median_past5_signed_atr"] = float(d.loc[episode & fut.valid, "past5_signed_atr"].median()) if (episode & fut.valid).any() else math.nan
                episode_rows.append(erow)
                if h == 12:
                    for pname, (start, end) in PERIODS.items():
                        pmask = mask & (d.time >= start) & (d.time < end)
                        prow = {"period": pname, "variant": variant, "threshold": threshold, "horizon_bars": h}
                        prow.update(summarize(fut, pmask))
                        period_rows.append(prow)
    threshold_df = pd.DataFrame(threshold_rows)
    episode_df = pd.DataFrame(episode_rows)
    period_df = pd.DataFrame(period_rows)
    bin_df = pd.DataFrame(bin_rows)
    threshold_df.to_csv(OUT/"threshold_metrics.csv", index=False)
    episode_df.to_csv(OUT/"episode_metrics.csv", index=False)
    period_df.to_csv(OUT/"period_metrics_12h.csv", index=False)
    bin_df.to_csv(OUT/"score_bin_reliability.csv", index=False)
    d[["time","close","atr","long_score","short_score","score","direction","strict_aligned","past5_signed_atr"]].to_csv(OUT/"scored_h1.csv", index=False)

    fut12 = future[12]
    eligible = (d.direction != 0) & fut12.valid
    diagnostics = {
        "auc_clean_trend": auc_rank(d.loc[eligible,"score"], fut12.loc[eligible,"clean_trend"].astype(float)),
        "spearman_score_vs_signed_return_atr": float(d.loc[eligible,"score"].corr(fut12.loc[eligible,"signed_return_atr"], method="spearman")),
        "spearman_score_vs_mfe_minus_mae": float(d.loc[eligible,"score"].corr(fut12.loc[eligible,"mfe_atr"]-fut12.loc[eligible,"mae_atr"], method="spearman")),
        "eligible_rows": int(eligible.sum()),
    }
    predictors = {
        "full_score": d.score,
        "structure_points": d[["long_structure_points","short_structure_points"]].max(axis=1),
        "directionality_no_structure": d.er_points + d.adx_points + d.move_points,
        "full_without_breakout": d.score - np.where(d.direction==1,d.long_break_points,d.short_break_points),
        "full_without_alignment": d.score - np.where(d.direction==1,d.long_align_points,d.short_align_points),
    }
    ablation = []
    for name, pred in predictors.items():
        p = pred[eligible]
        cutoff = float(p.quantile(.90))
        top = eligible & (pred >= cutoff)
        row = {"predictor": name, "top_decile_cutoff": cutoff, "auc_clean_trend": auc_rank(p, fut12.loc[eligible,"clean_trend"].astype(float))}
        row.update(summarize(fut12, top))
        ablation.append(row)
    pd.DataFrame(ablation).to_csv(OUT/"component_ablation_12h.csv", index=False)
    summary = {
        "data_quality": quality,
        "method": {
            "base_timeframe":"H1", "context_timeframe":"H4",
            "pivot_confirmation":"3 left / 3 right bars; exposed only after confirmation",
            "horizons_bars":HORIZONS,
            "clean_trend_definition":"signed endpoint >=0.5 ATR, forward efficiency >=0.30, MAE <=0.75 ATR",
            "first_pass_definition":"+1 ATR before -0.5 ATR; same-bar collision is a failure",
            "large_gap_filter":"forward windows with any gap >3h excluded",
        },
        "diagnostics_12h": diagnostics,
    }
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print("BACKTEST_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print("\nSTRICT THRESHOLDS — 12H BAR LEVEL")
    print(threshold_df[(threshold_df.variant=="strict_h1_h4_alignment") & (threshold_df.horizon_bars==12)].to_string(index=False))
    print("\nSTRICT THRESHOLDS — 12H NON-OVERLAPPING ONSETS")
    print(episode_df[(episode_df.variant=="strict_h1_h4_alignment") & (episode_df.horizon_bars==12)].to_string(index=False))
    print("\nSCORE BINS — 12H")
    print(bin_df[bin_df.horizon_bars==12].to_string(index=False))
    print("\nPERIOD STABILITY — 12H STRICT")
    print(period_df[period_df.variant=="strict_h1_h4_alignment"].to_string(index=False))
    print("\nABLATION — 12H")
    print(pd.DataFrame(ablation).to_string(index=False))


if __name__ == "__main__":
    main()
