from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TICK_SIZE = 0.25
COST_TICKS = 4.0
RNG = np.random.default_rng(20260828)

NUMERIC_FEATURES = [
    "prior_return_atr",
    "prior_range_atr",
    "overnight_range_atr",
    "overnight_return_atr",
    "overnight_close_location",
    "gap_atr",
    "preopen_return_atr",
    "or_range_atr",
    "or_return_atr",
    "or_close_location",
    "or_volume_ratio",
    "overnight_volume_ratio",
    "dist_prev_high_atr",
    "dist_prev_low_atr",
    "dist_on_high_atr",
    "dist_on_low_atr",
    "dist_vwap_atr",
    "h1_ema20_50_atr",
    "h1_dmi_spread",
    "h1_adx",
    "h1_mom6_atr",
    "h1_mom12_atr",
]

CLASS_COLUMNS = {
    "UP_CLEAN": "up_clean",
    "DOWN_CLEAN": "down_clean",
    "ANY_CLEAN_TREND": "any_clean",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_ohlcv(path: Path, source: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    tcol = next((c for c in d.columns if c in {"datetime", "timestamp", "time", "date"} or "timestamp" in c), None)
    if tcol is None:
        raise ValueError(f"No timestamp column found in {path}: {list(d.columns)}")
    parsed = pd.to_datetime(d[tcol], errors="coerce", utc=(source == "holdout_utc"))
    if source == "holdout_utc":
        parsed = parsed.dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["ts"] = parsed
    rename: dict[str, str] = {}
    for k in ["open", "high", "low", "close", "volume"]:
        if k not in d.columns:
            alt = next((c for c in d.columns if c.endswith(k)), None)
            if alt:
                rename[alt] = k
    d = d.rename(columns=rename)
    need = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    d = d[need].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=need).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    invalid = (d.high < d[["open", "close"]].max(axis=1)) | (d.low > d[["open", "close"]].min(axis=1))
    if invalid.any():
        raise ValueError(f"Invalid OHLC rows: {int(invalid.sum())}")
    return d


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_h1_indicators(m: pd.DataFrame) -> pd.DataFrame:
    x = m.copy()
    x["bucket_end"] = x.ts.dt.floor("h") + pd.Timedelta(hours=1)
    h = x.groupby("bucket_end", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index().rename(columns={"bucket_end": "ts"})
    prev = h.close.shift(1)
    tr = pd.concat([(h.high - h.low), (h.high - prev).abs(), (h.low - prev).abs()], axis=1).max(axis=1)
    h["atr"] = rma(tr, 14)
    up = h.high.diff(); down = -h.low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    pdi = 100.0 * rma(pdm, 14) / h.atr.replace(0, np.nan)
    mdi = 100.0 * rma(mdm, 14) / h.atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    h["adx"] = rma(dx, 14)
    h["dmi_spread"] = (pdi - mdi) / 100.0
    h["ema20"] = h.close.ewm(span=20, adjust=False, min_periods=20).mean()
    h["ema50"] = h.close.ewm(span=50, adjust=False, min_periods=50).mean()
    h["ema20_50_atr"] = (h.ema20 - h.ema50) / h.atr.replace(0, np.nan)
    h["mom6_atr"] = (h.close - h.close.shift(6)) / h.atr.replace(0, np.nan)
    h["mom12_atr"] = (h.close - h.close.shift(12)) / h.atr.replace(0, np.nan)
    return h


def sessionize(m: pd.DataFrame) -> pd.DataFrame:
    x = m.copy()
    x["minute"] = x.ts.dt.hour * 60 + x.ts.dt.minute
    x["session_date"] = x.ts.dt.normalize()
    x.loc[x.minute >= 1080, "session_date"] += pd.Timedelta(days=1)
    return x


def first_last(group: pd.DataFrame) -> tuple[float, float]:
    return float(group.iloc[0].open), float(group.iloc[-1].close)


def build_observations(m: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = sessionize(m)
    rth_mask = (x.minute >= 570) & (x.minute < 960)
    on_mask = (x.minute >= 1080) | (x.minute < 570)
    pre_mask = (x.minute >= 540) & (x.minute < 570)
    or_mask = (x.minute >= 570) & (x.minute < 600)
    pre10_mask = (x.minute >= 1080) | (x.minute < 600)

    rth = x[rth_mask].groupby("session_date", sort=True).agg(
        rth_open=("open", "first"), rth_high=("high", "max"), rth_low=("low", "min"),
        rth_close=("close", "last"), rth_volume=("volume", "sum"), rth_count=("ts", "size"),
        rth_first_ts=("ts", "min"), rth_last_ts=("ts", "max"),
    )
    rth["prev_close_for_tr"] = rth.rth_close.shift(1)
    rth["true_range"] = pd.concat([
        rth.rth_high - rth.rth_low,
        (rth.rth_high - rth.prev_close_for_tr).abs(),
        (rth.rth_low - rth.prev_close_for_tr).abs(),
    ], axis=1).max(axis=1)
    rth["prior_atr20"] = rth.true_range.shift(1).rolling(20, min_periods=20).mean()
    rth["prev_rth_open"] = rth.rth_open.shift(1)
    rth["prev_rth_high"] = rth.rth_high.shift(1)
    rth["prev_rth_low"] = rth.rth_low.shift(1)
    rth["prev_rth_close"] = rth.rth_close.shift(1)
    rth["prev_rth_return"] = (rth.rth_close - rth.rth_open).shift(1)
    rth["prev_rth_range"] = (rth.rth_high - rth.rth_low).shift(1)

    on = x[on_mask].groupby("session_date", sort=True).agg(
        on_open=("open", "first"), on_high=("high", "max"), on_low=("low", "min"),
        on_close=("close", "last"), on_volume=("volume", "sum"), on_count=("ts", "size"),
        on_first_ts=("ts", "min"), on_last_ts=("ts", "max"),
    )
    on["on_volume_median20"] = on.on_volume.shift(1).rolling(20, min_periods=20).median()

    pre = x[pre_mask].groupby("session_date", sort=True).agg(
        pre_open=("open", "first"), pre_close=("close", "last"), pre_count=("ts", "size"),
        pre_first_ts=("ts", "min"), pre_last_ts=("ts", "max"),
    )

    opn = x[or_mask].groupby("session_date", sort=True).agg(
        or_open=("open", "first"), or_high=("high", "max"), or_low=("low", "min"),
        or_close=("close", "last"), or_volume=("volume", "sum"), or_count=("ts", "size"),
        or_first_ts=("ts", "min"), or_last_ts=("ts", "max"),
    )
    opn["or_volume_median20"] = opn.or_volume.shift(1).rolling(20, min_periods=20).median()

    pre10 = x[pre10_mask].copy()
    pre10["pv"] = pre10.close * pre10.volume
    vw = pre10.groupby("session_date", sort=True).agg(pre10_pv=("pv", "sum"), pre10_volume=("volume", "sum"), pre10_last_ts=("ts", "max"))
    vw["vwap_1000"] = vw.pre10_pv / vw.pre10_volume.replace(0, np.nan)

    start_rows = x[(x.minute >= 600) & (x.minute < 606)].sort_values("ts").groupby("session_date", sort=True).first()
    start_rows = start_rows[["ts", "open"]].rename(columns={"ts": "start_ts", "open": "start_price"})

    future_rows: list[dict[str, Any]] = []
    for session_date, g in x[(x.minute >= 600) & (x.minute < 960)].groupby("session_date", sort=True):
        g = g.sort_values("ts")
        if g.empty:
            continue
        start_price = float(g.iloc[0].open)
        end_price = float(g.iloc[-1].close)
        path = np.abs(np.diff(np.r_[start_price, g.close.to_numpy(float)])).sum()
        signed_points = end_price - start_price
        efficiency = abs(signed_points) / path if path > 0 else 0.0
        if signed_points > 0:
            adverse_points = max(0.0, start_price - float(g.low.min()))
        elif signed_points < 0:
            adverse_points = max(0.0, float(g.high.max()) - start_price)
        else:
            adverse_points = max(float(g.high.max()) - start_price, start_price - float(g.low.min()))
        future_rows.append({
            "session_date": session_date, "future_start_ts": g.iloc[0].ts, "future_end_ts": g.iloc[-1].ts,
            "end_price": end_price, "future_count": int(len(g)), "signed_points": signed_points,
            "path_efficiency": efficiency, "adverse_points": adverse_points,
            "future_high": float(g.high.max()), "future_low": float(g.low.min()),
        })
    future = pd.DataFrame(future_rows).set_index("session_date") if future_rows else pd.DataFrame()

    obs = rth.join(on, how="left").join(pre, how="left").join(opn, how="left").join(vw, how="left").join(start_rows, how="left").join(future, how="left")
    obs = obs.reset_index()
    obs["decision_ts"] = obs.session_date + pd.Timedelta(hours=10)
    atr = obs.prior_atr20.replace(0, np.nan)
    obs["prior_return_atr"] = obs.prev_rth_return / atr
    obs["prior_range_atr"] = obs.prev_rth_range / atr
    obs["overnight_range_atr"] = (obs.on_high - obs.on_low) / atr
    obs["overnight_return_atr"] = (obs.on_close - obs.on_open) / atr
    obs["overnight_close_location"] = (obs.on_close - obs.on_low) / (obs.on_high - obs.on_low).replace(0, np.nan)
    obs["gap_atr"] = (obs.rth_open - obs.prev_rth_close) / atr
    obs["preopen_return_atr"] = (obs.pre_close - obs.pre_open) / atr
    obs["or_range_atr"] = (obs.or_high - obs.or_low) / atr
    obs["or_return_atr"] = (obs.or_close - obs.or_open) / atr
    obs["or_close_location"] = (obs.or_close - obs.or_low) / (obs.or_high - obs.or_low).replace(0, np.nan)
    obs["or_volume_ratio"] = obs.or_volume / obs.or_volume_median20.replace(0, np.nan)
    obs["overnight_volume_ratio"] = obs.on_volume / obs.on_volume_median20.replace(0, np.nan)
    obs["dist_prev_high_atr"] = (obs.start_price - obs.prev_rth_high) / atr
    obs["dist_prev_low_atr"] = (obs.start_price - obs.prev_rth_low) / atr
    obs["dist_on_high_atr"] = (obs.start_price - obs.on_high) / atr
    obs["dist_on_low_atr"] = (obs.start_price - obs.on_low) / atr
    obs["dist_vwap_atr"] = (obs.start_price - obs.vwap_1000) / atr
    obs["signed_endpoint_atr"] = obs.signed_points / atr
    obs["adverse_atr"] = obs.adverse_points / atr
    obs["up_clean"] = ((obs.signed_endpoint_atr >= 0.60) & (obs.path_efficiency >= 0.25) & (obs.adverse_atr <= 0.75)).astype(int)
    obs["down_clean"] = ((obs.signed_endpoint_atr <= -0.60) & (obs.path_efficiency >= 0.25) & (obs.adverse_atr <= 0.75)).astype(int)
    obs["any_clean"] = ((obs.up_clean == 1) | (obs.down_clean == 1)).astype(int)
    obs["dow"] = obs.session_date.dt.dayofweek

    h1 = add_h1_indicators(m)
    h1_keep = h1[["ts", "ema20_50_atr", "dmi_spread", "adx", "mom6_atr", "mom12_atr"]].rename(columns={
        "ema20_50_atr": "h1_ema20_50_atr", "dmi_spread": "h1_dmi_spread", "adx": "h1_adx",
        "mom6_atr": "h1_mom6_atr", "mom12_atr": "h1_mom12_atr",
    })
    obs = pd.merge_asof(obs.sort_values("decision_ts"), h1_keep.sort_values("ts"), left_on="decision_ts", right_on="ts", direction="backward")
    obs = obs.drop(columns=["ts"], errors="ignore")

    required = NUMERIC_FEATURES + ["dow", "up_clean", "down_clean", "any_clean", "start_price", "end_price", "prior_atr20"]
    before = len(obs)
    obs = obs[(obs.rth_count >= 300) & (obs.future_count >= 300) & (obs.or_count >= 25) & (obs.pre_count >= 20) & (obs.on_count >= 300)].copy()
    obs = obs[np.abs(obs.gap_atr) <= 3.0].copy()
    obs = obs.dropna(subset=required).reset_index(drop=True)

    causality_violations = int((obs.or_last_ts >= obs.decision_ts).sum())
    causality_violations += int((obs.pre_last_ts >= obs.decision_ts).sum())
    causality_violations += int((obs.pre10_last_ts >= obs.decision_ts).sum())
    causality_violations += int((obs.future_start_ts < obs.decision_ts).sum())

    audit = {
        "source": source_name,
        "raw_rows": int(len(m)), "raw_start": str(m.ts.min()), "raw_end": str(m.ts.max()),
        "sessions_before_filters": int(before), "sessions_after_filters": int(len(obs)),
        "causality_violations": causality_violations,
        "median_rth_count": float(obs.rth_count.median()) if len(obs) else None,
        "median_future_count": float(obs.future_count.median()) if len(obs) else None,
        "hourly_volume_peak_hour": int(m.assign(hour=m.ts.dt.hour).groupby("hour").volume.mean().idxmax()) if len(m) else None,
        "class_rates": {
            "up_clean": float(obs.up_clean.mean()) if len(obs) else None,
            "down_clean": float(obs.down_clean.mean()) if len(obs) else None,
            "any_clean": float(obs.any_clean.mean()) if len(obs) else None,
        },
    }
    return obs, audit


def atomic_mask(df: pd.DataFrame, atom: dict[str, Any]) -> np.ndarray:
    if atom["kind"] == "numeric":
        values = df[atom["feature"]].to_numpy(float)
        if atom["op"] == "le":
            return np.isfinite(values) & (values <= float(atom["threshold"]))
        return np.isfinite(values) & (values >= float(atom["threshold"]))
    return df.dow.to_numpy(int) == int(atom["value"])


def rule_mask(df: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for atom in rule["atoms"]:
        mask &= atomic_mask(df, atom)
    return mask


def metric_for_mask(df: pd.DataFrame, mask: np.ndarray, label_col: str) -> dict[str, float]:
    y = df[label_col].to_numpy(int)
    support = int(mask.sum())
    base = float(y.mean()) if len(y) else np.nan
    precision = float(y[mask].mean()) if support else np.nan
    lift = precision / base if support and base > 0 else np.nan
    return {"support": support, "base": base, "precision": precision, "lift": lift, "improvement": precision - base if support else np.nan}


def make_atoms(discovery: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    quantiles: dict[str, Any] = {}
    seen: set[tuple[Any, ...]] = set()
    for feature in NUMERIC_FEATURES:
        q = discovery[feature].quantile([0.25, 0.50, 0.75]).to_dict()
        quantiles[feature] = {"q25": float(q[0.25]), "q50": float(q[0.50]), "q75": float(q[0.75])}
        candidates = [
            ("le", "q25", q[0.25]), ("le", "q50", q[0.50]),
            ("ge", "q50", q[0.50]), ("ge", "q75", q[0.75]),
        ]
        for op, qname, threshold in candidates:
            key = (feature, op, round(float(threshold), 10))
            if key in seen:
                continue
            seen.add(key)
            atoms.append({"kind": "numeric", "feature": feature, "op": op, "threshold": float(threshold), "quantile": qname, "id": f"{feature}_{op}_{qname}"})
    for dow in range(5):
        atoms.append({"kind": "categorical", "feature": "dow", "value": dow, "id": f"dow_eq_{dow}"})
    return atoms, quantiles


def mine_rules(discovery: pd.DataFrame, val2024: pd.DataFrame, confirm2025: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    atoms, quantiles = make_atoms(discovery)
    atomic_masks = {a["id"]: atomic_mask(discovery, a) for a in atoms}
    candidate_atoms: list[list[dict[str, Any]]] = [[a] for a in atoms]
    for a, b in combinations(atoms, 2):
        if a["feature"] == b["feature"]:
            continue
        candidate_atoms.append([a, b])

    selected: dict[str, list[dict[str, Any]]] = {}
    for class_name, label_col in CLASS_COLUMNS.items():
        candidates: list[dict[str, Any]] = []
        y = discovery[label_col].to_numpy(int)
        base = float(y.mean())
        for atoms_for_rule in candidate_atoms:
            mask = np.ones(len(discovery), dtype=bool)
            for atom in atoms_for_rule:
                mask &= atomic_masks[atom["id"]]
            support = int(mask.sum())
            if support < 20:
                continue
            precision = float(y[mask].mean())
            lift = precision / base if base > 0 else 0.0
            if lift < 1.20 or precision - base < 0.05:
                continue
            rule = {"class": class_name, "atoms": atoms_for_rule}
            m24 = metric_for_mask(val2024, rule_mask(val2024, rule), label_col)
            m25 = metric_for_mask(confirm2025, rule_mask(confirm2025, rule), label_col)
            if m24["support"] < 12 or m25["support"] < 12:
                continue
            if not (m24["lift"] > 1.0 and m25["lift"] > 1.0 and m24["improvement"] > 0 and m25["improvement"] > 0):
                continue
            score = min(m24["lift"], m25["lift"]) * math.sqrt(min(m24["support"], m25["support"]))
            candidates.append({
                **rule,
                "discovery": {"support": support, "base": base, "precision": precision, "lift": lift, "improvement": precision - base},
                "validation_2024": m24, "confirmation_2025": m25, "selection_score": score,
                "id": "__AND__".join(a["id"] for a in atoms_for_rule),
            })
        candidates.sort(key=lambda r: (r["selection_score"], r["validation_2024"]["precision"], r["confirmation_2025"]["precision"]), reverse=True)
        retained: list[dict[str, Any]] = []
        combined = pd.concat([val2024, confirm2025], ignore_index=True)
        retained_masks: list[np.ndarray] = []
        for rule in candidates:
            mask = rule_mask(combined, rule)
            redundant = False
            for old_mask in retained_masks:
                union = int((mask | old_mask).sum())
                jaccard = float((mask & old_mask).sum() / union) if union else 1.0
                if jaccard >= 0.80:
                    redundant = True
                    break
            if not redundant:
                retained.append(rule)
                retained_masks.append(mask)
            if len(retained) >= 3:
                break
        selected[class_name] = retained
    selected["_quantiles"] = quantiles  # type: ignore[assignment]
    selected["_candidate_atom_count"] = len(atoms)  # type: ignore[assignment]
    selected["_candidate_rule_count"] = len(candidate_atoms)  # type: ignore[assignment]
    return selected


def apply_engine(df: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    gate = np.zeros(len(out), dtype=bool)
    up_votes = np.zeros(len(out), dtype=int)
    down_votes = np.zeros(len(out), dtype=int)
    for rule in rules.get("ANY_CLEAN_TREND", []):
        gate |= rule_mask(out, rule)
    for rule in rules.get("UP_CLEAN", []):
        up_votes += rule_mask(out, rule).astype(int)
    for rule in rules.get("DOWN_CLEAN", []):
        down_votes += rule_mask(out, rule).astype(int)
    prediction = np.zeros(len(out), dtype=int)
    prediction[gate & (up_votes > down_votes)] = 1
    prediction[gate & (down_votes > up_votes)] = -1
    out["trend_gate"] = gate.astype(int)
    out["up_votes"] = up_votes
    out["down_votes"] = down_votes
    out["prediction"] = prediction
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def bootstrap_mean(x: np.ndarray, reps: int = 10000) -> list[float | None]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [None, None]
    means = RNG.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def prediction_metrics(df: pd.DataFrame, pred_col: str = "prediction") -> dict[str, Any]:
    pred = df[pred_col].to_numpy(int)
    mask = pred != 0
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "coverage": 0.0}
    up = df.up_clean.to_numpy(int) == 1
    down = df.down_clean.to_numpy(int) == 1
    correct = mask & (((pred == 1) & up) | ((pred == -1) & down))
    wrong = mask & (((pred == 1) & down) | ((pred == -1) & up))
    no_clean = mask & ~(up | down)
    signed_return = pred[mask] * df.signed_endpoint_atr.to_numpy(float)[mask]
    random_direction_base = 0.5 * float((up | down).mean())
    always_long_base = float(up.mean())
    precision = float(correct.sum() / n)
    return {
        "n": n, "coverage": float(n / len(df)),
        "correct_clean_direction": int(correct.sum()), "wrong_clean_direction": int(wrong.sum()), "no_clean_trend": int(no_clean.sum()),
        "correct_clean_direction_precision": precision,
        "wrong_clean_direction_rate": float(wrong.sum() / n), "no_clean_trend_rate": float(no_clean.sum() / n),
        "wilson95_correct_precision": wilson(int(correct.sum()), n),
        "mean_signed_return_atr": float(signed_return.mean()), "median_signed_return_atr": float(np.median(signed_return)),
        "bootstrap90_mean_signed_return_atr": bootstrap_mean(signed_return),
        "random_direction_base_rate": random_direction_base, "always_long_base_rate": always_long_base,
        "lift_vs_random_direction_base": precision / random_direction_base if random_direction_base > 0 else None,
        "lift_vs_always_long_base": precision / always_long_base if always_long_base > 0 else None,
    }


def benchmark_predictions(df: pd.DataFrame) -> dict[str, Any]:
    b: dict[str, Any] = {}
    always = df.copy(); always["bench"] = 1
    b["always_long"] = prediction_metrics(always.rename(columns={"bench": "prediction"}))
    or_dir = np.sign(df.or_return_atr.to_numpy(float)).astype(int)
    z = df.copy(); z["prediction"] = or_dir
    b["opening_range_direction"] = prediction_metrics(z)
    ema_dir = np.sign(df.h1_ema20_50_atr.to_numpy(float)).astype(int)
    z = df.copy(); z["prediction"] = ema_dir
    b["h1_ema20_50_direction"] = prediction_metrics(z)
    return b


def random_coverage_benchmark(df: pd.DataFrame, n_predictions: int, observed_precision: float, reps: int = 10000) -> dict[str, Any]:
    if n_predictions <= 0 or n_predictions > len(df):
        return {"reps": reps, "n_predictions": n_predictions, "percentile": None}
    up = df.up_clean.to_numpy(int) == 1
    down = df.down_clean.to_numpy(int) == 1
    rates = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = RNG.choice(len(df), size=n_predictions, replace=False)
        direction = RNG.choice(np.array([-1, 1]), size=n_predictions)
        correct = ((direction == 1) & up[idx]) | ((direction == -1) & down[idx])
        rates[i] = correct.mean()
    percentile = float((np.sum(rates <= observed_precision) + 1) / (reps + 1))
    return {
        "reps": reps, "n_predictions": n_predictions, "mean_random_precision": float(rates.mean()),
        "p05": float(np.quantile(rates, 0.05)), "p95": float(np.quantile(rates, 0.95)),
        "observed_percentile": percentile,
    }


def economic_diagnostic(df: pd.DataFrame, minute_data: pd.DataFrame) -> dict[str, Any]:
    x = sessionize(minute_data)
    trades: list[dict[str, Any]] = []
    for row in df[df.prediction != 0].itertuples(index=False):
        g = x[(x.session_date == row.session_date) & (x.minute >= 600) & (x.minute < 960)].sort_values("ts")
        if g.empty:
            continue
        entry = float(g.iloc[0].open); side = int(row.prediction); atr = float(row.prior_atr20)
        risk_points = 0.75 * atr; target_points = 1.00 * atr
        stop = entry - side * risk_points; target = entry + side * target_points
        exit_price = float(g.iloc[-1].close); reason = "time"; exit_ts = g.iloc[-1].ts
        for bar in g.itertuples(index=False):
            hit_stop = bar.low <= stop if side == 1 else bar.high >= stop
            hit_target = bar.high >= target if side == 1 else bar.low <= target
            if hit_stop:
                exit_price = min(stop, float(bar.open)) if side == 1 else max(stop, float(bar.open))
                reason = "stop"; exit_ts = bar.ts; break
            if hit_target:
                exit_price = target; reason = "target"; exit_ts = bar.ts; break
        gross_r = side * (exit_price - entry) / risk_points
        cost_r = COST_TICKS * TICK_SIZE / risk_points
        trades.append({"session_date": row.session_date, "side": side, "gross_r": gross_r, "net_r": gross_r - cost_r, "reason": reason, "exit_ts": exit_ts})
    if not trades:
        return {"n": 0}
    t = pd.DataFrame(trades).sort_values("session_date")
    r = t.net_r.to_numpy(float)
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    eq = np.cumsum(r); peak = np.maximum.accumulate(np.r_[0.0, eq]); dd = peak[1:] - eq
    streak = best = 0
    for v in r:
        if v < 0:
            streak += 1; best = max(best, streak)
        else:
            streak = 0
    return {
        "n": int(len(t)), "win_rate": float((r > 0).mean()), "expectancy_net_r": float(r.mean()),
        "median_net_r": float(np.median(r)), "profit_factor": float(gains / losses) if losses > 0 else None,
        "max_drawdown_r": float(dd.max()) if len(dd) else 0.0, "longest_loss_streak": int(best),
        "exit_reasons": {str(k): int(v) for k, v in t.reason.value_counts().to_dict().items()},
    }


def describe_rules(rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in ["ANY_CLEAN_TREND", "UP_CLEAN", "DOWN_CLEAN"]:
        for rank, rule in enumerate(rules.get(class_name, []), start=1):
            text_parts = []
            for atom in rule["atoms"]:
                if atom["kind"] == "numeric":
                    symbol = "≤" if atom["op"] == "le" else "≥"
                    text_parts.append(f"{atom['feature']} {symbol} {atom['threshold']:.6g}")
                else:
                    text_parts.append(f"dow = {atom['value']}")
            rows.append({"class": class_name, "rank": rank, "rule": " AND ".join(text_parts), "id": rule["id"], "selection_score": rule["selection_score"], "discovery": rule["discovery"], "validation_2024": rule["validation_2024"], "confirmation_2025": rule["confirmation_2025"]})
    return rows


def discover(primary_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = load_ohlcv(primary_path, "primary_et")
    obs, audit = build_observations(primary, "primary_2022_2025")
    discovery = obs[(obs.session_date >= "2023-01-01") & (obs.session_date < "2024-01-01")].copy()
    val2024 = obs[(obs.session_date >= "2024-01-01") & (obs.session_date < "2025-01-01")].copy()
    confirm2025 = obs[(obs.session_date >= "2025-01-01") & (obs.session_date < "2025-12-01")].copy()
    if min(len(discovery), len(val2024), len(confirm2025)) < 100:
        raise RuntimeError(f"Insufficient internal sessions: {len(discovery)}, {len(val2024)}, {len(confirm2025)}")
    rules = mine_rules(discovery, val2024, confirm2025)
    internal = {}
    for name, frame in [("discovery_2023", discovery), ("validation_2024", val2024), ("confirmation_2025", confirm2025)]:
        scored = apply_engine(frame, rules)
        internal[name] = {"sessions": int(len(frame)), "metrics": prediction_metrics(scored), "benchmarks": benchmark_predictions(frame)}
        scored.to_csv(out_dir / f"{name}_scored.csv", index=False)
    spec = {
        "version": "V12-preregistered",
        "preregistration_commit": "4393516427810801724b2ae8bac050da9d21c2a1",
        "data_audit": audit,
        "internal_splits": {"discovery_2023": len(discovery), "validation_2024": len(val2024), "confirmation_2025": len(confirm2025)},
        "rules": rules,
        "rule_descriptions": describe_rules(rules),
        "internal_results": internal,
    }
    (out_dir / "v12_frozen_spec.json").write_text(json.dumps(json_safe(spec), indent=2), encoding="utf-8")
    pd.DataFrame(describe_rules(rules)).to_csv(out_dir / "v12_selected_rules.csv", index=False)
    print("V12_DISCOVERY_COMPLETE")
    print(json.dumps(json_safe({"audit": audit, "selected_rule_counts": {k: len(rules.get(k, [])) for k in CLASS_COLUMNS}, "internal_results": internal}), indent=2))


def evaluate(holdout_path: Path, frozen_spec_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(frozen_spec_path.read_text(encoding="utf-8"))
    rules = spec["rules"]
    holdout = load_ohlcv(holdout_path, "holdout_utc")
    obs, audit = build_observations(holdout, "untouched_2026_topstepx")
    scored = apply_engine(obs, rules)
    metrics = prediction_metrics(scored)
    benchmarks = benchmark_predictions(obs)
    random_benchmark = random_coverage_benchmark(scored, int(metrics.get("n", 0)), float(metrics.get("correct_clean_direction_precision", 0.0)))
    economic = economic_diagnostic(scored, holdout)
    random_percentile = random_benchmark.get("observed_percentile")
    base = metrics.get("random_direction_base_rate")
    precision = metrics.get("correct_clean_direction_precision")
    success_checks = {
        "at_least_12_predictions": bool(metrics.get("n", 0) >= 12),
        "precision_plus_10pp_vs_random_base": bool(precision is not None and base is not None and precision >= base + 0.10),
        "positive_mean_signed_return": bool(metrics.get("mean_signed_return_atr", -np.inf) > 0),
        "random_percentile_at_least_95": bool(random_percentile is not None and random_percentile >= 0.95),
        "positive_economic_expectancy_after_costs": bool(economic.get("expectancy_net_r", -np.inf) > 0),
        "zero_causality_violations": bool(audit.get("causality_violations") == 0),
    }
    final = {
        "version": "V12-preregistered",
        "preregistration_commit": spec.get("preregistration_commit"),
        "holdout_audit": audit,
        "holdout_sessions": int(len(obs)),
        "selected_rules": spec.get("rule_descriptions", []),
        "primary_metrics": metrics,
        "benchmarks": benchmarks,
        "random_coverage_benchmark": random_benchmark,
        "economic_diagnostic": economic,
        "success_checks": success_checks,
        "validated": bool(all(success_checks.values())),
    }
    scored.to_csv(out_dir / "v12_2026_scored_sessions.csv", index=False)
    (out_dir / "v12_holdout_results.json").write_text(json.dumps(json_safe(final), indent=2), encoding="utf-8")
    print("V12_HOLDOUT_COMPLETE")
    print(json.dumps(json_safe(final), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--primary", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--holdout", type=Path, required=True)
    e.add_argument("--spec", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "discover":
        discover(args.primary, args.out)
    else:
        evaluate(args.holdout, args.spec, args.out)


if __name__ == "__main__":
    main()
