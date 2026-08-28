from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from nq_trend_detector_v3 import (
    D1_URL, H1_URL, H4_URL, compute_market_outcomes, load_data, merge_context, prepare,
)

OUT = Path("trend_backtest_v6_results")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)
HORIZON = 12
STRIDE = 3
PERIODS = {
    "train_2013_2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-01-01")),
    "validation_2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
    "test_2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
}


def enrich(raw: pd.DataFrame, minutes: int, reg_n: int, mom_n: int) -> pd.DataFrame:
    d = prepare(raw, minutes, reg_n=reg_n, mom_n=mom_n)
    atr = d["atr"].replace(0, np.nan)
    d["dmi_balance"] = (d["plus_di"] - d["minus_di"]) / (d["plus_di"] + d["minus_di"]).replace(0, np.nan)
    d["ema_gap"] = (d["ema20"] - d["ema50"]) / atr
    d["close_ema20"] = (d["close"] - d["ema20"]) / atr
    d["reg_move"] = np.sign(d["reg_slope"]) * d["reg_move_atr"]
    d["mom_move"] = d["momentum"] / atr
    d["adx_change"] = d["adx"] - d["adx"].shift(3)
    d["er_change"] = d["er10"] - d["er10"].shift(3)
    d["atr_ratio"] = d["atr"] / d["atr"].rolling(50, min_periods=50).median().replace(0, np.nan)
    ret = d["close"].pct_change()
    for n in (3, 6, 12, 24):
        path = d["close"].diff().abs().rolling(n, min_periods=n).sum()
        move = d["close"] - d["close"].shift(n)
        d[f"past_er{n}"] = move.abs() / path.replace(0, np.nan)
        d[f"past_move{n}"] = move / atr
        d[f"past_range{n}"] = (d["high"].rolling(n).max() - d["low"].rolling(n).min()) / atr
        d[f"past_rv{n}"] = ret.rolling(n).std() * np.sqrt(n) * d["close"] / atr
    hour = d["time"].dt.hour + d["time"].dt.minute / 60
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["time"].dt.dayofweek / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["time"].dt.dayofweek / 7)
    return d


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    h1 = enrich(load_data(H1_URL), 60, 24, 12)
    h4 = enrich(load_data(H4_URL), 240, 12, 6)
    d1 = enrich(load_data(D1_URL), 1440, 10, 10)
    core = ["adx", "er10", "er20", "reg_r2", "dmi_balance", "ema_gap", "close_ema20", "reg_move", "mom_move", "ema_sep_atr", "extension_atr", "adx_change", "er_change", "atr_ratio"]
    past = [f"{kind}{n}" for n in (3, 6, 12, 24) for kind in ("past_er", "past_move", "past_range", "past_rv")]
    h1_features = core + past + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    h4_features = core + [x for x in past if not x.endswith("24")]
    d1_features = ["adx", "er10", "reg_r2", "dmi_balance", "ema_gap", "close_ema20", "reg_move", "mom_move", "atr_ratio"]
    d = merge_context(h1, h4, "h4_", h4_features)
    d = merge_context(d, d1, "d1_", d1_features)
    for name in ("ema_gap", "reg_move", "mom_move", "dmi_balance"):
        d[f"align_{name}"] = np.where(np.sign(d[name]) == np.sign(d[f"h4_{name}"]), np.sign(d[name]), 0)
    d["strength_product"] = d["er10"] * d["h4_er10"]
    d["r2_product"] = d["reg_r2"] * d["h4_reg_r2"]
    features = h1_features + [f"h4_{x}" for x in h4_features] + [f"d1_{x}" for x in d1_features] + [f"align_{x}" for x in ("ema_gap", "reg_move", "mom_move", "dmi_balance")] + ["strength_product", "r2_product"]

    out = compute_market_outcomes(d, HORIZON, 3 * 3600)
    endpoint = out["long_ret"]
    abs_endpoint = np.abs(endpoint)
    direction = np.where(np.isfinite(endpoint), np.sign(endpoint), 0).astype(int)
    adverse = np.where(endpoint >= 0, out["long_mae"], out["short_mae"])
    d["future_valid"] = out["valid"]
    d["future_endpoint_atr"] = endpoint
    d["future_efficiency"] = out["efficiency"]
    d["future_adverse_atr"] = adverse
    d["future_direction"] = direction
    d["target_clean_trend"] = (out["valid"] & (abs_endpoint >= .75) & (out["efficiency"] >= .35) & (adverse <= .75)).astype(int)
    d["target_persistent_range"] = (out["valid"] & (abs_endpoint < .50) & (out["long_mfe"] < 1) & (out["short_mfe"] < 1) & (out["efficiency"] < .30)).astype(int)
    d["target_expansion"] = (out["valid"] & (np.maximum(out["long_mfe"], out["short_mfe"]) >= 1)).astype(int)
    d["future_time"] = d["time"].shift(-HORIZON)
    return d, features


def mask_for(d: pd.DataFrame, period: str) -> np.ndarray:
    start, end = PERIODS[period]
    return ((d["time"] >= start) & (d["time"] < end) & (d["future_time"] < end) & d["future_valid"] & ((np.arange(len(d)) % STRIDE) == 0)).to_numpy()


def models() -> dict[str, Pipeline]:
    return {
        "logit": Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(C=.1, class_weight="balanced", max_iter=4000, random_state=20260828))]),
        "hgb7": Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=2, random_state=20260828))]),
        "hgb15": Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=5, random_state=20260828))]),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"n": int(len(y)), "prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan, "average_precision": float(average_precision_score(y, p)) if y.sum() else math.nan, "brier": float(brier_score_loss(y, p))}


def top_metrics(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    n = max(1, int(math.ceil(len(y) * frac)))
    chosen = np.argsort(-p)[:n]
    precision = float(y[chosen].mean())
    prevalence = float(y.mean())
    return {"fraction": frac, "selected_n": n, "precision": precision, "recall": float(y[chosen].sum() / max(1, y.sum())), "lift": precision / prevalence if prevalence else math.nan}


def fit_target(X: pd.DataFrame, y: np.ndarray, masks: dict[str, np.ndarray], target: str):
    rows, fitted, predictions = [], {}, {}
    for name, model in models().items():
        kwargs = {}
        if name.startswith("hgb"):
            kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y[masks["train_2013_2018"]])
        model.fit(X.loc[masks["train_2013_2018"]], y[masks["train_2013_2018"]], **kwargs)
        fitted[name] = model
        for period, mask in masks.items():
            p = model.predict_proba(X.loc[mask])[:, 1]
            predictions[(name, period)] = p
            rows.append({"target": target, "model": name, "period": period, **metrics(y[mask], p)})
    result = pd.DataFrame(rows)
    val = result[result.period == "validation_2019_2020"].sort_values(["average_precision", "roc_auc"], ascending=False)
    best = str(val.iloc[0].model)
    return result, best, fitted[best], predictions


def direction_rules(d: pd.DataFrame) -> dict[str, np.ndarray]:
    sign = lambda s: np.sign(s.fillna(0).to_numpy(float)).astype(int)
    h1 = {"ema": sign(d.ema_gap), "reg": sign(d.reg_move), "mom": sign(d.mom_move), "dmi": sign(d.dmi_balance)}
    aligned = {k: np.where(h1[k] == sign(d[f"h4_{'ema_gap' if k == 'ema' else 'reg_move' if k == 'reg' else 'mom_move' if k == 'mom' else 'dmi_balance'}"]), h1[k], 0) for k in h1}
    votes = np.vstack([h1["ema"], h1["reg"], h1["mom"], h1["dmi"], aligned["ema"], aligned["reg"]])
    majority = np.where(votes.sum(axis=0) >= 3, 1, np.where(votes.sum(axis=0) <= -3, -1, 0))
    return {"always_long": np.ones(len(d), int), "h1_ema": h1["ema"], "h1_h4_ema": aligned["ema"], "h1_h4_reg": aligned["reg"], "h1_h4_momentum": aligned["mom"], "h1_h4_dmi": aligned["dmi"], "majority_4of6": majority}


def top_global(mask: np.ndarray, scores: np.ndarray, frac: float = .10) -> np.ndarray:
    idx = np.flatnonzero(mask)
    n = max(1, int(math.ceil(len(idx) * frac)))
    out = np.zeros(len(mask), bool)
    out[idx[np.argsort(-scores)[:n]]] = True
    return out


def directional_metrics(y_trend: np.ndarray, actual: np.ndarray, pred: np.ndarray, selected: np.ndarray) -> dict:
    active = selected & (pred != 0)
    true_trend = active & (y_trend == 1)
    correct = true_trend & (pred == actual)
    return {"selected_n": int(selected.sum()), "signal_n": int(active.sum()), "trend_precision": float(y_trend[active].mean()) if active.any() else math.nan, "directional_precision": float(correct.sum() / active.sum()) if active.any() else math.nan, "direction_accuracy_given_trend": float(correct.sum() / true_trend.sum()) if true_trend.any() else math.nan}


def monthly_bootstrap(time: pd.Series, y: np.ndarray, actual: np.ndarray, pred: np.ndarray, selected: np.ndarray, reps: int = 4000) -> dict:
    active = selected & (pred != 0)
    f = pd.DataFrame({"month": time.dt.to_period("M").astype(str), "active": active, "correct": active & (y == 1) & (pred == actual), "trend": active & (y == 1)})
    g = f.groupby("month").agg(n=("active", "sum"), correct=("correct", "sum"), trend=("trend", "sum"))
    g = g[g.n > 0]
    if len(g) < 3:
        return {"precision_ci_low": math.nan, "precision_ci_high": math.nan, "trend_precision_ci_low": math.nan, "trend_precision_ci_high": math.nan}
    arr = g[["n", "correct", "trend"]].to_numpy(float)
    p, tp = np.empty(reps), np.empty(reps)
    for i in range(reps):
        draw = arr[RNG.integers(0, len(arr), len(arr))].sum(axis=0)
        p[i] = draw[1] / draw[0]
        tp[i] = draw[2] / draw[0]
    return {"precision_ci_low": float(np.quantile(p, .025)), "precision_ci_high": float(np.quantile(p, .975)), "trend_precision_ci_low": float(np.quantile(tp, .025)), "trend_precision_ci_high": float(np.quantile(tp, .975))}


def main() -> None:
    d, features = build_dataset()
    X = d[features]
    masks = {name: mask_for(d, name) for name in PERIODS}
    prevalence = []
    for period, mask in masks.items():
        prevalence.append({"period": period, "n": int(mask.sum()), "clean_trend_prevalence": float(d.loc[mask, "target_clean_trend"].mean()), "range_prevalence": float(d.loc[mask, "target_persistent_range"].mean()), "expansion_prevalence": float(d.loc[mask, "target_expansion"].mean())})
    pd.DataFrame(prevalence).to_csv(OUT / "class_prevalence.csv", index=False)

    comparisons, top_rows, best = [], [], {}
    for col, target in (("target_clean_trend", "clean_trend"), ("target_persistent_range", "persistent_range")):
        y = d[col].to_numpy(int)
        comp, name, model, preds = fit_target(X, y, masks, target)
        comparisons.append(comp)
        best[target] = (name, model, preds)
        for period, mask in masks.items():
            for frac in (.05, .10, .20):
                top_rows.append({"target": target, "model": name, "period": period, **top_metrics(y[mask], preds[(name, period)], frac)})
    comparison = pd.concat(comparisons, ignore_index=True)
    top = pd.DataFrame(top_rows)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    top.to_csv(OUT / "top_fraction_metrics.csv", index=False)

    negative = []
    for col, target in (("target_clean_trend", "clean_trend"), ("target_persistent_range", "persistent_range")):
        y = d[col].to_numpy(int)
        name = best[target][0]
        model = models()[name]
        shuffled = RNG.permutation(y[masks["train_2013_2018"]])
        kwargs = {"clf__sample_weight": compute_sample_weight("balanced", shuffled)} if name.startswith("hgb") else {}
        model.fit(X.loc[masks["train_2013_2018"]], shuffled, **kwargs)
        for period in ("validation_2019_2020", "test_2021_2023"):
            mask = masks[period]
            negative.append({"target": target, "model": name, "period": period, **metrics(y[mask], model.predict_proba(X.loc[mask])[:, 1])})
    pd.DataFrame(negative).to_csv(OUT / "negative_control.csv", index=False)

    trend_y = d.target_clean_trend.to_numpy(int)
    actual = d.future_direction.to_numpy(int)
    trend_name, trend_model, trend_preds = best["clean_trend"]
    rules = direction_rules(d)
    direction_rows, selections = [], {}
    for period, mask in masks.items():
        selected = top_global(mask, trend_preds[(trend_name, period)])
        selections[period] = selected
        for rule, pred in rules.items():
            direction_rows.append({"period": period, "trend_model": trend_name, "direction_rule": rule, **directional_metrics(trend_y, actual, pred, selected)})
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(OUT / "direction_rules.csv", index=False)
    val = direction[(direction.period == "validation_2019_2020") & (direction.signal_n >= 30)].sort_values(["directional_precision", "direction_accuracy_given_trend"], ascending=False)
    selected_rule = str(val.iloc[0].direction_rule)
    boot = []
    for period in ("validation_2019_2020", "test_2021_2023"):
        boot.append({"period": period, "direction_rule": selected_rule, **monthly_bootstrap(d.time, trend_y, actual, rules[selected_rule], selections[period])})
    pd.DataFrame(boot).to_csv(OUT / "directional_block_bootstrap.csv", index=False)

    test_mask = masks["test_2021_2023"]
    perm = permutation_importance(trend_model, X.loc[test_mask], trend_y[test_mask], scoring="average_precision", n_repeats=5, random_state=20260828, n_jobs=-1)
    importance = pd.DataFrame({"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUT / "trend_feature_importance.csv", index=False)

    pred = d.loc[test_mask, ["time", "close", "future_endpoint_atr", "future_efficiency", "future_adverse_atr", "future_direction", "target_clean_trend", "target_persistent_range", "target_expansion"]].copy()
    pred["clean_trend_score"] = trend_preds[(trend_name, "test_2021_2023")]
    range_name, _, range_preds = best["persistent_range"]
    pred["persistent_range_score"] = range_preds[(range_name, "test_2021_2023")]
    pred["selected_direction"] = rules[selected_rule][test_mask]
    pred.to_csv(OUT / "blind_test_predictions.csv", index=False)

    summary = {
        "status": "BACKTEST_V6_COMPLETE",
        "data": {"rows": int(len(d)), "start": str(d.time.min()), "end": str(d.time.max()), "instrument": "USATECHIDXUSD Nasdaq-100 proxy; not CME NQ", "sampling_stride_hours": STRIDE},
        "labels": {"clean_trend": "12h abs endpoint >=0.75 ATR, efficiency >=0.35, adverse <=0.75 ATR", "persistent_range": "12h abs endpoint <0.50 ATR, both excursions <1 ATR, efficiency <0.30"},
        "selection_protocol": "Models and direction rule selected on 2019-2020 validation; 2021-2023 test remained blind.",
        "best_clean_trend_model": trend_name,
        "best_range_model": range_name,
        "selected_direction_rule": selected_rule,
        "blind_clean_model": comparison[(comparison.target == "clean_trend") & (comparison.model == trend_name) & (comparison.period == "test_2021_2023")].iloc[0].to_dict(),
        "blind_range_model": comparison[(comparison.target == "persistent_range") & (comparison.model == range_name) & (comparison.period == "test_2021_2023")].iloc[0].to_dict(),
        "blind_clean_top_decile": top[(top.target == "clean_trend") & (top.period == "test_2021_2023") & (top.fraction == .10)].iloc[0].to_dict(),
        "blind_range_top_decile": top[(top.target == "persistent_range") & (top.period == "test_2021_2023") & (top.fraction == .10)].iloc[0].to_dict(),
        "blind_direction": direction[(direction.period == "test_2021_2023") & (direction.direction_rule == selected_rule)].iloc[0].to_dict(),
        "negative_control_blind": pd.DataFrame(negative)[pd.DataFrame(negative).period == "test_2021_2023"].to_dict(orient="records"),
        "top_features_descriptive": importance.head(12).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("BACKTEST_V6_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))
    print("\nTOP FRACTIONS\n", top.to_string(index=False))
    print("\nDIRECTION\n", direction.to_string(index=False))
    print("\nBOOTSTRAP\n", pd.DataFrame(boot).to_string(index=False))


if __name__ == "__main__":
    main()
