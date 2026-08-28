from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from nq_trend_detector_v6 import build_dataset, mask_for

OUT = Path("trend_backtest_v7_results")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)
PERIODS = ["train_2013_2018", "validation_2019_2020", "test_2021_2023"]


def binary_models() -> dict[str, Pipeline]:
    return {
        "logit": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=.1, class_weight="balanced", max_iter=4000, random_state=20260828)),
        ]),
        "hgb7": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=35, l2_regularization=3, random_state=20260828)),
        ]),
        "hgb15": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=6, random_state=20260828)),
        ]),
    }


def multiclass_models() -> dict[str, Pipeline]:
    return {
        "logit": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=.1, class_weight="balanced", max_iter=5000, random_state=20260828)),
        ]),
        "hgb7": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=4, random_state=20260828)),
        ]),
        "hgb15": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=350, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=8, random_state=20260828)),
        ]),
    }


def conditional_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= .5).astype(int)
    return {
        "n": int(len(y)),
        "up_prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
    }


def confidence_metrics(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    confidence = np.abs(p - .5)
    n = max(1, int(math.ceil(len(y) * frac)))
    chosen = np.argsort(-confidence)[:n]
    pred = (p[chosen] >= .5).astype(int)
    return {
        "fraction": frac,
        "selected_n": int(n),
        "accuracy": float((pred == y[chosen]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y[chosen], pred)) if len(np.unique(y[chosen])) == 2 else math.nan,
        "median_confidence": float(np.median(confidence[chosen])),
        "predicted_long_rate": float(pred.mean()),
    }


def multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = np.argmax(p, axis=1)
    return {
        "n": int(len(y)),
        "no_trend_prevalence": float((y == 0).mean()),
        "up_prevalence": float((y == 1).mean()),
        "down_prevalence": float((y == 2).mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_roc_auc_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
    }


def actionable_metrics(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    direction_prob = p[:, 1:3]
    direction_class = np.argmax(direction_prob, axis=1) + 1
    direction_conf = np.max(direction_prob, axis=1)
    n = max(1, int(math.ceil(len(y) * frac)))
    chosen = np.argsort(-direction_conf)[:n]
    actual = y[chosen]
    pred = direction_class[chosen]
    is_clean = actual != 0
    correct = actual == pred
    return {
        "fraction": frac,
        "selected_n": int(n),
        "clean_trend_precision": float(is_clean.mean()),
        "directional_precision_all_selected": float(correct.mean()),
        "direction_accuracy_given_clean_trend": float(correct[is_clean].mean()) if is_clean.any() else math.nan,
        "predicted_long_rate": float((pred == 1).mean()),
        "median_direction_probability": float(np.median(direction_conf[chosen])),
    }


def monthly_bootstrap(time: pd.Series, y: np.ndarray, p: np.ndarray, frac: float, reps: int = 5000) -> dict:
    direction_prob = p[:, 1:3]
    direction_class = np.argmax(direction_prob, axis=1) + 1
    direction_conf = np.max(direction_prob, axis=1)
    n = max(1, int(math.ceil(len(y) * frac)))
    selected = np.zeros(len(y), dtype=bool)
    selected[np.argsort(-direction_conf)[:n]] = True
    clean = selected & (y != 0)
    correct = selected & (y == direction_class)
    f = pd.DataFrame({
        "month": time.dt.to_period("M").astype(str).to_numpy(),
        "selected": selected,
        "clean": clean,
        "correct": correct,
    })
    g = f.groupby("month").agg(selected=("selected", "sum"), clean=("clean", "sum"), correct=("correct", "sum"))
    g = g[g.selected > 0]
    arr = g[["selected", "clean", "correct"]].to_numpy(float)
    clean_precision = np.empty(reps)
    direction_precision = np.empty(reps)
    for i in range(reps):
        draw = arr[RNG.integers(0, len(arr), len(arr))].sum(axis=0)
        clean_precision[i] = draw[1] / draw[0]
        direction_precision[i] = draw[2] / draw[0]
    return {
        "clean_precision_ci_low": float(np.quantile(clean_precision, .025)),
        "clean_precision_ci_high": float(np.quantile(clean_precision, .975)),
        "directional_precision_ci_low": float(np.quantile(direction_precision, .025)),
        "directional_precision_ci_high": float(np.quantile(direction_precision, .975)),
    }


def main() -> None:
    d, features = build_dataset()
    X = d[features]
    masks = {name: mask_for(d, name) for name in PERIODS}
    clean = d["target_clean_trend"].to_numpy(int) == 1
    future_up = (d["future_direction"].to_numpy(int) == 1).astype(int)

    conditional_rows = []
    conditional_conf_rows = []
    conditional_predictions = {}
    for name, model in binary_models().items():
        train_mask = masks["train_2013_2018"] & clean
        kwargs = {}
        if name.startswith("hgb"):
            kwargs["clf__sample_weight"] = compute_sample_weight("balanced", future_up[train_mask])
        model.fit(X.loc[train_mask], future_up[train_mask], **kwargs)
        for period, base_mask in masks.items():
            mask = base_mask & clean
            p = model.predict_proba(X.loc[mask])[:, 1]
            conditional_predictions[(name, period)] = p
            conditional_rows.append({"model": name, "period": period, **conditional_metrics(future_up[mask], p)})
            for frac in (.05, .10, .20, .50):
                conditional_conf_rows.append({"model": name, "period": period, **confidence_metrics(future_up[mask], p, frac)})
    conditional = pd.DataFrame(conditional_rows)
    conditional_conf = pd.DataFrame(conditional_conf_rows)
    conditional.to_csv(OUT / "conditional_direction_models.csv", index=False)
    conditional_conf.to_csv(OUT / "conditional_direction_confidence.csv", index=False)
    val_cond = conditional[conditional.period == "validation_2019_2020"].sort_values(["roc_auc", "balanced_accuracy"], ascending=False)
    best_cond = str(val_cond.iloc[0].model)

    shuffled_rows = []
    train_mask = masks["train_2013_2018"] & clean
    shuffled_y = RNG.permutation(future_up[train_mask])
    neg = binary_models()[best_cond]
    kwargs = {"clf__sample_weight": compute_sample_weight("balanced", shuffled_y)} if best_cond.startswith("hgb") else {}
    neg.fit(X.loc[train_mask], shuffled_y, **kwargs)
    for period in ("validation_2019_2020", "test_2021_2023"):
        mask = masks[period] & clean
        p = neg.predict_proba(X.loc[mask])[:, 1]
        shuffled_rows.append({"model": best_cond, "period": period, **conditional_metrics(future_up[mask], p)})
    pd.DataFrame(shuffled_rows).to_csv(OUT / "conditional_negative_control.csv", index=False)

    y3 = np.zeros(len(d), dtype=int)
    y3[clean & (d["future_direction"].to_numpy(int) == 1)] = 1
    y3[clean & (d["future_direction"].to_numpy(int) == -1)] = 2
    multi_rows = []
    action_rows = []
    multi_predictions = {}
    for name, model in multiclass_models().items():
        train_mask = masks["train_2013_2018"]
        kwargs = {}
        if name.startswith("hgb"):
            kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y3[train_mask])
        model.fit(X.loc[train_mask], y3[train_mask], **kwargs)
        for period, mask in masks.items():
            p = model.predict_proba(X.loc[mask])
            multi_predictions[(name, period)] = p
            multi_rows.append({"model": name, "period": period, **multiclass_metrics(y3[mask], p)})
            for frac in (.01, .02, .05, .10, .20):
                action_rows.append({"model": name, "period": period, **actionable_metrics(y3[mask], p, frac)})
    multi = pd.DataFrame(multi_rows)
    action = pd.DataFrame(action_rows)
    multi.to_csv(OUT / "multiclass_models.csv", index=False)
    action.to_csv(OUT / "multiclass_actionable_fractions.csv", index=False)
    val_multi = multi[multi.period == "validation_2019_2020"].sort_values(["macro_roc_auc_ovr", "balanced_accuracy"], ascending=False)
    best_multi = str(val_multi.iloc[0].model)

    val_action = action[(action.period == "validation_2019_2020") & (action.selected_n >= 100)].copy()
    val_action["selection_score"] = val_action["directional_precision_all_selected"] + .5 * val_action["clean_trend_precision"]
    selected_action = val_action.sort_values("selection_score", ascending=False).iloc[0]
    selected_action_model = str(selected_action.model)
    selected_fraction = float(selected_action.fraction)

    boot_rows = []
    for period in ("validation_2019_2020", "test_2021_2023"):
        mask = masks[period]
        p = multi_predictions[(selected_action_model, period)]
        boot_rows.append({
            "period": period,
            "model": selected_action_model,
            "fraction": selected_fraction,
            **monthly_bootstrap(d.loc[mask, "time"].reset_index(drop=True), y3[mask], p, selected_fraction),
        })
    pd.DataFrame(boot_rows).to_csv(OUT / "multiclass_block_bootstrap.csv", index=False)

    test_mask = masks["test_2021_2023"]
    test_cond = conditional[(conditional.model == best_cond) & (conditional.period == "test_2021_2023")].iloc[0].to_dict()
    test_multi = multi[(multi.model == best_multi) & (multi.period == "test_2021_2023")].iloc[0].to_dict()
    test_action = action[(action.model == selected_action_model) & (action.period == "test_2021_2023") & (np.isclose(action.fraction, selected_fraction))].iloc[0].to_dict()
    baseline_up = float(future_up[test_mask & clean].mean())
    summary = {
        "status": "BACKTEST_V7_COMPLETE",
        "instrument": "USATECHIDXUSD Nasdaq-100 proxy; not CME NQ",
        "protocol": "Train 2013-2018, select model/rule/coverage on 2019-2020, blind test 2021-2023; 3-hour sampling, 12-hour forward label.",
        "conditional_direction": {
            "best_model_validation": best_cond,
            "blind_test": test_cond,
            "blind_always_long_accuracy": baseline_up,
            "negative_control_blind": pd.DataFrame(shuffled_rows)[pd.DataFrame(shuffled_rows).period == "test_2021_2023"].iloc[0].to_dict(),
        },
        "direct_multiclass": {
            "best_model_validation": best_multi,
            "blind_test": test_multi,
            "selected_action_model": selected_action_model,
            "selected_fraction_on_validation": selected_fraction,
            "blind_actionable": test_action,
            "bootstrap": boot_rows,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("BACKTEST_V7_COMPLETE")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("\nCONDITIONAL DIRECTION\n", conditional.to_string(index=False))
    print("\nCONDITIONAL CONFIDENCE\n", conditional_conf.to_string(index=False))
    print("\nMULTICLASS\n", multi.to_string(index=False))
    print("\nACTIONABLE FRACTIONS\n", action.to_string(index=False))
    print("\nBOOTSTRAP\n", pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
