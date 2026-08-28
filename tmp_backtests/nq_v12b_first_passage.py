from __future__ import annotations

import argparse
import gzip
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nq_v12_preregistered_patterns as base

OUTCOME_COLUMNS = {"UP_EVENT": "up_event", "DOWN_EVENT": "down_event"}
TARGET_ATR = 0.50
STOP_ATR = 0.35
TICK_SIZE = 0.25
COST_TICKS = 4.0
FRESH_START = pd.Timestamp("2026-04-16")
RNG = np.random.default_rng(20260828)


def json_safe(x: Any) -> Any:
    return base.json_safe(x)


def first_passage_for_path(g: pd.DataFrame, entry: float, atr: float) -> tuple[int, int, int]:
    up_target = entry + TARGET_ATR * atr
    up_stop = entry - STOP_ATR * atr
    down_target = entry - TARGET_ATR * atr
    down_stop = entry + STOP_ATR * atr

    up_resolution: str | None = None
    down_resolution: str | None = None
    ambiguous = 0

    for bar in g.itertuples(index=False):
        if up_resolution is None:
            hit_target = float(bar.high) >= up_target
            hit_stop = float(bar.low) <= up_stop
            if hit_target and hit_stop:
                ambiguous = 1
                up_resolution = "ambiguous"
            elif hit_target:
                up_resolution = "target"
            elif hit_stop:
                up_resolution = "stop"

        if down_resolution is None:
            hit_target = float(bar.low) <= down_target
            hit_stop = float(bar.high) >= down_stop
            if hit_target and hit_stop:
                ambiguous = 1
                down_resolution = "ambiguous"
            elif hit_target:
                down_resolution = "target"
            elif hit_stop:
                down_resolution = "stop"

        if up_resolution is not None and down_resolution is not None:
            break

    if ambiguous:
        return 0, 0, 1
    up_event = int(up_resolution == "target")
    down_event = int(down_resolution == "target")
    if up_event and down_event:
        return 0, 0, 1
    return up_event, down_event, 0


def add_first_passage_labels(obs: pd.DataFrame, minute_data: pd.DataFrame) -> pd.DataFrame:
    x = base.sessionize(minute_data)
    paths = {
        session_date: g.sort_values("ts")
        for session_date, g in x[(x.minute >= 600) & (x.minute < 960)].groupby("session_date", sort=False)
    }
    up_events: list[int] = []
    down_events: list[int] = []
    ambiguous: list[int] = []
    for row in obs.itertuples(index=False):
        g = paths.get(row.session_date)
        if g is None or g.empty or not np.isfinite(row.prior_atr20) or row.prior_atr20 <= 0:
            up_events.append(0); down_events.append(0); ambiguous.append(1)
            continue
        entry = float(g.iloc[0].open)
        up, down, amb = first_passage_for_path(g, entry, float(row.prior_atr20))
        up_events.append(up); down_events.append(down); ambiguous.append(amb)
    out = obs.copy()
    out["up_event"] = up_events
    out["down_event"] = down_events
    out["ambiguous_event"] = ambiguous
    out["any_event"] = ((out.up_event == 1) | (out.down_event == 1)).astype(int)
    return out


def mine_rules(discovery: pd.DataFrame, val2024: pd.DataFrame, confirm2025: pd.DataFrame) -> dict[str, Any]:
    atoms, quantiles = base.make_atoms(discovery)
    discovery_masks = {a["id"]: base.atomic_mask(discovery, a) for a in atoms}
    candidates_atoms: list[list[dict[str, Any]]] = [[a] for a in atoms]
    for a, b in combinations(atoms, 2):
        if a["feature"] != b["feature"]:
            candidates_atoms.append([a, b])

    selected: dict[str, Any] = {}
    combined = pd.concat([val2024, confirm2025], ignore_index=True)
    for class_name, label_col in OUTCOME_COLUMNS.items():
        y = discovery[label_col].to_numpy(int)
        base_rate = float(y.mean())
        candidates: list[dict[str, Any]] = []
        for rule_atoms in candidates_atoms:
            mask = np.ones(len(discovery), dtype=bool)
            for atom in rule_atoms:
                mask &= discovery_masks[atom["id"]]
            support = int(mask.sum())
            if support < 20:
                continue
            precision = float(y[mask].mean())
            lift = precision / base_rate if base_rate > 0 else 0.0
            improvement = precision - base_rate
            if lift < 1.20 or improvement < 0.07:
                continue
            rule = {
                "class": class_name,
                "atoms": rule_atoms,
                "id": "__AND__".join(a["id"] for a in rule_atoms),
            }
            m24 = base.metric_for_mask(val2024, base.rule_mask(val2024, rule), label_col)
            m25 = base.metric_for_mask(confirm2025, base.rule_mask(confirm2025, rule), label_col)
            if m24["support"] < 12 or m25["support"] < 12:
                continue
            if not (m24["lift"] > 1.0 and m25["lift"] > 1.0 and m24["improvement"] > 0 and m25["improvement"] > 0):
                continue
            score = min(m24["lift"], m25["lift"]) * math.sqrt(min(m24["support"], m25["support"]))
            candidates.append({
                **rule,
                "discovery": {
                    "support": support, "base": base_rate, "precision": precision,
                    "lift": lift, "improvement": improvement,
                },
                "validation_2024": m24,
                "confirmation_2025": m25,
                "selection_score": score,
            })

        candidates.sort(
            key=lambda r: (
                r["selection_score"],
                r["validation_2024"]["precision"],
                r["confirmation_2025"]["precision"],
            ),
            reverse=True,
        )
        retained: list[dict[str, Any]] = []
        retained_masks: list[np.ndarray] = []
        for rule in candidates:
            mask = base.rule_mask(combined, rule)
            redundant = False
            for prior_mask in retained_masks:
                union = int((mask | prior_mask).sum())
                jaccard = float((mask & prior_mask).sum() / union) if union else 1.0
                if jaccard >= 0.80:
                    redundant = True
                    break
            if not redundant:
                retained.append(rule)
                retained_masks.append(mask)
            if len(retained) >= 4:
                break
        selected[class_name] = retained

    selected["_quantiles"] = quantiles
    selected["_candidate_atom_count"] = len(atoms)
    selected["_candidate_rule_count"] = len(candidates_atoms)
    return selected


def apply_engine(df: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    up_votes = np.zeros(len(out), dtype=int)
    down_votes = np.zeros(len(out), dtype=int)
    for rule in rules.get("UP_EVENT", []):
        up_votes += base.rule_mask(out, rule).astype(int)
    for rule in rules.get("DOWN_EVENT", []):
        down_votes += base.rule_mask(out, rule).astype(int)
    prediction = np.zeros(len(out), dtype=int)
    prediction[(up_votes >= 1) & (up_votes > down_votes)] = 1
    prediction[(down_votes >= 1) & (down_votes > up_votes)] = -1
    out["up_votes"] = up_votes
    out["down_votes"] = down_votes
    out["prediction"] = prediction
    return out


def wilson(k: int, n: int) -> list[float | None]:
    return base.wilson(k, n)


def prediction_metrics(df: pd.DataFrame) -> dict[str, Any]:
    pred = df.prediction.to_numpy(int)
    mask = pred != 0
    n = int(mask.sum())
    up = df.up_event.to_numpy(int) == 1
    down = df.down_event.to_numpy(int) == 1
    any_event = up | down
    random_base = 0.5 * float(any_event.mean()) if len(df) else np.nan
    if n == 0:
        return {
            "n": 0,
            "coverage": 0.0,
            "unconditional_up_event_rate": float(up.mean()) if len(df) else None,
            "unconditional_down_event_rate": float(down.mean()) if len(df) else None,
            "unconditional_random_direction_base": random_base,
        }
    correct = mask & (((pred == 1) & up) | ((pred == -1) & down))
    wrong = mask & (((pred == 1) & down) | ((pred == -1) & up))
    no_event = mask & ~any_event
    signed = pred[mask] * df.signed_endpoint_atr.to_numpy(float)[mask]
    precision = float(correct.sum() / n)
    long_mask = mask & (pred == 1)
    short_mask = mask & (pred == -1)

    def side_metrics(side_mask: np.ndarray, side: int) -> dict[str, Any]:
        nn = int(side_mask.sum())
        if nn == 0:
            return {"n": 0}
        side_correct = side_mask & (up if side == 1 else down)
        side_wrong = side_mask & (down if side == 1 else up)
        return {
            "n": nn,
            "precision": float(side_correct.sum() / nn),
            "wrong_event_rate": float(side_wrong.sum() / nn),
            "mean_signed_return_atr": float((side * df.signed_endpoint_atr.to_numpy(float)[side_mask]).mean()),
        }

    return {
        "n": n,
        "coverage": float(n / len(df)),
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "no_event": int(no_event.sum()),
        "correct_precision": precision,
        "wrong_event_rate": float(wrong.sum() / n),
        "no_event_rate": float(no_event.sum() / n),
        "wilson95_precision": wilson(int(correct.sum()), n),
        "mean_signed_return_atr": float(signed.mean()),
        "median_signed_return_atr": float(np.median(signed)),
        "bootstrap90_mean_signed_return_atr": base.bootstrap_mean(signed),
        "unconditional_up_event_rate": float(up.mean()),
        "unconditional_down_event_rate": float(down.mean()),
        "unconditional_random_direction_base": random_base,
        "precision_improvement_vs_random_base": precision - random_base,
        "lift_vs_random_base": precision / random_base if random_base > 0 else None,
        "long": side_metrics(long_mask, 1),
        "short": side_metrics(short_mask, -1),
    }


def random_benchmark(df: pd.DataFrame, observed: dict[str, Any], reps: int = 10000) -> dict[str, Any]:
    n = int(observed.get("n", 0))
    if n <= 0 or n > len(df):
        return {"reps": reps, "n_predictions": n, "observed_percentile": None}
    up = df.up_event.to_numpy(int) == 1
    down = df.down_event.to_numpy(int) == 1
    rates = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = RNG.choice(len(df), size=n, replace=False)
        direction = RNG.choice(np.array([-1, 1]), size=n)
        rates[i] = np.mean(((direction == 1) & up[idx]) | ((direction == -1) & down[idx]))
    obs_precision = float(observed["correct_precision"])
    return {
        "reps": reps,
        "n_predictions": n,
        "mean_random_precision": float(rates.mean()),
        "p05": float(np.quantile(rates, 0.05)),
        "p95": float(np.quantile(rates, 0.95)),
        "observed_percentile": float((np.sum(rates <= obs_precision) + 1) / (reps + 1)),
    }


def economic_diagnostic(scored: pd.DataFrame, minute_data: pd.DataFrame) -> dict[str, Any]:
    x = base.sessionize(minute_data)
    rows: list[dict[str, Any]] = []
    for row in scored[scored.prediction != 0].itertuples(index=False):
        g = x[(x.session_date == row.session_date) & (x.minute >= 600) & (x.minute < 960)].sort_values("ts")
        if g.empty:
            continue
        entry = float(g.iloc[0].open)
        side = int(row.prediction)
        atr = float(row.prior_atr20)
        target_points = TARGET_ATR * atr
        stop_points = STOP_ATR * atr
        target = entry + side * target_points
        stop = entry - side * stop_points
        exit_price = float(g.iloc[-1].close)
        reason = "time"
        for bar in g.itertuples(index=False):
            hit_target = float(bar.high) >= target if side == 1 else float(bar.low) <= target
            hit_stop = float(bar.low) <= stop if side == 1 else float(bar.high) >= stop
            if hit_target and hit_stop:
                exit_price = min(stop, float(bar.open)) if side == 1 else max(stop, float(bar.open))
                reason = "same_bar_stop"
                break
            if hit_stop:
                exit_price = min(stop, float(bar.open)) if side == 1 else max(stop, float(bar.open))
                reason = "stop"
                break
            if hit_target:
                exit_price = target
                reason = "target"
                break
        gross_r = side * (exit_price - entry) / stop_points
        cost_r = COST_TICKS * TICK_SIZE / stop_points
        rows.append({"session_date": row.session_date, "side": side, "net_r": gross_r - cost_r, "gross_r": gross_r, "reason": reason})
    if not rows:
        return {"n": 0}
    trades = pd.DataFrame(rows).sort_values("session_date")
    r = trades.net_r.to_numpy(float)
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    eq = np.cumsum(r); peak = np.maximum.accumulate(np.r_[0.0, eq]); dd = peak[1:] - eq
    streak = best = 0
    for value in r:
        if value < 0:
            streak += 1; best = max(best, streak)
        else:
            streak = 0
    return {
        "n": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "expectancy_net_r": float(r.mean()),
        "median_net_r": float(np.median(r)),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "max_drawdown_r": float(dd.max()) if len(dd) else 0.0,
        "longest_loss_streak": int(best),
        "exit_reasons": {str(k): int(v) for k, v in trades.reason.value_counts().to_dict().items()},
    }


def rule_descriptions(rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in ["UP_EVENT", "DOWN_EVENT"]:
        for rank, rule in enumerate(rules.get(class_name, []), start=1):
            parts: list[str] = []
            for atom in rule["atoms"]:
                if atom["kind"] == "numeric":
                    symbol = "≤" if atom["op"] == "le" else "≥"
                    parts.append(f"{atom['feature']} {symbol} {atom['threshold']:.6g}")
                else:
                    parts.append(f"dow = {atom['value']}")
            rows.append({
                "class": class_name,
                "rank": rank,
                "rule": " AND ".join(parts),
                "id": rule["id"],
                "selection_score": rule["selection_score"],
                "discovery": rule["discovery"],
                "validation_2024": rule["validation_2024"],
                "confirmation_2025": rule["confirmation_2025"],
            })
    return rows


def parse_ev_json(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            raw = raw["data"]
        elif all(isinstance(v, list) for v in raw.values()):
            raw = pd.DataFrame(raw).to_dict("records")
        else:
            raise ValueError(f"Unsupported EV JSON dictionary keys: {list(raw)[:20]}")
    d = pd.DataFrame(raw)
    required = ["ts", "o", "h", "l", "c", "v"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"EV data missing {missing}; columns={list(d.columns)}")
    ts = d.ts
    if pd.api.types.is_numeric_dtype(ts):
        max_abs = float(pd.to_numeric(ts, errors="coerce").abs().max())
        unit = "ns" if max_abs >= 1e17 else "us" if max_abs >= 1e14 else "ms" if max_abs >= 1e11 else "s"
        base_utc = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
    else:
        unit = "string"
        base_utc = pd.to_datetime(ts, errors="coerce", utc=True)
    out = pd.DataFrame({
        "base_utc": base_utc,
        "open": pd.to_numeric(d.o, errors="coerce"),
        "high": pd.to_numeric(d.h, errors="coerce"),
        "low": pd.to_numeric(d.l, errors="coerce"),
        "close": pd.to_numeric(d.c, errors="coerce"),
        "volume": pd.to_numeric(d.v, errors="coerce"),
    }).dropna().sort_values("base_utc").drop_duplicates("base_utc", keep="last").reset_index(drop=True)
    return out, {"json_rows": int(len(d)), "parsed_rows": int(len(out)), "timestamp_unit": unit, "columns": list(d.columns)}


def map_candidate(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    base_utc = raw.base_utc
    if name == "utc_to_new_york":
        ts = base_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif name == "raw_clock_as_new_york":
        ts = base_utc.dt.tz_localize(None)
    elif name.startswith("fixed_offset_"):
        offset = int(name.split("_")[-1])
        ts = (base_utc + pd.Timedelta(hours=offset)).dt.tz_localize(None)
    else:
        raise ValueError(name)
    out = raw[["open", "high", "low", "close", "volume"]].copy()
    out.insert(0, "ts", ts)
    return out.sort_values("ts").reset_index(drop=True)


def timezone_score(mapped: pd.DataFrame) -> dict[str, Any]:
    warm = mapped[mapped.ts < FRESH_START].copy()
    warm["minute"] = warm.ts.dt.hour * 60 + warm.ts.dt.minute
    warm["date"] = warm.ts.dt.normalize()
    rth = warm[(warm.minute >= 570) & (warm.minute < 960)]
    counts = rth.groupby("date").size()
    complete = int((counts >= 300).sum())
    weekdays = max(1, int(len(pd.bdate_range(warm.ts.min().normalize(), (FRESH_START - pd.Timedelta(days=1)).normalize())))) if len(warm) else 1
    hourly = warm.assign(hour=warm.ts.dt.hour).groupby("hour").volume.mean()
    peak_hour = int(hourly.idxmax()) if len(hourly) else -1
    peak_ok = peak_hour in {9, 10, 11}
    return {
        "warmup_rows": int(len(warm)),
        "complete_rth_sessions": complete,
        "complete_share_of_weekdays": float(complete / weekdays),
        "median_rth_bars": float(counts.median()) if len(counts) else 0.0,
        "peak_volume_hour": peak_hour,
        "peak_ok": peak_ok,
    }


def choose_ev_timezone(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    names = ["utc_to_new_york", "raw_clock_as_new_york"] + [f"fixed_offset_{x}" for x in range(-5, 3)]
    scores: dict[str, Any] = {}
    for name in names:
        mapped = map_candidate(raw, name)
        scores[name] = timezone_score(mapped)
    def rank(name: str) -> tuple[Any, ...]:
        s = scores[name]
        return (
            int(s["peak_ok"]),
            s["complete_rth_sessions"],
            s["complete_share_of_weekdays"],
            s["median_rth_bars"],
            -abs(s["peak_volume_hour"] - 10),
            int(name == "utc_to_new_york"),
        )
    chosen = max(names, key=rank)
    return map_candidate(raw, chosen), {"chosen": chosen, "candidate_scores": scores}


def discover(primary_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    minute = base.load_ohlcv(primary_path, "primary_et")
    obs, audit = base.build_observations(minute, "internal_primary_2022_2025")
    obs = add_first_passage_labels(obs, minute)
    discovery = obs[(obs.session_date >= "2023-01-01") & (obs.session_date < "2024-01-01")].copy()
    val2024 = obs[(obs.session_date >= "2024-01-01") & (obs.session_date < "2025-01-01")].copy()
    confirm2025 = obs[(obs.session_date >= "2025-01-01") & (obs.session_date < "2025-12-01")].copy()
    rules = mine_rules(discovery, val2024, confirm2025)
    internal: dict[str, Any] = {}
    for name, frame in [("discovery_2023", discovery), ("validation_2024", val2024), ("confirmation_2025", confirm2025)]:
        scored = apply_engine(frame, rules)
        internal[name] = {
            "sessions": int(len(frame)),
            "up_event_rate": float(frame.up_event.mean()),
            "down_event_rate": float(frame.down_event.mean()),
            "ambiguous_rate": float(frame.ambiguous_event.mean()),
            "metrics": prediction_metrics(scored),
            "economic": economic_diagnostic(scored, minute),
        }
        scored.to_csv(out_dir / f"{name}_scored.csv", index=False)
    spec = {
        "version": "V12b-first-passage",
        "preregistration_commit": "339f348f51629cad444610f7321d4ab1d22eb9ef",
        "label": {"target_atr": TARGET_ATR, "adverse_atr": STOP_ATR, "decision_time_et": "10:00", "end_time_et": "16:00"},
        "audit": audit,
        "rules": rules,
        "rule_descriptions": rule_descriptions(rules),
        "internal_results": internal,
    }
    (out_dir / "v12b_frozen_spec.json").write_text(json.dumps(json_safe(spec), indent=2), encoding="utf-8")
    pd.DataFrame(rule_descriptions(rules)).to_csv(out_dir / "v12b_selected_rules.csv", index=False)
    print("V12B_DISCOVERY_COMPLETE")
    print(json.dumps(json_safe({
        "split_sizes": {"2023": len(discovery), "2024": len(val2024), "2025": len(confirm2025)},
        "selected_rule_counts": {k: len(rules.get(k, [])) for k in OUTCOME_COLUMNS},
        "internal_results": internal,
    }), indent=2))


def evaluate(ev_path: Path, spec_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    raw, parse_audit = parse_ev_json(ev_path)
    mapped, timezone_audit = choose_ev_timezone(raw)
    obs, audit = base.build_observations(mapped, "fresh_EV_NQ_2026_proxy")
    obs = add_first_passage_labels(obs, mapped)
    fresh = obs[obs.session_date >= FRESH_START].copy().reset_index(drop=True)
    scored = apply_engine(fresh, spec["rules"])
    metrics = prediction_metrics(scored)
    random = random_benchmark(scored, metrics)
    economic = economic_diagnostic(scored, mapped)
    long_n = int(metrics.get("long", {}).get("n", 0))
    short_n = int(metrics.get("short", {}).get("n", 0))
    checks = {
        "at_least_15_predictions": bool(metrics.get("n", 0) >= 15),
        "precision_plus_10pp_vs_random_base": bool(metrics.get("precision_improvement_vs_random_base", -np.inf) >= 0.10),
        "random_percentile_at_least_95": bool(random.get("observed_percentile") is not None and random["observed_percentile"] >= 0.95),
        "positive_mean_signed_return": bool(metrics.get("mean_signed_return_atr", -np.inf) > 0),
        "positive_net_expectancy": bool(economic.get("expectancy_net_r", -np.inf) > 0),
        "zero_causality_violations": bool(audit.get("causality_violations") == 0),
        "both_long_and_short_present": bool(long_n > 0 and short_n > 0),
    }
    final = {
        "version": "V12b-first-passage",
        "preregistration_commit": spec.get("preregistration_commit"),
        "source_limitation": "EV Trading Labs NQ is an independent public Nasdaq feed and is not asserted to be a CME-certified exchange tape.",
        "parse_audit": parse_audit,
        "timezone_audit": timezone_audit,
        "feature_audit": audit,
        "fresh_start": str(FRESH_START.date()),
        "fresh_sessions": int(len(fresh)),
        "fresh_start_actual": str(fresh.session_date.min()) if len(fresh) else None,
        "fresh_end_actual": str(fresh.session_date.max()) if len(fresh) else None,
        "selected_rules": spec.get("rule_descriptions", []),
        "primary_metrics": metrics,
        "random_benchmark": random,
        "economic_diagnostic": economic,
        "success_checks": checks,
        "validated": bool(all(checks.values())),
        "classification": "complete_direction_engine" if long_n > 0 and short_n > 0 else "one_sided_or_no_predictions",
    }
    scored.to_csv(out_dir / "v12b_fresh_scored_sessions.csv", index=False)
    (out_dir / "v12b_fresh_results.json").write_text(json.dumps(json_safe(final), indent=2), encoding="utf-8")
    print("V12B_FRESH_HOLDOUT_COMPLETE")
    print(json.dumps(json_safe(final), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--primary", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--ev", type=Path, required=True)
    e.add_argument("--spec", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "discover":
        discover(args.primary, args.out)
    else:
        evaluate(args.ev, args.spec, args.out)


if __name__ == "__main__":
    main()
