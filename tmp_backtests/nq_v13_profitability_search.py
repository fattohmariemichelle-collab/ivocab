from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nq_v12_preregistered_patterns as base
import nq_v12b_first_passage as v12b

TICK_SIZE = 0.25
BASE_COST_TICKS = 4.0
RNG = np.random.default_rng(20260829)

# V12b rules were frozen before the EV Apr-Jul 2026 holdout was opened.
U2_OR_RANGE_ATR = 0.36017789500799474
D1_OR_VOL = 1.1220418366080693
D1_ON_VOL = 1.1855622112485902
D3_DIST_ON_HIGH = -0.21258134123326072


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    gate: str
    trigger: str
    stop_mode: str
    stop_value: float
    rr: float
    deadline_minute: int
    buffer_atr: float = 0.0
    buffer_points: float = 0.0
    day_filter: str = "all"
    require_score3: bool = False
    require_gap_alignment: bool = False
    require_breakout_relvol: float = 0.0


def json_safe(x: Any) -> Any:
    return base.json_safe(x)


def load_ev(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, parse_audit = v12b.parse_ev_json(path)
    mapped, timezone_audit = v12b.choose_ev_timezone(raw)
    return mapped, {"parse": parse_audit, "timezone": timezone_audit}


def load_topstep(path: Path) -> pd.DataFrame:
    return base.load_ohlcv(path, "holdout_utc")


def minute_with_session(minute: pd.DataFrame) -> pd.DataFrame:
    x = base.sessionize(minute)
    x = x.sort_values("ts").copy()
    x["typical"] = (x.high + x.low + x.close) / 3.0
    rth = (x.minute >= 570) & (x.minute < 960)
    x["rth_pv"] = np.where(rth, x.typical * x.volume, 0.0)
    x["rth_vol"] = np.where(rth, x.volume, 0.0)
    x["cum_rth_pv"] = x.groupby("session_date").rth_pv.cumsum()
    x["cum_rth_vol"] = x.groupby("session_date").rth_vol.cumsum()
    x["rth_vwap_live"] = x.cum_rth_pv / x.cum_rth_vol.replace(0, np.nan)
    return x


def add_day_context(minute: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame], dict[str, Any]]:
    x = minute_with_session(minute)
    obs, audit = base.build_observations(minute, source_name)

    rth = x[(x.minute >= 570) & (x.minute < 960)].copy()
    daily = rth.groupby("session_date", sort=True).agg(
        prevday_high=("high", "max"), prevday_low=("low", "min"),
        prevday_close=("close", "last"), prevday_pv=("rth_pv", "sum"),
        prevday_vol=("rth_vol", "sum"),
    )
    daily["prevday_vwap"] = daily.prevday_pv / daily.prevday_vol.replace(0, np.nan)
    daily = daily.shift(1)

    or15 = x[(x.minute >= 570) & (x.minute < 585)].groupby("session_date", sort=True).agg(
        or15_open=("open", "first"), or15_high=("high", "max"), or15_low=("low", "min"),
        or15_close=("close", "last"), or15_volume=("volume", "sum"), or15_count=("ts", "size"),
        or15_last_ts=("ts", "max"),
    )
    vwap935 = x[x.minute == 575].groupby("session_date").last()[["rth_vwap_live"]].rename(columns={"rth_vwap_live": "vwap_935"})
    vwap944 = x[x.minute == 584].groupby("session_date").last()[["rth_vwap_live"]].rename(columns={"rth_vwap_live": "vwap_944"})

    obs = obs.set_index("session_date").join(daily, how="left").join(or15, how="left").join(vwap935, how="left").join(vwap944, how="left").reset_index()
    obs["or15_range_points"] = obs.or15_high - obs.or15_low
    obs["or15_range_atr"] = obs.or15_range_points / obs.prior_atr20.replace(0, np.nan)
    obs["gap15_points"] = ((obs.or15_high + obs.or15_low) / 2.0) - obs.prev_rth_close
    obs["gap15_atr"] = obs.gap15_points / obs.prior_atr20.replace(0, np.nan)

    # Frozen V12b component-rule hits.
    obs["U2"] = (obs.or_range_atr >= U2_OR_RANGE_ATR).astype(int)
    obs["D1"] = ((obs.or_volume_ratio >= D1_OR_VOL) & (obs.overnight_volume_ratio >= D1_ON_VOL)).astype(int)
    obs["D3"] = ((obs.or_volume_ratio >= D1_OR_VOL) & (obs.dist_on_high_atr <= D3_DIST_ON_HIGH)).astype(int)
    obs["D13"] = ((obs.D1 == 1) & (obs.D3 == 1)).astype(int)

    def confidence(row: pd.Series, side: int) -> int:
        values = [row.prevday_high, row.prevday_low, row.prevday_close, row.prevday_vwap, row.or15_close]
        if not all(np.isfinite(v) for v in values):
            return 0
        H, L, C = float(row.prevday_high), float(row.prevday_low), float(row.prevday_close)
        px = float(row.or15_close)
        P = (H + L + C) / 3.0
        R1 = 2 * P - L; R2 = P + (H - L)
        S1 = 2 * P - H; S2 = P - (H - L)
        score = 0
        if (side == 1 and px > P) or (side == -1 and px < P): score += 1
        if (side == 1 and px > row.prevday_vwap) or (side == -1 and px < row.prevday_vwap): score += 1
        if (side == 1 and R1 <= px <= R2) or (side == -1 and S2 <= px <= S1): score += 1
        slope = row.vwap_944 - row.vwap_935
        if np.isfinite(slope) and ((side == 1 and slope > 0) or (side == -1 and slope < 0)): score += 1
        return score

    obs["confidence_long"] = obs.apply(lambda r: confidence(r, 1), axis=1)
    obs["confidence_short"] = obs.apply(lambda r: confidence(r, -1), axis=1)
    obs["dow"] = pd.to_datetime(obs.session_date).dt.dayofweek

    day_paths = {
        pd.Timestamp(k): g.sort_values("ts").reset_index(drop=True)
        for k, g in x[(x.minute >= 570) & (x.minute < 960)].groupby("session_date", sort=False)
    }
    return obs, day_paths, audit


def pass_day_filter(row: pd.Series, mode: str) -> bool:
    dow = int(row.dow)
    if mode == "all": return True
    if mode == "no_mon": return dow != 0
    if mode == "no_fri": return dow != 4
    if mode == "tue_thu": return dow in {1, 2, 3}
    if mode == "no_mon_fri": return dow not in {0, 4}
    raise ValueError(mode)


def gate_side(row: pd.Series, gate: str) -> int:
    if gate == "short_d13":
        return -1 if row.D13 == 1 else 0
    if gate == "long_u2":
        return 1 if row.U2 == 1 else 0
    if gate == "mix_d13_else_u2":
        if row.D13 == 1: return -1
        if row.U2 == 1: return 1
        return 0
    if gate == "wide_market":
        return 2 if row.U2 == 1 else 0  # 2 = market chooses side
    if gate == "public":
        return 2
    if gate == "highvol_market":
        return 2 if row.or_volume_ratio >= D1_OR_VOL else 0
    if gate == "pm_market":
        return 2
    if gate == "asia_market":
        return 2
    raise ValueError(gate)


def trigger_immediate(row: pd.Series, path: pd.DataFrame, side: int, cfg: Config) -> tuple[pd.Timestamp, int] | None:
    g = path[path.minute >= 600]
    if g.empty or side not in {-1, 1}: return None
    return pd.Timestamp(g.iloc[0].ts) - pd.Timedelta(seconds=1), side


def first_break(path: pd.DataFrame, start_minute: int, deadline: int, high: float, low: float,
                buffer: float, allowed_side: int, relvol_floor: float = 0.0,
                require_vwap_side: bool = False) -> tuple[pd.Timestamp, int] | None:
    g = path[(path.minute >= start_minute) & (path.minute <= deadline)]
    if g.empty: return None
    for bar in g.itertuples(index=False):
        local_med = path[(path.ts < bar.ts) & (path.ts >= bar.ts - pd.Timedelta(minutes=20))].volume.median()
        relvol = float(bar.volume / local_med) if np.isfinite(local_med) and local_med > 0 else 1.0
        long_ok = allowed_side in {1, 2} and float(bar.close) > high + buffer
        short_ok = allowed_side in {-1, 2} and float(bar.close) < low - buffer
        if require_vwap_side:
            long_ok = long_ok and float(bar.close) > float(bar.rth_vwap_live)
            short_ok = short_ok and float(bar.close) < float(bar.rth_vwap_live)
        if relvol < relvol_floor:
            long_ok = short_ok = False
        if long_ok: return pd.Timestamp(bar.ts), 1
        if short_ok: return pd.Timestamp(bar.ts), -1
    return None


def first_rejection(path: pd.DataFrame, start_minute: int, deadline: int, high: float, low: float,
                    buffer: float, allowed_side: int) -> tuple[pd.Timestamp, int] | None:
    g = path[(path.minute >= start_minute) & (path.minute <= deadline)]
    if g.empty: return None
    for bar in g.itertuples(index=False):
        if allowed_side in {-1, 2}:
            if float(bar.high) >= high + buffer and float(bar.close) < high and float(bar.close) < float(bar.rth_vwap_live):
                return pd.Timestamp(bar.ts), -1
        if allowed_side in {1, 2}:
            if float(bar.low) <= low - buffer and float(bar.close) > low and float(bar.close) > float(bar.rth_vwap_live):
                return pd.Timestamp(bar.ts), 1
    return None


def event_for_config(row: pd.Series, path: pd.DataFrame, cfg: Config) -> tuple[pd.Timestamp, int] | None:
    if not pass_day_filter(row, cfg.day_filter): return None
    side = gate_side(row, cfg.gate)
    if side == 0: return None
    atr = float(row.prior_atr20)
    if not np.isfinite(atr) or atr <= 0: return None
    buffer = cfg.buffer_points + cfg.buffer_atr * atr

    if cfg.trigger == "immediate10":
        return trigger_immediate(row, path, side, cfg)

    if cfg.trigger == "or30_side_break":
        return first_break(path, 600, cfg.deadline_minute, float(row.or_high), float(row.or_low), buffer, side,
                           cfg.require_breakout_relvol, True)
    if cfg.trigger == "or30_any_break":
        return first_break(path, 600, cfg.deadline_minute, float(row.or_high), float(row.or_low), buffer, 2,
                           cfg.require_breakout_relvol, True)
    if cfg.trigger == "or30_rejection":
        return first_rejection(path, 600, cfg.deadline_minute, float(row.or_high), float(row.or_low), buffer, side)

    if cfg.trigger.startswith("or15"):
        if not (np.isfinite(row.or15_high) and np.isfinite(row.or15_low)): return None
        # Public-style range and gap gates.
        if cfg.family.startswith("public"):
            if not (55.0 <= float(row.or15_range_points) <= 110.0): return None
            gap = float(row.gap15_points)
            if cfg.require_gap_alignment:
                if side == 2:
                    # Direction is chosen only when the breakout agrees with the gap.
                    pass
                elif side * gap <= 20.0:
                    return None
        elif cfg.family.startswith("atr_orb"):
            if not (0.12 <= float(row.or15_range_atr) <= 0.35): return None

        event = first_break(path, 586, cfg.deadline_minute, float(row.or15_high), float(row.or15_low), buffer, 2,
                            cfg.require_breakout_relvol, False)
        if event is None: return None
        ts, detected_side = event
        if cfg.require_gap_alignment:
            gap = float(row.gap15_points)
            if detected_side * gap <= 20.0: return None
        if cfg.require_score3:
            score = int(row.confidence_long if detected_side == 1 else row.confidence_short)
            if score < 3: return None
        return ts, detected_side

    if cfg.trigger == "pm_orb":
        pm = path[(path.minute >= 780) & (path.minute < 795)]
        if len(pm) < 12: return None
        high, low = float(pm.high.max()), float(pm.low.min())
        rng = high - low
        if not (15.0 <= rng <= 60.0): return None
        return first_break(path, 795, cfg.deadline_minute, high, low, cfg.buffer_points, 2, 0.0, False)

    if cfg.trigger == "asia_gap":
        # Not available in RTH-only day path; handled separately.
        return None
    raise ValueError(cfg.trigger)


def stop_points(row: pd.Series, cfg: Config) -> float:
    if cfg.stop_mode == "atr": return cfg.stop_value * float(row.prior_atr20)
    if cfg.stop_mode == "points": return cfg.stop_value
    raise ValueError(cfg.stop_mode)


def simulate(path: pd.DataFrame, signal_ts: pd.Timestamp, side: int, row: pd.Series,
             cfg: Config, cost_ticks: float) -> dict[str, Any] | None:
    future = path[path.ts > signal_ts].copy()
    if future.empty: return None
    entry_bar = future.iloc[0]
    if (pd.Timestamp(entry_bar.ts) - signal_ts).total_seconds() > 5 * 60: return None
    entry = float(entry_bar.open)
    risk = stop_points(row, cfg)
    if not np.isfinite(risk) or risk <= TICK_SIZE: return None
    target_distance = cfg.rr * risk
    stop = entry - side * risk
    target = entry + side * target_distance
    exit_price = float(future.iloc[-1].close)
    reason = "time"
    exit_ts = pd.Timestamp(future.iloc[-1].ts)
    for bar in future.itertuples(index=False):
        if int(bar.minute) >= 955:
            exit_price = float(bar.close); exit_ts = pd.Timestamp(bar.ts); reason = "time"; break
        hit_stop = float(bar.low) <= stop if side == 1 else float(bar.high) >= stop
        hit_target = float(bar.high) >= target if side == 1 else float(bar.low) <= target
        if hit_stop:
            exit_price = stop; exit_ts = pd.Timestamp(bar.ts); reason = "stop"; break
        if hit_target:
            exit_price = target; exit_ts = pd.Timestamp(bar.ts); reason = "target"; break
    signed_points = side * (exit_price - entry) - cost_ticks * TICK_SIZE
    net_r = signed_points / risk
    return {
        "date": str(pd.Timestamp(row.session_date).date()), "config": cfg.name,
        "family": cfg.family, "side": side, "signal_ts": signal_ts,
        "entry_ts": pd.Timestamp(entry_bar.ts), "exit_ts": exit_ts,
        "entry": entry, "exit": exit_price, "stop_points": risk,
        "target_points": target_distance, "net_r": net_r, "reason": reason,
        "or30_range_atr": float(row.or_range_atr), "or15_range_atr": float(row.or15_range_atr),
        "or_volume_ratio": float(row.or_volume_ratio), "overnight_volume_ratio": float(row.overnight_volume_ratio),
        "confidence": int(row.confidence_long if side == 1 else row.confidence_short),
    }


def configs() -> list[Config]:
    out: list[Config] = []
    # The key repair: reduce V12's high-coverage vote engine to consensus gates and wait for price confirmation.
    for stop in [0.25, 0.35, 0.45]:
        for rr in [1.5, 2.0, 2.5]:
            suffix = f"s{stop:.2f}_r{rr:.1f}"
            out += [
                Config(f"d13_short_immediate_{suffix}", "v12_consensus", "short_d13", "immediate10", "atr", stop, rr, 600),
                Config(f"d13_short_break_{suffix}", "v12_consensus", "short_d13", "or30_side_break", "atr", stop, rr, 690, buffer_atr=.02, require_breakout_relvol=.8),
                Config(f"mix_confirmed_{suffix}", "v12_consensus", "mix_d13_else_u2", "or30_side_break", "atr", stop, rr, 690, buffer_atr=.02, require_breakout_relvol=.8),
                Config(f"wide_market_break_{suffix}", "market_direction", "wide_market", "or30_any_break", "atr", stop, rr, 690, buffer_atr=.02, require_breakout_relvol=.8),
                Config(f"d13_rejection_{suffix}", "rejection", "short_d13", "or30_rejection", "atr", stop, rr, 690, buffer_atr=.01),
            ]

    # Public ORB benchmark and ATR-normalized versions. Direction is chosen by the actual breakout.
    for rr in [2.0, 3.0]:
        out.append(Config(f"public_orb_score3_r{rr:.0f}", "public_orb", "public", "or15_break", "points", 27.0, rr, 630,
                          buffer_points=4.0, day_filter="no_mon_fri", require_score3=True,
                          require_gap_alignment=True, require_breakout_relvol=1.0))
        out.append(Config(f"public_orb_raw_r{rr:.0f}", "public_orb", "public", "or15_break", "points", 27.0, rr, 630,
                          buffer_points=4.0, day_filter="no_mon_fri", require_score3=False,
                          require_gap_alignment=True, require_breakout_relvol=1.0))
    for stop in [.08, .10, .12]:
        for rr in [2.0, 2.5, 3.0]:
            out.append(Config(f"atr_orb_score3_s{stop:.2f}_r{rr:.1f}", "atr_orb", "public", "or15_break", "atr", stop, rr, 660,
                              buffer_atr=.01, day_filter="no_mon_fri", require_score3=True,
                              require_gap_alignment=False, require_breakout_relvol=1.0))

    # Afternoon ORB benchmark from an independent public research implementation.
    for rr in [2.0, 2.5, 3.0]:
        out.append(Config(f"pm_orb_r{rr:.1f}", "pm_orb", "pm_market", "pm_orb", "points", 22.0, rr, 855,
                          buffer_points=2.0, day_filter="no_mon_fri"))
    return out


def summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades: return {"n": 0}
    r = np.array([t["net_r"] for t in trades], float)
    equity = np.cumsum(r); peak = np.maximum.accumulate(np.r_[0.0, equity]); dd = peak[1:] - equity
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    streak = cur = 0
    for v in r:
        if v <= 0: cur += 1; streak = max(streak, cur)
        else: cur = 0
    return {
        "n": int(len(r)), "win_rate": float((r > 0).mean()), "expectancy_r": float(r.mean()),
        "median_r": float(np.median(r)), "profit_factor": float(gains / losses) if losses > 0 else None,
        "max_drawdown_r": float(dd.max()) if len(dd) else 0.0, "longest_loss_streak": int(streak),
        "total_r": float(r.sum()), "long_n": int(sum(t["side"] == 1 for t in trades)),
        "short_n": int(sum(t["side"] == -1 for t in trades)),
        "bootstrap90_mean_r": [float(x) for x in np.quantile([
            np.mean(RNG.choice(r, size=len(r), replace=True)) for _ in range(3000)
        ], [.05, .95])] if len(r) >= 5 else [None, None],
    }


def evaluate_source(obs: pd.DataFrame, paths: dict[pd.Timestamp, pd.DataFrame], cfgs: list[Config],
                    cost_ticks: float = BASE_COST_TICKS) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    rows = []
    all_trades: dict[str, list[dict[str, Any]]] = {}
    for cfg in cfgs:
        trades: list[dict[str, Any]] = []
        for row in obs.itertuples(index=False):
            s = pd.Series(row._asdict())
            path = paths.get(pd.Timestamp(s.session_date))
            if path is None or path.empty: continue
            event = event_for_config(s, path, cfg)
            if event is None: continue
            signal_ts, side = event
            trade = simulate(path, signal_ts, side, s, cfg, cost_ticks)
            if trade is not None: trades.append(trade)
        all_trades[cfg.name] = trades
        rows.append({"config": cfg.name, "family": cfg.family, "cost_ticks": cost_ticks, **summarize(trades)})
    return pd.DataFrame(rows), all_trades


def subset_year(obs: pd.DataFrame, year: int) -> pd.DataFrame:
    return obs[pd.to_datetime(obs.session_date).dt.year == year].copy()


def selection_table(primary_obs: pd.DataFrame, primary_paths: dict[pd.Timestamp, pd.DataFrame], cfgs: list[Config]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, dict[str, list[dict[str, Any]]]]]:
    metrics: dict[int, pd.DataFrame] = {}; trades: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for year in [2023, 2024, 2025]:
        o = subset_year(primary_obs, year)
        p = {k: v for k, v in primary_paths.items() if pd.Timestamp(k).year == year}
        metrics[year], trades[year] = evaluate_source(o, p, cfgs)
        metrics[year]["year"] = year
    merged = metrics[2023][["config", "family", "n", "expectancy_r", "profit_factor", "max_drawdown_r"]].rename(columns=lambda c: f"2023_{c}" if c not in {"config", "family"} else c)
    for year in [2024, 2025]:
        m = metrics[year][["config", "n", "expectancy_r", "profit_factor", "max_drawdown_r"]].rename(columns=lambda c: f"{year}_{c}" if c != "config" else c)
        merged = merged.merge(m, on="config", how="left")
    merged["eligible_2023_2024"] = (
        (merged["2023_n"] >= 15) & (merged["2024_n"] >= 15)
        & (merged["2023_expectancy_r"] > 0) & (merged["2024_expectancy_r"] > 0)
        & (merged["2023_profit_factor"] > 1.05) & (merged["2024_profit_factor"] > 1.05)
    )
    merged["selection_score"] = np.minimum(merged["2023_expectancy_r"], merged["2024_expectancy_r"]) + 0.25 * np.minimum(merged["2023_profit_factor"] - 1, merged["2024_profit_factor"] - 1)
    merged = merged.sort_values(["eligible_2023_2024", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    return merged, metrics, trades


def aggregate_external(selected: list[str], external: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]], cfg_by_name: dict[str, Config]) -> pd.DataFrame:
    rows=[]
    for source, (obs, paths) in external.items():
        cfgs=[cfg_by_name[n] for n in selected]
        m4,_=evaluate_source(obs, paths, cfgs, 4.0); m4["source"]=source; m4["stress"]="4ticks"; rows.append(m4)
        m8,_=evaluate_source(obs, paths, cfgs, 8.0); m8["source"]=source; m8["stress"]="8ticks"; rows.append(m8)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--ev2024", type=Path, required=True)
    ap.add_argument("--ev2025", type=Path, required=True)
    ap.add_argument("--ev2026", type=Path, required=True)
    ap.add_argument("--topstep-nq", type=Path, required=True)
    ap.add_argument("--topstep-mnq", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("v13_results"))
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    cfgs=configs(); cfg_by_name={c.name:c for c in cfgs}
    primary=base.load_ohlcv(args.primary,"primary_et")
    primary_obs,primary_paths,primary_audit=add_day_context(primary,"primary_2022_2025")
    selection,year_metrics,year_trades=selection_table(primary_obs,primary_paths,cfgs)
    selection.to_csv(args.out/"primary_selection.csv",index=False)
    pd.concat(year_metrics.values(),ignore_index=True).to_csv(args.out/"primary_year_metrics.csv",index=False)

    # Select without using 2025 or any external source. Preserve family diversity.
    eligible=selection[selection.eligible_2023_2024].copy()
    selected=[]
    for family,g in eligible.groupby("family",sort=False):
        selected.extend(g.head(2).config.tolist())
    selected=list(dict.fromkeys(selected))[:12]

    external: dict[str,tuple[pd.DataFrame,dict[pd.Timestamp,pd.DataFrame]]]={}
    audits={"primary":primary_audit}
    for year,path in [(2024,args.ev2024),(2025,args.ev2025),(2026,args.ev2026)]:
        minute,a=load_ev(path); obs,paths,audit=add_day_context(minute,f"ev_{year}")
        audits[f"ev_{year}"]={"load":a,"features":audit}
        if year==2026:
            external["ev_2026_jan_apr"]=(obs[pd.to_datetime(obs.session_date)<pd.Timestamp("2026-04-16")].copy(),{k:v for k,v in paths.items() if k<pd.Timestamp("2026-04-16")})
            external["ev_2026_apr_jul"]=(obs[pd.to_datetime(obs.session_date)>=pd.Timestamp("2026-04-16")].copy(),{k:v for k,v in paths.items() if k>=pd.Timestamp("2026-04-16")})
        else:
            external[f"ev_{year}_cross_source"]=(obs,paths)
    for label,path in [("topstep_nq_2026",args.topstep_nq),("topstep_mnq_2026",args.topstep_mnq)]:
        minute=load_topstep(path); obs,paths,audit=add_day_context(minute,label); audits[label]=audit; external[label]=(obs,paths)

    external_metrics=aggregate_external(selected,external,cfg_by_name)
    external_metrics.to_csv(args.out/"external_metrics.csv",index=False)

    # Robustness summary: primary 2025 confirmation plus independent/cross-source results.
    robust=[]
    for name in selected:
        row=selection[selection.config==name].iloc[0].to_dict()
        ext=external_metrics[(external_metrics.config==name)&(external_metrics.stress=="4ticks")]
        ext8=external_metrics[(external_metrics.config==name)&(external_metrics.stress=="8ticks")]
        sufficient=ext[ext.n>=8]
        sufficient8=ext8[ext8.n>=8]
        row.update({
            "external_sources_n":int(len(sufficient)),
            "external_positive_sources":int((sufficient.expectancy_r>0).sum()),
            "external_min_expectancy":float(sufficient.expectancy_r.min()) if len(sufficient) else None,
            "external_mean_expectancy":float(sufficient.expectancy_r.mean()) if len(sufficient) else None,
            "external_8tick_positive_sources":int((sufficient8.expectancy_r>0).sum()),
            "external_8tick_min_expectancy":float(sufficient8.expectancy_r.min()) if len(sufficient8) else None,
            "confirmed_primary_2025":bool(row.get("2025_n",0)>=12 and row.get("2025_expectancy_r",-99)>0 and row.get("2025_profit_factor",0)>1),
        })
        row["robust_candidate"] = bool(row["confirmed_primary_2025"] and row["external_sources_n"]>=3 and row["external_positive_sources"]==row["external_sources_n"] and row["external_8tick_positive_sources"]==len(sufficient8))
        robust.append(row)
    robust_df=pd.DataFrame(robust).sort_values(["robust_candidate","external_min_expectancy","selection_score"],ascending=[False,False,False])
    robust_df.to_csv(args.out/"robustness_ranking.csv",index=False)

    # Save selected trade lists for audit.
    trade_rows=[]
    for year in [2023,2024,2025]:
        for name in selected:
            for t in year_trades[year].get(name,[]): trade_rows.append({"source":f"primary_{year}",**t})
    pd.DataFrame(trade_rows).to_csv(args.out/"selected_primary_trades.csv",index=False)

    summary={
        "version":"V13-profitability-search",
        "method":{
            "selection":"Candidate configurations selected on primary 2023-2024 only; 2025 and all external feeds are confirmation diagnostics.",
            "causality":"All features use data available before the entry trigger; breakout/rejection entries execute at the next one-minute open.",
            "collision":"Stop wins any same-bar stop/target collision.",
            "costs":"4 ticks base and 8 ticks stress, deducted from every trade.",
            "global_limitation":"The overall research process has previously inspected aggregate 2025-2026 behavior, so these results are not equivalent to a pristine prospective live trial.",
        },
        "selected_configs":selected,
        "robust_candidates":robust_df[robust_df.robust_candidate].config.tolist() if len(robust_df) else [],
        "top_ranking":robust_df.head(15).to_dict("records") if len(robust_df) else [],
        "audits":audits,
    }
    (args.out/"summary.json").write_text(json.dumps(json_safe(summary),indent=2),encoding="utf-8")
    print("V13_COMPLETE")
    print(json.dumps(json_safe({"selected":selected,"robust":summary["robust_candidates"],"top":summary["top_ranking"][:8]}),indent=2))


if __name__=="__main__":
    main()
