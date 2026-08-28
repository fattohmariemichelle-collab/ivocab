from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import nq_v13_fast_runner as fast

v13 = fast.v13


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--ev2024", type=Path, required=True)
    ap.add_argument("--ev2025", type=Path, required=True)
    ap.add_argument("--ev2026", type=Path, required=True)
    ap.add_argument("--topstep-nq", type=Path, required=True)
    ap.add_argument("--topstep-mnq", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("v13b_results"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    cfgs = [c for c in v13.configs() if c.family == "public_orb"]
    sources = {}
    audits = {}

    primary = v13.base.load_ohlcv(args.primary, "primary_et")
    obs, paths, audit = v13.add_day_context(primary, "primary")
    audits["primary"] = audit
    for year in [2023, 2024, 2025]:
        sources[f"primary_{year}"] = (
            obs[pd.to_datetime(obs.session_date).dt.year == year].copy(),
            {k: v for k, v in paths.items() if pd.Timestamp(k).year == year},
        )

    for year, path in [(2024, args.ev2024), (2025, args.ev2025), (2026, args.ev2026)]:
        minute, la = v13.load_ev(path)
        o, p, a = v13.add_day_context(minute, f"ev_{year}")
        audits[f"ev_{year}"] = {"load": la, "features": a}
        if year == 2026:
            sources["ev_2026_jan_apr"] = (o[pd.to_datetime(o.session_date) < pd.Timestamp("2026-04-16")], {k:v for k,v in p.items() if k < pd.Timestamp("2026-04-16")})
            sources["ev_2026_apr_jul"] = (o[pd.to_datetime(o.session_date) >= pd.Timestamp("2026-04-16")], {k:v for k,v in p.items() if k >= pd.Timestamp("2026-04-16")})
        else:
            sources[f"ev_{year}"] = (o, p)

    for label, path in [("topstep_nq", args.topstep_nq), ("topstep_mnq", args.topstep_mnq)]:
        minute = v13.load_topstep(path)
        o, p, a = v13.add_day_context(minute, label)
        audits[label] = a
        sources[label] = (o, p)

    frames=[]; trade_rows=[]
    for source, (o,p) in sources.items():
        for cost in [4.0, 8.0, 12.0]:
            m, trades = v13.evaluate_source(o,p,cfgs,cost)
            m["source"] = source; m["cost_ticks"] = cost
            frames.append(m)
            for name, rows in trades.items():
                for row in rows: trade_rows.append({"source":source,"cost_ticks":cost,**row})
    metrics=pd.concat(frames,ignore_index=True)
    metrics.to_csv(args.out/"orb_metrics_all_sources.csv",index=False)
    pd.DataFrame(trade_rows).to_csv(args.out/"orb_trades_all_sources.csv",index=False)

    summary_rows=[]
    for name,g in metrics[metrics.cost_ticks==4].groupby("config"):
        large=g[g.n>=15]
        all_nonempty=g[g.n>0]
        stress=metrics[(metrics.config==name)&(metrics.cost_ticks==12)&(metrics.n>0)]
        summary_rows.append({
            "config":name,
            "large_sources":int(len(large)),
            "positive_large_sources":int((large.expectancy_r>0).sum()),
            "min_large_expectancy":float(large.expectancy_r.min()) if len(large) else None,
            "all_nonempty_sources":int(len(all_nonempty)),
            "positive_all_nonempty_sources":int((all_nonempty.expectancy_r>0).sum()),
            "pooled_trades":int(all_nonempty.n.sum()),
            "weighted_expectancy":float((all_nonempty.expectancy_r*all_nonempty.n).sum()/all_nonempty.n.sum()) if all_nonempty.n.sum() else None,
            "positive_12tick_sources":int((stress.expectancy_r>0).sum()),
            "total_12tick_sources":int(len(stress)),
        })
    ranking=pd.DataFrame(summary_rows).sort_values(["positive_large_sources","min_large_expectancy","weighted_expectancy"],ascending=[False,False,False])
    ranking.to_csv(args.out/"orb_crossfeed_ranking.csv",index=False)
    result={"version":"V13b-public-ORB-audit","ranking":ranking.to_dict("records"),"audits":audits}
    (args.out/"summary.json").write_text(json.dumps(v13.json_safe(result),indent=2),encoding="utf-8")
    print("V13B_COMPLETE")
    print(ranking.to_string(index=False))
    print("\nDETAIL")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
