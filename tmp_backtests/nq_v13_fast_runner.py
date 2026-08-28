from __future__ import annotations

import pandas as pd
import numpy as np

import nq_v13_profitability_search as v13

_original_minute_with_session = v13.minute_with_session


def fast_minute_with_session(minute: pd.DataFrame) -> pd.DataFrame:
    x = _original_minute_with_session(minute)
    x["volume_med20_prior"] = (
        x.groupby("session_date", group_keys=False)["volume"]
        .apply(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    )
    return x


def fast_first_break(path: pd.DataFrame, start_minute: int, deadline: int, high: float, low: float,
                     buffer: float, allowed_side: int, relvol_floor: float = 0.0,
                     require_vwap_side: bool = False):
    g = path[(path.minute >= start_minute) & (path.minute <= deadline)]
    if g.empty:
        return None
    for bar in g.itertuples(index=False):
        med = getattr(bar, "volume_med20_prior", np.nan)
        relvol = float(bar.volume / med) if np.isfinite(med) and med > 0 else 1.0
        long_ok = allowed_side in {1, 2} and float(bar.close) > high + buffer
        short_ok = allowed_side in {-1, 2} and float(bar.close) < low - buffer
        if require_vwap_side:
            long_ok = long_ok and float(bar.close) > float(bar.rth_vwap_live)
            short_ok = short_ok and float(bar.close) < float(bar.rth_vwap_live)
        if relvol < relvol_floor:
            long_ok = short_ok = False
        if long_ok:
            return pd.Timestamp(bar.ts), 1
        if short_ok:
            return pd.Timestamp(bar.ts), -1
    return None


v13.minute_with_session = fast_minute_with_session
v13.first_break = fast_first_break

if __name__ == "__main__":
    v13.main()
