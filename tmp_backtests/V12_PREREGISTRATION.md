# V12 preregistration — causal NQ pattern discovery

Committed before loading or scoring the 2026 TopstepX holdout.

## Objective

Use future price action only as a label, never as a feature, to discover repeatable causal configurations present at 10:00 New York time that predict a clean directional move from 10:00 to 16:00 ET.

## Data

### Internal research data
- Public NQ 1-minute continuous dataset: `tgtanalytics/nq-futures-1min-bar-2022-2025`.
- 2022: warm-up only.
- 2023: discovery/training.
- 2024: validation.
- 2025 through 30 November: internal confirmation.
- These years are not claimed to be pristine because previous research inspected them.

### Final untouched test
- Public TopstepX/ProjectX NQ 1-minute file from `axb0306/cme-futures-ohlc`.
- File known only by metadata before this preregistration: `NQ/NQ_1min_20260120_20260415.csv`.
- No 2026 bar values or outcomes will be inspected before the algorithm below is fixed.
- Timestamps are converted from UTC to `America/New_York` using timezone-aware conversion.

## Unit of observation

One observation per complete trading session at exactly 10:00 ET, after the 09:30–10:00 opening range has completed.

Sessions are excluded when:
- fewer than 300 RTH one-minute bars are present between 09:30 and 16:00;
- prior 20-session ATR is unavailable;
- absolute session gap exceeds 3 prior ATR, to reduce continuous-contract rollover contamination;
- any required OHLC field is missing.

## Labels

Reference price: first available open at or after 10:00 ET.
End price: final close at or before 16:00 ET.
Scale: mean true range of the previous 20 completed RTH sessions; current-day future data are not used in the scale.

For the 10:00–16:00 path:
- signed endpoint move = `(end - start) / prior20_RTH_ATR`;
- path efficiency = `abs(end - start) / sum(abs(minute close changes))`;
- adverse excursion is measured against the eventual direction and scaled by prior ATR.

Classes:
- `UP_CLEAN`: signed move >= +0.60 ATR, efficiency >= 0.25, adverse excursion <= 0.75 ATR;
- `DOWN_CLEAN`: signed move <= -0.60 ATR, efficiency >= 0.25, adverse excursion <= 0.75 ATR;
- `NO_CLEAN_TREND`: all other sessions.

Primary target is a correct clean trend in the predicted direction among all predictions, not accuracy conditional on a trend having occurred.

## Causal features available at 10:00 ET

All features are computed only from data timestamped no later than 10:00 ET:

1. prior-session return / prior ATR;
2. prior-session range / prior ATR;
3. overnight range / prior ATR;
4. overnight return / prior ATR;
5. overnight close location within overnight range;
6. gap from prior RTH close to 09:30 open / prior ATR;
7. 09:00–09:30 return / prior ATR;
8. opening-range width / prior ATR;
9. opening-range return / prior ATR;
10. opening-range close location within its high-low range;
11. opening-range volume / rolling 20-session median opening-range volume;
12. overnight volume / rolling 20-session median overnight volume;
13. 10:00 price distance from prior-session high and low / prior ATR;
14. 10:00 price distance from overnight high and low / prior ATR;
15. 10:00 price distance from causal session VWAP / prior ATR;
16. H1 EMA20−EMA50 / H1 ATR using completed hourly bars only;
17. H1 DMI spread and ADX using completed hourly bars only;
18. prior 6-hour and 12-hour momentum / H1 ATR;
19. day of week encoded categorically.

Month-of-year, exact calendar date, future opening-range values, future session high/low, and current-session final volume are forbidden.

## Pattern-search language

The main V12 engine is an interpretable rule miner.

Candidate atomic conditions are restricted to:
- feature <= discovery-set q25;
- feature <= discovery-set q50;
- feature >= discovery-set q50;
- feature >= discovery-set q75;
- day-of-week equals a specific weekday.

Candidate rules contain one or two atomic conditions only.

Rules are mined separately for `UP_CLEAN`, `DOWN_CLEAN`, and `ANY_CLEAN_TREND`.

Discovery filters on 2023:
- minimum support: 20 sessions;
- minimum lift over class base rate: 1.20;
- minimum precision improvement: +5 percentage points.

Validation filters, applied without changing thresholds:
- minimum support: 12 sessions in 2024 and 12 in 2025;
- lift > 1.00 in both 2024 and 2025;
- precision improvement > 0 in both 2024 and 2025;
- same predicted class in all periods.

Selection score:
`min(lift_2024, lift_2025) * sqrt(min(support_2024, support_2025))`.

At most three non-redundant rules per class are retained. A rule is redundant when its fired-session set has Jaccard similarity >= 0.80 with a higher-ranked retained rule.

## Prediction logic

At 10:00 ET:
1. `trend_gate` fires when at least one retained `ANY_CLEAN_TREND` rule fires.
2. `up_vote` is the number of retained UP rules firing.
3. `down_vote` is the number of retained DOWN rules firing.
4. Predict UP when gate fires and `up_vote > down_vote`.
5. Predict DOWN when gate fires and `down_vote > up_vote`.
6. Otherwise abstain.

No parameter or rule may be changed after the 2026 file is loaded.

## Benchmarks

The final report must compare V12 with:
- always LONG at 10:00;
- opening-range direction;
- H1 EMA20/EMA50 direction;
- random direction with identical prediction coverage, repeated 10,000 times.

## Primary holdout metrics

On 2026:
- number of predictions and coverage;
- correct clean-direction precision among all predictions;
- wrong clean-direction rate;
- no-clean-trend rate;
- mean signed 10:00–16:00 return in prior-ATR units;
- Wilson 95% interval for correct clean-direction precision;
- bootstrap 90% interval for mean signed return;
- lift versus unconditional correct-direction base rate;
- percentile versus the random-coverage benchmark.

## Fixed economic diagnostic

This is secondary, not a selection criterion:
- enter first open after 10:00 in predicted direction;
- initial stop = 0.75 prior ATR;
- target = 1.00 prior ATR;
- exit at 16:00 if neither is hit;
- same-bar target/stop collision is counted as a stop;
- cost = 4 NQ ticks round trip.

Report expectancy in R, profit factor, maximum drawdown in R, and longest loss streak.

## Success criteria

V12 is considered promising, not production-valid, only if all are true on the untouched 2026 holdout:

1. at least 12 predictions;
2. correct clean-direction precision exceeds its unconditional baseline by at least 10 percentage points;
3. mean signed return is positive;
4. random-coverage percentile >= 95%;
5. fixed economic diagnostic expectancy is positive after 4 ticks;
6. no causality audit violation is detected.

Failure of any criterion means the preregistered V12 pattern engine is not validated.
