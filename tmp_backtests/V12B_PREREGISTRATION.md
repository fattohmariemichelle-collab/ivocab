# V12b preregistration — corrected first-passage pattern study

Committed after V12 produced zero labels, and before downloading or inspecting the new post-15-April-2026 holdout.

## Why V12b is necessary

V12 defined path efficiency using every one-minute close change and required efficiency >= 0.25. On the internal 2023–2025 data, the maximum observed value was below that threshold, so no session could ever be labelled as a trend. This was a label-calibration failure, not evidence for or against a market pattern.

V12b removes the unusable efficiency label and replaces it with a first-passage definition fixed below.

## Internal data and splits

- Internal continuous NQ one-minute data: public 2022–2025 dataset previously used.
- 2022: warm-up.
- 2023: discovery.
- 2024: validation.
- 2025 through 30 November: confirmation.
- These internal periods are not claimed to be pristine.

## Fresh external holdout

- A separate public `NQ M1 2026` feed from EV Trading Labs will be downloaded only after this preregistration is committed.
- Only observations dated **2026-04-16 or later** will be used, so they do not overlap the already-inspected TopstepX holdout ending 2026-04-15.
- This source may be a broker/CFD-style NQ feed rather than an exchange-certified CME tape. It is used only as an independent replication proxy, and that limitation must be reported.
- The engine must infer and document timestamp units/timezone from file metadata and session-volume structure without moving timestamps to improve results. Permitted timezone candidates are UTC, America/New_York, and fixed UTC offsets from -5 to +2. Selection is made mechanically on the warm-up portion: maximize the share of full 390-minute RTH sessions and require the volume peak to fall between 09:00 and 11:00 ET. The selected mapping is frozen before scoring the post-16-April segment.

## Observation time

One decision per complete trading session at 10:00 New York time, after the 09:30–10:00 opening range is complete.

## Scale

`prior20_RTH_ATR` is the mean true range of the preceding 20 completed RTH sessions. No current-day post-decision data enter the scale.

## Future labels, fixed before the fresh holdout

Reference price: first available open at or after 10:00 ET.
Window: 10:00–16:00 ET.

- `UP_EVENT`: price reaches `entry + 0.50 × prior20_RTH_ATR` before reaching `entry - 0.35 × prior20_RTH_ATR`.
- `DOWN_EVENT`: price reaches `entry - 0.50 × prior20_RTH_ATR` before reaching `entry + 0.35 × prior20_RTH_ATR`.
- If both barriers are touched inside the same one-minute bar, the session is labelled ambiguous and neither event is assigned.
- If neither directional event occurs by 16:00, the label is `NO_EVENT`.

These future labels are outcomes only. They are never features.

## Causal feature set at 10:00

The same V12 causal features are retained:

1. prior-session return and range / prior ATR;
2. overnight range, return, close location and relative volume;
3. gap from prior RTH close to 09:30 open;
4. 09:00–09:30 return;
5. opening-range width, return, close location and relative volume;
6. distance at 10:00 from prior-session high/low;
7. distance at 10:00 from overnight high/low;
8. distance at 10:00 from causal session VWAP;
9. completed-H1 EMA20−EMA50 / H1 ATR;
10. completed-H1 DMI spread, ADX, 6h and 12h momentum;
11. day of week.

Every feature must pass the timestamp audit: source timestamp <= 10:00 ET.

## Rule-mining language

- Atomic numeric conditions: discovery-set q25/q50/q75 comparisons only.
- Day-of-week equality is permitted.
- One or two atomic conditions per rule.
- UP and DOWN rules are mined separately.

Discovery requirements on 2023:
- support >= 20 sessions;
- precision improvement >= 7 percentage points over the class base rate;
- lift >= 1.20.

Validation requirements without threshold changes:
- support >= 12 in 2024 and >= 12 in 2025;
- precision improvement > 0 in both years;
- lift > 1 in both years.

At most four non-redundant rules per direction are retained. Jaccard similarity >= 0.80 is treated as redundant.

## Prediction logic

- `up_vote` = retained UP rules firing.
- `down_vote` = retained DOWN rules firing.
- Predict LONG when up_vote > down_vote and up_vote >= 1.
- Predict SHORT when down_vote > up_vote and down_vote >= 1.
- Otherwise abstain.

No thresholds or rules may be changed after the external 2026 file is downloaded.

## Primary fresh-holdout metrics

On sessions from 2026-04-16 onward:

- predictions and coverage;
- correct first-passage direction precision;
- wrong first-passage direction rate;
- no-event rate;
- mean signed 10:00–16:00 return / prior ATR;
- Wilson 95% precision interval;
- random-direction/random-coverage benchmark with 10,000 repetitions;
- result by LONG and SHORT separately.

## Fixed trade diagnostic

For every prediction:
- enter first open at/after 10:00;
- target = +0.50 prior ATR in predicted direction;
- stop = -0.35 prior ATR;
- same-bar collision counts as a stop;
- close at 16:00 if neither barrier is hit;
- cost = 4 NQ ticks round trip.

Report expectancy in R, profit factor, drawdown and loss streak.

## Success criteria

V12b is only considered promising if all are true on the fresh post-April holdout:

1. at least 15 predictions;
2. correct-direction precision is at least 10 percentage points above the unconditional random-direction event baseline;
3. random benchmark percentile >= 95%;
4. mean signed return > 0;
5. net expectancy > 0 after 4 ticks;
6. no causality violation;
7. both LONG and SHORT predictions are present, or the result is explicitly classified as one-sided and not a complete direction engine.

Failure of any criterion means V12b is not validated.
