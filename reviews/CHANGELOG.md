# Changes

## 2027.2.0 (2026-11-24)

- Added: comparing retraining schedules by how forecast error grows with the age of the model, on `model-age`, `retraining-schedules` and `when-models-go-stale` (day 3, session 4), with a new session outcome.
- Added: `labs/_tools/retrain_schedules.py`, which replays the boosted model of Lab 3.2 over 48 weekly origins of 2025 under four schedules, and its result `labs/data/reference/retrain_schedules.json`.
- Added: a stretch exercise in Lab 3.2, retraining at the second origin and comparing the second fold, with its reference solution and check.
- Added: practice question `q-retraining-schedule-01`.
- Tightened: the callout and notes of `accuracy-is-one-column` and the notes of `run-time-upkeep-licence` (day 3).
