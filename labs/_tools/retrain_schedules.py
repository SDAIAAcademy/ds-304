"""How forecast error grows with the age of the model, under four retraining schedules.

    uv run python _tools/retrain_schedules.py        # from labs/, about a quarter of an hour on a laptop
    SHAMS_QUICK=1 uv run python _tools/retrain_schedules.py   # eight origins, a check that the code runs

The boosted model of Lab 3.2, with the same recipe (27 training origins 14 days apart, rows whose target day is
observed on the training day, the same settings), forecasts the 388 established feeders 28 days ahead from 48
weekly origins of 2025. One model is trained at every origin. A retraining schedule decides which of those
models serves each origin: the newest one trained on a retraining day of the schedule. Four schedules:

    weekly          a new model at every origin                          48 trainings in the year
    every 4 weeks                                                        12 trainings
    every 13 weeks                                                        4 trainings
    once            the model of the first origin serves the whole year   1 training

Reported for each schedule: the mean scaled error over the 48 origins, and the gap to a model trained that
same week, by the age of the model on the day it forecast. The gap is paired, origin by origin, so the season
of the origin cancels out: an old model forecasting in summer is compared with a new model forecasting the
same summer weeks. The same table is computed for each kind of feeder, and for a contrast: the same models
with each feeder's level kept from the training day instead of recomputed at the origin, as a model that
learns its level when it is fitted would keep it. The model trained at the first held-out origin of Lab 3.2
is checked against that lab's scores.

Writes data/reference/retrain_schedules.json (full run only), which the Day 3 slides on retraining quote.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shams as U  # noqa: E402
import lightgbm as lgb  # noqa: E402

ROUNDS = U.pick(300, 40)
N = U.pick(48, 8)                                                     # weekly forecast origins
SCHEDULES = {"weekly": 1, "every 4 weeks": 4, "every 13 weeks": 13, "once": N}
AGES = [(0, 0, "same week"), (1, 3, "1 to 3 weeks"), (4, 12, "4 to 12 weeks"), (13, 25, "13 to 25 weeks"),
        (26, 99, "26 weeks or more")]
KINDS = ["residential", "commercial", "industrial"]

W = U.feeders()[U.established()]
if U.QUICK:                                                           # quick mode: three regions
    m = U.meta()
    W = W[[f for f in W.columns if f in set(m.feeder[m.region.isin(["R01", "R05", "R11"])])]]
kind_of = U.meta().set_index("feeder").kind
last_origin = U.LAST - pd.Timedelta(days=U.H)                         # the last origin with 28 observed days
FC = [last_origin - pd.Timedelta(days=7 * k) for k in range(N - 1, -1, -1)]


def train_origins(r):
    """The recipe of Lab 3.2: 27 origins 14 days apart, the last one 14 days before the training day."""
    return [r - pd.Timedelta(days=14 * k) for k in range(27, 0, -1)]


ALL = sorted({o for r in FC for o in train_origins(r)} | set(FC))
t0 = time.perf_counter()
frame = U.make_frame(W, ALL)
print(f"{len(frame):,} rows, {len(ALL)} origins, built in {time.perf_counter() - t0:.0f} s", flush=True)
frame["kind_name"] = frame.feeder.map(kind_of)
tests = {o: frame[frame.origin == o] for o in FC}
level_at = {o: t.groupby("feeder")["level"].first() for o, t in tests.items()}   # each feeder's level on each origin

params = {**U.LGB, "objective": "l1"}
errors = {}                       # (trained at, forecast origin, kind or "all") -> mean scaled error over feeders
fits = []
for i, r in enumerate(FC):
    rows = frame[frame.origin.isin(train_origins(r)) & (frame.date <= r)]
    t1 = time.perf_counter()
    model = lgb.train(params, lgb.Dataset(rows[U.FEATURES], rows.y), num_boost_round=ROUNDS)
    fits.append(time.perf_counter() - t1)
    for o in FC[i:]:                                                  # every later origin a schedule could give it
        t = tests[o]
        p = model.predict(t[U.FEATURES])
        e = np.abs(t.y - p) * t.level / t.scale
        errors[(r, o, "all")] = float(e.mean())
        for k, g in e.groupby(t.kind_name):
            errors[(r, o, k)] = float(g.mean())
        # contrast: the same model with each feeder's level kept from its training day, as a model that learns
        # its level when it is fitted (a regression without recent lags) would keep it
        fixed = t.feeder.map(level_at[r]).to_numpy()
        errors[(r, o, "level fixed at training")] = float((np.abs(t.y * t.level - p * fixed) / t.scale).mean())
    print(f"model {i + 1}/{N} trained at {r.date()} on {len(rows):,} rows in {fits[-1]:.1f} s", flush=True)

if not U.QUICK:                                                       # the model of Lab 3.2, reproduced
    lab = U.reference("d3_2")["by_fold"]["boosted"]
    r0 = U.FEEDER_ORIGINS[0]
    mine = [round(errors[(r0, o, "all")], 3) for o in U.FEEDER_ORIGINS]
    print("Lab 3.2 by fold:", lab, " this script, model trained at the first held-out origin:", mine)
    U.check(np.allclose(mine, lab, atol=0.005), "the model of 5 November reproduces the scores of Lab 3.2")


def table(kind: str) -> dict:
    res = {}
    for name, every in SCHEDULES.items():
        recs = []
        for j, o in enumerate(FC):
            trained = FC[(j // every) * every]
            recs.append(dict(age=(o - trained).days // 7, e=errors[(trained, o, kind)],
                             gap=errors[(trained, o, kind)] - errors[(o, o, kind)]))
        d = pd.DataFrame(recs)
        res[name] = dict(trainings=int(-(-N // every)), mase=round(float(d.e.mean()), 3),
                         gap_to_weekly=round(float(d.gap.mean()), 3),
                         worst_gap=round(float(d.gap.max()), 3))
    # every model at every age, whatever the schedule: how fast one model goes stale, paired by origin
    curve = pd.DataFrame([dict(age=(o - r).days // 7, gap=e - errors[(o, o, kind)])
                          for (r, o, k), e in errors.items() if k == kind])
    by_age = {}
    for lo, hi, label in AGES:
        g = curve[(curve.age >= lo) & (curve.age <= hi)].gap
        if len(g):
            by_age[label] = dict(gap=round(float(g.mean()), 3), share_worse=round(float((g > 0).mean()), 2), n=int(len(g)))
    return dict(schedules=res, gap_by_age=by_age)


fresh = np.array([errors[(o, o, "all")] for o in FC])
out = {"origins": [str(FC[0].date()), str(FC[-1].date())], "n_origins": N, "feeders": int(W.shape[1]),
       "seconds_per_training": round(float(np.mean(fits)), 1),
       "spread_between_origins": round(float(fresh.std(ddof=1)), 3),
       "all": table("all"), "by_kind": {k: table(k) for k in KINDS},
       "level_fixed_at_training": table("level fixed at training")}
print(json.dumps(out, indent=1))
if not U.QUICK:
    (U.REF / "retrain_schedules.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
