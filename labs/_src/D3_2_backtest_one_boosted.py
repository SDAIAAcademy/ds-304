# %% [markdown]
# # Lab 3.2 · One boosted model across feeders
#
# Day 3 · Session 2 (Global boosted forecasters) · 25 min.
#
# You train one LightGBM model on all established feeders at once. Every row is a feeder, an origin and a day
# of the horizon; every feature is something known on the origin day. You backtest it on the eight held-out
# weeks against the feeders' floor, then add quantile models and measure their coverage.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/feeders.npz`, `data/feeders_meta.csv`, `data/regions.csv` and `data/calendar.csv`.
# Nothing from your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Read the frame: one row per feeder, origin and horizon day (4')
# - [ ] 2 · Train one boosted model on scaled targets (6')
# - [ ] 3 · Backtest on eight held-out weeks against the floor (6')
# - [ ] 4 · Train quantile models and measure coverage (6')
# - [ ] 5 · Read the feature importances (3')
# - [ ] ✱ Stretch · Train without scaling the target and compare
# - [ ] ✱ Stretch · Retrain at the second origin and compare the second fold

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-304.git"
    if not Path("/content/forecasting").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/forecasting"], check=True)
    os.chdir("/content/forecasting/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("shams.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import shams as U

plt = U.style()
import time
import lightgbm as lgb

W = U.feeders()[U.established()]
if U.QUICK:                                        # quick mode: three regions, fewer training origins
    m = U.meta()
    W = W[[f for f in W.columns if f in set(m.feeder[m.region.isin(["R01", "R05", "R11"])])]]
TEST = U.FEEDER_ORIGINS                            # two folds of 28 days: the eight held-out weeks
# Training origins every 14 days, the last one 14 days before the first test origin
TRAIN_ORIGINS = [TEST[0] - pd.Timedelta(days=14 * k) for k in range(U.pick(27, 8), 0, -1)]
ROUNDS = U.pick(300, 80)
t0 = time.perf_counter()
frame = U.make_frame(W, TRAIN_ORIGINS + TEST)
seconds_frame = time.perf_counter() - t0
print(f"{len(frame):,} rows for {W.shape[1]} feeders and {len(TRAIN_ORIGINS) + len(TEST)} origins, built in {seconds_frame:.0f} s")

# %% [markdown]
# ## 1 · Read the frame (4')
#
# | Column | What it is, as known on the origin day |
# |---|---|
# | `h`, `dow`, `weekend`, `ramadan`, `eid`, `national` | the target day: steps ahead and its calendar |
# | `level` | the feeder's mean demand over the 28 days up to the origin |
# | `last7` | its last week against that level |
# | `dow_profile` | its usual value on the target's weekday, against that level |
# | `year_ago` | the target day one year earlier, against the level one year earlier |
# | `cdd_change` | cooling degrees expected on the target day (temperature known at the origin) minus the last 28 days' |
# | `kind`, `region` | what the feeder is and where |
# | `y` | **the target**: demand on the target day divided by `level` |
#
# The frame holds training origins up to 14 days before the first test origin. Their horizons reach 28 days
# ahead, so some of their target days fall **inside the test window**.
#
# ✏️ **Exercise** · Return the rows a model may be trained on when the first forecast is made at
# `first_test_origin`: rows whose target day is known by then.

# %%
def training_rows(frame: pd.DataFrame, first_test_origin: pd.Timestamp) -> pd.DataFrame:
    # BEGIN SOLUTION
    return frame[frame.date <= first_test_origin]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    train = training_rows(frame, TEST[0])
except NotImplementedError:
    train = frame[frame.date <= TEST[0]]
test = frame[frame.origin.isin(TEST)].copy()
leaky = frame[(frame.origin < TEST[0]) & (frame.date > TEST[0])]
print(f"training rows: {len(train):,}; test rows: {len(test):,}; rows that would have leaked: {len(leaky):,}")
U.check(train.date.max() <= TEST[0], "no training target lies after the first test origin")
U.check(len(leaky) > 0 and not set(leaky.index) & set(train.index), f"{len(leaky):,} rows of the last training origin are left out: their targets are test days")

# %% [markdown]
# **Look at.** Splitting by origin (`origin < first test origin`) looks right and leaks: the last training
# origin's days 15 to 28 are the first test fold's days 1 to 14.

# %% [markdown]
# ## 2 · One model for every feeder (6')
#
# The target is scaled by the feeder's own level, so a feeder of 20 MWh and one of 200 MWh look alike and one
# model can learn from both. `U.LGB` holds the settings; `U.FEATURES` the feature columns.
#
# ✏️ **Exercise** · Train a LightGBM model with `lgb.train(params, lgb.Dataset(X, y), num_boost_round=ROUNDS)`.
# `objective` is `"l1"` (absolute error) for the point forecast; for a quantile model it is `"quantile"` with
# the parameter `alpha` set to the quantile.

# %%
def train_model(rows: pd.DataFrame, objective: str = "l1", alpha: float | None = None, target=None):
    # BEGIN SOLUTION
    params = {**U.LGB, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    y = rows.y if target is None else target
    return lgb.train(params, lgb.Dataset(rows[U.FEATURES], y), num_boost_round=ROUNDS)
    # END SOLUTION

# %%
def ref_train_model(rows, objective="l1", alpha=None, target=None):
    params = {**U.LGB, "objective": objective, **({"alpha": alpha} if alpha is not None else {})}
    return lgb.train(params, lgb.Dataset(rows[U.FEATURES], rows.y if target is None else target), num_boost_round=ROUNDS)


t0 = time.perf_counter()
try:
    model = train_model(train)
except NotImplementedError:
    train_model = ref_train_model
    model = train_model(train)
seconds_fit = time.perf_counter() - t0
print(f"one model, {model.num_trees()} trees, trained on {len(train):,} rows in {seconds_fit:.1f} s")
U.check(model.num_trees() == ROUNDS, "the model is trained")

# %% [markdown]
# ## 3 · Backtest on the eight held-out weeks (6')
#
# The model predicts the scaled target. ✏️ **Exercise** · Return the forecast in MWh for the given rows.

# %%
def predict_mwh(model, rows: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    return model.predict(rows[U.FEATURES]) * rows.level.to_numpy()
    # END SOLUTION

# %%
try:
    yhat = predict_mwh(model, test)
except NotImplementedError:
    predict_mwh = lambda model, rows: model.predict(rows[U.FEATURES]) * rows.level.to_numpy()
    yhat = predict_mwh(model, test)
bt = test[["feeder", "origin", "date", "h", "scale"]].assign(y=test.y * test.level, yhat=yhat)
floor = pd.concat([U.feeder_long(W, o, U.feeder_floor(W, o)) for o in TEST], ignore_index=True)
by_fold = {"floor": [round(U.mase(floor[floor.origin == o]), 3) for o in TEST],
           "boosted": [round(U.mase(bt[bt.origin == o]), 3) for o in TEST]}
print(f"floor (same weekday last year): {U.mase(floor):.3f}  by fold {by_fold['floor']}")
print(f"one boosted model:              {U.mase(bt):.3f}  by fold {by_fold['boosted']}")
skill = 1 - U.mase(bt) / U.mase(floor)
print(f"skill against the floor: {skill:.2f}")
if not U.QUICK:
    U.check(skill > 0.1, f"the boosted model removes {skill:.0%} of the floor's error on feeders it was never tuned for one by one")

# %% [markdown]
# **Look at.** The second fold is trained on nothing newer than the first: the same model serves both, 28 days
# apart. In production you would retrain as new data arrives; a backtest should say which of the two it did.

# %% [markdown]
# ## 4 · Quantile models (6')
#
# A boosted model trained with the pinball loss of quantile `alpha` predicts that quantile. Train one model
# per quantile: 0.1, 0.5 and 0.9.

# %%
t0 = time.perf_counter()
for q in U.PQS:
    bt[f"q{q}"] = predict_mwh(train_model(train, "quantile", q), test)
seconds_quantiles = time.perf_counter() - t0
sc = U.scores(bt, qs=U.PQS)
print(f"three quantile models in {seconds_quantiles:.1f} s:", sc)

# %% [markdown]
# ✏️ **Exercise** · Quantile models are trained separately, so nothing forces `q0.1 <= q0.5 <= q0.9` on every
# row. Return the share of rows where the three are out of order.

# %%
def crossing_share(bt: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    return float(((bt["q0.1"] > bt["q0.5"]) | (bt["q0.5"] > bt["q0.9"])).mean())
    # END SOLUTION

# %%
try:
    crossing = crossing_share(bt)
except NotImplementedError:
    crossing = float(((bt["q0.1"] > bt["q0.5"]) | (bt["q0.5"] > bt["q0.9"])).mean())
print(f"80% interval covers {sc['cover80']:.1%}; quantiles out of order on {crossing:.2%} of rows")
U.check(0 <= crossing < 0.2, "quantile crossing is rare here; sort the three per row before delivering them")
if not U.QUICK:
    U.check(0.6 < sc["cover80"] < 0.9, "coverage is measured, not assumed: read it before you promise 80%")

# %% [markdown]
# ## 5 · What the model uses (3')

# %%
gain = pd.Series(model.feature_importance("gain"), index=U.FEATURES)
gain = (gain / gain.sum()).sort_values(ascending=False).round(3)
print(gain.to_string())

# %% [markdown]
# ✏️ **Exercise** · Return the names of the three features with the largest share of the gain.

# %%
def top_three(gain: pd.Series) -> list:
    # BEGIN SOLUTION
    return list(gain.sort_values(ascending=False).index[:3])
    # END SOLUTION

# %%
try:
    top = top_three(gain)
except NotImplementedError:
    top = list(gain.index[:3])
print("top three:", top)
U.check(len(top) == 3 and top[0] == gain.index[0], "the largest share first")

# The stretch, done: the same model on the unscaled target (MWh)
raw = train_model(train, target=train.y * train.level)
bt_raw = bt.assign(yhat=raw.predict(test[U.FEATURES]))
print(f"trained on MWh without scaling: {U.mase(bt_raw):.3f}, against {U.mase(bt):.3f} with the scaled target")
U.keep("d3_2", dict(scores=sc, by_fold=by_fold, floor=round(U.mase(floor), 3), skill=round(skill, 3), crossing=crossing,
                    seconds=dict(frame=round(seconds_frame, 1), fit=round(seconds_fit, 1), quantiles=round(seconds_quantiles, 1)),
                    rows=dict(train=len(train), test=len(test), leaky=len(leaky)), fits=4, forecasts=2 * W.shape[1],
                    importance=gain.to_dict(), unscaled=round(U.mase(bt_raw), 3)))

# %% [markdown]
# **Look at.** Without scaling, none of the features tells the model how large a feeder is, and one tree
# cannot serve 20 MWh and 200 MWh at once. Scaling the target is what makes a global model possible.

# %% [markdown]
# ## ✱ Stretch · Retrain at the second origin
#
# Step 3 forecast the second fold with a model trained 28 days earlier. On the second origin, the 28 days of
# the first fold are observed and could be training rows.
#
# ✏️ **Exercise** · Train a new point model on the rows known at the second origin (`TEST[1]`), forecast the
# second fold with it, and return its scaled error. Reuse `training_rows`, `train_model` and `predict_mwh`.

# %%
def second_fold_retrained(frame: pd.DataFrame, test: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    fresh = train_model(training_rows(frame, TEST[1]))
    fold2 = test[test.origin == TEST[1]]
    b = fold2[["feeder", "origin", "date", "h", "scale"]].assign(y=fold2.y * fold2.level, yhat=predict_mwh(fresh, fold2))
    return U.mase(b)
    # END SOLUTION

# %%
try:
    fold2_fresh = second_fold_retrained(frame, test)
except NotImplementedError:
    fresh_rows = frame[frame.date <= TEST[1]]
    fold2 = test[test.origin == TEST[1]]
    fold2_fresh = U.mase(fold2[["feeder", "origin", "date", "h", "scale"]].assign(
        y=fold2.y * fold2.level, yhat=ref_train_model(fresh_rows).predict(fold2[U.FEATURES]) * fold2.level.to_numpy()))
fold2_old = U.mase(bt[bt.origin == TEST[1]])
print(f"second fold: model 28 days old {fold2_old:.3f}, model retrained at the second origin {fold2_fresh:.3f}")
U.check(abs(fold2_fresh - fold2_old) < 0.5, "both models scored on the same 388 feeders and 28 days")
U.keep("d3_2_retrain", dict(old=round(fold2_old, 3), fresh=round(fold2_fresh, 3)))

# %% [markdown]
# **Look at.** One fold is one comparison, inside the spread between folds that step 3 showed. Whether a
# retraining schedule pays is measured over many origins and set against what each training costs:
# Session 4 does that for a whole year.

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why is `year_ago` divided by the level of a year ago, not by today's level?</summary>
# The feature should say "this day was 20% above its level then". Dividing by today's level would mix in a
# year of growth.</details>
#
# <details><summary>2 · Which rows would leak with a split by origin?</summary>
# The rows of the last training origin whose target days fall after the first test origin. Their targets are
# test days.</details>
#
# <details><summary>3 · One model was trained for 388 feeders. What does a new feeder need?</summary>
# 28 days of history for its level and weekday profile. It needs no model of its own: Lab 4.2 uses this.</details>
#
# ## Carry forward
#
# The scores and run times of this lab are a row of the scorecard in Lab 3.4, supplied there as a reference
# file.
