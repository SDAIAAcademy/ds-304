# %% [markdown]
# # Lab 2.3 · Weather and Hijri regressors
#
# Day 2 · Session 3 (Weather and moving holidays) · 25 min.
#
# You give the ARIMA model what it cannot learn from the past of demand alone: the temperature, and the
# Hijri calendar. You measure the gain on Eid days, and you compare a temperature the desk could really have
# had at the origin with the one measured afterwards.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/regions.csv`, `data/calendar.csv` and the supplied floor,
# `data/reference/bt_floor.csv`. Nothing from your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Build cooling degrees and holiday regressors (5')
# - [ ] 2 · Fit the regression with weather only (5')
# - [ ] 3 · Add Ramadan, Eid and national days; measure the gain on Eid days (6')
# - [ ] 4 · Compare the honest temperature with the one measured later (6')
# - [ ] 5 · Read the coefficients in MWh (3')
# - [ ] ✱ Stretch · Give Eid al-Fitr and Eid al-Adha separate regressors

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/forecasting.git"
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
from statsforecast.models import ARIMA

RID = "R01"
r = U.region(RID)
y = r["demand"]
cal = U.calendar()
ORIGINS = U.origins(U.pick(13, 4))                 # quick mode: the last four folds
floor = U.bt("floor")
floor = floor[floor.origin.isin(ORIGINS)]


def weekly_arima():
    return ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)


def future_days(origin, h):
    return pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")

# %% [markdown]
# ## 1 · Regressors (5')
#
# - **Cooling degrees**: how far the day's temperature is above 22 °C, and 0 below it.
# - **Heating degrees**: how far it is below 16 °C, and 0 above it.
# - **Calendar**: `ramadan` (0 or 1), `eid` (a weight from 0 to 1: 1 on the first day of each Eid, lower on
#   the days around it), `national` (National Day and Founding Day).
#
# ✏️ **Exercise** · Write the two temperature transforms. They return Series.

# %%
def degrees(temp: pd.Series) -> tuple:
    # BEGIN SOLUTION
    cooling = (temp - 22).clip(lower=0)
    heating = (16 - temp).clip(lower=0)
    return cooling, heating
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    cdd, hdd = degrees(r.temp_c)
except NotImplementedError:
    cdd, hdd = (r.temp_c - 22).clip(lower=0), (16 - r.temp_c).clip(lower=0)
X_all = U.exog(RID, r.index)                       # the same transforms, plus the calendar columns
U.check(np.allclose(cdd, X_all.cdd) and np.allclose(hdd, X_all.hdd), "cooling and heating degrees match the helper's")
print(X_all.loc["2025-03-28":"2025-04-02"].round(1).to_string())

# %% [markdown]
# `U.exog(region, dates, origin=..., holidays=..., weather=...)` returns these columns. For days after
# `origin` it does **not** use the measured temperature: it uses `U.temp_known`, the normal for the date plus
# the anomaly of the origin day, fading by a fifth each day. That is what a forecaster without a weather
# service can know.

# %% [markdown]
# ## 2 · Weather only (5')
#
# A regression with ARIMA errors: demand = coefficients × regressors + an ARIMA series for what is left.
#
# ✏️ **Exercise** · Write the forecast function with weather regressors only (`holidays=False`). `X` covers
# the training days; `Xf` covers the horizon and must be given the origin.

# %%
def dyn_weather(train: pd.Series, h: int, origin: pd.Timestamp) -> pd.DataFrame:
    # BEGIN SOLUTION
    X = U.exog(RID, train.index, holidays=False)
    Xf = U.exog(RID, future_days(origin, h), origin=origin, holidays=False)
    return U.sf(weekly_arima(), train, h, X, Xf)
    # END SOLUTION

# %%
try:
    bt_w = U.backtest(y, dyn_weather, ORIGINS)
except NotImplementedError:
    bt_w = U.bt("dyn_weather")
    bt_w = bt_w[bt_w.origin.isin(ORIGINS)]
U.keep_bt("dyn_weather", bt_w)
print("weather only:", U.scores(bt_w), "| floor:", round(U.mase(floor), 3))
if not U.QUICK:
    U.check(U.mase(bt_w) < U.mase(floor), f"temperature alone beats the floor: {U.mase(bt_w):.2f} against {U.mase(floor):.2f}")

# %% [markdown]
# ## 3 · Add the Hijri calendar, and score the Eid days (6')
#
# ✏️ **Exercise** · Write the same function with the calendar columns too (the default of `U.exog`). Then
# write `score_where`: the scaled error of a backtest on the rows where `mask` is true.

# %%
def dyn_full(train: pd.Series, h: int, origin: pd.Timestamp) -> pd.DataFrame:
    # BEGIN SOLUTION
    X = U.exog(RID, train.index)
    Xf = U.exog(RID, future_days(origin, h), origin=origin)
    return U.sf(weekly_arima(), train, h, X, Xf)
    # END SOLUTION


def score_where(bt: pd.DataFrame, mask) -> float:
    # BEGIN SOLUTION
    rows = bt[np.asarray(mask)]
    return float((np.abs(rows.y - rows.yhat) / rows["scale"]).mean())
    # END SOLUTION

# %%
try:
    bt_full = U.backtest(y, dyn_full, ORIGINS)
except NotImplementedError:
    bt_full = U.bt("dynreg")
    bt_full = bt_full[bt_full.origin.isin(ORIGINS)]
U.keep_bt("dynreg", bt_full)


def ref_score_where(bt, mask):
    rows = bt[np.asarray(mask)]
    return float((np.abs(rows.y - rows.yhat) / rows["scale"]).mean())


try:
    score_where(bt_full, np.ones(len(bt_full), bool))
except NotImplementedError:
    score_where = ref_score_where
is_eid = cal.eid_weight.reindex(bt_full.date).to_numpy() > 0
gain = pd.DataFrame({name: {"all days": round(U.mase(b), 3), "Eid days": round(score_where(b, is_eid), 3),
                            "other days": round(score_where(b, ~is_eid), 3)}
                     for name, b in (("floor", floor), ("weather only", bt_w), ("weather + calendar", bt_full))}).T
print(f"{int(is_eid.sum())} Eid days in the test folds")
print(gain.to_string())
if not U.QUICK:
    U.check(gain.loc["weather + calendar", "Eid days"] < 0.75 * gain.loc["weather only", "Eid days"],
            f"on Eid days the error falls from {gain.loc['weather only', 'Eid days']} to {gain.loc['weather + calendar', 'Eid days']}")
    U.check(gain.loc["weather + calendar", "Eid days"] < 0.5 * gain.loc["floor", "Eid days"], "and it is less than half the floor's error there")

# %% [markdown]
# **Look at.** The calendar moves the score on all days a little and the score on Eid days a lot. An average
# over the year would have hidden most of the gain.

# %% [markdown]
# ## 4 · The honest temperature and the one measured later (6')
#
# `oracle=True` makes `U.exog` use the temperature that was measured on each future day. No forecaster has
# it at the origin. It is here so that you can see what the mistake looks like in a score.
#
# ✏️ **Exercise** · Write the oracle version of `dyn_full`, then return how much better it looks.

# %%
def dyn_oracle(train: pd.Series, h: int, origin: pd.Timestamp) -> pd.DataFrame:
    # BEGIN SOLUTION
    X = U.exog(RID, train.index)
    Xf = U.exog(RID, future_days(origin, h), origin=origin, oracle=True)
    return U.sf(weekly_arima(), train, h, X, Xf)
    # END SOLUTION

# %%
try:
    bt_oracle = U.backtest(y, dyn_oracle, ORIGINS)
except NotImplementedError:
    bt_oracle = U.bt("dynreg_oracle")
    bt_oracle = bt_oracle[bt_oracle.origin.isin(ORIGINS)]
U.keep_bt("dynreg_oracle", bt_oracle)
honest, oracle = U.mase(bt_full), U.mase(bt_oracle)
print(f"honest temperature: {honest:.3f}   measured later: {oracle:.3f}   flattery: {honest - oracle:.3f}")
print("by horizon week, honest:", U.by_week(bt_full).tolist(), "| measured later:", U.by_week(bt_oracle).tolist())
if not U.QUICK:
    U.check(honest - oracle > 0.1, f"the leak flatters the score by {honest - oracle:.2f}: a number you will not see in production")

# %% [markdown]
# **Look at.** The honest score is the one to report. The gap is also useful: it is what a good weather
# forecast could be worth to the desk.

# %% [markdown]
# ## 5 · The coefficients, in MWh (3')
#
# The coefficients of a regression with ARIMA errors read like any regression: MWh per unit of the regressor.

# %%
last_train = y.loc[:ORIGINS[-1]].iloc[-1096:]
X_last = U.exog(RID, last_train.index)
fit = weekly_arima().fit(last_train.to_numpy(), X=X_last.to_numpy())
coef = {name: round(float(fit.model_["coef"][f"ex_{k + 1}"]), 1) for k, name in enumerate(X_last.columns)}
print("MWh per unit:", coef, "| mean daily demand:", round(float(last_train.mean())), "MWh")

# %% [markdown]
# ✏️ **Exercise** · Return, as a share of mean daily demand in percent: the effect of one more cooling
# degree, and the effect of the first day of Eid.

# %%
def effects_percent(coef: dict, mean_demand: float) -> tuple:
    # BEGIN SOLUTION
    return coef["cdd"] / mean_demand * 100, coef["eid"] / mean_demand * 100
    # END SOLUTION

# %%
try:
    per_degree, eid_day = effects_percent(coef, float(last_train.mean()))
except NotImplementedError:
    per_degree, eid_day = coef["cdd"] / float(last_train.mean()) * 100, coef["eid"] / float(last_train.mean()) * 100
print(f"one more cooling degree: {per_degree:+.1f}% of a mean day; the first day of Eid: {eid_day:+.1f}%")
U.check(per_degree > 0 and eid_day < -5, "hotter days raise demand; Eid lowers it")
U.keep("d2_3", dict(gain=gain.T.to_dict(), honest=honest, oracle=oracle, coef=coef, per_degree=per_degree, eid_day=eid_day,
                    n_eid=int(is_eid.sum()), mean_demand=float(last_train.mean()), scores=U.scores(bt_full),
                    weeks_honest=U.by_week(bt_full).tolist(), weeks_oracle=U.by_week(bt_oracle).tolist()))

# %%
# The stretch, done: separate regressors for the two Eids (full mode only)
if not U.QUICK:
    def exog_two(dates, origin=None):
        X = U.exog(RID, dates, origin=origin)
        c = cal.reindex(pd.DatetimeIndex(dates))
        X["eid_fitr"] = (c.eid_weight * c.eid_fitr).to_numpy()
        X["eid_adha"] = (c.eid_weight * c.eid_adha).to_numpy()
        return X.drop(columns="eid")

    bt_two = U.backtest(y, lambda tr, h, origin: U.sf(weekly_arima(), tr, h, exog_two(tr.index), exog_two(future_days(origin, h), origin)), ORIGINS)
    two = dict(all_days=round(U.mase(bt_two), 3), eid_days=round(ref_score_where(bt_two, is_eid), 3))
    U.keep("d2_3_stretch", two)
    print("two Eid regressors:", two)

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why may the Eid regressor use future dates when the temperature may not use future values?</summary>
# The calendar is known at the origin for every future day. The temperature of a future day is not: only a
# forecast of it is.</details>
#
# <details><summary>2 · The oracle model scores far better. Why not report that score with a note?</summary>
# It answers a different question: how good the model would be with a perfect weather forecast. The planner
# will get the honest model's errors, so that is the score that describes the product.</details>
#
# <details><summary>3 · What does the fading anomaly assume?</summary>
# That a warm spell seen on the origin day continues and dies out over about two weeks. It is a simple
# stand-in for a weather forecast. A real desk would test its own weather feed in the same backtest.</details>
#
# ## Carry forward
#
# `bt_dynreg.csv`, the honest model, is saved as a reference file: the third backtest of Lab 2.4 and Lab 4.3.
# Lab 4.1 hides the oracle mistake inside a notebook for you to find.
