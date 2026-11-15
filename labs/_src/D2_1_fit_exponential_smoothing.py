# %% [markdown]
# # Lab 2.1 · Exponential smoothing against the floor
#
# Day 2 · Session 1 (Exponential smoothing) · 25 min.
#
# You write simple exponential smoothing by hand, fit an ETS model with a weekly season, and find that it
# loses to the floor. Then you give it the annual season through a decomposition and score it again. Every
# score is read against the floor of Lab 1.4.
#
# **AI policy: AI-tutor.** An assistant may explain output; it does not write the backtest.
#
# **Reference state.** `data/regions.csv` and the supplied floor, `data/reference/bt_floor.csv`. Nothing from
# your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Write simple exponential smoothing by hand (6')
# - [ ] 2 · Fit ETS with a weekly season and score it (6')
# - [ ] 3 · Add the annual season: decompose, then smooth (6')
# - [ ] 4 · Compare with the floor by horizon week (4')
# - [ ] 5 · Read the smoothing weights (3')
# - [ ] ✱ Stretch · Compare a damped trend with an undamped one at 28 days

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
from statsforecast.models import AutoETS, MSTL

y = U.region("R01")["demand"]
ORIGINS = U.origins(U.pick(13, 3))                 # quick mode: the last three folds
floor = U.bt("floor")
floor = floor[floor.origin.isin(ORIGINS)]
print(f"the floor on these {len(ORIGINS)} folds: scaled error {U.mase(floor):.3f}")

# %% [markdown]
# ## 1 · Simple exponential smoothing by hand (6')
#
# The level after each new value is a weighted average of that value and the previous level:
#
# `level = alpha * value + (1 - alpha) * level`
#
# The forecast for every future day is the last level. With `alpha` near 0 the level forgets slowly; with
# `alpha = 1` it is the last value: the naive forecast.
#
# ✏️ **Exercise** · Return the last level. Start the level at the first value.

# %%
def ses_level(values, alpha: float) -> float:
    # BEGIN SOLUTION
    level = values[0]
    for x in values[1:]:
        level = alpha * x + (1 - alpha) * level
    return float(level)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_ses_level(values, alpha):
    level = values[0]
    for x in values[1:]:
        level = alpha * x + (1 - alpha) * level
    return float(level)


try:
    ses_level([10.0, 20.0, 30.0], 0.5)
except NotImplementedError:
    ses_level = ref_ses_level
U.check(ses_level([10.0, 20.0, 30.0], 0.5) == 22.5, "levels 10, 15, 22.5 with alpha = 0.5")
U.check(ses_level([10.0, 20.0, 30.0], 1.0) == 30.0, "alpha = 1 is the naive forecast")

# Which alpha forecasts tomorrow best? One-step error inside the last three years, for six values of alpha.
v = y.iloc[-1096:].to_numpy()


def one_step_error(v, alpha):
    level, errors = v[0], []
    for x in v[1:]:
        errors.append(abs(x - level))
        level = alpha * x + (1 - alpha) * level
    return float(np.mean(errors))


grid = {a: round(one_step_error(v, a), 1) for a in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)}
print("one-step error in MWh by alpha:", grid)
best_alpha = min(grid, key=grid.get)
U.check(best_alpha >= 0.9, f"the best alpha is {best_alpha}: on daily demand, smoothing barely beats 'tomorrow equals today'")

# %% [markdown]
# **Look at.** The best alpha is at the top of the grid. Daily demand moves in long weather spells, so the
# newest value is the best guide to tomorrow. Remember this when you read the 28-day score.

# %% [markdown]
# ## 2 · ETS with a weekly season (6')
#
# ETS models add a seasonal part (and optionally a trend) to the level, each with its own smoothing weight.
# `AutoETS(season_length=7, model="ANA")` is additive errors, no trend, additive weekly season.
# `U.sf(model, train, h)` fits one model and returns the forecast with its 80% and 95% intervals.
#
# ✏️ **Exercise** · Write the forecast function for the backtest.

# %%
def ets_weekly(train: pd.Series, h: int) -> pd.DataFrame:
    # BEGIN SOLUTION
    return U.sf(AutoETS(season_length=7, model="ANA"), train, h)
    # END SOLUTION

# %%
try:
    bt_ets = U.backtest(y, ets_weekly, ORIGINS)
except NotImplementedError:
    bt_ets = U.bt("ets_weekly")
    bt_ets = bt_ets[bt_ets.origin.isin(ORIGINS)]
U.keep_bt("ets_weekly", bt_ets)
print("ETS, weekly season:", U.scores(bt_ets))
U.check(len(bt_ets) == len(ORIGINS) * 28 and "q0.9" in bt_ets, "one row per test day, with quantile columns")
if not U.QUICK:
    U.check(U.mase(bt_ets) > U.mase(floor) + 0.3, f"ETS with a weekly season scores {U.mase(bt_ets):.2f}: far worse than the floor at {U.mase(floor):.2f}")

# %% [markdown]
# **Look at.** A fitted model with intervals loses to "the same weekday last year". It knows the week and
# nothing about the summer: 28 days ahead, that is the larger part.

# %% [markdown]
# ## 3 · Add the annual season: decompose, then smooth (6')
#
# ETS cannot hold a 365-day season on daily data. So split the work: a decomposition takes out the weekly
# and annual seasons (as in Lab 1.3), ETS forecasts what is left, and the seasons are added back.
# `MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"))` does the three steps.
#
# ✏️ **Exercise** · Fit it on the logarithm of the training data and return the forecast on the original
# scale (`np.exp` of the whole table, quantiles included).

# %%
def stl_ets(train: pd.Series, h: int) -> pd.DataFrame:
    # BEGIN SOLUTION
    model = MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"))
    return np.exp(U.sf(model, np.log(train), h))
    # END SOLUTION

# %%
try:
    bt_stl = U.backtest(y, stl_ets, ORIGINS)
except NotImplementedError:
    bt_stl = U.bt("stl_ets")
    bt_stl = bt_stl[bt_stl.origin.isin(ORIGINS)]
U.keep_bt("stl_ets", bt_stl)
print("decompose, then ETS:", U.scores(bt_stl))
if not U.QUICK:
    U.check(U.mase(bt_stl) < U.mase(floor), f"with the annual season it scores {U.mase(bt_stl):.2f} and beats the floor at {U.mase(floor):.2f}")

# %% [markdown]
# ## 4 · Against the floor, by horizon week (4')
#
# **Skill** is the share of the floor's error a method removes: `1 - error / error of the floor`. Positive
# beats the floor, negative loses to it.
#
# ✏️ **Exercise** · Write it for two backtest tables (use `U.mase`).

# %%
def skill(bt: pd.DataFrame, floor: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    return 1 - U.mase(bt) / U.mase(floor)
    # END SOLUTION

# %%
try:
    skills = {"ETS, weekly season": skill(bt_ets, floor), "decompose, then ETS": skill(bt_stl, floor)}
except NotImplementedError:
    skills = {"ETS, weekly season": 1 - U.mase(bt_ets) / U.mase(floor), "decompose, then ETS": 1 - U.mase(bt_stl) / U.mase(floor)}
table = pd.DataFrame({"floor": U.by_week(floor), "ETS, weekly season": U.by_week(bt_ets), "decompose, then ETS": U.by_week(bt_stl)}).T
table.columns = [f"week {w}" for w in table.columns]
table.insert(0, "MASE", [round(U.mase(b), 3) for b in (floor, bt_ets, bt_stl)])
table["skill"] = [0.0] + [round(s, 3) for s in skills.values()]
print(table.to_string())
U.keep("d2_1", dict(table=table.T.to_dict(), grid=grid, scores_ets=U.scores(bt_ets), scores_stl=U.scores(bt_stl)))
U.check(skills["ETS, weekly season"] < skills["decompose, then ETS"], "the annual season is worth more than the model class")

# %% [markdown]
# ## 5 · Read the smoothing weights (3')
#
# Simple exponential smoothing gives the value k days back the weight `alpha * (1 - alpha) ** k`. So the last
# 7 days together carry `1 - (1 - alpha) ** 7` of the weight.
#
# ✏️ **Exercise** · Return the share of the weight on the last `k` days.

# %%
def weight_on_last(alpha: float, k: int) -> float:
    # BEGIN SOLUTION
    return 1 - (1 - alpha) ** k
    # END SOLUTION

# %%
fit = AutoETS(season_length=7, model="ANA").fit(y.iloc[-1096:].to_numpy())
alpha_fitted = float(fit.model_["par"][0])
try:
    w = {a: round(weight_on_last(a, 7), 3) for a in (0.1, 0.3, round(alpha_fitted, 3))}
except NotImplementedError:
    w = {a: round(1 - (1 - a) ** 7, 3) for a in (0.1, 0.3, round(alpha_fitted, 3))}
print(f"fitted alpha of the weekly ETS: {alpha_fitted:.3f}")
print("share of the weight on the last 7 days, by alpha:", w)
U.keep("d2_1_alpha", dict(alpha=alpha_fitted, weights={str(k): v for k, v in w.items()}))
U.check(alpha_fitted > 0.9, "the fitted alpha is close to 1: this ETS is the naive forecast with a weekday profile")

# %% [markdown]
# **Look at.** That is why its score sits beside the naive forecasts of Lab 1.4 and not beside the floor.

# %%
# The stretch, done: a damped trend against an undamped one (full mode only)
if not U.QUICK:
    damped = U.backtest(y, lambda tr, h: U.sf(AutoETS(season_length=7, model="AAA", damped=True), tr, h), ORIGINS)
    undamped = U.backtest(y, lambda tr, h: U.sf(AutoETS(season_length=7, model="AAA"), tr, h), ORIGINS)
    st = {"damped trend": U.by_week(damped).to_dict(), "undamped trend": U.by_week(undamped).to_dict()}
    U.keep("d2_1_stretch", st)
    print(pd.DataFrame(st).T.to_string())

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why does ETS with a weekly season lose to the floor at 28 days?</summary>
# It forecasts a flat level with a weekday pattern. Over four weeks the temperature moves the level, and the
# model has no annual season to follow it.</details>
#
# <details><summary>2 · What does an alpha close to 1 tell you?</summary>
# The level is almost the last value. The model forgets everything older than a day or two: it is close to a
# naive forecast.</details>
#
# <details><summary>3 · Why fit on the logarithm in task 3?</summary>
# The seasons multiply the level (Lab 1.3). On the log scale they add, which is what the decomposition assumes.
# The forecast and its quantiles are turned back with the exponential.</details>
#
# ## Carry forward
#
# Two backtests are saved for later labs as reference files: `bt_ets_weekly.csv` and `bt_stl_ets.csv`.
# Lab 2.4 measures whether their intervals hold.
