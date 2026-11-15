# %% [markdown]
# # Lab 4.1 · Three leaks in a backtest
#
# Day 4 · Session 1 (Leakage across time) · 25 min.
#
# A colleague hands you a backtest of Al-Rimal with a score far better than anything you measured this week.
# Three faults are seeded in it. Each lets something from after the origin reach the forecast. You find and
# fix them one at a time, record the score after every fix, and write the review memo.
#
# **AI policy: AI-tutor.** Find the faults yourself; an assistant may explain a function.
#
# **Reference state.** `data/regions.csv`, `data/calendar.csv` and `data/reference/bt_windows.csv` (the
# colleague's candidate runs for the third fault). Nothing from your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Run the faulty backtest and record its score (3')
# - [ ] 2 · Find and fix the feature that sees the future (6')
# - [ ] 3 · Find and fix the regressor not known at the origin (6')
# - [ ] 4 · Find and fix the selection on the test window (6')
# - [ ] 5 · Write the review memo with before and after (4')
# - [ ] ✱ Stretch · Write a test that fails when a feature uses a day after the origin

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
from statsforecast.models import ARIMA

RID = "R01"
data = U.region(RID)                               # demand, temp_c, temp_normal, indexed by day
y = data["demand"]
cal = U.calendar()
ORIGINS = U.origins(U.pick(13, 4))                 # quick mode: the last four folds
floor = U.bt("floor")
floor = floor[floor.origin.isin(ORIGINS)]


def run_backtest(features) -> pd.DataFrame:
    """Backtest the regression with ARIMA errors, with regressors from features(data, dates, origin)."""
    def forecast(train, h, origin):
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
        model = ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)
        return U.sf(model, train, h, features(data, train.index, origin), features(data, future, origin))
    return U.backtest(y, forecast, ORIGINS)

# %% [markdown]
# ## 1 · The backtest as you received it (3')
#
# Read the colleague's feature function before you run it. `dates` are the days the regressors are needed
# for: the training days, or the 28 days after `origin`.

# %%
def colleague_features(data: pd.DataFrame, dates, origin) -> pd.DataFrame:
    X = pd.DataFrame(index=dates)
    X["cdd"] = (data.temp_c.reindex(dates) - 22).clip(lower=0).to_numpy()
    X["smooth"] = data.demand.rolling(7, center=True, min_periods=1).mean().reindex(dates).to_numpy()
    X["ramadan"] = cal.ramadan.reindex(dates).astype(float).to_numpy()
    X["eid"] = cal.eid_weight.reindex(dates).astype(float).to_numpy()
    return X


bt_faulty = run_backtest(colleague_features)
scores = {"as received": round(U.mase(bt_faulty), 3)}
print(f"as received: scaled error {scores['as received']}   (the floor: {U.mase(floor):.3f}; the regression of Lab 2.3: 0.766)")
U.check(scores["as received"] < 0.5, "a score this far under everything measured so far is a reason to review, not to celebrate")

# %% [markdown]
# ## 2 · The feature that sees the future (6')
#
# `smooth` is a 7-day rolling mean of demand, **centred**: the value for a day averages the three days before
# it, the day itself and the three days after it. For a test day that includes the demand being forecast.
#
# ✏️ **Exercise** · Rewrite the function so that `smooth` uses only days known at any origin of a 28-day
# forecast: a trailing 7-day mean (`rolling(7)`, not centred), shifted by 28 days. Change nothing else yet.

# %%
def features_fix1(data: pd.DataFrame, dates, origin) -> pd.DataFrame:
    # BEGIN SOLUTION
    X = pd.DataFrame(index=dates)
    X["cdd"] = (data.temp_c.reindex(dates) - 22).clip(lower=0).to_numpy()
    X["smooth"] = data.demand.rolling(7).mean().shift(28).reindex(dates).to_numpy()
    X["ramadan"] = cal.ramadan.reindex(dates).astype(float).to_numpy()
    X["eid"] = cal.eid_weight.reindex(dates).astype(float).to_numpy()
    return X
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_features_fix1(data, dates, origin):
    X = colleague_features(data, dates, origin)
    X["smooth"] = data.demand.rolling(7).mean().shift(28).reindex(dates).to_numpy()
    return X


try:
    features_fix1(data, data.index[-60:], data.index[-29])
except NotImplementedError:
    features_fix1 = ref_features_fix1
probe_day = pd.Timestamp("2025-07-10")
expected = data.demand.loc[probe_day - pd.Timedelta(days=34): probe_day - pd.Timedelta(days=28)].mean()
got = features_fix1(data, pd.DatetimeIndex([probe_day]), probe_day - pd.Timedelta(days=5))["smooth"].iloc[0]
U.check(abs(got - expected) < 1e-6, "smooth for 10 July is the mean of 6 to 12 June: nothing later than 28 days before")
bt_fix1 = run_backtest(features_fix1)
scores["after fix 1 (feature)"] = round(U.mase(bt_fix1), 3)
print(scores)
U.check(scores["after fix 1 (feature)"] > scores["as received"] + 0.1, "without the centred feature the score rises: that feature carried the answer")

# %% [markdown]
# ## 3 · The regressor not known at the origin (6')
#
# `cdd` is computed from `temp_c`, the temperature **measured** on each day. For the 28 test days nobody has
# it at the origin. `U.temp_known(RID, origin, dates)` returns what can be known: the normal for the date plus
# the origin day's anomaly, fading.
#
# ✏️ **Exercise** · Starting from your fix 1, use the measured temperature for days up to the origin and
# `U.temp_known` for days after it.

# %%
def features_fix2(data: pd.DataFrame, dates, origin) -> pd.DataFrame:
    # BEGIN SOLUTION
    dates = pd.DatetimeIndex(dates)
    X = features_fix1(data, dates, origin)
    temp = data.temp_c.reindex(dates)
    temp = temp.where(dates <= origin, U.temp_known(RID, origin, dates))
    X["cdd"] = (temp - 22).clip(lower=0).to_numpy()
    return X
    # END SOLUTION

# %%
def ref_features_fix2(data, dates, origin):
    dates = pd.DatetimeIndex(dates)
    X = features_fix1(data, dates, origin)
    temp = data.temp_c.reindex(dates).where(dates <= origin, U.temp_known(RID, origin, dates))
    X["cdd"] = (temp - 22).clip(lower=0).to_numpy()
    return X


try:
    features_fix2(data, data.index[-60:], data.index[-29])
except NotImplementedError:
    features_fix2 = ref_features_fix2
o = ORIGINS[-1]
fut = pd.date_range(o + pd.Timedelta(days=1), periods=28, freq="D")
U.check(np.allclose(features_fix2(data, fut, o)["cdd"], (U.temp_known(RID, o, fut) - 22).clip(lower=0)),
        "future cooling degrees come from the temperature known at the origin")
U.check(np.allclose(features_fix2(data, data.index[-400:-300], o)["cdd"], (data.temp_c.iloc[-400:-300] - 22).clip(lower=0)),
        "past cooling degrees still come from the measured temperature")
bt_fix2 = run_backtest(features_fix2)
scores["after fix 2 (regressor)"] = round(U.mase(bt_fix2), 3)
print(scores)
U.check(scores["after fix 2 (regressor)"] > scores["after fix 1 (feature)"], "the honest temperature costs accuracy: the earlier score was not available to anyone")

# %% [markdown]
# ## 4 · Selection on the test window (6')
#
# The colleague also tried five lengths of training window. The runs are supplied: `bt_windows.csv` holds the
# backtest of every window over the 13 folds of 2025 (the test year) and over the 13 folds of 2024. Their
# notebook then did this:

# %%
windows = pd.read_csv(U.REF / "bt_windows.csv", parse_dates=["origin", "date"])
windows["e"] = (windows.y - windows.yhat).abs() / windows["scale"]
test_year, year_before = windows[windows.year == 2025], windows[windows.year == 2024]
per_fold = test_year.groupby(["origin", "window"]).e.mean().unstack()          # folds by windows
as_reported = float(per_fold.min(axis=1).mean())                               # the best window of each fold
print(per_fold.round(2).to_string())
print(f"\nas the colleague reported it: the best window in every test fold, averaged: {as_reported:.3f}")

# %% [markdown]
# For every test fold it kept the window that did best **on that fold**, and reported the mean of those
# winners. No forecaster could have picked those windows before seeing the fold.
#
# ✏️ **Exercise** · Choose **one** window using the year before the test year only, then return that window
# and its score on the test year.

# %%
def honest_selection(year_before: pd.DataFrame, test_year: pd.DataFrame) -> tuple:
    # BEGIN SOLUTION
    chosen = year_before.groupby("window").e.mean().idxmin()
    return int(chosen), float(test_year.e[test_year.window == chosen].mean())
    # END SOLUTION

# %%
try:
    chosen, honest = honest_selection(year_before, test_year)
except NotImplementedError:
    chosen = int(year_before.groupby("window").e.mean().idxmin())
    honest = float(test_year.e[test_year.window == chosen].mean())
by_window = pd.DataFrame({"2024 (used to choose)": year_before.groupby("window").e.mean(), "2025 (test)": test_year.groupby("window").e.mean()}).round(3)
print(by_window.to_string())
print(f"\nchosen on 2024: a window of {chosen} days; its score on 2025: {honest:.3f}; as reported: {as_reported:.3f}")
U.check(chosen in set(windows.window), "one window, chosen without the test year")
U.check(honest > as_reported, f"selection on the test window flattered the score by {honest - as_reported:.3f}")
scores["as reported with per-fold selection"] = round(as_reported, 3)
scores["after fix 3 (selection)"] = round(honest, 3)

# %% [markdown]
# **Look at.** The honest score of the chosen window is also not the best score in the 2025 column. That is
# what choosing in advance costs, and it is the number a planner will experience.

# %% [markdown]
# ## 5 · The review memo (4')
#
# ✏️ **Exercise** · Return the memo as text. It names each fault, where it was, the fix, and the score before
# and after, and ends with the score you would sign.

# %%
def review_memo(scores: dict, chosen: int) -> str:
    # BEGIN SOLUTION
    return "\n".join([
        "Review of the Al-Rimal backtest (28 days ahead, 13 origins over 2025)",
        "",
        f"Score as received: {scores['as received']}. Three faults let information from after the origin reach the forecast.",
        "",
        f"1. Feature `smooth`: a centred 7-day mean of demand, which contains the days being forecast. "
        f"Replaced by a trailing mean shifted by the horizon. Score after the fix: {scores['after fix 1 (feature)']}.",
        f"2. Regressor `cdd`: computed from the temperature measured on the test days. Replaced by the temperature "
        f"known at the origin. Score after the fix: {scores['after fix 2 (regressor)']}.",
        f"3. Selection: the training window was chosen per fold on the test fold itself (reported {scores['as reported with per-fold selection']}). "
        f"Chosen once on the 2024 folds ({chosen} days) and scored on 2025: {scores['after fix 3 (selection)']}.",
        "",
        f"Score to report: {scores['after fix 3 (selection)']} (training window of {chosen} days, chosen on 2024) against a floor of "
        f"{round(U.mase(floor), 3)}. The corrected backtest is attached.",
    ])
    # END SOLUTION

# %%
try:
    memo = review_memo(scores, chosen)
except NotImplementedError:
    memo = "Review memo not written yet.\n" + "\n".join(f"{k}: {v}" for k, v in scores.items())
print(memo)
U.save("d4_1_review_memo.md", memo + "\n")
test_year[test_year.window == chosen].drop(columns=["e"]).to_csv(U.OUT / "d4_1_corrected_backtest.csv", index=False)
U.keep("d4_1", dict(scores=scores, chosen=chosen, floor=round(U.mase(floor), 3), by_window=by_window.T.to_dict()))
U.check(all(k in scores for k in ("as received", "after fix 1 (feature)", "after fix 2 (regressor)", "after fix 3 (selection)")),
        "the memo has a score before and after every fix")

# %%
# The stretch, done: a test that fails when a feature uses a day after the origin
def leaking_columns(features, origin=pd.Timestamp("2025-06-18")) -> list:
    """Change everything measured after the origin and see which feature columns change."""
    dates = pd.date_range(origin - pd.Timedelta(days=60), origin + pd.Timedelta(days=28), freq="D")
    altered = data.copy()
    after = altered.index > origin
    altered.loc[after, "demand"] = altered.loc[after, "demand"] * 1.5
    altered.loc[after, "temp_c"] = altered.loc[after, "temp_c"] + 5
    a, b = features(data, dates, origin), features(altered, dates, origin)
    return [c for c in a.columns if not np.allclose(a[c].fillna(0), b[c].fillna(0))]


print("columns that change when the future changes:")
for name, fn in (("as received", colleague_features), ("after fix 1", features_fix1), ("after fix 2", features_fix2)):
    print(f"  {name}: {leaking_columns(fn)}")
U.check(leaking_columns(features_fix2) == [], "after both fixes no feature depends on anything after the origin")

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why is a centred rolling mean harmless in a plot and harmful as a feature?</summary>
# In a plot you look at the past with hindsight. As a feature for a test day it contains that day and the
# three days after it: the answer.</details>
#
# <details><summary>2 · The third fault changed no feature. Why is it a leak?</summary>
# The test folds decided which model's score was reported. Information from the test window reached the
# reported number through the choice.</details>
#
# <details><summary>3 · What is the fastest way to catch the first two kinds of leak?</summary>
# The test of the stretch: change the data after the origin and check that no feature changes.</details>
#
# ## Carry forward
#
# `out/d4_1_review_memo.md` and `out/d4_1_corrected_backtest.csv` are LO4's evidence: a review memo and a
# corrected backtest. The capstone asks you
# to run the same review on your own work before you present it.
