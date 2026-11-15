# %% [markdown]
# # Lab 1.4 · The benchmark floor (AI-off)
#
# Day 1 · Session 4 (The benchmark floor) · 25 min.
#
# You build the measuring instrument of the course by hand: rolling origins, naive forecasts and a scaled
# error. Then you backtest three naive methods over 13 origins of 28 days. The best of them is the floor:
# from tomorrow, a method that does not beat it is not worth its upkeep.
#
# **AI policy: AI-off.** This is the first backtest: write it yourself, with no assistant. You will be asked
# to explain every line of it in the capstone defence.
#
# **Reference state.** `data/regions.csv`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Write the rolling origins (5')
# - [ ] 2 · Write the naive and seasonal-naive forecasts (5')
# - [ ] 3 · Write the scaled error (5')
# - [ ] 4 · Backtest three naive methods over 13 origins (6')
# - [ ] 5 · Read the error by horizon week and name the floor (4')
# - [ ] ✱ Stretch · Score the Eid days separately from the other days

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

y = U.region("R01")["demand"]
H = 28
print(y.tail(3))

# %% [markdown]
# ## 1 · Rolling origins (5')
#
# An **origin** is the last day the forecaster has seen. Fold k is trained on the days up to its origin and
# tested on the next `h` days. With 13 origins 28 days apart, the test days cover the last 364 days of data
# exactly once.
#
# ✏️ **Exercise** · Return the list of `n` origins, oldest first, so that the last fold ends on `last_day`.

# %%
def rolling_origins(last_day: pd.Timestamp, n: int = 13, h: int = 28) -> list:
    # BEGIN SOLUTION
    return [last_day - pd.Timedelta(days=h * k) for k in range(n, 0, -1)]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    origins = rolling_origins(y.index[-1])
except NotImplementedError:
    origins = U.origins()
U.check(origins == U.origins(), f"13 origins from {origins[0].date()} to {origins[-1].date()}")
U.check(origins[-1] + pd.Timedelta(days=H) == y.index[-1], "the last fold ends on the last day of data")

# %% [markdown]
# ## 2 · Naive and seasonal naive (5')
#
# - **Naive**: every future day equals the last observed day.
# - **Seasonal naive with period m**: each future day equals the day m days before it. With `m = 7`, next
#   Friday equals last Friday. With `m = 364` (52 weeks), it equals the same weekday a year ago.
#
# ✏️ **Exercise** · Write both. `train` is a Series that ends on the origin. Return `h` values.

# %%
def naive(train: pd.Series, h: int) -> np.ndarray:
    # BEGIN SOLUTION
    return np.repeat(train.iloc[-1], h)
    # END SOLUTION


def seasonal_naive(train: pd.Series, h: int, m: int) -> np.ndarray:
    # BEGIN SOLUTION
    last = train.to_numpy()[-m:]
    return np.resize(last, h)            # repeats the last m values when h is longer than m
    # END SOLUTION

# %%
toy = pd.Series(np.arange(1.0, 22.0))    # 1, 2, ..., 21
try:
    a, b = naive(toy, 3), seasonal_naive(toy, 9, 7)
except NotImplementedError:
    a, b = np.repeat(toy.iloc[-1], 3), U.seasonal_naive(toy, 9, 7)
    naive = lambda train, h: np.repeat(train.iloc[-1], h)
    seasonal_naive = U.seasonal_naive
U.check(list(a) == [21, 21, 21], "naive repeats the last value")
U.check(list(b) == [15, 16, 17, 18, 19, 20, 21, 15, 16], "seasonal naive repeats the last week, twice if needed")

# %% [markdown]
# ## 3 · The scaled error (5')
#
# An error of 150 MWh is small in Al-Rimal and large in Al-Qimma. The **mean absolute scaled error** divides
# the mean absolute error of the forecast by the mean absolute error the weekly seasonal naive made **inside
# the training data** (each day against the day 7 days before it):
#
# `MASE = mean(|y - forecast|) / mean(|train[t] - train[t - 7]|)`
#
# Below 1: better than last week's value was, on average, in the past. It has no unit, so regions and
# feeders can share a table.
#
# ✏️ **Exercise** · Write it.

# %%
def scaled_error(actual, forecast, train: pd.Series, m: int = 7) -> float:
    # BEGIN SOLUTION
    v = train.to_numpy()
    scale = np.mean(np.abs(v[m:] - v[:-m]))
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))) / scale)
    # END SOLUTION

# %%
train0, test0 = y.loc[:origins[0]].iloc[-1096:], y.loc[origins[0] + pd.Timedelta(days=1): origins[0] + pd.Timedelta(days=H)]
try:
    e0 = scaled_error(test0, seasonal_naive(train0, H, 7), train0)
except NotImplementedError:
    e0 = float(np.mean(np.abs(test0 - U.seasonal_naive(train0, H, 7))) / U.scale(train0))
    scaled_error = lambda actual, forecast, train, m=7: float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))) / U.scale(train, m))
e0_ref = float(np.mean(np.abs(test0 - U.seasonal_naive(train0, H, 7))) / U.scale(train0))
U.check(abs(e0 - e0_ref) < 1e-9, f"first fold, weekly seasonal naive: scaled error {e0:.3f}")

# %% [markdown]
# ## 4 · Backtest three naive methods (6')
#
# ✏️ **Exercise** · For each origin: train on the 1,096 days up to the origin, forecast 28 days with the
# method, and keep one row per test day with `origin`, `h` (1 to 28), `y`, `yhat` and `scale` (the
# denominator of the scaled error for that fold). Return the table.

# %%
def backtest(y: pd.Series, method, origins: list, h: int = 28, window: int = 1096) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for o in origins:
        train = y.loc[:o].iloc[-window:]
        test = y.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=h)]
        v = train.to_numpy()
        rows.append(pd.DataFrame({"origin": o, "date": test.index, "h": np.arange(1, h + 1), "y": test.to_numpy(),
                                  "yhat": method(train, h), "scale": np.mean(np.abs(v[7:] - v[:-7]))}))
    return pd.concat(rows, ignore_index=True)
    # END SOLUTION

# %%
methods = {
    "naive (last day)": lambda tr, h: naive(tr, h),
    "last week (m=7)": lambda tr, h: seasonal_naive(tr, h, 7),
    "same weekday last year (m=364)": lambda tr, h: seasonal_naive(tr, h, 364),
}
try:
    bts = {name: backtest(y, fn, origins) for name, fn in methods.items()}
except NotImplementedError:
    bts = {name: U.backtest(y, fn, origins) for name, fn in methods.items()}
U.check(all(len(b) == 13 * 28 for b in bts.values()), "364 test days per method, each day tested once")
table = pd.DataFrame({name: {"MASE": round(U.mase(b), 3), **{f"week {w}": v for w, v in U.by_week(b).items()}}
                      for name, b in bts.items()}).T
print(table.to_string())
U.keep("d1_4_floor", table.T.to_dict())
U.keep_bt("floor", bts["same weekday last year (m=364)"])
U.keep_bt("weekly_naive", bts["last week (m=7)"])

# %% [markdown]
# ## 5 · Name the floor (4')
#
# ✏️ **Exercise** · Return the name of the method with the lowest overall MASE.

# %%
def floor_method(table: pd.DataFrame) -> str:
    # BEGIN SOLUTION
    return table["MASE"].idxmin()
    # END SOLUTION

# %%
try:
    floor = floor_method(table)
except NotImplementedError:
    floor = table["MASE"].idxmin()
U.check(floor == "same weekday last year (m=364)", f"the floor is: {floor}, MASE {table.loc[floor, 'MASE']}")
U.check(table.loc["last week (m=7)", "week 4"] > 1.5 * table.loc[floor, "week 4"], "in week 4 of the horizon, last week's values are far worse than last year's")

fig, ax = plt.subplots()
for name in table.index:
    ax.plot([1, 2, 3, 4], table.loc[name, ["week 1", "week 2", "week 3", "week 4"]], marker="o", label=name)
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("week of the 28-day horizon"); ax.set_ylabel("scaled error"); ax.legend(frameon=False)
ax.set_title("Al-Rimal: three naive methods, 13 origins", loc="left"); plt.show()

# The stretch, done: Eid days against the other days, for the floor
cal = U.calendar()
b = bts[floor].copy()
b["eid"] = cal.eid_weight.reindex(b.date).to_numpy() > 0
eid_split = (np.abs(b.y - b.yhat) / b["scale"]).groupby(b.eid).mean().round(2)
U.keep("d1_4_eid", {"eid_days": int(b.eid.sum()), "eid": eid_split[True], "other": eid_split[False]})
print(f"floor on {int(b.eid.sum())} Eid days: {eid_split[True]}; on the other days: {eid_split[False]}")

# %% [markdown]
# **Look at.** A scaled error above 1 is not a bug: the denominator is a one-week-ahead error, and you are
# forecasting up to four weeks ahead. What matters from now on is the distance to the floor.

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why does "last week" lose to "last year" at 28 days in Al-Rimal?</summary>
# Demand follows the temperature, and the temperature four weeks from now is closer to the same weeks last
# year than to last week.</details>
#
# <details><summary>2 · Why 364 days and not 365?</summary>
# 364 is 52 whole weeks: a Friday is matched with a Friday. With 365 the weekday pattern would shift by a day.</details>
#
# <details><summary>3 · Where does the floor fail?</summary>
# On Eid days: last year's Eid was 11 days later. The floor's error there is about twice its error elsewhere.</details>
#
# ## Carry forward
#
# The floor table is the reference line of Days 2 to 4: `data/reference/bt_floor.csv` is supplied to every
# later lab, so none depends on your run.
