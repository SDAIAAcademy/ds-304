# %% [markdown]
# # Lab 2.2 · A seasonal ARIMA with checked residuals
#
# Day 2 · Session 2 (ARIMA-family models) · 25 min.
#
# You difference the series and read its autocorrelation, fit a seasonal ARIMA model, test whether its
# residuals still hold a pattern, and add annual Fourier terms. Each step is scored against the floor.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/regions.csv` and the supplied floor, `data/reference/bt_floor.csv`. Nothing from
# your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Difference the series and read the autocorrelation (5')
# - [ ] 2 · Fit a seasonal ARIMA and score it (6')
# - [ ] 3 · Test the residuals for leftover correlation (5')
# - [ ] 4 · Add annual Fourier terms and score again (6')
# - [ ] 5 · Record both scores against the floor (3')
# - [ ] ✱ Stretch · Let an automatic search choose the orders and compare time and score

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
from statsforecast.models import ARIMA, AutoARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

y = U.region("R01")["demand"]
ORIGINS = U.origins(U.pick(13, 3))                 # quick mode: the last three folds
floor = U.bt("floor")
floor = floor[floor.origin.isin(ORIGINS)]
last_train = y.loc[:ORIGINS[-1]].iloc[-1096:]      # the training window of the last fold


def weekly_arima():
    """ARIMA(1,0,1)(0,1,1) with a period of 7: one seasonal difference, short memory around it."""
    return ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)

# %% [markdown]
# ## 1 · Difference and read the autocorrelation (5')
#
# The autocorrelation at lag k is the correlation of the series with itself k days earlier. A series whose
# autocorrelation stays near 1 has a level that wanders: ARIMA removes that by differencing. A seasonal
# difference with period 7 subtracts the same weekday of the week before.
#
# ✏️ **Exercise** · Return the seasonal difference, without the missing values at the start.

# %%
def seasonal_difference(y: pd.Series, m: int = 7) -> pd.Series:
    # BEGIN SOLUTION
    return (y - y.shift(m)).dropna()
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    d7 = seasonal_difference(last_train)
except NotImplementedError:
    d7 = (last_train - last_train.shift(7)).dropna()
acf_raw, acf_d7 = acf(last_train, nlags=14), acf(d7, nlags=14)
print("lag:             ", list(range(1, 15)))
print("demand:          ", acf_raw[1:].round(2).tolist())
print("after difference:", acf_d7[1:].round(2).tolist())
U.check(len(d7) == len(last_train) - 7, "the difference is 7 days shorter")
U.check(acf_raw[7] > 0.9 and abs(acf_d7[7]) < 0.5, f"lag 7: {acf_raw[7]:.2f} before, {acf_d7[7]:.2f} after differencing")
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, a, t in zip(axes, (acf_raw, acf_d7), ("demand", "after a seasonal difference")):
    ax.bar(range(1, 15), a[1:], color="#16324f"); ax.set_title(t, loc="left", fontsize=9); ax.set_ylim(-0.4, 1)
plt.show()

# %% [markdown]
# **Look at.** After differencing, the correlation dies out over about five days: short memory, which a small
# autoregressive and moving-average part can model.

# %% [markdown]
# ## 2 · Fit a seasonal ARIMA and score it (6')
#
# `weekly_arima()` returns the model. ✏️ **Exercise** · Write the forecast function for the backtest with
# `U.sf(model, train, h)`.

# %%
def arima_weekly(train: pd.Series, h: int) -> pd.DataFrame:
    # BEGIN SOLUTION
    return U.sf(weekly_arima(), train, h)
    # END SOLUTION

# %%
try:
    bt_arima = U.backtest(y, arima_weekly, ORIGINS)
except NotImplementedError:
    bt_arima = U.bt("arima_weekly")
    bt_arima = bt_arima[bt_arima.origin.isin(ORIGINS)]
U.keep_bt("arima_weekly", bt_arima)
print("seasonal ARIMA, weekly:", U.scores(bt_arima), "| floor:", round(U.mase(floor), 3))
if not U.QUICK:
    U.check(U.mase(bt_arima) > U.mase(floor) + 0.3, f"like the weekly ETS, it loses to the floor: {U.mase(bt_arima):.2f} against {U.mase(floor):.2f}")

# %% [markdown]
# ## 3 · Test the residuals (5')
#
# If the model has used everything the past holds, its one-step errors inside the training data (the
# residuals) look like noise. The Ljung-Box test asks whether the first `lags` autocorrelations of the
# residuals are together larger than noise would give. A small p-value means a pattern is left, and the
# model's intervals, which assume noise, are not to be trusted.
#
# ✏️ **Exercise** · Return the p-value with `acorr_ljungbox(residuals, lags=[lags], model_df=model_df)`:
# it returns a table with a column `lb_pvalue`. `model_df` is the number of fitted ARMA coefficients.

# %%
def ljung_box_p(residuals, lags: int = 14, model_df: int = 3) -> float:
    # BEGIN SOLUTION
    return float(acorr_ljungbox(residuals, lags=[lags], model_df=model_df)["lb_pvalue"].iloc[0])
    # END SOLUTION

# %%
fit_plain = weekly_arima().fit(last_train.to_numpy())
res_plain = fit_plain.model_["residuals"][7:]            # the first week has no seasonal difference
try:
    p_plain = ljung_box_p(res_plain)
except NotImplementedError:
    p_plain = float(acorr_ljungbox(res_plain, lags=[14], model_df=3)["lb_pvalue"].iloc[0])
print(f"Ljung-Box p-value, weekly ARIMA: {p_plain:.4f}")
U.check(p_plain < 0.05, "the residuals are not noise: something systematic is left")

# %% [markdown]
# ## 4 · Annual Fourier terms (6')
#
# What is left is the summer. A pair of sine and cosine waves with a period of one year, and a second pair
# with half that period, can draw the annual shape. They are known for any future date, so they can be
# regressors: columns the model is given for the training days (`X`) and for the horizon (`Xf`).
#
# ✏️ **Exercise** · Write the forecast function. It receives the origin as a third argument. Build the
# future dates, then call `U.sf(model, train, h, X, Xf)` with `fourier` on both sets of dates.

# %%
def fourier(dates, K: int = 2) -> pd.DataFrame:
    """K pairs of annual sine and cosine waves for the given days."""
    t = np.asarray((pd.DatetimeIndex(dates) - pd.Timestamp("2017-01-01")).days, float)
    return pd.DataFrame({f"{f}{k}": getattr(np, f)(2 * np.pi * k * t / 365.25)
                         for k in range(1, K + 1) for f in ("sin", "cos")}, index=dates)


def arima_fourier(train: pd.Series, h: int, origin: pd.Timestamp) -> pd.DataFrame:
    # BEGIN SOLUTION
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
    return U.sf(weekly_arima(), train, h, fourier(train.index), fourier(future))
    # END SOLUTION

# %%
try:
    bt_four = U.backtest(y, arima_fourier, ORIGINS)
except NotImplementedError:
    bt_four = U.bt("arima_fourier")
    bt_four = bt_four[bt_four.origin.isin(ORIGINS)]
U.keep_bt("arima_fourier", bt_four)
fit_four = weekly_arima().fit(last_train.to_numpy(), X=fourier(last_train.index).to_numpy())
res_four = fit_four.model_["residuals"][7:]
try:
    p_four = ljung_box_p(res_four)
except NotImplementedError:
    p_four = float(acorr_ljungbox(res_four, lags=[14], model_df=3)["lb_pvalue"].iloc[0])
print("ARIMA with annual Fourier terms:", U.scores(bt_four), f"| Ljung-Box p-value: {p_four:.3f}")
if not U.QUICK:
    U.check(U.mase(bt_four) < U.mase(floor), f"with the annual terms it scores {U.mase(bt_four):.2f} and beats the floor at {U.mase(floor):.2f}")
U.check(p_four > 0.05, "and its residuals pass the test")

# %% [markdown]
# ## 5 · Record both against the floor (3')

# %%
rows = {"floor (same weekday last year)": dict(MASE=round(U.mase(floor), 3), **U.by_week(floor).rename(lambda w: f"week {w}").to_dict(), ljung_box_p=np.nan),
        "ARIMA, weekly": dict(MASE=round(U.mase(bt_arima), 3), **U.by_week(bt_arima).rename(lambda w: f"week {w}").to_dict(), ljung_box_p=round(p_plain, 4)),
        "ARIMA, weekly + annual Fourier": dict(MASE=round(U.mase(bt_four), 3), **U.by_week(bt_four).rename(lambda w: f"week {w}").to_dict(), ljung_box_p=round(p_four, 4))}
table = pd.DataFrame(rows).T
print(table.to_string())
U.keep("d2_2", dict(table={k: {a: (None if pd.isna(b) else b) for a, b in v.items()} for k, v in rows.items()},
                    acf_raw7=float(acf_raw[7]), acf_d7=[float(a) for a in acf_d7[1:8]],
                    coef={k: float(v) for k, v in fit_plain.model_["coef"].items()},
                    scores_four=U.scores(bt_four), scores_arima=U.scores(bt_arima)))

# %%
# The stretch, done: an automatic search of the orders, on two folds (full mode only)
if not U.QUICK:
    t0 = time.perf_counter()
    bt_auto = U.backtest(y, lambda tr, h, origin: U.sf(AutoARIMA(season_length=7), tr, h, fourier(tr.index),
                                                       fourier(pd.date_range(origin + pd.Timedelta(days=1), periods=h))), ORIGINS[-2:])
    secs_auto = time.perf_counter() - t0
    same = bt_four[bt_four.origin.isin(ORIGINS[-2:])]
    U.keep("d2_2_stretch", dict(auto=round(U.mase(bt_auto), 3), fixed=round(U.mase(same), 3), seconds=round(secs_auto, 1)))
    print(f"automatic search on two folds: {U.mase(bt_auto):.3f} in {secs_auto:.0f} s; fixed orders on the same folds: {U.mase(same):.3f}")

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · The weekly ARIMA fails the residual test. Does that alone explain its poor score?</summary>
# It points to the cause. The test says a pattern is left in the one-step errors; the plot of demand says
# which one: the summer. Over 28 days that missing pattern is most of the error.</details>
#
# <details><summary>2 · Why can Fourier terms be used as regressors for the future?</summary>
# They depend on the date only, and the date of every future day is known at the origin.</details>
#
# <details><summary>3 · The model with Fourier terms passes the test. Are its intervals now right?</summary>
# The test is passed inside the training data, one step ahead. Whether 80% intervals hold 80% of the time
# 28 days ahead is measured in the backtest: Lab 2.4.</details>
#
# ## Carry forward
#
# `bt_arima_fourier.csv` is saved as a reference file. It is one of the three backtests Lab 2.4 and Lab 4.3
# start from.
