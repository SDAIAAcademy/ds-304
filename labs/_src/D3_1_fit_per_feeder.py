# %% [markdown]
# # Lab 3.1 · Feeders that add up to their region
#
# Day 3 · Session 1 (Many series and hierarchies) · 25 min.
#
# You forecast every feeder of one region and the region itself, see that the two sets of forecasts
# disagree, and reconcile them three ways. Then you compare the errors at the feeder level and at the regional
# level, over the eight held-out weeks.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/feeders.npz`, `data/feeders_meta.csv`, `data/regions.csv` and `data/calendar.csv`.
# Nothing from your own earlier labs: the regional model of Lab 2.3 is supplied as `U.dynreg`.
#
# **Tasks**
# - [ ] 1 · Forecast every feeder of one region, and the region (6')
# - [ ] 2 · Show that the forecasts do not add up (3')
# - [ ] 3 · Reconcile bottom-up and top-down (6')
# - [ ] 4 · Reconcile by least squares (6')
# - [ ] 5 · Compare errors at both levels (4')
# - [ ] ✱ Stretch · Repeat for all 12 regions

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
from statsforecast.models import AutoETS

RID = "R05"                                        # Al-Dana: residential, commercial and industrial feeders
established = set(U.established())                 # feeders with three years of history or more
ORIGINS = U.FEEDER_ORIGINS                         # two folds of 28 days: the eight held-out weeks
names = U.region_names()


def load(rid):
    """The established feeders of a region (wide table) and their sum, the regional series."""
    W = U.feeders(rid)
    W = W[[f for f in W.columns if f in established]]
    return W, W.sum(axis=1)


W, region = load(RID)
print(f"{names[RID]}: {W.shape[1]} feeders, folds start after {[o.date().isoformat() for o in ORIGINS]}")
print(U.meta().set_index("feeder").loc[W.columns, "kind"].value_counts().to_dict())

# %% [markdown]
# ## 1 · Base forecasts at both levels (6')
#
# Forecast each level with the method that suits it:
#
# - every **feeder** with ETS and a weekly season: `U.sf_many(model, train)` fits the same model on every
#   column of a wide table and returns 28 rows, one column per feeder;
# - the **region** with the regression of Lab 2.3 (weather known at the origin, plus the calendar):
#   `U.dynreg(RID, train, 28, origin)` returns a table whose column `yhat` is the forecast.
#
# Both train on the 1,096 days up to the origin.
#
# ✏️ **Exercise** · Return the feeder forecasts (a wide table) and the regional forecast (an array of 28).

# %%
def base_forecasts(W: pd.DataFrame, region: pd.Series, rid: str, origin: pd.Timestamp) -> tuple:
    # BEGIN SOLUTION
    feeders_fc = U.sf_many(AutoETS(season_length=7, model="ANA"), W.loc[:origin].iloc[-1096:])
    region_fc = U.dynreg(rid, region.loc[:origin].iloc[-1096:], 28, origin).yhat.to_numpy()
    return feeders_fc, region_fc
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_base_forecasts(W, region, rid, origin):
    return (U.sf_many(AutoETS(season_length=7, model="ANA"), W.loc[:origin].iloc[-1096:]),
            U.dynreg(rid, region.loc[:origin].iloc[-1096:], 28, origin).yhat.to_numpy())


try:
    base = {o: base_forecasts(W, region, RID, o) for o in ORIGINS}
except NotImplementedError:
    base = {o: ref_base_forecasts(W, region, RID, o) for o in ORIGINS}
f0, r0 = base[ORIGINS[0]]
U.check(f0.shape == (28, W.shape[1]) and len(r0) == 28, f"28 days by {W.shape[1]} feeders, and 28 regional values")

# %% [markdown]
# ## 2 · They do not add up (3')
#
# ✏️ **Exercise** · Return the mean absolute gap between the regional forecast and the sum of the feeder
# forecasts, as a percentage of the regional forecast.

# %%
def gap_percent(feeders_fc: pd.DataFrame, region_fc: np.ndarray) -> float:
    # BEGIN SOLUTION
    total = feeders_fc.sum(axis=1).to_numpy()
    return float(np.mean(np.abs(region_fc - total)) / np.mean(region_fc) * 100)
    # END SOLUTION

# %%
try:
    gaps = [gap_percent(*base[o]) for o in ORIGINS]
except NotImplementedError:
    gaps = [float(np.mean(np.abs(r - f.sum(axis=1).to_numpy())) / np.mean(r) * 100) for f, r in base.values()]
print("gap between the two levels, % of the regional forecast, by fold:", np.round(gaps, 1).tolist())
U.check(min(gaps) > 0.5, "the two levels disagree: a planner who adds up the feeders gets another regional number")

# %% [markdown]
# **Look at.** Neither number is "the" forecast yet. Purchasing reads the regional one, operations read the
# feeders, and the two desks would plan for different totals.

# %% [markdown]
# ## 3 · Bottom-up and top-down (6')
#
# - **Bottom-up**: keep the feeder forecasts; the regional forecast becomes their sum.
# - **Top-down** (by forecast proportions): keep the regional forecast; split it among the feeders in the
#   proportions of their own forecasts, day by day.
#
# ✏️ **Exercise** · Write both. Each returns the reconciled feeder forecasts as a wide table; the reconciled
# regional forecast is always their sum.

# %%
def bottom_up(feeders_fc: pd.DataFrame, region_fc: np.ndarray) -> pd.DataFrame:
    # BEGIN SOLUTION
    return feeders_fc.copy()
    # END SOLUTION


def top_down(feeders_fc: pd.DataFrame, region_fc: np.ndarray) -> pd.DataFrame:
    # BEGIN SOLUTION
    share = feeders_fc.div(feeders_fc.sum(axis=1), axis=0)
    return share.mul(region_fc, axis=0)
    # END SOLUTION

# %%
def ref_top_down(feeders_fc, region_fc):
    return feeders_fc.div(feeders_fc.sum(axis=1), axis=0).mul(region_fc, axis=0)


try:
    bottom_up(f0, r0), top_down(f0, r0)
except NotImplementedError:
    bottom_up, top_down = (lambda f, r: f.copy()), ref_top_down
U.check(np.allclose(top_down(f0, r0).sum(axis=1), r0), "top-down: the feeders now add up to the regional forecast")
U.check(np.allclose(bottom_up(f0, r0), f0), "bottom-up: the feeder forecasts are unchanged")

# %% [markdown]
# ## 4 · Least squares (6')
#
# Bottom-up trusts only the feeders; top-down trusts only the region. Least-squares reconciliation moves
# every forecast as little as possible, in total, until they add up. For one total over `n` feeders it has a
# closed form: the gap `regional forecast - sum of feeders` is shared equally among the `n + 1` forecasts.
# Each feeder moves by `gap / (n + 1)`, and the region is again the sum of the feeders.
#
# ✏️ **Exercise** · Write it.

# %%
def least_squares(feeders_fc: pd.DataFrame, region_fc: np.ndarray) -> pd.DataFrame:
    # BEGIN SOLUTION
    n = feeders_fc.shape[1]
    gap = region_fc - feeders_fc.sum(axis=1).to_numpy()
    return feeders_fc.add(gap / (n + 1), axis=0)
    # END SOLUTION

# %%
def ref_least_squares(feeders_fc, region_fc):
    gap = region_fc - feeders_fc.sum(axis=1).to_numpy()
    return feeders_fc.add(gap / (feeders_fc.shape[1] + 1), axis=0)


try:
    ls0 = least_squares(f0, r0)
except NotImplementedError:
    least_squares = ref_least_squares
    ls0 = least_squares(f0, r0)
n = f0.shape[1]
moved = ls0.sum(axis=1).to_numpy() - f0.sum(axis=1).to_numpy()
U.check(np.allclose(moved, (r0 - f0.sum(axis=1).to_numpy()) * n / (n + 1)), f"the regional total moves {n} parts in {n + 1} of the way to the regional forecast")

# %% [markdown]
# **Look at.** With 37 feeders the total lands almost on the regional forecast: one large, well-forecast
# series outweighs many small ones. Methods that also weigh each series by how well it is forecast go further
# (minimum trace reconciliation); the idea is the same.

# %% [markdown]
# ## 5 · Errors at both levels (4')
#
# For every method: the scaled error of the feeders, and the scaled error of the region (the sum of the
# reconciled feeders against the sum of the actual feeders).

# %%
def evaluate(W, region, base, methods) -> pd.DataFrame:
    rows = {}
    for name, fn in methods.items():
        feeder_tables, region_errors = [], []
        for o, (f, r) in base.items():
            rec = fn(f, r)
            feeder_tables.append(U.feeder_long(W, o, rec))
            actual = region.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=28)].to_numpy()
            total = r if name == "base (not coherent)" else rec.sum(axis=1).to_numpy()
            region_errors.append(np.mean(np.abs(actual - total)) / U.scale(region.loc[:o].iloc[-364:]))
        rows[name] = {"feeders": round(U.mase(pd.concat(feeder_tables)), 3), "region": round(float(np.mean(region_errors)), 3)}
    return pd.DataFrame(rows).T


METHODS = {"base (not coherent)": lambda f, r: f, "bottom-up": bottom_up, "top-down": top_down, "least squares": least_squares}
table = evaluate(W, region, base, METHODS)
print(table.to_string())
U.keep("d3_1", dict(region=RID, name=names[RID], feeders=int(W.shape[1]), gaps=[round(g, 1) for g in gaps], table=table.T.to_dict()))
U.check(table.loc["top-down", "region"] == table.loc["base (not coherent)", "region"], "top-down keeps the regional forecast, so it keeps its error")
U.check(table.loc["bottom-up", "feeders"] == table.loc["base (not coherent)", "feeders"], "bottom-up keeps the feeder forecasts, so it keeps their error")

# %% [markdown]
# ✏️ **Exercise** · Return the name of the coherent method with the lowest feeder error, and the one with the
# lowest regional error.

# %%
def best_coherent(table: pd.DataFrame) -> tuple:
    # BEGIN SOLUTION
    coherent = table.drop(index="base (not coherent)")
    return coherent["feeders"].idxmin(), coherent["region"].idxmin()
    # END SOLUTION

# %%
try:
    best_f, best_r = best_coherent(table)
except NotImplementedError:
    best_f, best_r = table.drop(index="base (not coherent)")["feeders"].idxmin(), table.drop(index="base (not coherent)")["region"].idxmin()
print(f"best coherent method for the feeders: {best_f}; for the region: {best_r}")
U.check(best_f != "bottom-up", "at the feeder level, borrowing from the regional forecast beats the feeder forecasts alone")

# %%
# The stretch, done: all 12 regions (full mode only; about four minutes)
if not U.QUICK:
    rows = {}
    for rid in sorted(names):
        Wr, reg = load(rid)
        b = {o: ref_base_forecasts(Wr, reg, rid, o) for o in ORIGINS}
        t = evaluate(Wr, reg, b, METHODS)
        rows[rid] = {"feeders: bottom-up": t.loc["bottom-up", "feeders"], "feeders: top-down": t.loc["top-down", "feeders"],
                     "feeders: least squares": t.loc["least squares", "feeders"],
                     "region: bottom-up": t.loc["bottom-up", "region"], "region: regional model": t.loc["top-down", "region"]}
    all12 = pd.DataFrame(rows).T
    all12.insert(0, "name", [names[r] for r in all12.index])
    print(all12.to_string())
    helped = int((all12["feeders: top-down"] < all12["feeders: bottom-up"]).sum())
    region_better = int((all12["region: regional model"] < all12["region: bottom-up"]).sum())
    print(f"top-down improves the feeder error in {helped} of 12 regions; the regional model beats the bottom-up total in {region_better} of 12")
    U.keep("d3_1_all", dict(table=all12.T.to_dict(), helped=helped, region_better=region_better))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why does top-down improve the feeder forecasts here?</summary>
# The regional model knows the weather and the calendar; the feeder models know only their own week. Top-down
# carries the regional model's level down to every feeder, in the feeders' own proportions.</details>
#
# <details><summary>2 · When would bottom-up be the better choice?</summary>
# When the feeder forecasts are good and the feeders behave differently from each other, so that a regional
# model averages away what matters. Run the stretch: it happens in some regions.</details>
#
# <details><summary>3 · Does reconciliation always reduce the error?</summary>
# No. It makes the forecasts coherent, always. It improves a level only when it brings in a better forecast
# from another level. Measure both levels, as in the table.</details>
#
# ## Carry forward
#
# The idea to keep: every level reads the same numbers. Lab 3.4 puts the feeder-level scores of several
# method families in one scorecard.
