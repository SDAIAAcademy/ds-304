# %% [markdown]
# # Lab 1.3 · Decompose and measure strength
#
# Day 1 · Session 3 (Decomposition) · 25 min.
#
# You split regional demand into a trend, a weekly season, an annual season and a remainder, and you give
# each a strength between 0 and 1 so that 12 regions can be compared in one table. Then you read the
# remainder: what a decomposition cannot explain is where the next model has work to do.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/regions.csv` and `data/calendar.csv`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Decompose log demand with weekly and annual periods (6')
# - [ ] 2 · Compute the strength of trend and of each season (6')
# - [ ] 3 · Rank the 12 regions by seasonal strength (5')
# - [ ] 4 · Read the largest remainder days and name them (5')
# - [ ] 5 · Find the region whose trend breaks (3')
# - [ ] ✱ Stretch · Decompose one feeder and compare its strengths with its region

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
from statsmodels.tsa.seasonal import MSTL

cal = U.calendar()
names = U.region_names()
YEARS = U.pick(slice("2017", "2025"), slice("2021", "2025"))   # quick mode: five years
RECENT = slice("2021", "2025")                                 # the 12-region table uses five years: under a minute


def decompose(y: pd.Series) -> pd.DataFrame:
    """Trend, weekly season, annual season and remainder of log demand (they add up to the log)."""
    res = MSTL(np.log(y), periods=(7, 365), stl_kwargs={"robust": True, "outer_iter": 3, "inner_iter": 2}).fit()
    return pd.DataFrame({"log_y": np.log(y), "trend": res.trend, "weekly": res.seasonal["seasonal_7"],
                         "annual": res.seasonal["seasonal_365"], "remainder": res.resid})

# %% [markdown]
# ## 1 · Decompose (6')
#
# Summer swings are larger in MWh when demand is higher, so the parts multiply. Taking the logarithm turns
# multiplication into addition: `log y = trend + weekly + annual + remainder`.
#
# ✏️ **Exercise** · Decompose Al-Rimal and return the table. Then verify that the four parts add up.

# %%
y = U.region("R01")["demand"].loc[YEARS]


def decompose_home(y: pd.Series) -> pd.DataFrame:
    # BEGIN SOLUTION
    return decompose(y)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    parts = decompose_home(y)
except NotImplementedError:
    parts = decompose(y)
gap = (parts.trend + parts.weekly + parts.annual + parts.remainder - parts.log_y).abs().max()
U.check(gap < 1e-8, "trend + weekly + annual + remainder = log demand")
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
for ax, col in zip(axes, ["trend", "annual", "weekly", "remainder"]):
    ax.plot(parts.index, parts[col], lw=0.7, color="#16324f"); ax.set_title(col, loc="left", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2 · Strength (6')
#
# The strength of a component compares the remainder with the component plus the remainder:
#
# `strength = max(0, 1 - Var(remainder) / Var(component + remainder))`
#
# Near 1: the component dominates the noise. Near 0: it is lost in it.
#
# ✏️ **Exercise** · Write the function.

# %%
def strength(component: pd.Series, remainder: pd.Series) -> float:
    # BEGIN SOLUTION
    return max(0.0, 1 - remainder.var() / (component + remainder).var())
    # END SOLUTION

# %%
def strengths(parts: pd.DataFrame, fn) -> dict:
    return {k: round(float(fn(parts[k], parts.remainder)), 3) for k in ("trend", "annual", "weekly")}


def ref_strength(component, remainder):
    return max(0.0, 1 - remainder.var() / (component + remainder).var())


try:
    home = strengths(parts, strength)
except NotImplementedError:
    home = strengths(parts, ref_strength)
print("Al-Rimal:", home)
U.check(home["annual"] > 0.9 and home["annual"] > home["weekly"], "the annual season is the strongest pattern in Al-Rimal")

# %% [markdown]
# ## 3 · Twelve regions (5')
#
# ✏️ **Exercise** · Build the table of strengths for every region (rows: region ids; columns: `trend`,
# `annual`, `weekly`). Use `decompose` and your `strength` on the years in `RECENT` (five years keep the run
# under a minute; a shorter window also changes the trend strength, so compare regions within this table).

# %%
REGIONS = U.pick(sorted(names), ["R01", "R03", "R05", "R11"])     # quick mode: four regions


def strength_table(regions: list) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for r in regions:
        p = decompose(U.region(r)["demand"].loc[RECENT])
        rows[r] = {k: round(float(strength(p[k], p.remainder)), 3) for k in ("trend", "annual", "weekly")}
    return pd.DataFrame(rows).T
    # END SOLUTION

# %%
try:
    table = strength_table(REGIONS)
except NotImplementedError:
    table = pd.DataFrame(U.reference("d1_3_strengths")).T.loc[REGIONS]
if not U.QUICK:
    U.keep("d1_3_strengths", table.T.to_dict())
table.insert(0, "name", [names[r] for r in table.index])
print(table.sort_values("annual", ascending=False).to_string())
U.check(table.loc["R03", "annual"] < table.loc["R01", "annual"], "the cool mountain region Al-Jabal has a weaker annual season than Al-Rimal")

# %% [markdown]
# **Look at.** Where the annual strength is lower, a forecaster gains less from temperature and the weekly
# pattern weighs more. One model setting will not suit all 12 regions.

# %% [markdown]
# ## 4 · What is in the remainder (5')
#
# ✏️ **Exercise** · Return the 12 days of Al-Rimal with the largest absolute remainder, with the calendar's
# `event` column beside them.

# %%
def largest_remainders(parts: pd.DataFrame, cal: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    # BEGIN SOLUTION
    top = parts.remainder.abs().sort_values(ascending=False).index[:n]
    out = parts.loc[top, ["remainder"]].round(3)
    out["event"] = cal.event.reindex(top)
    return out
    # END SOLUTION

# %%
try:
    top = largest_remainders(parts, cal)
except NotImplementedError:
    t = parts.remainder.abs().sort_values(ascending=False).index[:12]
    top = parts.loc[t, ["remainder"]].round(3).assign(event=cal.event.reindex(t))
print(top.to_string())
n_eid = int(top.event.str.startswith("eid").sum())
eid_days = cal.index[(cal.eid_weight >= 0.7) & (cal.index <= parts.index[-1]) & (cal.index >= parts.index[0])]
eid_rem = round(float(parts.remainder.reindex(eid_days).mean()), 3)
other_rem = round(float(parts.remainder.drop(eid_days).mean()), 3)
print(f"mean remainder on the {len(eid_days)} main Eid days: {eid_rem}; on all other days: {other_rem}")
U.check(eid_rem < -0.08, f"on Eid days the remainder averages {eid_rem} in logs (about {(np.exp(eid_rem) - 1) * 100:.0f}%): the decomposition has no place for a moving holiday")
U.check(pd.Timestamp("2023-03-14") in top.index, "the outage of 14 March 2023 is among the 12 largest remainders")
U.check(n_eid < 6, f"only {n_eid} of the 12 largest are Eid days: the others are weather swings and the outage")

# %% [markdown]
# ## 5 · The trend that breaks (3')
#
# ✏️ **Exercise** · For each region, compute the growth of yearly mean demand from 2021 to 2023, in percent.
# Return the region with the largest growth and the table.

# %%
def growth_2021_2023() -> tuple:
    # BEGIN SOLUTION
    df = U.regions()
    yearly = df.groupby(["region", df.date.dt.year]).demand.mean().unstack()
    g = ((yearly[2023] / yearly[2021] - 1) * 100).round(1)
    return g.idxmax(), g
    # END SOLUTION

# %%
try:
    broken, g = growth_2021_2023()
except NotImplementedError:
    g = pd.Series(U.reference("d1_3_growth"))
    broken = g.idxmax()
U.keep("d1_3_growth", g.to_dict())
U.keep("d1_3_home", dict(home=home, n_eid=n_eid, eid_rem=eid_rem, other_rem=other_rem))
print(g.sort_values(ascending=False).head(4).to_string())
U.check(broken == "R11", f"{names[broken]} ({broken}) grew {g[broken]:.0f}% in two years: a step, not a trend")
s = U.region("R11")["demand"].loc["2021":"2023"]
fig, ax = plt.subplots(); ax.plot(s.rolling(28).mean(), color="#16324f")
ax.axvline(pd.Timestamp("2022-07-01"), color="#e4572e"); ax.set_title("Al-Mina, 28-day mean: six industrial feeders join on 1 July 2022", loc="left"); plt.show()

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why decompose the logarithm and not demand itself?</summary>
# The seasonal swing grows with the level. On the log scale it is a constant addition, which is what the
# decomposition assumes.</details>
#
# <details><summary>2 · The Eid days sit in the remainder, about a tenth below zero. What does that tell you about a seasonal model?</summary>
# A fixed 365-day season cannot hold an event that moves 11 days a year. It needs its own regressor (Lab 2.3).</details>
#
# <details><summary>3 · A model is trained on Al-Mina from 2019. What goes wrong?</summary>
# It reads the step of July 2022 as fast growth, or averages two different levels. Train after the break, or
# tell the model about it.</details>
#
# ## Carry forward
#
# The strength table says where the annual season matters most. The remainder says what is still missing:
# moving holidays and outages.
