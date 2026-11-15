# %% [markdown]
# # Lab 1.2 · Three time scales
#
# Day 1 · Session 2 (Seeing structure) · 25 min.
#
# You look at Al-Rimal's daily demand at three time scales (nine years, one year, four weeks) before any
# model. Each scale shows a different pattern. You shade Ramadan and the two Eids, measure the weekday
# profile and the Eid dip of every year, and find one dip that no calendar explains.
#
# **AI policy: AI-tutor.** An assistant may explain a plot; you write the measurements.
#
# **Reference state.** `data/regions.csv` and `data/calendar.csv`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Plot nine years and name the trend and the annual shape (5')
# - [ ] 2 · Plot one year against temperature (5')
# - [ ] 3 · Plot four weeks and compute the weekday profile (6')
# - [ ] 4 · Shade Ramadan and Eid and measure the Eid dip per year (6')
# - [ ] 5 · Find the dip that is not a holiday (3')
# - [ ] ✱ Stretch · Repeat for Al-Jabal, a cool mountain region, and compare

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

df = U.region("R01")          # Al-Rimal: demand (MWh per day), temp_c, temp_normal
y = df["demand"]
cal = U.calendar()
print(df.tail(3))

# %% [markdown]
# ## 1 · Nine years (5')
#
# At this scale you see the trend and the annual shape, and nothing about weekdays.

# %%
fig, ax = plt.subplots()
ax.plot(y.index, y, lw=0.6, color="#16324f")
ax.plot(y.rolling(365, center=True).mean(), color="#e4572e", lw=2, label="365-day mean")
ax.set_title("Al-Rimal, daily demand, 2017 to 2025", loc="left"); ax.set_ylabel("MWh per day"); ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ✏️ **Exercise** · Return the average growth per year, in percent, from the mean of 2017 to the mean of 2025
# (eight years of compounding).

# %%
def annual_growth(y: pd.Series) -> float:
    # BEGIN SOLUTION
    first, last = y.loc["2017"].mean(), y.loc["2025"].mean()
    return ((last / first) ** (1 / 8) - 1) * 100
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    growth = annual_growth(y)
except NotImplementedError:
    growth = U.reference("d1_2")["growth"]
U.check(2.0 < growth < 3.5, f"demand grows about {growth:.1f}% a year")
peak_month = int(y.groupby(y.index.month).mean().idxmax())
ratio = float(y.loc["2025-07"].mean() / y.loc["2025-01"].mean())
print(f"highest month: {peak_month}; July 2025 is {ratio:.1f} times January 2025")

# %% [markdown]
# ## 2 · One year, against temperature (5')

# %%
one = df.loc["2024"]
fig, ax = plt.subplots()
ax.plot(one.index, one.demand, color="#16324f", lw=1)
ax2 = ax.twinx(); ax2.plot(one.index, one.temp_c, color="#e4572e", lw=0.8, alpha=0.7); ax2.grid(False)
ax.set_ylabel("MWh per day"); ax2.set_ylabel("°C"); ax.set_title("Al-Rimal, 2024: demand (dark) and temperature", loc="left")
plt.show()

# %% [markdown]
# ✏️ **Exercise** · Demand follows cooling, and cooling starts above about 22 °C. Return the correlation between
# demand and "cooling degrees" (`temp_c` minus 22, never below zero) over the whole series.

# %%
def cooling_correlation(df: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    cdd = (df.temp_c - 22).clip(lower=0)
    return float(np.corrcoef(cdd, df.demand)[0, 1])
    # END SOLUTION

# %%
try:
    corr = cooling_correlation(df)
except NotImplementedError:
    corr = U.reference("d1_2")["corr"]
U.check(corr > 0.9, f"correlation of demand with cooling degrees: {corr:.2f}")

# %% [markdown]
# ## 3 · Four weeks, and the weekday profile (6')
#
# ✏️ **Exercise** · Return the mean demand of each weekday divided by the overall mean, as a Series indexed
# `Mon` to `Sun`. Use the last three years only, so growth does not blur it.

# %%
def weekday_profile(y: pd.Series) -> pd.Series:
    # BEGIN SOLUTION
    recent = y.loc["2023":"2025"]
    prof = recent.groupby(recent.index.strftime("%a")).mean() / recent.mean()
    return prof.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    # END SOLUTION

# %%
try:
    prof = weekday_profile(y)
except NotImplementedError:
    prof = pd.Series(U.reference("d1_2")["profile"])
print(prof.round(3).to_string())
U.check(prof.idxmin() == "Fri" and prof["Sat"] < prof["Sun"], "Friday is the lowest day, then Saturday: the weekend is Friday and Saturday")

four = y.loc["2025-10-05":"2025-11-01"]
fig, ax = plt.subplots()
ax.plot(four.index, four, marker="o", color="#16324f")
for d in four.index[four.index.dayofweek == 4]:
    ax.axvline(d, color="#e4572e", alpha=0.4)
ax.set_title("Four weeks in autumn 2025; the lines are Fridays", loc="left"); plt.show()

# %% [markdown]
# ## 4 · Ramadan, Eid, and the dip that moves (6')

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6.5))
for ax, yr in zip(axes, (2019, 2022, 2025)):
    s = y.loc[f"{yr}-02-01":f"{yr}-09-15"]
    ax.plot(s.index, s, color="#16324f", lw=1)
    U.shade_events(ax, s.index[0], s.index[-1])
    ax.set_title(f"{yr}: Ramadan (light), Eid al-Fitr and Eid al-Adha (dark)", loc="left", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ✏️ **Exercise** · For one year, find the first day of Eid al-Fitr (`eid_fitr == 1` and `eid_weight == 1`) and
# return its date and the dip: demand on that day against the mean of the same weekday one week before and
# one week after, in percent (negative for a dip). Taking both neighbours cancels the spring warming.

# %%
def eid_dip(y: pd.Series, cal: pd.DataFrame, year: int) -> tuple:
    # BEGIN SOLUTION
    c = cal.loc[str(year)]
    day = c.index[(c.eid_fitr == 1) & (c.eid_weight == 1.0)][0]
    around = (y[day - pd.Timedelta(days=7)] + y[day + pd.Timedelta(days=7)]) / 2
    return day, (y[day] / around - 1) * 100
    # END SOLUTION

# %%
try:
    dips = {yr: eid_dip(y, cal, yr) for yr in range(2017, 2026)}
    dips = {yr: (d.strftime("%Y-%m-%d"), round(float(v), 1)) for yr, (d, v) in dips.items()}
except NotImplementedError:
    dips = {int(k): tuple(v) for k, v in U.reference("d1_2")["dips"].items()}
table = pd.DataFrame(dips, index=["first day of Eid al-Fitr", "dip %"]).T
print(table.to_string())
U.check(table["dip %"].astype(float).max() < -5, f"every Eid al-Fitr is a dip: from {table['dip %'].astype(float).max()}% to {table['dip %'].astype(float).min()}%")
shift = pd.to_datetime(table.iloc[:, 0]).dt.dayofyear.diff().dropna()
print("days earlier each year:", (-shift).astype(int).tolist())

# %% [markdown]
# **Look at.** The dip is as regular as a season, but it arrives 10 to 12 days earlier each Gregorian year. A
# model with a 365-day season will look for it on the wrong days.

# %% [markdown]
# ## 5 · The dip that is not a holiday (3')
#
# In Al-Rimal a cool week can cut demand by a quarter, so a drop alone proves nothing. Look for a drop the
# weather does not explain.
#
# ✏️ **Exercise** · Compute each day's change against the mean of the same weekday one week before and one
# week after, for demand and for temperature. Among the days with no calendar event (`event == ""`) that were
# not cooler than their two neighbours, return the date of the largest drop in demand.

# %%
def strange_day(r: pd.DataFrame, cal: pd.DataFrame) -> pd.Timestamp:
    # BEGIN SOLUTION
    def against_neighbours(s):
        return s - (s.shift(7) + s.shift(-7)) / 2
    drop = against_neighbours(r.demand) / r.demand
    plain = cal.reindex(r.index).event == ""
    not_cooler = against_neighbours(r.temp_c) >= 0
    return drop[plain & not_cooler].idxmin()
    # END SOLUTION

# %%
try:
    odd = strange_day(df, cal)
except NotImplementedError:
    odd = pd.Timestamp(U.reference("d1_2")["odd"])
odd_drop = round(float((y[odd] / ((y[odd - pd.Timedelta(days=7)] + y[odd + pd.Timedelta(days=7)]) / 2) - 1) * 100), 1)
print("largest drop the weather does not explain:", odd.date(), f"{odd_drop}%")
U.check(odd == pd.Timestamp("2023-03-14"), "the drop of 14 March 2023 is in no calendar: ask operations (a sandstorm outage)")
U.keep("d1_2", dict(growth=growth, corr=corr, profile=prof.round(4).to_dict(), dips=dips, odd=odd, odd_drop=odd_drop,
                    peak_month=peak_month, ratio=ratio))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Which pattern can you see only in the four-week plot?</summary>
# The weekly one: Friday low, Saturday a little low. At nine years it is invisible.</details>
#
# <details><summary>2 · Why is "the same date last year" a poor guide to the Eid dip?</summary>
# The Hijri year is about 11 days shorter than the Gregorian year, so Eid falls on a different date each year.</details>
#
# <details><summary>3 · What do you do with 14 March 2023?</summary>
# Ask before deleting. Operations confirm an outage: demand was not low, supply was. Keep a note of it; a model
# should not learn it as a pattern.</details>
#
# ## Carry forward
#
# Three facts later labs use: the weekend is Friday and Saturday; demand follows cooling degrees above 22 °C;
# Eid moves. Lab 1.3 measures how strong each pattern is.
