# %% [markdown]
# # Lab 4.2 · Feeders with eight weeks of history
#
# Day 4 · Session 2 (New and short series) · 25 min.
#
# Twelve feeders were commissioned on 11 September 2025, one in each region. On 5 November they have eight
# weeks of history: no last year to repeat, no annual season to fit. You forecast their next 28 days four
# ways: from their own history, by borrowing the shape of similar feeders, with the global boosted model and
# with the pretrained forecaster. Then you say what the planner should be told.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/feeders.npz`, `data/feeders_meta.csv`, `data/regions.csv` and `data/calendar.csv`.
# Nothing from your own earlier labs: the global model is trained again here, in a few seconds.
#
# **Tasks**
# - [ ] 1 · Meet the twelve feeders with eight weeks of history (3')
# - [ ] 2 · Forecast each from its own history (5')
# - [ ] 3 · Pool: borrow the profile of similar feeders (7')
# - [ ] 4 · Add the global model and the pretrained forecaster (6')
# - [ ] 5 · Compare, and say what to tell the planner (4')
# - [ ] ✱ Stretch · Cut the history to four weeks and compare again

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
import lightgbm as lgb
from statsforecast.models import AutoETS

meta = U.meta().set_index("feeder")
ALL = U.feeders()
NEW, OLD = U.new_feeders(), U.established()
ORIGIN = U.FEEDER_ORIGINS[0]                       # 2025-11-05: the new feeders have 56 days of history
DAYS = pd.date_range(ORIGIN + pd.Timedelta(days=1), periods=28, freq="D")
Wn = ALL[NEW]

# %% [markdown]
# ## 1 · The twelve new feeders (3')

# %%
hist = Wn.loc[:ORIGIN].dropna(how="all")
print(f"{len(NEW)} feeders, {len(hist)} days of history up to {ORIGIN.date()}")
print(meta.loc[NEW, ["region", "kind"]].assign(mean_mwh=hist.mean().round(1)).to_string())
U.check(len(hist) == 56, "eight weeks of history at the origin")
fig, ax = plt.subplots()
ax.plot(Wn.loc["2025-09-01":ORIGIN + pd.Timedelta(days=28), NEW[:3]])
ax.axvline(ORIGIN, color="#e4572e"); ax.set_title("three of the twelve; the line is the origin", loc="left"); plt.show()

# %% [markdown]
# **Look at.** The floor of this course, "the same weekday last year", does not exist for these feeders.
# Whatever you deliver must come with that sentence.

# %% [markdown]
# ## 2 · From their own history (5')
#
# Two forecasts that use the feeder's 56 days and nothing else: last week repeated, and ETS with a weekly
# season.
#
# ✏️ **Exercise** · Return "last week repeated" for every new feeder: a table of 28 days by 12 feeders.

# %%
def last_week(history: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    values = np.tile(history.to_numpy()[-7:], (4, 1))
    return pd.DataFrame(values, index=DAYS, columns=history.columns)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_last_week(history):
    return pd.DataFrame(np.tile(history.to_numpy()[-7:], (4, 1)), index=DAYS, columns=history.columns)


try:
    fc_week = last_week(hist)
except NotImplementedError:
    last_week = ref_last_week
    fc_week = last_week(hist)
U.check(fc_week.shape == (28, 12) and np.allclose(fc_week.iloc[7], hist.iloc[-7]), "day 8 of the forecast equals day 1: the last week, four times")
fc_ets = U.sf_many(AutoETS(season_length=7, model="ANA"), hist)
fc_ets.index = DAYS
results = {"last week repeated": U.feeder_long(Wn, ORIGIN, fc_week), "ETS, weekly, own history": U.feeder_long(Wn, ORIGIN, fc_ets)}
print({k: round(U.mase(v), 3) for k, v in results.items()})

# %% [markdown]
# ## 3 · Pool: borrow the shape of similar feeders (7')
#
# A new residential feeder in Al-Rimal will follow the path of the other residential feeders in Al-Rimal:
# the same weather, the same holidays. Those feeders have a last year. So:
#
# 1. For every established feeder of the **same region and kind** (of the same region, when fewer than three
#    share the kind), take its floor forecast for the 28 days (the same weekdays a year earlier) and divide
#    by its own mean over the 28 days up to the origin. That is its **shape**: where it is expected to go,
#    relative to where it is now.
# 2. Take the median shape over those feeders, day by day.
# 3. Multiply by the new feeder's own mean over its last 28 days.
#
# ✏️ **Exercise** · Write it for one new feeder. `U.feeder_floor(wide, origin)` returns the floor forecasts of
# a wide table.

# %%
def similar_to(feeder: str) -> list:
    """Established feeders of the same region and kind; of the same region when fewer than three share the kind."""
    same = [f for f in OLD if meta.region[f] == meta.region[feeder] and meta.kind[f] == meta.kind[feeder]]
    return same if len(same) >= 3 else [f for f in OLD if meta.region[f] == meta.region[feeder]]


def pooled(feeder: str) -> np.ndarray:
    # BEGIN SOLUTION
    Ws = ALL[similar_to(feeder)]
    shape = U.feeder_floor(Ws, ORIGIN) / Ws.loc[:ORIGIN].iloc[-28:].mean()
    own_level = ALL[feeder].loc[:ORIGIN].iloc[-28:].mean()
    return (shape.median(axis=1) * own_level).to_numpy()
    # END SOLUTION

# %%
def ref_pooled(feeder):
    Ws = ALL[similar_to(feeder)]
    shape = U.feeder_floor(Ws, ORIGIN) / Ws.loc[:ORIGIN].iloc[-28:].mean()
    return (shape.median(axis=1) * ALL[feeder].loc[:ORIGIN].iloc[-28:].mean()).to_numpy()


try:
    fc_pool = pd.DataFrame({f: pooled(f) for f in NEW}, index=DAYS)
except NotImplementedError:
    pooled = ref_pooled
    fc_pool = pd.DataFrame({f: pooled(f) for f in NEW}, index=DAYS)
n_similar = {f: len(similar_to(f)) for f in NEW}
print("similar established feeders per new feeder:", n_similar)
results["pooled shape of similar feeders"] = U.feeder_long(Wn, ORIGIN, fc_pool)
print({k: round(U.mase(v), 3) for k, v in results.items()})
U.check(fc_pool.shape == (28, 12) and fc_pool.notna().all().all(), "a pooled forecast for every new feeder")
U.check(U.mase(results["pooled shape of similar feeders"]) < U.mase(results["last week repeated"]),
        "borrowing the neighbours' last year beats repeating the feeder's own last week")

# %% [markdown]
# **Look at.** The new feeder supplied one number, its level. Everything else came from feeders that have a
# history. That is pooling in its simplest form.

# %% [markdown]
# ## 4 · The global model and the pretrained forecaster (6')
#
# The boosted model of Lab 3.2 was trained across feeders on features that need 28 days of history. A new
# feeder has them. Its `year_ago` feature is missing, which the model accepts.

# %%
TRAIN_ORIGINS = [ORIGIN - pd.Timedelta(days=28 * k) for k in range(U.pick(13, 4), 0, -1)]
Wo = ALL[OLD] if not U.QUICK else ALL[[f for f in OLD if meta.region[f] in ("R01", "R05", "R11")]]
train = U.make_frame(Wo, TRAIN_ORIGINS)
train = train[train.date <= ORIGIN]                # no target after the origin
rows_new = U.make_frame(Wn, [ORIGIN])
print(f"trained on {len(train):,} rows of {Wo.shape[1]} established feeders; {len(rows_new)} rows to forecast; "
      f"year_ago missing on {rows_new.year_ago.isna().mean():.0%} of them")

# %% [markdown]
# ✏️ **Exercise** · Train the three quantile models (0.1, 0.5, 0.9) on `train` and forecast `rows_new`, in
# MWh. Return a dictionary `{"q0.1": array, "q0.5": array, "q0.9": array}`. Use
# `lgb.train({**U.LGB, "objective": "quantile", "alpha": q}, lgb.Dataset(X, y), num_boost_round=ROUNDS)`.

# %%
ROUNDS = U.pick(300, 80)


def global_quantiles(train: pd.DataFrame, rows: pd.DataFrame) -> dict:
    # BEGIN SOLUTION
    out = {}
    for q in U.PQS:
        model = lgb.train({**U.LGB, "objective": "quantile", "alpha": q}, lgb.Dataset(train[U.FEATURES], train.y), num_boost_round=ROUNDS)
        out[f"q{q}"] = model.predict(rows[U.FEATURES]) * rows.level.to_numpy()
    return out
    # END SOLUTION

# %%
def ref_global_quantiles(train, rows):
    return {f"q{q}": lgb.train({**U.LGB, "objective": "quantile", "alpha": q}, lgb.Dataset(train[U.FEATURES], train.y),
                               num_boost_round=ROUNDS).predict(rows[U.FEATURES]) * rows.level.to_numpy() for q in U.PQS}


try:
    gq = global_quantiles(train, rows_new)
except NotImplementedError:
    gq = ref_global_quantiles(train, rows_new)
g = rows_new[["feeder", "origin", "date", "h", "scale"]].assign(y=rows_new.y * rows_new.level, yhat=gq["q0.5"], **gq)
results["global boosted model"] = g
U.check(len(g) == 12 * 28 and g.yhat.notna().all(), "the global model forecasts feeders it never saw in training")

contexts = [hist[f].to_numpy() for f in NEW]       # 56 days each: all the history there is
pq, info = U.pretrained(contexts)
quantiles = {f"q{tau}": pq[:, j, :].T for j, tau in enumerate(U.PQS)}
results["pretrained, zero-shot"] = U.feeder_long(Wn, ORIGIN, quantiles["q0.5"], quantiles)
print("pretrained forecaster answered by:", info["model"])
if not info["live"]:
    print("!! STAND-IN: its row below is not a pretrained model's result. Run this notebook in Colab, after installing "
          "chronos-forecasting as in Lab 3.3, for the real one.")

# %% [markdown]
# ## 5 · Compare, and what to tell the planner (4')

# %%
table = pd.DataFrame({name: U.scores(b, qs=U.PQS) for name, b in results.items()}).T
table["measured"] = [True, True, True, True, bool(info["live"])]
print(table.to_string())

# %% [markdown]
# ✏️ **Exercise** · Return the name of the method with the lowest scaled error among the measured rows.

# %%
def best_method(table: pd.DataFrame) -> str:
    # BEGIN SOLUTION
    return table[table.measured]["mase"].astype(float).idxmin()
    # END SOLUTION

# %%
try:
    best = best_method(table)
except NotImplementedError:
    best = table[table.measured]["mase"].astype(float).idxmin()
own = float(table.loc["last week repeated", "mase"])
print(f"best measured method: {best} ({float(table.loc[best, 'mase']):.3f}); from own history alone: {own:.3f}")
U.check(best not in ("last week repeated", "ETS, weekly, own history"), "a method that borrows from other series wins on short history")
note = (f"Twelve new feeders, eight weeks of history. Forecast by: {best}. Scaled error {float(table.loc[best, 'mase']):.2f} on one fold of "
        f"28 days (336 feeder-days): one fold is thin evidence. No last-year benchmark exists for these feeders. "
        f"Review after each further month of data.")
print("\nFor the planner:", note)
U.save("d4_2_note.md", note + "\n")
U.keep("d4_2", dict(table=table.T.to_dict(), best=best, n_similar=n_similar, live=bool(info["live"]),
                    missing_year_ago=float(rows_new.year_ago.isna().mean())))

# %%
# The stretch, done: four weeks of history (the pooled shape and the own-history methods)
hist4 = hist.iloc[-28:]
four = {"last week repeated": U.mase(U.feeder_long(Wn, ORIGIN, ref_last_week(hist4))),
        "ETS, weekly, own history": U.mase(U.feeder_long(Wn, ORIGIN, U.sf_many(AutoETS(season_length=7, model="ANA"), hist4))),
        "pooled shape of similar feeders": U.mase(results["pooled shape of similar feeders"])}
print("with four weeks of history:", {k: round(v, 3) for k, v in four.items()}, "(the pooled shape already used only the last 28 days)")
U.keep("d4_2_stretch", {k: round(v, 3) for k, v in four.items()})

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why can the floor not be computed for these feeders?</summary>
# The floor repeats the same weekday of last year. A feeder commissioned in September has no last year.</details>
#
# <details><summary>2 · What did the pooled forecast take from the new feeder, and what from its neighbours?</summary>
# Its level from the new feeder; the path over the next 28 days from established feeders of the same region
# and kind.</details>
#
# <details><summary>3 · The comparison rests on one fold of 28 days. What follows?</summary>
# Report it as thin evidence, deliver wide intervals, and repeat the comparison as each month of data arrives.
# A method choice made on 336 points can flip.</details>
#
# ## Carry forward
#
# The capstone datasets hold new series too: charging hubs that opened during the data. `out/d4_2_note.md` is
# the kind of sentence your planner's note needs for them.
