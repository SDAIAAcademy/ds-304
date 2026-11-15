# %% [markdown]
# # Lab 3.3 · A pretrained forecaster, zero-shot
#
# Day 3 · Session 3 (Pretrained zero-shot forecasters) · 25 min.
#
# A pretrained forecaster was trained on a large collection of other people's series. You give it the history
# of a feeder it has never seen and it returns quantiles: no training on your data. You run a small one on the
# feeders, time it, score it on the eight held-out weeks like any other method, and read the licences of four
# models from three families.
#
# **Where to run it.** In Google Colab: the cell below installs the package and the weights (about nine
# million parameters) download into the Colab runtime. Nothing is downloaded to your laptop. Outside Colab,
# or when the package is absent, a **stand-in** answers so that the notebook still runs: it repeats each
# feeder's weekday profile. The stand-in is not a pretrained model and every result says which one answered.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/feeders.npz`, `data/feeders_meta.csv` and `data/model_cards.json`. Nothing from
# your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Load the forecaster and forecast one feeder (5')
# - [ ] 2 · Forecast the feeders zero-shot and time the run (6')
# - [ ] 3 · Score point, quantile and coverage on the held-out weeks (6')
# - [ ] 4 · Record the licence of four models from three families (5')
# - [ ] 5 · Write the scorecard row (3')
# - [ ] ✱ Stretch · Shorten the context to eight weeks and score again

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

# %%
import json
import time

if IN_COLAB:                                       # the pretrained forecaster's package, in the Colab runtime only
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting"], check=True)

W = U.feeders()[U.established()]
if U.QUICK:
    W = W.iloc[:, :40]                             # quick mode: 40 feeders, and the stand-in
TEST = U.FEEDER_ORIGINS                            # two folds of 28 days: the eight held-out weeks
print(f"{W.shape[1]} feeders; the forecaster is {U.PRETRAINED}; quantiles {U.PQS}")

# %% [markdown]
# ## 1 · One feeder (5')
#
# The model's only input is the **context**: the feeder's own history up to the origin, as an array. No
# calendar, no temperature, no feeder id.
#
# ✏️ **Exercise** · Return the last `length` values of a feeder up to and including the origin.

# %%
def context_of(W: pd.DataFrame, feeder: str, origin: pd.Timestamp, length: int = 512) -> np.ndarray:
    # BEGIN SOLUTION
    return W[feeder].loc[:origin].dropna().to_numpy()[-length:]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_context_of(W, feeder, origin, length=512):
    return W[feeder].loc[:origin].dropna().to_numpy()[-length:]


one = W.columns[0]
try:
    ctx = context_of(W, one, TEST[0])
except NotImplementedError:
    context_of = ref_context_of
    ctx = context_of(W, one, TEST[0])
U.check(len(ctx) == 512 and ctx[-1] == W.loc[TEST[0], one], "512 days, the last one is the origin day: nothing after it")

q, info = U.pretrained([ctx])                      # array [series, quantile, day], and who answered
print("answered by:", info["model"], "| shape:", q.shape, "| seconds:", round(info["seconds"], 2))
if not info["live"]:
    print("!! STAND-IN: these are not a pretrained model's forecasts. Run this notebook in Colab for the real ones.")
days = pd.date_range(TEST[0] + pd.Timedelta(days=1), periods=28, freq="D")
fc = pd.DataFrame({"date": days, "q0.1": q[0, 0], "q0.5": q[0, 1], "q0.9": q[0, 2], "y": W[one].reindex(days).to_numpy()})
fig, ax = plt.subplots()
U.fan(ax, W[one].loc[TEST[0] - pd.Timedelta(days=55): TEST[0]], fc, f"{one}: eight weeks of history, 28 days zero-shot (80% band)")
plt.show()

# %% [markdown]
# **Look at.** This model family was trained to predict the quantiles 0.1 to 0.9. Its package returns the 0.1
# and 0.9 values for anything further out, so a "95% interval" from it would be its 80% interval under another
# name. That is why Day 3 scores three quantiles and the 80% interval for every method.

# %% [markdown]
# ## 2 · All feeders, timed (6')
#
# `U.pretrained(contexts)` takes a list of contexts and returns an array `[series, quantile, day]` for the
# quantiles 0.1, 0.5, 0.9.
#
# ✏️ **Exercise** · Forecast one fold: build the list of contexts, call the forecaster, and return a long
# table with `U.feeder_long(W, origin, yhat, quantiles)`. `yhat` is the median as a table of 28 days by
# feeders; `quantiles` is a dictionary `{"q0.1": ..., "q0.5": ..., "q0.9": ...}` of such tables. Return the
# table and the `info`.

# %%
def forecast_fold(W: pd.DataFrame, origin: pd.Timestamp, length: int = 512) -> tuple:
    # BEGIN SOLUTION
    contexts = [context_of(W, f, origin, length) for f in W.columns]
    q, info = U.pretrained(contexts)
    quantiles = {f"q{tau}": q[:, j, :].T for j, tau in enumerate(U.PQS)}
    return U.feeder_long(W, origin, quantiles["q0.5"], quantiles), info
    # END SOLUTION

# %%
def ref_forecast_fold(W, origin, length=512):
    q, info = U.pretrained([ref_context_of(W, f, origin, length) for f in W.columns])
    quantiles = {f"q{tau}": q[:, j, :].T for j, tau in enumerate(U.PQS)}
    return U.feeder_long(W, origin, quantiles["q0.5"], quantiles), info


t0 = time.perf_counter()
try:
    folds = [forecast_fold(W, o) for o in TEST]
except NotImplementedError:
    forecast_fold = ref_forecast_fold
    folds = [forecast_fold(W, o) for o in TEST]
seconds = time.perf_counter() - t0
bt = pd.concat([f for f, _ in folds], ignore_index=True)
info = folds[0][1]
print(f"{2 * W.shape[1]} forecasts of 28 days in {seconds:.1f} s, by: {info['model']}")
U.check(len(bt) == 2 * 28 * W.shape[1] and {"q0.1", "q0.5", "q0.9"} <= set(bt.columns), "one row per feeder, fold and day, with three quantiles")

# %% [markdown]
# ## 3 · Score it like any other method (6')
#
# ✏️ **Exercise** · Return the skill against the floor: `1 - error / error of the floor`, with `U.mase`.

# %%
def skill(bt: pd.DataFrame, floor: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    return 1 - U.mase(bt) / U.mase(floor)
    # END SOLUTION

# %%
floor = pd.concat([U.feeder_long(W, o, U.feeder_floor(W, o)) for o in TEST], ignore_index=True)
try:
    sk = skill(bt, floor)
except NotImplementedError:
    sk = 1 - U.mase(bt) / U.mase(floor)
sc = U.scores(bt, qs=U.PQS)
by_fold = [round(U.mase(bt[bt.origin == o]), 3) for o in TEST]
print("answered by:", info["model"])
print("scores:", sc, "| by fold:", by_fold)
print(f"floor: {U.mase(floor):.3f} | skill against the floor: {sk:.2f}")
U.check(np.isfinite(sc["mase"]) and 0 <= sc["cover80"] <= 1, "scaled error, pinball loss and coverage are computed on the held-out weeks")

# %% [markdown]
# **Look at.** Compare the two folds. A model that sees only the feeder's own past cannot know that the
# cooling season is ending or that a holiday is coming: it has no calendar and no temperature.

# %% [markdown]
# ## 4 · Licences (5')
#
# Weights you can download are not weights you may use for anything. The model card states the licence.
# `data/model_cards.json` records four models of three families as read from their cards on the date in the file.
#
# ✏️ **Exercise** · Shams Power is a company and would use the forecasts in its operations. Return True when
# a licence allows that. Among these four cards the licences are `apache-2.0` (permissive: commercial use
# allowed) and `cc-by-nc-4.0` (NC means non-commercial).

# %%
def commercial_use(licence: str) -> bool:
    # BEGIN SOLUTION
    return licence.lower() in {"apache-2.0", "mit", "bsd-3-clause"}
    # END SOLUTION

# %%
cards = U.js("model_cards")
try:
    commercial_use("apache-2.0")
except NotImplementedError:
    commercial_use = lambda licence: licence.lower() in {"apache-2.0", "mit", "bsd-3-clause"}
lic = pd.DataFrame(cards["models"])[["family", "model", "parameters_millions", "licence"]]
lic["commercial use"] = lic.licence.map(commercial_use)
print(f"read from the model cards on {cards['checked']}")
print(lic.to_string(index=False))
U.check(lic.set_index("family")["commercial use"].to_dict() == {"Chronos-Bolt": True, "Chronos-2": True, "TimesFM": True, "Moirai": False},
        "three families are permissive; one is for non-commercial use only")

# %% [markdown]
# **Look at.** A licence is a fact with a date. Read the card again on the day you decide: names, versions and
# terms of these models change within months.

# %% [markdown]
# ## 5 · The scorecard row (3')

# %%
row = dict(method="pretrained, zero-shot", model=info["model"], live=bool(info["live"]), mase=sc["mase"], pinball=sc["pinball"],
           cover80=sc["cover80"], by_fold=by_fold, seconds=round(seconds, 1), fits=0, forecasts=2 * W.shape[1],
           licence=lic.licence[lic.model == U.PRETRAINED].iloc[0], floor=round(U.mase(floor), 3), skill=round(sk, 3))
print(json.dumps(row, indent=1))
U.save("d3_3_row.json", json.dumps(row, indent=1) + "\n")
if not U.QUICK:
    U.keep("d3_3_row", row)
if not row["live"]:
    print("!! This row was produced by the STAND-IN. Do not put it in a scorecard as a pretrained model's result.")

# %%
# The stretch, done: eight weeks of context
short = pd.concat([forecast_fold(W, o, 56)[0] for o in TEST], ignore_index=True)
print(f"context of 56 days: {U.scores(short, qs=U.PQS)}; context of 512 days: {sc}")
if not U.QUICK:
    U.keep("d3_3_short", dict(short=U.scores(short, qs=U.PQS), long=sc, live=bool(info["live"])))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · What does "zero-shot" mean here?</summary>
# The model's weights were never updated on Shams Power data. It forecasts from the context alone.</details>
#
# <details><summary>2 · Why score it with the same folds, scale and quantiles as the other methods?</summary>
# So that the scorecard compares methods and not measuring instruments. A leaderboard rank on other people's
# data says little about your feeders.</details>
#
# <details><summary>3 · The card says cc-by-nc-4.0. What can the desk do with that model?</summary>
# Study it and compare against it. Not run it for the company's operations. Use a permissively licensed model
# for that, or ask the publisher for other terms.</details>
#
# ## Carry forward
#
# `out/d3_3_row.json` is your row for the scorecard of Lab 3.4. If it says `"live": false`, it came from the
# stand-in: run this notebook in Colab to get the real row.
