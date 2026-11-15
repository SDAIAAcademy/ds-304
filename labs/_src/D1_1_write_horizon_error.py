# %% [markdown]
# # Lab 1.1 · Frame three requests
#
# Day 1 · Session 1 (What is forecast, and why) · 25 min.
#
# Three people at Shams Power ask the forecast desk "what will demand be?". Each means something different.
# You turn each request into a forecast frame: the decision it serves, the horizon, the level, how the error
# is measured and which quantile the decision needs. No model yet.
#
# **AI policy: AI-tutor.** Write the frames yourself; an assistant may explain a term.
#
# **Reference state.** `data/requests.json` and `data/regions.csv`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Read the three requests and name the decision behind each (5')
# - [ ] 2 · Set horizon, granularity and level for each (6')
# - [ ] 3 · Turn the cost of each error into a target quantile (6')
# - [ ] 4 · Count the days the data delay takes from the horizon (4')
# - [ ] 5 · Write one framing line per request and swap it (4')
# - [ ] ✱ Stretch · Frame a fourth request from your own workplace

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

requests = U.js("requests")
for r in requests:
    print(f"\n[{r['id']}] {r['who']}\n    {r['text']}")

# %% [markdown]
# ## 1 · The decision behind each request (5')
#
# A forecast is worth something only through the decision it changes. For each request write, in your own
# words, the decision and what happens when the forecast is too low and when it is too high.
#
# ✏️ **Exercise** · Fill the three dictionaries. Free text: one short sentence each.

# %%
def decisions() -> dict:
    # BEGIN SOLUTION
    return {
        "A": dict(decision="How much energy to nominate for each of the next 28 days, per region",
                  too_low="Buy the gap on the spot market at about three times the contract price",
                  too_high="Resell the surplus at about half price"),
        "B": dict(decision="On which day of next week feeder F017 can be taken out for maintenance",
                  too_low="Neighbouring feeders overload on the chosen day: customers lose supply",
                  too_high="A safe day is refused and the maintenance waits"),
        "C": dict(decision="How much transformer capacity to add in Al-Rimal over three years",
                  too_low="The peak summer day exceeds capacity: load shedding",
                  too_high="Capital spent a year or two early"),
    }
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    D = decisions()
except NotImplementedError:
    D = U.reference("d1_1_decisions")
U.keep("d1_1_decisions", D)
U.check(set(D) == {"A", "B", "C"} and all(len(v["decision"]) > 15 for v in D.values()),
        "three decisions, each a full sentence")

# %% [markdown]
# ## 2 · Horizon, granularity and level (6')
#
# - **Horizon**: how many days ahead the furthest forecast is.
# - **Granularity**: the time step of the forecast (`"day"`, or `"year"` for one number per year).
# - **Level**: `"region"` or `"feeder"`.
# - **What is forecast**: `"total"` (the day's energy), `"upper"` (how high it could go) or `"peak"` (the
#   highest day of a period).
#
# ✏️ **Exercise** · Return the frame of each request.

# %%
def frames() -> dict:
    # BEGIN SOLUTION
    return {
        "A": dict(horizon_days=28, granularity="day", level="region", target="total"),
        "B": dict(horizon_days=7, granularity="day", level="feeder", target="upper"),
        "C": dict(horizon_days=3 * 365, granularity="year", level="region", target="peak"),
    }
    # END SOLUTION

# %%
try:
    F = frames()
except NotImplementedError:
    F = U.reference("d1_1_frames")
U.keep("d1_1_frames", F)
U.check(F["A"]["horizon_days"] == 28 and F["B"]["horizon_days"] == 7, "A looks 28 days ahead, B looks 7")
U.check(F["B"]["level"] == "feeder" and F["A"]["level"] == F["C"]["level"] == "region", "B is a feeder, A and C are regions")
U.check(F["C"]["granularity"] == "year" and F["C"]["target"] == "peak", "C asks for one peak per year, not 1,095 daily values")
pd.DataFrame(F).T

# %% [markdown]
# ## 3 · From the cost of each error to a quantile (6')
#
# When being one MWh short costs `cu` and being one MWh long costs `co`, the forecast that minimises the
# expected cost is not the middle of the distribution. It is the quantile `cu / (cu + co)`.
#
# Omar's numbers: a missing MWh costs three contract prices instead of one (so `cu = 2`), a surplus MWh is
# resold at half price (so `co = 0.5`).
#
# ✏️ **Exercise** · Write the function, then read the quantile of each request.

# %%
def target_quantile(cost_under: float, cost_over: float) -> float:
    # BEGIN SOLUTION
    return cost_under / (cost_under + cost_over)
    # END SOLUTION

# %%
try:
    Q = {r["id"]: round(target_quantile(r["cost_under"], r["cost_over"]), 2) for r in requests}
except NotImplementedError:
    Q = U.reference("d1_1_quantiles")
U.keep("d1_1_quantiles", Q)
U.check(Q == {"A": 0.8, "B": 0.9, "C": 0.8}, f"target quantiles {Q}")

# %% [markdown]
# **Look at.** None of the three planners should be given the mean. Omar should nominate the 80th percentile:
# a number demand stays under on four days out of five. Hamad needs the 90th percentile of a single feeder.

# %% [markdown]
# ## 4 · What the data delay takes from the horizon (4')
#
# Regional meter data reaches the forecast desk two days late. A forecast made on Thursday for the next 28
# days starts from Tuesday's data: it is really a forecast 30 days ahead.
#
# ✏️ **Exercise** · Return the number of steps ahead the model must forecast.

# %%
def steps_ahead(horizon_days: int, data_delay_days: int) -> int:
    # BEGIN SOLUTION
    return horizon_days + data_delay_days
    # END SOLUTION

# %%
try:
    steps = {r["id"]: steps_ahead(F[r["id"]]["horizon_days"], r["data_delay_days"]) for r in requests}
except NotImplementedError:
    steps = U.reference("d1_1_steps")
U.keep("d1_1_steps", steps)
U.check(steps["A"] == 30 and steps["B"] == 8, f"steps ahead {steps}")

# How fast does today's value stop describing the future? Mean absolute change after h days, Al-Rimal.
y = U.region()["demand"]
stale = {h: round(float((y.shift(-h) - y).abs().mean() / y.mean() * 100), 1) for h in (1, 2, 7, 28, 30)}
U.keep("d1_1_stale", stale)
print("mean absolute change after h days, % of mean demand:", stale)

# %% [markdown]
# **Look at.** Two more days cost little at 28 days and a lot at 1 day. A delay matters most for short horizons.

# %% [markdown]
# ## 5 · One framing line per request (4')
#
# Write the line you would send back to each planner before any modelling, then swap with a neighbour:
# can they tell what you will deliver?

# %%
lines = []
for r in requests:
    k = r["id"]
    lines.append(f"[{k}] For: {D[k]['decision']}. Forecast: {F[k]['target']} per {F[k]['granularity']}, "
                 f"{F[k]['level']} level, {F[k]['horizon_days']} days ahead ({steps[k]} steps from the last "
                 f"data). Delivered as the {int(Q[k] * 100)}th percentile, scored by pinball loss.")
print("\n".join(lines))
U.save("d1_1_frames.md", "\n\n".join(lines) + "\n")

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why is request C not answered by a 28-day model run 39 times?</summary>
# It asks for the highest day of each of three summers. That is a different target (an annual peak), driven
# by growth and by how hot the summer is, not by last week's demand.</details>
#
# <details><summary>2 · Omar's surplus price falls to a quarter of the contract price. Which way does his quantile move?</summary>
# Being long now costs 0.75 per MWh, so the quantile is 2 / 2.75 = 0.73: lower than 0.8.</details>
#
# <details><summary>3 · Which of the three requests can use aggregated data only?</summary>
# A and C. B needs one feeder. A single feeder can serve few customers: ask the data protection officer
# whether its series can identify a household before using it outside the desk.</details>
#
# ## Carry forward
#
# The frame of request A (28 days, daily, regional, 80th percentile) is the frame of the whole course.
# `out/d1_1_frames.md` holds your three lines.
