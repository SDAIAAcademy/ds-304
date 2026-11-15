# %% [markdown]
# # Lab 4.4 · Fan chart and planner's note
#
# Day 4 · Session 4 (Fan charts and forecast notes) · 25 min.
#
# You make the forecast the week has been leading to: Al-Rimal, the 28 days after the last day of data, with
# bands rebuilt from two years of backtest errors. You draw it as a fan chart, state in plain words how often
# such bands held, write the note the load planning manager reads, and choose your capstone dataset.
#
# **AI policy: AI-tutor.** The note is yours; an assistant may comment on its wording.
#
# **Reference state.** `data/regions.csv`, `data/calendar.csv`, the supplied backtests `bt_dynreg_2024.csv`,
# `bt_dynreg.csv`, `bt_dynreg_rebuilt.csv` and `bt_floor.csv` in `data/reference/`, and `capstone/`. Nothing
# from your own earlier labs.
#
# **Tasks**
# - [ ] 1 · Draw the fan chart for the next 28 days (7')
# - [ ] 2 · State the backtest coverage in plain words (4')
# - [ ] 3 · Write the planner's note (8')
# - [ ] 4 · Check the note against its six required parts (3')
# - [ ] 5 · Choose the capstone dataset and three method families (3')
# - [ ] ✱ Stretch · Write the note in Arabic with the same numbers

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

RID = "R01"
y = U.region(RID)["demand"]
ORIGIN = U.LAST                                    # 2025-12-31: the last day of data
DAYS = pd.date_range(ORIGIN + pd.Timedelta(days=1), periods=28, freq="D")
point = U.dynreg(RID, y.iloc[-1096:], 28, ORIGIN)  # the regression of Lab 2.3, trained on the last three years
history = pd.concat([U.bt("dynreg_2024"), U.bt("dynreg")], ignore_index=True)   # its backtest errors: 26 folds
print(f"forecast for {DAYS[0].date()} to {DAYS[-1].date()}; point forecast from {point.yhat.min():.0f} to {point.yhat.max():.0f} MWh")

# %% [markdown]
# ## 1 · The fan chart (7')
#
# A fan chart shows the median and the bands, widening with the horizon. The bands come from the errors the
# same model made in its backtest (Lab 2.4): for each week of the horizon, the quantiles of the scaled errors
# of all 26 folds, added to the point forecast.
#
# ✏️ **Exercise** · Build the table of the forecast: columns `date`, `q0.025`, `q0.1`, `q0.5`, `q0.9`,
# `q0.975`. `error_quantiles` holds the quantile of the scaled error for each horizon week (rows 1 to 4) and
# each quantile (columns); `scale` is the model's scaled-error denominator for this training window.

# %%
history["week"] = (history.h - 1) // 7 + 1
history["e"] = (history.y - history.yhat) / history["scale"]
error_quantiles = history.groupby("week").e.quantile(list(U.QS)).unstack()
scale = U.scale(y.iloc[-1096:])
print(error_quantiles.round(2).to_string())


def fan_table(point: pd.DataFrame, error_quantiles: pd.DataFrame, scale: float) -> pd.DataFrame:
    # BEGIN SOLUTION
    week = (np.arange(28) // 7) + 1
    out = pd.DataFrame({"date": DAYS})
    for q in U.QS:
        out[f"q{q}"] = point.yhat.to_numpy() + error_quantiles.loc[week, q].to_numpy() * scale
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_fan_table(point, error_quantiles, scale):
    week = (np.arange(28) // 7) + 1
    out = pd.DataFrame({"date": DAYS})
    for q in U.QS:
        out[f"q{q}"] = point.yhat.to_numpy() + error_quantiles.loc[week, q].to_numpy() * scale
    return out


try:
    fc = fan_table(point, error_quantiles, scale)
except NotImplementedError:
    fc = ref_fan_table(point, error_quantiles, scale)
U.check((fc["q0.025"] < fc["q0.1"]).all() and (fc["q0.1"] < fc["q0.9"]).all() and (fc["q0.9"] < fc["q0.975"]).all(), "the quantiles are in order on every day")
U.check((fc["q0.9"] - fc["q0.1"]).iloc[-7:].mean() > (fc["q0.9"] - fc["q0.1"]).iloc[:7].mean(), "the 80% band is wider in week 4 than in week 1")
fig, ax = plt.subplots(figsize=(10, 4))
U.fan(ax, y.iloc[-56:], fc, "Al-Rimal: eight weeks measured, 28 days forecast")
U.shade_events(ax, y.index[-56], DAYS[-1])
OUT_DIR = U.OUT; OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "d4_4_fan.png", dpi=150, bbox_inches="tight"); plt.show()
fc.round(1).to_csv(OUT_DIR / "d4_4_forecast.csv", index=False)

# %% [markdown]
# **Look at.** The median line is the least important thing on the chart. Omar buys near the upper edge of
# the dark band; Shahad reads how wide the fan is in week 4.

# %% [markdown]
# ## 2 · Coverage in plain words (4')
#
# "80% interval" is a label. What the planner can use is how often such a band held when it was tested.
# `bt_dynreg_rebuilt.csv` is the 2025 backtest with bands built this way from earlier folds only.
#
# ✏️ **Exercise** · Return a sentence of the form "In the 2025 backtest the dark band held on N of 100 days
# and the light band on M of 100." with N and M computed from the table (`U.coverage(bt, 80)` and
# `U.coverage(bt, 95)` return shares).

# %%
rebuilt = U.bt("dynreg_rebuilt")


def coverage_sentence(bt: pd.DataFrame) -> str:
    # BEGIN SOLUTION
    n, m = round(U.coverage(bt, 80) * 100), round(U.coverage(bt, 95) * 100)
    return f"In the 2025 backtest the dark band held on {n} of 100 days and the light band on {m} of 100."
    # END SOLUTION

# %%
try:
    sentence = coverage_sentence(rebuilt)
except NotImplementedError:
    sentence = (f"In the 2025 backtest the dark band held on {round(U.coverage(rebuilt, 80) * 100)} of 100 days "
                f"and the light band on {round(U.coverage(rebuilt, 95) * 100)} of 100.")
print(sentence)
U.check(str(round(U.coverage(rebuilt, 80) * 100)) in sentence and str(round(U.coverage(rebuilt, 95) * 100)) in sentence, "the sentence carries the two measured numbers")

# %% [markdown]
# ## 3 · The planner's note (8')
#
# One page for Shahad Al-Tamimi, the load planning manager. Six parts, in this order:
#
# 1. **decision**: what the forecast is for, and for whom
# 2. **forecast**: the numbers: the range of the median, and the day and size of the highest upper quantile
# 3. **uncertainty**: the bands in plain words, with the coverage sentence
# 4. **evidence**: the backtest score against the floor
# 5. **limits**: when not to use this forecast
# 6. **source**: data up to which day, method in one line, who to ask
#
# The numbers you need are computed below.

# %%
floor_bt = U.bt("floor")
facts = {
    "first_day": DAYS[0].strftime("%d %B %Y"), "last_day": DAYS[-1].strftime("%d %B %Y"),
    "median_low": round(float(fc["q0.5"].min())), "median_high": round(float(fc["q0.5"].max())),
    "q90_peak": round(float(fc["q0.9"].max())), "q90_peak_day": fc.date[fc["q0.9"].idxmax()].strftime("%d %B"),
    "band80_week1": round(float((fc["q0.9"] - fc["q0.1"]).iloc[:7].mean())), "band80_week4": round(float((fc["q0.9"] - fc["q0.1"]).iloc[-7:].mean())),
    "score": round(U.mase(rebuilt), 2), "floor": round(U.mase(floor_bt), 2),
    "skill_percent": round((1 - U.mase(rebuilt) / U.mase(floor_bt)) * 100),
    "events": sorted(set(U.calendar().event.reindex(DAYS)) - {""}),
}
print(facts)

# %% [markdown]
# ✏️ **Exercise** · Return the note as a dictionary with the six keys above. Use the numbers in `facts` and
# your `sentence`. Write for a manager: no model names in the first five parts.

# %%
def planners_note(facts: dict, sentence: str) -> dict:
    # BEGIN SOLUTION
    return {
        "decision": (f"Daily energy demand of Al-Rimal for {facts['first_day']} to {facts['last_day']}, for the weekly purchase "
                     "nomination of energy purchasing and the reserve decision of load planning."),
        "forecast": (f"The middle forecast runs between {facts['median_low']:,} and {facts['median_high']:,} MWh a day. "
                     f"The upper edge of the dark band, under which demand is expected on 9 days in 10, reaches {facts['q90_peak']:,} MWh on the highest day, {facts['q90_peak_day']}."),
        "uncertainty": (f"The dark band is where demand is expected on 8 days in 10, the light band on 19 in 20. The dark band is about "
                        f"{facts['band80_week1']:,} MWh wide in the first week and {facts['band80_week4']:,} MWh in the fourth. {sentence}"),
        "evidence": (f"Tested on every day of 2025, 28 days ahead: scaled error {facts['score']} against {facts['floor']} for repeating the same "
                     f"weekday of last year, {facts['skill_percent']}% better."),
        "limits": ("Do not use it for a single feeder, for hours within the day, or beyond 28 days. It assumes ordinary weather around the "
                   "seasonal normal: in a declared heat wave or dust storm, ask for a rerun. It knows the announced calendar: if a holiday "
                   "date moves, ask for a rerun. Outages are not forecast."),
        "source": (f"Regional meter data to 31 December 2025. Method: regression on temperature and the Hijri and Gregorian calendar with "
                   "ARIMA errors; bands from two years of backtest errors. Forecast desk Marqab, Nada Al-Jasser."),
    }
    # END SOLUTION

# %%
REQUIRED = ["decision", "forecast", "uncertainty", "evidence", "limits", "source"]
try:
    note = planners_note(facts, sentence)
except NotImplementedError:
    note = U.reference("d4_4_note")
print("\n\n".join(f"{k.upper()}\n{v}" for k, v in note.items()))

# %% [markdown]
# ## 4 · Check the note (3')
#
# ✏️ **Exercise** · Return the list of problems: a required part that is missing or shorter than 40
# characters; a note without the coverage sentence's two numbers; a `limits` part that names fewer than two
# situations (count the sentences).

# %%
def problems(note: dict, rebuilt: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    out = [f"part missing or too short: {k}" for k in REQUIRED if len(note.get(k, "")) < 40]
    n, m = str(round(U.coverage(rebuilt, 80) * 100)), str(round(U.coverage(rebuilt, 95) * 100))
    if n not in note.get("uncertainty", "") or m not in note.get("uncertainty", ""):
        out.append("the uncertainty part does not state the measured coverage")
    if note.get("limits", "").count(".") < 2:
        out.append("the limits part names fewer than two situations")
    return out
    # END SOLUTION

# %%
def ref_problems(note, rebuilt):
    out = [f"part missing or too short: {k}" for k in REQUIRED if len(note.get(k, "")) < 40]
    n, m = str(round(U.coverage(rebuilt, 80) * 100)), str(round(U.coverage(rebuilt, 95) * 100))
    if n not in note.get("uncertainty", "") or m not in note.get("uncertainty", ""):
        out.append("the uncertainty part does not state the measured coverage")
    if note.get("limits", "").count(".") < 2:
        out.append("the limits part names fewer than two situations")
    return out


try:
    found = problems(note, rebuilt)
except NotImplementedError:
    problems = ref_problems
    found = problems(note, rebuilt)
print("problems:", found or "none")
U.check(found == [], "the note has its six parts, the measured coverage and its limits")
U.check(problems({**note, "limits": "Use with care."}, rebuilt) != [], "a note whose limits say nothing is caught")
text = "# Forecast note: Al-Rimal, next 28 days\n\n" + "\n\n".join(f"**{k.capitalize()}.** {v}" for k, v in note.items()) + "\n\n![fan chart](d4_4_fan.png)\n"
U.save("d4_4_note.md", text)
U.keep("d4_4_note", note)
U.keep("d4_4", dict(facts=facts, sentence=sentence, error_quantiles=error_quantiles.round(3).T.to_dict()))
if U.WRITE_REF:
    fc.round(1).to_csv(U.REF / "d4_4_forecast.csv", index=False, lineterminator="\n")

# %% [markdown]
# ## 5 · Your capstone (3')
#
# Two datasets are supplied in `capstone/data/`; `capstone/README.md` holds the planner's request for each. A
# public dataset is allowed when its open-data terms permit your use and you state them.

# %%
hubs = pd.read_csv(U.LABS / "capstone" / "data" / "ev_hubs.csv", parse_dates=["date"])
plants = pd.read_csv(U.LABS / "capstone" / "data" / "cooling_plants.csv", parse_dates=["date"])
print(f"ev_hubs.csv: {hubs.hub.nunique()} charging hubs, {hubs.date.min().date()} to {hubs.date.max().date()}, "
      f"history per hub from {hubs.groupby('hub').size().min()} to {hubs.groupby('hub').size().max()} days")
print(f"cooling_plants.csv: {plants.plant.nunique()} district-cooling plants, {plants.date.min().date()} to {plants.date.max().date()}")

# %% [markdown]
# ✏️ **Exercise** · Fill your choice: the dataset, the decision your forecast serves, the horizon in days,
# and three method families (at least one statistical and one that learns across series or is pretrained).

# %%
def capstone_choice() -> dict:
    # BEGIN SOLUTION
    return {"dataset": "ev_hubs.csv", "decision": "energy to reserve for each charging hub over the next 14 days",
            "horizon_days": 14, "families": ["seasonal naive floor", "ETS per hub", "one boosted model across hubs"]}
    # END SOLUTION

# %%
try:
    cap = capstone_choice()
except NotImplementedError:
    cap = {"dataset": "(not chosen yet)", "decision": "", "horizon_days": 0, "families": []}
print(cap)
if cap["families"]:
    U.check(len(cap["families"]) == 3 and cap["horizon_days"] > 0 and len(cap["decision"]) > 15, "a dataset, a decision, a horizon and three families")
    U.save("d4_4_capstone_choice.md", "\n".join(f"- **{k}**: {v}" for k, v in cap.items()) + "\n")

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why does the note say "8 days in 10" and also the measured coverage?</summary>
# The first is the promise; the second is how the promise was kept when tested. A planner needs both, and
# trusts the second.</details>
#
# <details><summary>2 · What belongs in "when not to use"?</summary>
# The conditions the backtest did not cover or the method cannot see: another level or horizon, extreme
# weather, a holiday that moves, outages.</details>
#
# <details><summary>3 · Why are model names kept out of the first five parts?</summary>
# The planner decides on numbers, evidence and limits. The method is one line in the source part for whoever
# needs to check or rerun it.</details>
#
# ## Carry forward
#
# `out/d4_4_note.md`, `out/d4_4_fan.png` and `out/d4_4_forecast.csv` are the model for the capstone's
# two-page note. `out/d4_4_capstone_choice.md` is what you bring to the capstone day.
