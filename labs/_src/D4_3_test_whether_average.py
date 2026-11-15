# %% [markdown]
# # Lab 4.3 · Does the average beat the best?
#
# Day 4 · Session 3 (Combining forecasts) · 25 min.
#
# You average three forecasts from different method families and ask a fair question: does the average beat
# the best single method, when "the best" has to be chosen before the test, not after it?
#
# **AI policy: AI-tutor.**
#
# **Reference state.** Three supplied backtests of Al-Rimal over the 13 origins of 2025:
# `data/reference/bt_stl_ets.csv`, `bt_arima_fourier.csv` and `bt_dynreg.csv`. Nothing from your own earlier
# labs.
#
# **Tasks**
# - [ ] 1 · Load three backtests and score each (4')
# - [ ] 2 · Average them and score the average (5')
# - [ ] 3 · Choose the best single method on early folds; test on late folds (7')
# - [ ] 4 · Average the quantiles and check coverage (5')
# - [ ] 5 · State the verdict with its numbers (4')
# - [ ] ✱ Stretch · Weight the three methods by their early-fold errors

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

BTS = {"decompose, then ETS": U.bt("stl_ets"), "ARIMA + annual Fourier": U.bt("arima_fourier"),
       "regression: weather + calendar": U.bt("dynreg")}
QCOLS = U.QCOLS
folds = sorted(BTS["decompose, then ETS"].origin.unique())
EARLY, LATE = folds[:6], folds[6:]                 # six folds to choose on, seven to test on
print(f"{len(folds)} folds; early: {pd.Timestamp(EARLY[0]).date()} to {pd.Timestamp(EARLY[-1]).date()}; "
      f"late: {pd.Timestamp(LATE[0]).date()} to {pd.Timestamp(LATE[-1]).date()}")

# %% [markdown]
# ## 1 · Three methods, one test year (4')

# %%
single = pd.DataFrame({name: U.scores(b) for name, b in BTS.items()}).T
print(single.to_string())
same = all(np.allclose(b.y, BTS["decompose, then ETS"].y) and (b.date.to_numpy() == BTS["decompose, then ETS"].date.to_numpy()).all() for b in BTS.values())
U.check(same, "the three backtests cover the same days, in the same order: they can be averaged row by row")

# %% [markdown]
# ## 2 · The simple average (5')
#
# ✏️ **Exercise** · Return a backtest table whose `yhat` is the mean of the three point forecasts, row by
# row. Keep `origin`, `date`, `h`, `y` and `scale` from the first table.

# %%
def average(bts: dict) -> pd.DataFrame:
    # BEGIN SOLUTION
    first = next(iter(bts.values()))
    out = first[["origin", "date", "h", "y", "scale"]].copy()
    out["yhat"] = np.mean([b.yhat.to_numpy() for b in bts.values()], axis=0)
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def ref_average(bts):
    out = next(iter(bts.values()))[["origin", "date", "h", "y", "scale"]].copy()
    out["yhat"] = np.mean([b.yhat.to_numpy() for b in bts.values()], axis=0)
    return out


try:
    avg = average(BTS)
except NotImplementedError:
    average = ref_average
    avg = average(BTS)
U.check(np.allclose(avg.yhat, sum(b.yhat.to_numpy() for b in BTS.values()) / 3), "yhat is the mean of the three forecasts")
all_year = {**{name: round(U.mase(b), 3) for name, b in BTS.items()}, "average of three": round(U.mase(avg), 3)}
print(all_year)
U.check(U.mase(avg) < np.mean([U.mase(b) for b in BTS.values()]), "the average forecast is better than the average of the three scores")

# %% [markdown]
# **Look at.** The average always beats the mean of the single scores: errors in opposite directions cancel.
# Whether it beats the **best** single method is another question, and the fair version of it is next.

# %% [markdown]
# ## 3 · "The best single method", chosen in advance (7')
#
# Looking at the table of task 1 you know which single method was best over the year. You know it **after**
# the year. A forecaster in June had only the folds up to June.
#
# ✏️ **Exercise** · Choose the single method with the lowest scaled error on the `EARLY` folds. Return its
# name, its score on the `LATE` folds, and the average's score on the `LATE` folds.

# %%
def fair_comparison(bts: dict, avg: pd.DataFrame, early: list, late: list) -> tuple:
    # BEGIN SOLUTION
    early_scores = {name: U.mase(b[b.origin.isin(early)]) for name, b in bts.items()}
    chosen = min(early_scores, key=early_scores.get)
    b = bts[chosen]
    return chosen, U.mase(b[b.origin.isin(late)]), U.mase(avg[avg.origin.isin(late)])
    # END SOLUTION

# %%
def ref_fair_comparison(bts, avg, early, late):
    early_scores = {name: U.mase(b[b.origin.isin(early)]) for name, b in bts.items()}
    chosen = min(early_scores, key=early_scores.get)
    return chosen, U.mase(bts[chosen][bts[chosen].origin.isin(late)]), U.mase(avg[avg.origin.isin(late)])


try:
    chosen, chosen_late, avg_late = fair_comparison(BTS, avg, EARLY, LATE)
except NotImplementedError:
    chosen, chosen_late, avg_late = ref_fair_comparison(BTS, avg, EARLY, LATE)
halves = pd.DataFrame({"early folds": {name: round(U.mase(b[b.origin.isin(EARLY)]), 3) for name, b in BTS.items()},
                       "late folds": {name: round(U.mase(b[b.origin.isin(LATE)]), 3) for name, b in BTS.items()}})
halves.loc["average of three"] = [round(U.mase(avg[avg.origin.isin(EARLY)]), 3), round(U.mase(avg[avg.origin.isin(LATE)]), 3)]
print(halves.to_string())
best_late_in_hindsight = halves.drop(index="average of three")["late folds"].idxmin()
print(f"\nchosen on the early folds: {chosen} -> {chosen_late:.3f} on the late folds; the average -> {avg_late:.3f}")
print(f"best single method on the late folds, known only afterwards: {best_late_in_hindsight}")
U.check(chosen in BTS and chosen == halves.drop(index="average of three")["early folds"].idxmin(), "the single method is chosen on the early folds only")

# %% [markdown]
# ## 4 · Average the quantiles (5')
#
# A combination needs intervals too. The simplest rule: average each quantile across the methods.
#
# ✏️ **Exercise** · Add the five quantile columns to the average: each is the mean of that column over the
# three methods.

# %%
def average_quantiles(bts: dict, avg: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    out = avg.copy()
    for c in QCOLS:
        out[c] = np.mean([b[c].to_numpy() for b in bts.values()], axis=0)
    return out
    # END SOLUTION

# %%
try:
    avg_q = average_quantiles(BTS, avg)
except NotImplementedError:
    avg_q = avg.copy()
    for c in QCOLS:
        avg_q[c] = np.mean([b[c].to_numpy() for b in BTS.values()], axis=0)
prob = pd.DataFrame({**{name: U.scores(b) for name, b in BTS.items()}, "average of three": U.scores(avg_q)}).T
print(prob.to_string())
U.check(set(QCOLS) <= set(avg_q.columns), "the average has its five quantiles")
U.check(abs(prob.loc["average of three", "cover80"] - 0.8) < abs(prob.loc["regression: weather + calendar", "cover80"] - 0.8),
        "the averaged 80% interval is closer to its promise than the interval of the best point forecast")

# %% [markdown]
# **Look at.** Averaging quantiles mixes one interval that is too narrow with two that are about right. It
# helps; it is not a calibration. Lab 2.4's rebuild from backtest errors can be applied to the average too.

# %% [markdown]
# ## 5 · The verdict (4')
#
# ✏️ **Exercise** · Return True if, on the late folds, the average beat the single method chosen on the early
# folds.

# %%
def average_wins(chosen_late: float, avg_late: float) -> bool:
    # BEGIN SOLUTION
    return avg_late < chosen_late
    # END SOLUTION

# %%
try:
    wins = bool(average_wins(chosen_late, avg_late))
except NotImplementedError:
    wins = bool(avg_late < chosen_late)
verdict = (f"On the seven late folds the average of three scored {avg_late:.3f} and the single method chosen on the six early folds "
           f"({chosen}) scored {chosen_late:.3f}: the average {'won' if wins else 'lost'} by {abs(chosen_late - avg_late):.3f}. "
           f"Over the whole year the average scored {all_year['average of three']} against {min(v for k, v in all_year.items() if k != 'average of three')} "
           f"for the best single method in hindsight. Pinball loss: average {prob.loc['average of three', 'pinball']}, "
           f"best single {prob.drop(index='average of three')['pinball'].min()}.")
print(verdict)
U.save("d4_3_verdict.md", verdict + "\n")

# The stretch, done: weights from the early folds (inverse error), tested on the late folds
inv = {name: 1 / U.mase(b[b.origin.isin(EARLY)]) for name, b in BTS.items()}
wts = {name: v / sum(inv.values()) for name, v in inv.items()}
weighted_fc = avg.copy()
weighted_fc["yhat"] = sum(wts[name] * b.yhat.to_numpy() for name, b in BTS.items())
weighted_late = U.mase(weighted_fc[weighted_fc.origin.isin(LATE)])
print("weights from the early folds:", {k: round(v, 2) for k, v in wts.items()}, f"-> {weighted_late:.3f} on the late folds (simple average: {avg_late:.3f})")
U.keep("d4_3", dict(all_year=all_year, halves=halves.T.to_dict(), chosen=chosen, chosen_late=chosen_late, avg_late=avg_late, wins=wins,
                    best_late_in_hindsight=best_late_in_hindsight, prob=prob.T.to_dict(), weights=wts, weighted_late=weighted_late))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why is "the average lost to the best method" an unfair comparison?</summary>
# The best method of the test year is known only after the test year. Comparing against it is selection on the
# test window, the third leak of Lab 4.1.</details>
#
# <details><summary>2 · When does averaging help most?</summary>
# When the methods are each reasonable and make different errors: different inputs, different model families.
# Averaging three copies of one idea gains nothing.</details>
#
# <details><summary>3 · What does a combination cost?</summary>
# Every member must be kept running, and the intervals need their own check. Put the combination in the
# scorecard with its upkeep, like any other method.</details>
#
# ## Carry forward
#
# `out/d4_3_verdict.md`. In the capstone, an average of your three method families is a cheap fourth row for
# the README table: score it under the same rule, chosen in advance.
