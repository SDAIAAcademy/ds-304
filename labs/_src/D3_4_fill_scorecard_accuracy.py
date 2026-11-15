# %% [markdown]
# # Lab 3.4 · The scorecard
#
# Day 3 · Session 4 (Choosing among families) · 25 min.
#
# Five method families were measured on the same 388 feeders and the same eight held-out weeks. You put
# accuracy, cost and licence in one scorecard, rank the families under the weightings of two different
# planners, and write down the choice and what would change it.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** The measured results, supplied in `data/reference/`: `fd_stats.json` (the floor and two
# statistical methods), `d3_2.json` (the boosted model) and `d3_3_row.json` (the pretrained forecaster).
# Nothing from your own earlier labs is needed. If you ran Lab 3.3 in Colab, paste your own row below.
#
# **Tasks**
# - [ ] 1 · Load the measured results of five methods (3')
# - [ ] 2 · Compute skill against the floor (5')
# - [ ] 3 · Add run time, upkeep and licence (5')
# - [ ] 4 · Rank under two planners' weightings (7')
# - [ ] 5 · Write the choice and what would change it (5')
# - [ ] ✱ Stretch · Change one weight until the ranking flips, and say what that weight would mean

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

# Paste the row your own Lab 3.3 printed in Colab here (a dictionary), or leave None to use the supplied one.
MY_PRETRAINED = None

# %% [markdown]
# ## 1 · The measured results (3')

# %%
stats, boosted = U.reference("fd_stats"), U.reference("d3_2")
pre = MY_PRETRAINED or U.reference("d3_3_row")
rows = {
    "floor: same weekday last year": dict(stats["floor"], pinball=np.nan, cover80=np.nan),
    "ETS, weekly season, per feeder": stats["ets_weekly"],
    "decompose then ETS, per feeder": stats["stl_ets"],
    "one boosted model, all feeders": dict(boosted["scores"], seconds=sum(boosted["seconds"].values()), fits=boosted["fits"],
                                          by_fold=boosted["by_fold"]["boosted"]),
    "pretrained, zero-shot": dict(mase=pre["mase"], pinball=pre["pinball"], cover80=pre["cover80"], seconds=pre["seconds"],
                                  fits=0, by_fold=pre["by_fold"]),
}
card = pd.DataFrame(rows).T[["mase", "pinball", "cover80", "seconds", "fits", "by_fold"]]
card["measured"] = [True, True, True, True, bool(pre.get("live"))]
print(card.to_string())
if not pre.get("live"):
    print("\n!! The pretrained row comes from the STAND-IN of Lab 3.3, not from a pretrained model. It is shown and "
          "left out of the ranking. Run Lab 3.3 in Colab and paste your row into MY_PRETRAINED to include it.")
U.check(len(card) == 5 and card.loc["floor: same weekday last year", "mase"] > 0, "five method families on the same feeders and folds")

# %% [markdown]
# **Look at.** `by_fold` first. The two folds are 28 days apart and the same method scores very differently
# on them. Keep that spread in mind when two methods differ by a few hundredths.

# %% [markdown]
# ## 2 · Skill against the floor (5')
#
# ✏️ **Exercise** · Return a Series: `1 - mase / mase of the floor` for every row.

# %%
def skill(card: pd.DataFrame) -> pd.Series:
    # BEGIN SOLUTION
    return 1 - card["mase"] / card.loc["floor: same weekday last year", "mase"]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    card["skill"] = skill(card).astype(float).round(3)
except NotImplementedError:
    card["skill"] = (1 - card["mase"] / card.loc["floor: same weekday last year", "mase"]).astype(float).round(3)
print(card[["mase", "skill", "pinball", "cover80"]].to_string())
U.check(card.loc["floor: same weekday last year", "skill"] == 0, "the floor has skill 0 by definition")
U.check(card.loc["ETS, weekly season, per feeder", "skill"] < 0, "a fitted model per feeder can lose to the floor: 388 fits for a negative skill")

# %% [markdown]
# ## 3 · Cost: run time, upkeep, licence (5')
#
# Three more columns:
#
# - **Run time** per 1,000 forecasts, from the measured seconds (776 forecasts of 28 days each: 388 feeders,
#   two folds). These are one machine's seconds: read the ratios, not the values.
# - **Upkeep**: what must be kept working every week, from 1 (nothing) to 4. The course's judgment, below.
# - **Licence** of the software or the weights.
#
# ✏️ **Exercise** · Return the seconds per 1,000 forecasts for every row.

# %%
UPKEEP = {
    "floor: same weekday last year": (1, "nothing to fit; needs one year of history", "none needed"),
    "ETS, weekly season, per feeder": (2, "a fit per feeder at every run", "Apache-2.0 (statsforecast)"),
    "decompose then ETS, per feeder": (3, "a slow fit per feeder at every run; three years of history", "Apache-2.0 (statsforecast)"),
    "one boosted model, all feeders": (3, "a feature pipeline and one training job; retrain as data arrives", "MIT (LightGBM)"),
    "pretrained, zero-shot": (2, "a pinned model version and its runtime; no training", "Apache-2.0 (Chronos-Bolt)"),
}
N_FORECASTS = 776


def seconds_per_thousand(card: pd.DataFrame) -> pd.Series:
    # BEGIN SOLUTION
    return card["seconds"].astype(float) / N_FORECASTS * 1000
    # END SOLUTION

# %%
try:
    card["s_per_1000"] = seconds_per_thousand(card).astype(float).round(1)
except NotImplementedError:
    card["s_per_1000"] = (card["seconds"].astype(float) / N_FORECASTS * 1000).round(1)
card["upkeep"] = [UPKEEP[k][0] for k in card.index]
card["what must be kept working"] = [UPKEEP[k][1] for k in card.index]
card["licence"] = [UPKEEP[k][2] for k in card.index]
print(card[["skill", "pinball", "cover80", "s_per_1000", "upkeep", "licence"]].to_string())
slowest, fastest_fitted = card["s_per_1000"].astype(float).idxmax(), card["s_per_1000"].astype(float).drop("floor: same weekday last year").idxmin()
print(f"slowest: {slowest}; fastest method that is not the floor: {fastest_fitted}")
U.check(card.loc["decompose then ETS, per feeder", "s_per_1000"] > 20 * card.loc["one boosted model, all feeders", "s_per_1000"],
        "the per-feeder decomposition costs more than 20 times the run time of the one boosted model")

# %% [markdown]
# ## 4 · Two planners, two weightings (7')
#
# Turn every criterion into a score from 0 (the worst row) to 1 (the best row), then take a weighted sum.
#
# | Criterion | Better when | Note |
# |---|---|---|
# | `skill` | higher | |
# | `pinball` | lower | the floor has no quantiles: it scores 0 |
# | `coverage_gap` | lower | distance of the 80% coverage from 0.80; the floor scores 0 |
# | `log_time` | lower | logarithm of the seconds per 1,000 forecasts |
# | `upkeep` | lower | |
#
# ✏️ **Exercise** · Write `to_score` (a column to 0-1, missing values to 0) and `weighted` (the weighted sum
# of the score columns). Only rows with a measured result are ranked.

# %%
def to_score(col: pd.Series, higher_is_better: bool) -> pd.Series:
    # BEGIN SOLUTION
    col = col.astype(float)
    lo, hi = col.min(), col.max()
    score = (col - lo) / (hi - lo) if higher_is_better else (hi - col) / (hi - lo)
    return score.fillna(0.0)
    # END SOLUTION


def weighted(scores: pd.DataFrame, weights: dict) -> pd.Series:
    # BEGIN SOLUTION
    return sum(scores[k] * w for k, w in weights.items())
    # END SOLUTION

# %%
def ref_to_score(col, higher_is_better):
    col = col.astype(float)
    lo, hi = col.min(), col.max()
    return ((col - lo) / (hi - lo) if higher_is_better else (hi - col) / (hi - lo)).fillna(0.0)


try:
    to_score(pd.Series([1.0, 2.0]), True), weighted(pd.DataFrame({"a": [1.0]}), {"a": 1.0})
except NotImplementedError:
    to_score, weighted = ref_to_score, (lambda scores, weights: sum(scores[k] * w for k, w in weights.items()))
U.check(to_score(pd.Series([1.0, 3.0, 2.0]), True).tolist() == [0.0, 1.0, 0.5], "to_score: the worst row is 0, the best is 1")
U.check(to_score(pd.Series([1.0, 3.0, np.nan]), False).tolist() == [1.0, 0.0, 0.0], "lower is better, and a missing value scores 0")

ranked = card[card.measured].copy()
crit = pd.DataFrame({"skill": to_score(ranked["skill"], True), "pinball": to_score(ranked["pinball"], False),
                     "coverage_gap": to_score((ranked["cover80"].astype(float) - 0.8).abs(), False),
                     "log_time": to_score(np.log(ranked["s_per_1000"].astype(float) + 0.1), False),
                     "upkeep": to_score(ranked["upkeep"], False)})
WEIGHTS = {
    "purchasing (Omar): buys on a quantile, 12 regions": dict(skill=0.2, pinball=0.4, coverage_gap=0.2, log_time=0.1, upkeep=0.1),
    "operations (Hamad): 400 feeders, a small team": dict(skill=0.25, pinball=0.15, coverage_gap=0.1, log_time=0.2, upkeep=0.3),
}
ranking = pd.DataFrame({who: weighted(crit, w).astype(float).round(3) for who, w in WEIGHTS.items()})
print(crit.round(2).to_string(), "\n")
print(ranking.to_string())
winners = {who: ranking[who].idxmax() for who in WEIGHTS}
print(winners)
U.check(all(abs(sum(w.values()) - 1) < 1e-9 for w in WEIGHTS.values()), "each planner's weights add up to 1")
U.check(ranking.loc["ETS, weekly season, per feeder"].max() < ranking.max().min(), "the weekly ETS per feeder wins under neither weighting")
print("the two weightings", "agree" if len(set(winners.values())) == 1 else "disagree", "on the first place")

# %% [markdown]
# ## 5 · The choice, and what would change it (5')
#
# Write the entry for the forecast desk's decision log. A scorecard does not decide: you do, and you say on
# what evidence.
#
# ✏️ **Exercise** · Return a dictionary with four entries: `chosen` (a row name of the scorecard), `why` (the
# criteria and numbers that decided it, and for whom), `simplest_within_noise` (is a simpler method within the
# fold-to-fold spread of the winner?) and `would_change_it` (one measured result or one change of terms that
# would reverse the choice).

# %%
def my_choice(card: pd.DataFrame, winners: dict) -> dict:
    # BEGIN SOLUTION
    # One defensible answer. The desk runs one method for both readers, and the two weightings disagree.
    floor_name, ops, buy = "floor: same weekday last year", "operations (Hamad): 400 feeders, a small team", "purchasing (Omar): buys on a quantile, 12 regions"
    best = winners[buy]
    return {
        "chosen": best,
        "why": (f"Purchasing buys on a quantile and the floor has none: {best} has the best pinball loss "
                f"({float(card.loc[best, 'pinball']):.3f}) and a skill of {card.loc[best, 'skill']:.2f} against the floor, at "
                f"{card.loc[best, 's_per_1000']:.0f} s per 1,000 forecasts, licence {card.loc[best, 'licence']}. Under the weights of "
                f"operations the first row is: {winners[ops]}. The desk runs one method for both readers and keeps the floor as "
                "the fallback when the pipeline fails."),
        "simplest_within_noise": (f"The floor is the simplest method. Its folds score {card.loc[floor_name, 'by_fold']} and those of the "
                                  f"chosen method {card.loc[best, 'by_fold']}: the chosen method is ahead in both folds by more than a "
                                  "tenth, so the floor is not within noise of it. Two folds are thin evidence all the same."),
        "would_change_it": ("If operations were the only reader, their weighting would keep the floor. A measured pretrained row "
                            "with a better pinball loss at lower upkeep, the 80% coverage of the chosen method falling below 70% "
                            "in a later backtest, or a change of licence of a component would each reopen the choice."),
    }
    # END SOLUTION

# %%
try:
    choice = my_choice(card, winners)
except NotImplementedError:
    choice = {"chosen": winners["operations (Hamad): 400 feeders, a small team"], "why": "(not written yet)",
              "simplest_within_noise": "(not written yet)", "would_change_it": "(not written yet)"}
for k, v in choice.items():
    print(f"{k}: {v}")
U.save("d3_4_choice.md", "\n\n".join(f"**{k}**: {v}" for k, v in choice.items()) + "\n")
U.check(choice["chosen"] in card.index and len(choice) == 4, "the choice names a row of the scorecard")
U.keep("d3_4", dict(card=card.drop(columns=["by_fold"]).T.to_dict(), by_fold=card["by_fold"].to_dict(), criteria=crit.round(3).T.to_dict(),
                    ranking=ranking.T.to_dict(), winners=winners, pretrained_live=bool(pre.get("live"))))

# %%
# The stretch, done: operations put 0.30 on upkeep. How far must that weight fall before the floor loses first place?
w = dict(WEIGHTS["operations (Hamad): 400 feeders, a small team"])
others = {k: v for k, v in w.items() if k != "upkeep"}
flip = None
for up in np.arange(0.30, -0.001, -0.025):
    trial = {**{k: v * (1 - up) / sum(others.values()) for k, v in others.items()}, "upkeep": up}
    first = weighted(crit, trial).astype(float).idxmax()
    if first != winners["operations (Hamad): 400 feeders, a small team"]:
        flip = (round(float(up), 3), first)
        break
print("weight on upkeep at which the first place of operations changes, and to which method:", flip)
U.keep("d3_4_flip", dict(upkeep_weight=None if flip is None else flip[0], new_first=None if flip is None else flip[1]))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why is run time on a logarithmic scale?</summary>
# The methods differ by factors of ten and more. On a linear scale the slowest method would set the scale and
# all the others would look equally fast.</details>
#
# <details><summary>2 · Two methods differ by 0.02 in scaled error. Is the lower one better?</summary>
# Look at the folds. The same method moves by far more than 0.02 from one fold to the next. Within that spread,
# choose on cost, upkeep and licence: the simplest method within noise of the best.</details>
#
# <details><summary>3 · What does a scorecard not show?</summary>
# What happens on the days that matter most (score Eid days apart), how the method fails, and who can repair
# it at night. Add the rows your organisation needs.</details>
#
# ## Carry forward
#
# `out/d3_4_choice.md` is your decision-log entry. The capstone README needs the same table for your three
# method families: accuracy, probabilistic accuracy, run time, licence.
