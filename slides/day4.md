---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/slides/day4.html"
---

# Forecasting and Time Series

## Day 4: Review and communicate

Review a backtest for leaks, forecast series with little history, combine methods and write the note the planner reads

#### [day4#title] Review and communicate

*SDAIA Academy · Forecasting and Time Series*

Day 4: Review a backtest for leaks, forecast series with little history, combine methods and write the note the planner reads

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-304

#### [day4#agenda] Review and communicate

*Today*

- **Session 1 · Leakage across time**: A backtest leaks when anything after the origin reaches the forecast: a feature, a regressor or the choice of model
- **Session 2 · New and short series**: A series with eight weeks of history borrows strength from its neighbours, from a global model or from pretraining
- **Session 3 · Combining forecasts**: An average of different methods is hard to beat, and "the best single method" is only known afterwards
- **Session 4 · Fan charts and forecast notes**: The planner reads a fan chart and a short note: the forecast, how often the bands held, and when not to use it

#### [day4#review-day-3] What may be said about these two methods?

*Review · Day 3*

**Question:** On two held-out folds, method P scores 0.64 (0.75 and 0.53 by fold) and method Q scores 0.62 (0.73 and 0.51). P is a table lookup; Q needs a feature pipeline. What is the sound reading?

- A. Q is the more accurate method and should replace P
- B. The gap is small beside the spread between folds: prefer the simpler P
- C. Two folds are too few to say anything about either method
- D. Q, because a trained model will improve as data arrives

### Session 1: Leakage across time

#### [day4#s1] Leakage across time

*Session 1*

- Find features and regressors that use data after the origin
- Recognise selection on the test window
- Correct a backtest and report the score before and after

#### [day4#a-score-too-good] A score far below every honest result is a finding to explain

*A backtest arrives for review*

- **1.06** the floor, Al-Rimal, 13 origins over 2025
- **0.77** the regression of Lab 2.3 with weather and calendar
- **0.24** the backtest you are asked to review

> **The reviewer's first question.** What does this forecast know on the origin day, and how does it know it?

#### [day4#three-doors-for-leakage] Three doors through which the future reaches a forecast

*Leakage across time*

- **1 · A feature.** A column computed from the series itself that reaches past the origin: a centred average, a lag shorter than the horizon.
- **2 · A regressor.** An outside variable used with values that were not known on the origin day: measured weather, revised figures.
- **3 · A choice.** A model, setting or window picked because it scored best on the test folds. No column leaks; the decision does.

#### [day4#feature-sees-the-future] A centred average contains the days being forecast

*Lab 4.1 · step 2*

```python
# file: as received
X["smooth"] = data.demand.rolling(7, center=True, min_periods=1).mean().reindex(dates)   # t-3 to t+3
```

```python
# file: fix 1
X["smooth"] = data.demand.rolling(7).mean().shift(28).reindex(dates)       # ends 28 days before t
```

**0.24 → 0.41** scaled error before and after the fix

> **The rule.** A feature built from the target must end at least one horizon before the day it describes: here 28 days.

#### [day4#regressor-from-the-future] The measured temperature of the test days was never available

*Lab 4.1 · step 3*

```python
# file: fix 2
temp = data.temp_c.reindex(dates)                                  # measured: fine up to the origin
temp = temp.where(dates <= origin, U.temp_known(RID, origin, dates))  # after it: what was known
X["cdd"] = (temp - 22).clip(lower=0)
```

**0.41 → 0.71** scaled error before and after the fix

> **The rule.** For every outside variable ask what its value was on the origin day, not what it turned out to be.

#### [day4#selection-on-the-test-window] The best setting in every test fold is known only afterwards

*Lab 4.1 · step 4*

| Training window, days | 2024 folds | 2025 folds |
| --- | --- | --- |
| 365 | 0.694 | 0.683 |
| 548 | 0.722 | 0.689 |
| 730 | 0.733 | 0.682 |
| 1,096 | 0.759 | 0.711 |
| 1,461 | 0.750 | 0.717 |

- **0.651** as reported: the best window of each 2025 fold, averaged
- **0.683** honest: one window chosen on 2024, scored on 2025

#### [day4#a-test-for-leaking-columns] Change the future, and see which columns change with it

*Lab 4.1 · stretch*

- Build the features for one origin.
- Replace every value after the origin by a different one and build them again.
- A column that changed is reading the future.

| Feature function | Columns that changed |
| --- | --- |
| As received | `cdd`, `smooth` |
| After fix 1 | `cdd` |
| After fix 2 | none |

> **What it cannot catch.** Selection on the test window is not in any column. That fault is found by asking how each setting was chosen.

#### [day4#the-review-memo] The memo reports every fault with the score before and after

*Lab 4.1 · step 5*

| State of the backtest | Scaled error | What changed |
| --- | --- | --- |
| As received | 0.244 |  |
| After fix 1 | 0.414 | feature ends 28 days before the target |
| After fix 2 | 0.711 | temperature as known at the origin |
| After fix 3: the score to report | 0.683 | one window, chosen on 2024 |
| The floor | 1.061 |  |

> **The model survives its review.** At 0.68 it is still the best regional model of the course. The claim of 0.24 does not.

#### [day4#check-selection-leak] Where is the leak?

*Checkpoint*

**Question:** An analyst backtests four models on the 13 folds of 2025. Every feature and regressor passes a known-at-the-origin test. She reports: "Our forecast scores 0.70: for each fold we use the model that was most accurate on it." What should the review memo say?

- A. Nothing: the features are clean, so the score stands
- B. Add more models, so that the best one per fold is better still
- C. The model was chosen on the test folds: choose on earlier folds and rescore
- D. Report the pinball loss in place of the scaled error

#### [day4#lab-4-1] Three leaks in a backtest

Lab 4.1 · 25 min

Review a backtest with three seeded faults, fix each one, and write a review memo with the score before and after every fix.

[D4_1_find_three_seeded.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D4_1_find_three_seeded.ipynb)

- [ ] Run the faulty backtest and record its score (3 min)
- [ ] Find and fix the feature that sees the future (6 min)
- [ ] Find and fix the regressor not known at the origin (6 min)
- [ ] Find and fix the selection on the test window (6 min)
- [ ] Write the review memo with before and after (4 min)
- [ ] ✱ Stretch: Write a test that fails when a feature uses a day after the origin

### Session 2: New and short series

#### [day4#s2] New and short series

*Session 2*

- Forecast a short series from its own history
- Pool information from similar series
- Compare pooled, global and pretrained forecasts on a new series

#### [day4#twelve-new-feeders] Twelve feeders with eight weeks of history, and four weeks to forecast

*The cold start*

Figure: One new feeder: 56 days up to the origin of 5 November 2025, then the 28 days to forecast (dashed).

- **What is missing.** No last year: the floor of this course does not exist. No annual season can be estimated.
- **What exists.** A level, eight weekday patterns, a region, a kind, and 388 older feeders.

#### [day4#from-its-own-history] From its own eight weeks, a feeder can repeat or smooth

*Lab 4.2 · step 2*

- **1.10** last week repeated four times
- **0.95** ETS with a weekly season on 56 days

> **Both are blind to the season.** The four weeks ahead are cooler than the eight behind. Nothing in 56 days of one feeder says so.

#### [day4#borrow-the-shape] Borrow the shape from similar feeders; keep your own level

*Lab 4.2 · step 3*

1. **1 · Similar** established feeders of the same region and kind
2. **2 · Their shape** each one's next 28 days from last year, divided by its recent level
3. **3 · Median** one shape for the group
4. **4 · Own level** multiply by the new feeder's last 28 days

**0.92** pooled shape of similar feeders

> **The new feeder supplied one number,** its level. The season came from between 7 and 29 neighbours.

#### [day4#global-model-on-new-series] The global model needs nothing more than 28 days of a new feeder

*Lab 4.2 · step 4*

- Trained on the 388 established feeders only.
- For a new feeder `year_ago` is missing on every row. The trees route missing values by themselves.
- The rest of the row exists: last week, weekday profile, calendar, expected change in cooling, region, kind.

- **0.75** scaled error on the 12 new feeders
- **60%** coverage of its 80% interval

#### [day4#short-history-comparison] The methods that borrow from other series come first

*Lab 4.2 · step 5 · 12 new feeders, one fold*

| Method | Uses | Scaled error |
| --- | --- | --- |
| Last week repeated | own 7 days | 1.10 |
| ETS, weekly season | own 56 days | 0.95 |
| Pooled shape | own level, neighbours' season | 0.92 |
| Global boosted model | own 28 days, 388 feeders' patterns | 0.75 |
| Pretrained, zero-shot | own 56 days, pretraining | your run |

> **Thin evidence.** 336 feeder-days in one fold. With four weeks of history ETS scored 0.85, better than with eight: that is noise, not a finding.

#### [day4#what-to-tell-the-planner] What Hamad is told about a forecast for a new feeder

*Lab 4.2 · step 5*

- **Say.** Which method. Its error on the one fold there is. That no last-year benchmark exists for these feeders. When it will be reviewed: after each further month of data.
- **Do not say.** "The model is 80% confident." Its 80% band held on 60% of days. Until the band is rebuilt from errors, give the point forecast and the size of past misses.

> **For the capstone.** A new series changes the floor, the amount of evidence and the wording of the note. State all three.

#### [day4#check-cold-start] Which forecast for a site that opened six weeks ago?

*Checkpoint*

**Question:** A charging hub opened six weeks ago, in spring. Forty older hubs in the same city have two years of history and a clear summer rise. The planner needs the next four weeks. Which approach is the soundest starting point?

- A. The older hubs' seasonal shape at this time of year, times the new hub's level
- B. A seasonal ARIMA with an annual season on the hub's six weeks
- C. The average demand of the forty older hubs
- D. No forecast until the hub has a full year of data

#### [day4#lab-4-2] Feeders with eight weeks of history

Lab 4.2 · 25 min

Forecast the twelve new feeders from their first eight weeks with their own history, a pooled profile, the global model and the pretrained forecaster, and compare.

[D4_2_forecast_feeder_eight.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D4_2_forecast_feeder_eight.ipynb)

- [ ] Meet the twelve feeders with eight weeks of history (3 min)
- [ ] Forecast each from its own history (5 min)
- [ ] Pool: borrow the profile of similar feeders (7 min)
- [ ] Add the global model and the pretrained forecaster (6 min)
- [ ] Compare, and say what to tell the planner (4 min)
- [ ] ✱ Stretch: Cut the history to four weeks and compare again

### Session 3: Combining forecasts

#### [day4#s3] Combining forecasts

*Session 3*

- Average forecasts from different method families
- Choose the best single method without looking at the test folds
- Combine quantiles and check coverage

#### [day4#why-averages-work] Methods that err in different directions cancel part of each other's error

*Combining forecasts*

- The error of an average is never above the average of the errors.
- The gain is larger the less the methods' errors move together.
- Three models from one family, fed the same inputs, cancel little.

- **The three in Lab 4.3.** Decompose then ETS: seasons only. ARIMA with annual Fourier terms: seasons and recent errors. Regression: temperature and calendar. Three views of one series.

> **The simple average has no settings.** Nothing to tune means nothing to tune on the test window.

#### [day4#average-of-three] Over the year the average beats two of the three, not the best one

*Lab 4.3 · steps 1 and 2 · Al-Rimal, 13 folds of 2025*

| Method | Scaled error |
| --- | --- |
| Decompose, then ETS | 0.984 |
| ARIMA with annual Fourier terms | 0.826 |
| Regression: weather and calendar | 0.766 |
| Average of the three | 0.793 |
| Mean of the three single scores | 0.859 |

> **"The average lost to the regression" is a comparison made in hindsight.** Nobody knew in January which single method would win the year.

#### [day4#choose-before-you-look] Choose the single method on early folds, then compare on late folds

*Lab 4.3 · step 3*

| Method | 6 early folds | 7 late folds |
| --- | --- | --- |
| Decompose, then ETS | 0.895 | 1.059 |
| ARIMA with annual Fourier terms | 0.686 | 0.945 |
| Regression: chosen on the early folds | 0.596 | 0.913 |
| Average of the three | 0.652 | 0.914 |

> **A tie.** On the late folds the average and the honestly chosen single method differ by 0.001.

#### [day4#average-the-quantiles] On the quantiles the average is the best row of the table

*Lab 4.3 · step 4*

| Method, all 13 folds | Pinball | 80% covers | 95% covers |
| --- | --- | --- | --- |
| Decompose, then ETS | 0.243 | 83.5% | 93.1% |
| ARIMA with annual Fourier terms | 0.190 | 78.8% | 93.1% |
| Regression: weather and calendar | 0.208 | 63.5% | 76.9% |
| Average of the three, quantile by quantile | 0.184 | 81.6% | 94.0% |

> **Why.** The regression has the best centre and intervals that are too narrow. The average keeps most of the centre and takes its width from the other two.

#### [day4#the-verdict-on-averaging] The verdict, with its numbers and its limits

*Lab 4.3 · step 5*

- **Point forecast.** A tie on the late folds: 0.914 for the average, 0.913 for the method chosen in advance.
- **Quantiles.** The average wins: pinball 0.184 against 0.190 for the best single method, with coverage near nominal.
- **Cost.** Three pipelines to keep running in place of one. That belongs in the scorecard of Lab 3.4.

> **Stretch.** Weights from early-fold errors (0.26, 0.34, 0.39) score 0.906 on the late folds against 0.914 for equal weights: a gain inside the noise.

#### [day4#check-best-in-hindsight] Is this comparison fair to the average?

*Checkpoint*

**Question:** A report says: "On the 2025 test folds the average of our four models scored 0.80. The best single model scored 0.76. We therefore deliver the best single model." What is the weakness of the argument?

- A. The average should have used weights fitted on the 2025 folds
- B. Four models are too few for an average to work
- C. The scaled error cannot compare an average with a single model
- D. "Best" was read off the test folds: the choice was not available in advance

#### [day4#lab-4-3] Does the average beat the best?

Lab 4.3 · 25 min

Average three supplied backtests, choose the best single method on the early folds only, and test on the late folds whether the average beats it.

[D4_3_test_whether_average.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D4_3_test_whether_average.ipynb)

- [ ] Load three backtests and score each (4 min)
- [ ] Average them and score the average (5 min)
- [ ] Choose the best single method on early folds; test on late folds (7 min)
- [ ] Average the quantiles and check coverage (5 min)
- [ ] State the verdict with its numbers (4 min)
- [ ] ✱ Stretch: Weight the three methods by their early-fold errors

### Session 4: Fan charts and forecast notes

#### [day4#s4] Fan charts and forecast notes

*Session 4*

- Draw a fan chart from quantile forecasts
- State interval coverage in plain words
- Write a forecast note that says when not to use the forecast

#### [day4#the-fan-chart] The planner's picture: where demand is expected, and how sure we are

*Lab 4.4 · step 1 · Al-Rimal, 1 to 28 January 2026*

Figure: Navy: the last eight weeks of 2025. Teal line: the median forecast. Dark band: 8 days in 10. Light band: 19 days in 20.

#### [day4#bands-from-backtest-errors] The bands are the backtest's own errors, by week of the horizon

*Lab 4.4 · step 1*

```python
# file: Lab 4.4 · step 1
week = (np.arange(28) // 7) + 1
for q in U.QS:      # 0.025, 0.1, 0.5, 0.9, 0.975
    out[f"q{q}"] = (point.yhat
        + error_quantiles.loc[week, q] * scale)
```

| Width of the dark band | MWh |
| --- | --- |
| Week 1 | 525 |
| Week 4 | 869 |

> **Nothing new.** The point forecast is the regression of Lab 2.3. The error quantiles are those of Lab 2.4, taken from the folds of 2024 and 2025.

#### [day4#coverage-in-plain-words] State what the bands did, in counts a planner can check

*Lab 4.4 · step 2*

- **Not this.** "The 80% prediction interval has an empirical coverage of 0.783 and the 95% interval of 0.945."
- **This.** "In the 2025 backtest the dark band held on 78 of 100 days and the light band on 95 of 100."

> **Three rules.** Counts, not probabilities. What was measured, not what was intended. The period it was measured on.

#### [day4#the-forecast-note] A forecast note has six parts, and the planner reads the first three

*Lab 4.4 · step 3*

- **1 · Decision.** What is forecast, for which days, for which decision.
- **2 · Forecast.** The range of the median, and the quantile the reader acts on.
- **3 · Uncertainty.** What the bands mean and how often they held.

- **4 · Evidence.** The backtest score against the floor, in one sentence.
- **5 · Limits.** When not to use it, and when to ask for a rerun.
- **6 · Source.** Data up to which day, the method in one line, who signs.

#### [day4#when-not-to-use-it] Say when the forecast should not be used

*The part most notes leave out*

- **Outside its scope:** not for one feeder, not for hours within the day, not beyond 28 days.
- **When its assumptions break:** a declared heat wave or dust storm; a holiday date that moves.
- **What it never forecasts:** outages.

- **Why it protects the reader.** Shahad Al-Tamimi asked for the peak day three years ahead. A 28-day forecast stretched to answer her would carry these bands into a question they were never measured on.

> **Each limit names a situation and an action:** do not use, or ask for a rerun.

#### [day4#capstone-brief] Your capstone: a dataset, a planner's request, three families, one note

*Lab 4.4 · step 5 · the capstone starts here*

| Supplied dataset | What it holds |
| --- | --- |
| `ev_hubs.csv` | 36 charging hubs, daily energy, 226 to 915 days each |
| `cooling_plants.csv` | 8 district-cooling plants, daily, five years |
| A public dataset | allowed when its terms permit, and you state them |

- A floor and three method families in one backtest.
- A README with point and probabilistic accuracy.
- A two-page planner's note with fan charts.
- A public repository that runs from a clean copy.

> **Defence.** Eight minutes, then four of questions: the assessor moves the origin or the horizon and asks what changes.

#### [day4#check-forecast-note] Which sentence belongs in the note?

*Checkpoint*

**Question:** A forecast's 80% band covered 71% of days in last year's backtest. The planner buys energy at the upper edge of that band. Which sentence should the note carry?

- A. "Demand will be inside the band with a probability of 80%."
- B. "Last year the band held on 71 days in 100; it is narrower than its name."
- C. "The interval is approximately calibrated."
- D. "The band shows the uncertainty of the forecast."

#### [day4#lab-4-4] Fan chart and planner's note

Lab 4.4 · 25 min

Draw the fan chart of the next 28 days for Al-Rimal, write the planner's note with its coverage statement and limits, and choose the capstone dataset.

[D4_4_draw_fan_chart.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D4_4_draw_fan_chart.ipynb)

- [ ] Draw the fan chart for the next 28 days (7 min)
- [ ] State the backtest coverage in plain words (4 min)
- [ ] Write the planner's note (8 min)
- [ ] Check the note against its six required parts (3 min)
- [ ] Choose the capstone dataset and three method families (3 min)
- [ ] ✱ Stretch: Write the note in Arabic with the same numbers

#### [day4#wrap-up] Three things to keep

*Day 4 · wrap-up*

- **Review before you report.** A score of 0.24 became 0.68 after three fixes. Features, regressors and choices must all be known at the origin.
- **Borrow when history is short.** On eight weeks of history the methods that learn from other series came first: 0.75 against 0.95 from the feeder alone.
- **Write for the reader.** A fan chart, coverage in counts, and the situations in which the forecast should not be used.

#### [day4#next] The capstone: your analysis, defended to a planner

*Next*

Build the backtest first and the models second. Bring a public repository that runs from a clean copy, a README with point and probabilistic accuracy against the floor, and a two-page note with fan charts. Eight minutes to present, four minutes of questions.
