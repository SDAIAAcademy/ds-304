---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.2.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/slides/day3.html"
---

# Forecasting and Time Series

## Day 3: Many series and pretrained models

Four hundred feeders: forecasts that add up, one model for all series, and a pretrained model as a baseline

#### [day3#title] Many series and pretrained models

*SDAIA Academy · Forecasting and Time Series*

Day 3: Four hundred feeders: forecasts that add up, one model for all series, and a pretrained model as a baseline

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-304

#### [day3#agenda] Many series and pretrained models

*Today*

- **Session 1 · Many series and hierarchies**: Forecasts made level by level do not add up; reconciliation makes them coherent and often better
- **Session 2 · Global boosted forecasters**: One boosted model trained across all feeders learns shared patterns from features known at the origin
- **Session 3 · Pretrained zero-shot forecasters**: A pretrained forecaster needs no training on your data: measure it like any other baseline, and read its licence
- **Session 4 · Choosing among families**: Accuracy is one column: choose a method family on accuracy, cost, licence and upkeep together

#### [day3#review-day-2] Which regressor may use its future values?

*Review · Day 2*

**Question:** A regression forecasts daily demand 28 days ahead. Its backtest scores 0.55 with four regressors. Which one makes that score dishonest?

- A. A column that is 1 on the days of Eid al-Fitr
- B. The day of the week of the target day
- C. The temperature measured on the target day
- D. Mean demand of the 28 days before the origin

### Session 1: Many series and hierarchies

#### [day3#s1] Many series and hierarchies

*Session 1*

- Forecast every series of a two-level hierarchy
- Reconcile bottom-up, top-down and by least squares
- Compare errors at the feeder and the regional level

#### [day3#from-one-to-many] Four hundred feeders, and every one of them adds up to a region

*The case grows*

- **400** feeders in 12 regions, daily energy since 2017
- **388** established: three years of history or more
- **12** new: commissioned in September 2025. They wait for Day 4

> **The frame for today.** Eight weeks held out: two folds of 28 days, with origins on 5 November and 3 December 2025. Every method is scored on the same feeders and the same days.

#### [day3#two-readers-two-levels] Purchasing reads the region; operations read the feeders

*Why two levels*

- **Omar Al-Khuzaim · energy purchasing.** Nominates energy by region. He needs the regional total and its upper quantile.
- **Hamad Al-Dawsari · network operations.** Plans maintenance and switching by feeder. He needs each feeder's next four weeks.

> **The problem.** If the feeder forecasts of a region do not add up to the regional forecast, two departments plan on two different futures.

#### [day3#base-forecasts-disagree] Forecast each level with its best model, and the two disagree

*Lab 3.1 · steps 1 and 2 · Al-Dana, 37 feeders*

```python
# file: Lab 3.1 · step 1
# 37 feeders: ETS with a weekly season, each
feeders_fc = U.sf_many(
    AutoETS(season_length=7, model="ANA"),
    W.loc[:origin].iloc[-1096:])
# the region: the regression of Lab 2.3
region_fc = U.dynreg(rid, region_train, 28, origin)
```

- **3.0%** mean gap between the regional forecast and the sum of feeders, first fold
- **2.4%** second fold

#### [day3#bottom-up-top-down] Two simple ways to make them add up

*Reconciliation*

- **Bottom-up.** Keep the feeder forecasts. The region is their sum. **Throws away** the regional model, with its temperature and calendar.
- **Top-down.** Keep the regional forecast. Split it by each feeder's share of the feeder forecasts, day by day. **Throws away** the level of every feeder model, and keeps its share.

```python
# file: Lab 3.1 · step 3
share = feeders_fc.div(feeders_fc.sum(axis=1), axis=0)
top_down = share.mul(region_fc, axis=0)
```

#### [day3#least-squares-reconciliation] Least squares: move every forecast as little as possible

*Use both levels*

- Find the coherent set of forecasts closest to the base forecasts, in squared distance.
- For one region with n feeders the answer is closed: every feeder takes an equal part of the gap.
- With 37 feeders the total lands almost on the regional forecast.

```python
# file: Lab 3.1 · step 4
n = feeders_fc.shape[1]
gap = region_fc - feeders_fc.sum(axis=1)
least_sq = feeders_fc.add(gap / (n + 1), axis=0)
```

> **The equal share is the weakness.** A feeder of 30 MWh and one of 175 MWh receive the same correction in MWh.

#### [day3#errors-at-both-levels] Reconciling made the feeder forecasts better, not only coherent

*Lab 3.1 · step 5 · Al-Dana, two folds*

| Forecasts | Scaled error, feeders | Scaled error, region | Coherent |
| --- | --- | --- | --- |
| Base: each level by its own model | 0.722 | 0.463 | no |
| Bottom-up | 0.722 | 0.634 | yes |
| Top-down | 0.650 | 0.463 | yes |
| Least squares | 0.651 | 0.465 | yes |

> **Why.** The regional model knows the temperature and the calendar. Reconciliation passes that knowledge down to feeder models that have neither.

#### [day3#reconciliation-twelve-regions] Top-down helps where the regional model is the better one

*Lab 3.1 · stretch · all 12 regions*

| Region | Feeders, bottom-up | Feeders, top-down |
| --- | --- | --- |
| Al-Rimal | 1.216 | 0.692 |
| Al-Badia | 1.319 | 0.592 |
| Al-Dana | 0.722 | 0.650 |
| Al-Wadi | 0.744 | 0.879 |
| Al-Marjan | 0.654 | 0.881 |

**8 of 12** regions where top-down lowers the feeder error

> **In the other four** the regional model lost to the sum of its feeders, and top-down spread its error over every feeder.

#### [day3#check-reconciliation] Which reconciliation, and on what evidence?

*Checkpoint*

**Question:** In one region the regional model scores 0.89 on the held-out weeks and the sum of the feeder forecasts scores 0.58 against the regional total. Purchasing and operations need one coherent set. Which is the defensible choice?

- A. Bottom-up: here the feeder models carry the better information
- B. Top-down, because it won in Al-Dana
- C. Keep both base forecasts and let each department use its own
- D. Least squares, because it uses both levels

#### [day3#lab-3-1] Feeders that add up to their region

Lab 3.1 · 25 min

Forecast every feeder of one region and the region itself, show that the forecasts disagree, reconcile them three ways and compare errors at both levels.

[D3_1_fit_per_feeder.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D3_1_fit_per_feeder.ipynb)

- [ ] Forecast every feeder of one region, and the region (6 min)
- [ ] Show that the forecasts do not add up (3 min)
- [ ] Reconcile bottom-up and top-down (6 min)
- [ ] Reconcile by least squares (6 min)
- [ ] Compare errors at both levels (4 min)
- [ ] ✱ Stretch: Repeat for all 12 regions

### Session 2: Global boosted forecasters

#### [day3#s2] Global boosted forecasters

*Session 2*

- Build lag and calendar features that use nothing after the origin
- Train one boosted model on scaled targets across series
- Backtest it with point and quantile scores on held-out weeks

#### [day3#local-and-global-models] A local model learns one series; a global model learns from all of them

*One model for many series*

- **Local: one model per feeder.** 388 fits. Each knows three years of one feeder. It cannot learn what Eid does from the feeder next door.
- **Global: one model, all feeders.** One fit on every feeder's rows. An Eid seen on 388 feeders over eight years is thousands of examples.

> **The price.** A global model needs a table: one row per forecast to make, with columns that describe it. Building that table is where the work and the risk are.

#### [day3#one-row-per-forecast] One row per feeder, origin and horizon day

*Lab 3.2 · step 1*

| Column | What it holds | Known at the origin because |
| --- | --- | --- |
| `last7`, `dow_profile` | the feeder's last week and weekday shape | the last 28 days up to the origin |
| `year_ago` | the same weekday 52 weeks before the target | 364 days back is before any origin |
| `dow`, `eid`, `ramadan`, `national` | the calendar of the target day | a function of the date |
| `cdd_change` | cooling degrees expected, less the recent mean | the temperature rule of Lab 2.3 |
| `h`, `kind`, `region` | days ahead; what the feeder is | fixed facts |

> **315,056 rows** for 388 feeders and 29 origins, built by `U.make_frame` in half a minute.

#### [day3#scale-the-target] Divide each feeder by its own recent level, and the series look alike

*Why the target is scaled*

Figure: Three feeders of Al-Dana, MWh a day: means of 33, 78 and 175.

Figure: The same three, each divided by its mean.

#### [day3#split-by-target-date] Split by the date of the target, not by the origin

*Lab 3.2 · step 1*

```python
# file: Lab 3.2 · step 1
# leaks: a row with an earlier origin can
# have a target day inside the test weeks
frame[frame.origin < first_test_origin]

# honest: the target is known by then
frame[frame.date <= first_test_origin]
```

**5,432** rows whose origin is before the first test origin and whose target day is inside the held-out weeks

> **Every one of them** is a label from the test period, handed to the model as training data.

#### [day3#one-boosted-model] One model, 300 trees, four seconds

*Lab 3.2 · steps 2 and 3*

```python
# file: Lab 3.2 · step 2
params = {**U.LGB, "objective": "l1"}                 # absolute error: the measure we score
model = lgb.train(params, lgb.Dataset(rows[U.FEATURES], rows.y), num_boost_round=300)
forecast_mwh = model.predict(test[U.FEATURES]) * test.level
```

- **0.753** the floor on 388 feeders: 0.878 and 0.628 by fold
- **0.622** one boosted model: 0.730 and 0.515
- **0.17** skill against the floor

#### [day3#boosted-quantiles] A quantile is one more model, with the pinball loss as its objective

*Lab 3.2 · step 4*

```python
# file: Lab 3.2 · step 4
for tau in (0.1, 0.5, 0.9):
    q[tau] = train_model(rows,
        objective="quantile", alpha=tau)
```

| Measure, 388 feeders | Value |
| --- | --- |
| Pinball loss, scaled | 0.196 |
| 80% interval covers | 72.1% |
| Quantiles out of order | 0.00% |

> **72 is not 80.** The interval is too narrow, like every model's own interval on Day 2. The repair is the same: rebuild it from backtest errors.

#### [day3#what-the-model-uses] Half of the model's gain comes from one feature

*Lab 3.2 · step 5*

| Item | Value |
| --- | --- |
| cdd_change | 50% |
| year_ago | 12% |
| dow_profile | 11% |
| kind | 7% |
| eight others | 20% |

> **Read it as a review.** The model leans on the expected change in cooling demand. If that column leaked, half the model would be built on it.

#### [day3#check-global-split] What is wrong with this training table?

*Checkpoint*

**Question:** A global model forecasts 28 days ahead from a table with one row per series, origin and horizon day. The analyst trains on all rows whose origin is before 1 June and tests at the origin of 1 June. The score is far better than any other's. What should be checked first?

- A. Whether the training table has enough rows per series
- B. Whether the number of trees was tuned
- C. Whether the target was scaled per series
- D. Whether training rows have target days after 1 June

#### [day3#lab-3-2] One boosted model across feeders

Lab 3.2 · 25 min

Train one LightGBM model on lag and calendar features across the feeders, backtest it on the eight held-out weeks against per-feeder floors and add quantile models.

[D3_2_backtest_one_boosted.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D3_2_backtest_one_boosted.ipynb)

- [ ] Read the frame: one row per feeder, origin and horizon day (4 min)
- [ ] Train one boosted model on scaled targets (6 min)
- [ ] Backtest on eight held-out weeks against the floor (6 min)
- [ ] Train quantile models and measure coverage (6 min)
- [ ] Read the feature importances (3 min)
- [ ] ✱ Stretch: Train without scaling the target and compare

### Session 3: Pretrained zero-shot forecasters

#### [day3#s3] Pretrained zero-shot forecasters

*Session 3*

- Run a pretrained forecaster zero-shot on held-out weeks
- Record accuracy, quantile scores and run time
- Read a model card for licence and permitted use

#### [day3#what-pretraining-buys] A forecaster trained on other people's series, used on yours as it is

*A third family*

1. **Pretraining** a large collection of real and synthetic series, done once by the publisher
2. **Context** you pass the recent history of one series
3. **Forecast** quantiles for the next days. No training on your data

> **Zero-shot** means the model has never seen this series or any series like it from you. What it knows of your data is the context you pass in.

#### [day3#what-the-model-sees] The model in the lab sees one thing: the series' own past

*Read the interface before the scores*

- **It can use.** Level, trend and repeating patterns inside the context: the week, and the year when the context is long enough to hold one.
- **It cannot know.** That Eid falls in the horizon. That a heat wave is expected. What kind of feeder this is. No calendar and no weather are passed in.

> **Predict before you run.** Where should a model like this do well against the floor, and where should it lose?

#### [day3#run-zero-shot] Three lines, no training, and a clock

*Lab 3.3 · steps 1 and 2*

```python
# file: Lab 3.3 · step 2
contexts = [context_of(W, f, origin, 512) for f in W.columns]   # 388 arrays of history
q, info = U.pretrained(contexts)        # shape: series x quantiles (0.1, 0.5, 0.9) x 28 days
print(info["model"], info["seconds"])
```

- **In Colab.** The setup cell installs the package and the weights (9 million parameters) load into the Colab runtime. Runs on its processor.
- **Anywhere else.** A stand-in answers and says so in capital letters. Its numbers are not a pretrained model's result.

#### [day3#score-it-like-any-method] Score it like any other method: the same feeders, days and columns

*Lab 3.3 · step 3*

| Method, 388 feeders, two folds | Scaled error | Pinball | 80% covers | Seconds |
| --- | --- | --- | --- | --- |
| Floor: same weekday last year | 0.753 | none | none | 0 |
| One boosted model | 0.622 | 0.196 | 72.1% | 49 |
| Pretrained, zero-shot | your run | your run | your run | your run |

> **Fill the last row from your own Colab run.** Then compare the two folds: which one did the model find harder, and what did it not know?

#### [day3#context-length] The context is the model's only knowledge of your series

*Lab 3.3 · stretch*

- **512 days.** Holds a full year. The model can see that demand fell at this time last year.
- **56 days.** Holds eight weeks. The model sees a weekly pattern and a short trend, and no year.

> **Measure the difference.** It tells you what the model would do for a feeder that is eight weeks old. That feeder exists: Lab 4.2.

#### [day3#read-the-model-card] The licence is on the model card, and it is not the same for every family

*Lab 3.3 · step 4 · model cards read on 15 November 2026*

| Family | Model | Parameters | Licence | Use by a company |
| --- | --- | --- | --- | --- |
| Chronos-Bolt | amazon/chronos-bolt-tiny | 8.7 M | Apache-2.0 | yes |
| Chronos-2 | amazon/chronos-2 | 119.5 M | Apache-2.0 | yes |
| TimesFM | google/timesfm-2.5-200m-pytorch | 231.3 M | Apache-2.0 | yes |
| Moirai | Salesforce/moirai-2.0-R-small | 11.4 M | CC BY-NC 4.0 | no: non-commercial |

> **A licence is a fact with a date.** Read the card again on the day you decide, and record the date beside the licence.

#### [day3#check-pretrained-baseline] What does this result allow you to say?

*Checkpoint*

**Question:** A pretrained forecaster, zero-shot, has a skill of 0.10 on a fold of ordinary weeks and of -0.30 on a fold with a moving holiday. It was given 512 days of history and nothing else. Which conclusion is supported?

- A. Pretrained forecasters do not work on energy demand
- B. It lacks the calendar: a baseline for ordinary weeks, not for holidays
- C. Its skill is -0.10 on average, so it is slightly worse than the floor
- D. It needs a longer context to learn the holiday

#### [day3#lab-3-3] A pretrained forecaster, zero-shot

Lab 3.3 · 25 min

Run a small pretrained forecaster zero-shot on the feeders, time it, score it on the eight held-out weeks, and record the licence of four models from three families, from their model cards.

[D3_3_run_pretrained_forecaster.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D3_3_run_pretrained_forecaster.ipynb)

- [ ] Load the forecaster and forecast one feeder (5 min)
- [ ] Forecast the feeders zero-shot and time the run (6 min)
- [ ] Score point, quantile and coverage on the held-out weeks (6 min)
- [ ] Record the licence of four models from three families (5 min)
- [ ] Write the scorecard row (3 min)
- [ ] ✱ Stretch: Shorten the context to eight weeks and score again

### Session 4: Choosing among families

#### [day3#s4] Choosing among families

*Session 4*

- Compute skill against the floor for every method
- Weigh accuracy against run time, upkeep and licence
- Defend the simplest method within noise of the best
- Compare how forecast error grows under different retraining schedules and choose one

#### [day3#accuracy-is-one-column] Accuracy is one column of the decision

*Choosing among families*

- **Accuracy.** Skill against the floor, pinball loss, coverage. On the same series and days.
- **Run time.** Seconds per 1,000 forecasts. Every week, for 400 feeders.
- **Upkeep.** What must be kept working: fits, a feature pipeline, a pinned model.
- **Licence.** Of the software and of the weights, with the date it was read.

> **This session's question.** Which family would you run weekly for a year, and what would change your mind?

#### [day3#skill-against-the-floor] 388 fitted models can lose to a table lookup

*Lab 3.4 · steps 1 and 2 · 388 feeders, two folds*

| Family | Scaled error | By fold | Skill |
| --- | --- | --- | --- |
| Floor: same weekday last year | 0.753 | 0.878, 0.628 | 0 |
| ETS, weekly season, per feeder | 0.898 | 1.071, 0.725 | -0.19 |
| Decompose then ETS, per feeder | 0.726 | 0.840, 0.612 | 0.04 |
| One boosted model, all feeders | 0.622 | 0.730, 0.515 | 0.17 |
| Pretrained, zero-shot | your run | your run | your run |

> **Read the folds.** Every method moves by 0.2 or more from one fold to the next. A skill of 0.04 is inside that spread; 0.17, ahead in both folds, is not.

#### [day3#run-time-upkeep-licence] The same five rows, by what they cost to run and to keep

*Lab 3.4 · step 3*

| Family | Seconds per 1,000 forecasts | What must be kept working | Licence |
| --- | --- | --- | --- |
| Floor | 0 | nothing to fit; one year of history | none needed |
| ETS, weekly, per feeder | 181 | a fit per feeder at every run | Apache-2.0 |
| Decompose then ETS | 1,884 | a slow fit per feeder; three years of history | Apache-2.0 |
| One boosted model | 63 | a feature pipeline and one training job | MIT |
| Pretrained, zero-shot | your run | a pinned model version and its runtime | Apache-2.0 |

> **One machine's seconds.** Read the ratios: the per-feeder decomposition costs 30 times the one boosted model, for a skill of 0.04 against 0.17.

#### [day3#two-planners-two-weightings] Two readers weigh the same scorecard differently

*Lab 3.4 · step 4*

| Criterion, each scored 0 to 1 | Purchasing · Omar | Operations · Hamad |
| --- | --- | --- |
| Skill against the floor | 0.20 | 0.25 |
| Pinball loss | 0.40 | 0.15 |
| Distance of coverage from 80% | 0.20 | 0.10 |
| Run time, on a logarithm | 0.10 | 0.20 |
| Upkeep | 0.10 | 0.30 |

> **Where the weights come from.** Omar buys on a quantile, so a method without quantiles is of little use to him. Hamad has 400 feeders and a small team.

#### [day3#the-ranking-flips] The two weightings do not agree on the first place

*Lab 3.4 · step 4 · four measured rows*

| Family | Purchasing | Operations |
| --- | --- | --- |
| Floor: same weekday last year | 0.31 | **0.63** |
| ETS, weekly season, per feeder | 0.07 | 0.20 |
| Decompose then ETS, per feeder | 0.58 | 0.35 |
| One boosted model, all feeders | **0.83** | 0.57 |

> **Stretch.** Lower the weight of upkeep for operations from 0.30 to 0.25 and the boosted model takes first place. A choice that turns on 0.05 of a weight is a choice to discuss with Hamad, not to compute.

#### [day3#write-the-choice] A scorecard does not decide: you do, and you write down why

*Lab 3.4 · step 5*

- **The decision-log entry.**
  - **Chosen:** a row of the scorecard.
  - **Why:** the criteria and the numbers, and for whom.
  - **Simplest within noise:** is a simpler method inside the fold-to-fold spread of the winner?
  - **Would change it:** one result or one change of terms.
- **One defensible answer.** The desk runs one method for both readers: the boosted model, ahead of the floor in both folds by more than a tenth, with the floor kept as the fallback. If operations were the only reader, the floor would stay.

#### [day3#model-age] A model ages from the day it is trained: measure what each week of age costs

*Running it for a year*

1. **Replay a year** 48 weekly origins of 2025; a model trained at every one
2. **Apply a schedule** each origin is forecast by the newest model the schedule allows
3. **Pair by origin** its error less that of a model trained the same week

> **Why pair.** Old models forecast late in the year, new ones early. Comparing on the same origins removes the season from the gap.

#### [day3#retraining-schedules] Retraining every week bought nothing this model needed

*Four schedules · 388 feeders · 48 weekly origins of 2025*

| Schedule | Trainings a year | Scaled error | Worst weekly gap to a fresh model |
| --- | --- | --- | --- |
| Every week | 48 | 0.801 | 0 |
| Every 4 weeks | 12 | 0.801 | 0.065 |
| Every 13 weeks | 4 | 0.799 | 0.037 |
| Once, in January | 1 | 0.797 | 0.068 |

> **Read it against the noise.** Across origins the weekly model's error has a standard deviation of 0.20; the four schedules' errors differ by at most 0.004.

#### [day3#when-models-go-stale] How fast a model ages depends on what it reads at the origin

*Same models, two ways to rescale · gap to a model trained that week*

| Age of the model | Level read at the origin | Level kept from training |
| --- | --- | --- |
| 1 to 3 weeks | -0.001 | 0.175 |
| 4 to 12 weeks | -0.005 | 1.235 |
| 13 to 25 weeks | -0.007 | 2.724 |
| 26 weeks or more | -0.013 | 1.957 |

> **The rule.** Measure the gap by age before choosing a schedule, then keep watching it: retrain early when the live error leaves the band of the backtest.

#### [day3#check-scorecard-choice] Which choice can be defended?

*Checkpoint*

**Question:** Skill against the floor on 200 series: X 0.21, Y 0.20, Z 0.05. X is a pretrained model under a non-commercial licence. Y is a boosted model under a permissive licence. Z is the fastest. The forecasts will run a utility's weekly purchasing. Which family?

- A. X: it has the highest skill
- B. Z: it is the fastest of the three
- C. Y: within 0.01 of the best, and its licence permits the use
- D. X now, and ask about the licence after the pilot

#### [day3#lab-3-4] The scorecard

Lab 3.4 · 25 min

Fill a scorecard of accuracy, cost and licence for five method families from the measured results, rank them under two planners' weightings and write the choice.

[D3_4_fill_scorecard_accuracy.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D3_4_fill_scorecard_accuracy.ipynb)

- [ ] Load the measured results of five methods (3 min)
- [ ] Compute skill against the floor (5 min)
- [ ] Add run time, upkeep and licence (5 min)
- [ ] Rank under two planners' weightings (7 min)
- [ ] Write the choice and what would change it (5 min)
- [ ] ✱ Stretch: Change one weight until the ranking flips, and say what that weight would mean

#### [day3#wrap-up] Three things to keep

*Day 3 · wrap-up*

- **Coherent, then better.** Reconciling passed the regional model's knowledge to the feeders: 0.72 to 0.65 in Al-Dana. It helped in 8 regions of 12, where the trusted level was the better one.
- **One model, many series.** One boosted model on features known at the origin: skill 0.17 on 388 feeders. Split by the date of the target.
- **A baseline and a gate.** A pretrained forecaster is one more row, scored like the others. Its licence is read before its score.

#### [day3#next] Day 4: Review and communicate

*Next*

A backtest with a score that is too good and three faults to find. Twelve feeders with eight weeks of history. Whether an average of methods beats the best one, chosen honestly. And the fan chart and note that a planner reads. The capstone dataset is chosen in the last lab.
