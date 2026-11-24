---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.2.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/"
---

# Forecasting and Time Series

Forecast daily demand with a backtest behind every number, and hand the planner a fan chart they can act on

Participants forecast with the discipline of a backtest. On synthetic daily electricity demand from Shams Power, a fictional distributor with 12 regions and 400 feeders, they decompose series, set a seasonal-naive benchmark floor, fit exponential smoothing and ARIMA-family models with weather and Hijri holidays, train one boosted model across many series, reconcile feeder and regional forecasts and run pretrained zero-shot forecasters as baselines. Every method is scored by rolling-origin cross-validation with point and probabilistic measures and reaches the planner as a fan chart.

- **Audience:** Specialists. Analysts and data scientists who have completed the level 200 set or can show equivalent work: an analysis or model they built, validated and can explain.
- **Prerequisites:** DS-201 Statistical Inference by Simulation; DS-204 Machine Learning for Tabular Data: Baselines and Validation; Resamples and reshapes time-indexed data in a dataframe
- **Case study:** Shams Power: Shams Power is a fictional electricity distributor with 12 regions and 400 feeders. Its forecast desk, Marqab, owes the load planners a 28-day daily forecast per region for energy purchasing, with honest intervals. Participants work on nine years of synthetic daily demand with temperature and a Hijri and Gregorian calendar: Eid dips that move through the year, a level shift, an outage, twelve new feeders and a notebook with three seeded leaks. All data is synthetic and generated with a fixed seed.

## Outcomes

- LO1 (Applied): Apply decomposition and a seasonal-naive benchmark floor scored by rolling-origin cross-validation, given regional daily demand with weekly and annual seasonality
- LO2 (Applied): Implement exponential smoothing, ARIMA-family and dynamic regression models with temperature and moving-holiday terms, given the regional series and a Hijri and Gregorian calendar
- LO3 (Applied): Measure point and quantile accuracy of a global boosted model and a pretrained zero-shot forecaster against statistical models, given 400 feeder series with eight weeks held out
- LO4 (Performance): Review a forecast evaluation for leakage across time and selection on the test window, and correct it, given a notebook with three seeded faults
- LO5 (Performance): Build a forecasting analysis for new series and defend method, horizon and uncertainty to a planner, given a supplied dataset and a planner's request

## Day 1: Look, frame and set the floor

From a planner's request to a series you have looked at and a floor every later method must beat

#### [day1#title] Look, frame and set the floor

*SDAIA Academy · Forecasting and Time Series*

Day 1: From a planner's request to a series you have looked at and a floor every later method must beat

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-304

#### [day1#four-days-one-backtest] Four days, one backtest that every method must pass, and a capstone

*The course at a glance*

- **Day 1 · Look and set the floor.** Frame the request, look at the series, decompose it, backtest the naive forecasts.
- **Day 2 · Statistical forecasters.** Exponential smoothing, ARIMA, weather and Hijri regressors, intervals that hold.
- **Day 3 · Many series.** Hierarchies, one boosted model for 400 feeders, a pretrained forecaster, a scorecard.
- **Day 4 · Review and communicate.** Leaks across time, short series, combinations, the fan chart and the planner's note.

> **Day 5 · Capstone.** Three method families against the floor on a new dataset, in a public repository, with a two-page planner's note, defended live for eight minutes.

#### [day1#backtest-before-belief] No forecast is believed before its backtest

*How this course works*

- **A floor first.** The best naive forecast is the price of entry. A method that does not beat it is not worth its upkeep.
- **One instrument.** Every method is scored by the same rolling origins, the same horizon and the same scaled error.
- **Uncertainty is part of the forecast.** A number without an interval, or an interval nobody checked, is not delivered.

> **AI policy.** AI off for the entry and exit checks and for writing the first backtest (Lab 1.4). In the other labs an assistant may explain output, not write the backtest. In the capstone anything goes, defended live.

#### [day1#shams-power] Shams Power and its forecast desk, Marqab

*One case study all week*

A fictional electricity distributor: 12 regions, 400 feeders, nine years of daily demand. You are the forecast desk, **Marqab** (مرقاب).

- **Regions.** Daily demand in MWh, the day's temperature and its normal, 2017 to 2025. A region is the exact sum of its feeders.
- **Feeders.** 400 series, each residential, commercial or industrial. Twelve were commissioned in September 2025.
- **Calendar.** Gregorian and Hijri: the weekend, Ramadan, both Eids, National Day and Founding Day, to March 2026.

#### [day1#setup-and-reference-state] Open the notebooks in Colab, or run them locally with uv

*Lab environment*

1. **Colab:** open a notebook from the course page. Its first cell clones the repository and installs one package.
2. **Locally:** the commands on the right. Run `00_setup_check.ipynb` once.
3. Each lab starts from a **supplied reference state**: data files and the results earlier labs would have produced. No lab depends on your earlier run.

```bash
# file: terminal (local option)
git clone https://github.com/SDAIAAcademy/ds-304
cd forecasting/labs
uv sync
uv run jupyter lab
```

#### [day1#agenda] Look, frame and set the floor

*Today*

- **Session 1 · What is forecast, and why**: A forecast serves a decision: fix the horizon, the level and the cost of each kind of error before any model
- **Session 2 · Seeing structure**: Look before you model: weekly, annual and Hijri patterns show at different time scales
- **Session 3 · Decomposition**: Decomposition splits a series into trend, seasons and remainder, and gives each a strength you can compare
- **Session 4 · The benchmark floor**: No method counts until it beats the best naive forecast in a rolling-origin backtest

### Session 1: What is forecast, and why

#### [day1#s1] What is forecast, and why

*Session 1*

- Separate a forecast from a plan and a target
- Fix horizon, granularity and level from the decision
- Turn the cost of over- and under-forecasting into a target quantile

#### [day1#forecast-plan-target] A forecast, a plan and a target are three different numbers

*Before any model*

- **Forecast.** What you expect to happen, with its uncertainty. It can be wrong; it must not be biased on purpose.
- **Plan.** What you decide to do about it: how much energy to buy, which day to maintain a feeder.
- **Target.** What someone wants to happen. A target that leaks into a forecast makes both useless.

> **What goes wrong.** "Give me one number" hides which of the three is asked for. Ask what will be done with it.

#### [day1#decision-behind-request] "What will demand be?" means something different to each planner

*Three requests at Shams Power*

| Who asks | The decision | If the forecast is too low | If it is too high |
| --- | --- | --- | --- |
| Omar, energy purchasing | Energy to nominate for each of the next 28 days | Buys the gap at about three times the price | Resells the surplus at about half price |
| Hamad, network operations | The day next week a feeder can be taken out | Neighbouring feeders overload | A safe day is refused |
| Shahad, load planning | Transformer capacity over three years | The peak day exceeds capacity | Capital is spent early |

> **A forecast is worth what the decision it changes is worth.** Write the decision down before the first plot.

#### [day1#horizon-granularity-level] The decision fixes horizon, time step, level and what is forecast

*The frame*

| Request | Horizon | Time step | Level | What is forecast |
| --- | --- | --- | --- | --- |
| Purchasing | 28 days | day | region | the day's energy |
| Maintenance | 7 days | day | one feeder | how high it could go |
| Capacity | 3 years | year | region | the highest day of each summer |

> **This course.** The highlighted row: 28 days ahead, daily, by region.

> **What goes wrong.** The capacity request is not a 28-day model run 39 times. An annual peak is another target.

#### [day1#cost-to-quantile] Unequal costs move the answer away from the middle

*Which number to hand over*

```python
# file: Lab 1.1 · step 3
def target_quantile(cost_under, cost_over):
    return cost_under / (cost_under + cost_over)

target_quantile(2.0, 0.5)   # Omar: 0.8
```

A missing MWh costs Omar two contract prices more; a surplus MWh loses half a price.

- **80th** percentile for purchasing: demand stays under it four days in five
- **90th** percentile for the maintenance request

#### [day1#data-delay-steps] A 28-day forecast from two-day-old data is a 30-step forecast

*What the desk knows, and when*

| Days ahead | Mean change in demand |
| --- | --- |
| 1 | 4.3% |
| 2 | 6.2% |
| 7 | 8.0% |
| 28 | 15.0% |
| 30 | 16.5% |

- Regional meter data reaches the desk **two days late**.
- The origin is the last day the model has seen, not the day the forecast is sent.
- A delay costs most at short horizons: tomorrow becomes step 3, and the change after 2 steps is already 6.2%, against 4.3% after 1.

#### [day1#feeder-level-personal-data] The finer the level, the closer the series is to a household

*Responsible use*

- A regional total describes a city. A meter describes when a family is home.
- A feeder that serves a handful of customers sits between the two.
- Before a fine-grained series leaves the desk, ask the data protection officer whether it can be linked to a person, and so falls under the Personal Data Protection Law.

- **In this course.** Every series is synthetic and aggregated: regions and feeders, never meters. The capstone uses supplied data, or public data under its stated open-data terms.

#### [day1#check-frame-first] What do you send back first?

*Checkpoint*

**Question:** A planner writes: "Send me next month's demand for Al-Sahil by Thursday." Being short costs her four times what being long costs. What does the desk reply with first?

- A. The most accurate model's mean forecast for each day
- B. One number for the month, the average of the last three years
- C. The frame: daily or monthly, which decision, and the 80th percentile
- D. The figure in the annual budget, so the two stay consistent

#### [day1#lab-1-1] Frame three requests

Lab 1.1 · 25 min

Turn three requests from Shams Power planners into a forecast frame each: the decision, horizon, level, error measure and target quantile.

[D1_1_write_horizon_error.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D1_1_write_horizon_error.ipynb)

- [ ] Read the three requests and name the decision behind each (5 min)
- [ ] Set horizon, granularity and level for each (6 min)
- [ ] Turn the cost of each error into a target quantile (6 min)
- [ ] Count the days the data delay takes from the horizon (4 min)
- [ ] Write one framing line per request and swap it (4 min)
- [ ] ✱ Stretch: Frame a fourth request from your own workplace

### Session 2: Seeing structure

#### [day1#s2] Seeing structure

*Session 2*

- Plot a series at three time scales and name what each shows
- Measure the weekday profile and the Eid dip
- Explain why a Hijri holiday is not a seasonal pattern

#### [day1#nine-years-trend-and-summer] Nine years show two things: growth and summer

*The longest time scale*

Figure: Al-Rimal, weekly mean of daily demand in MWh, 2017 to 2025 (synthetic).

#### [day1#summer-is-temperature] The annual shape is the temperature, not the calendar

*One year*

- **2.3×** July 2025 demand against January 2025
- **0.97** correlation of daily demand with cooling degrees above 22 °C
- **2.7%** growth per year, 2017 to 2025

> **Cooling degrees.** How far the day's temperature is above the point where cooling starts: `max(0, temp - 22)`. Below that point temperature barely matters.

#### [day1#friday-is-the-low-day] The week only shows when you zoom in, and it starts on Sunday

*Four weeks*

Figure: Al-Rimal, four weeks in autumn 2025. The dashed lines are Fridays.

- **0.94** Friday, against the mean day
- **0.98** Saturday

#### [day1#hijri-holidays-move] Eid arrives 10 to 12 days earlier every Gregorian year

*A pattern that is not a season*

| Year | First day of Eid al-Fitr |
| --- | --- |
| 2021 | 13 May |
| 2022 | 2 May |
| 2023 | 21 April |
| 2024 | 10 April |
| 2025 | 30 March |

- The Hijri year is about 11 days shorter than the Gregorian year.
- So Ramadan and both Eids drift through the seasons, a full circle in about 33 years.
- A 365-day season looks for last year's Eid on the wrong days.

#### [day1#eid-dip-measured] Every Eid al-Fitr is a dip, from 8% to 28% in Al-Rimal

*Measure it*

Figure: Al-Rimal, February to July 2025. Light: Ramadan. Dark: Eid al-Fitr and Eid al-Adha.

- **How the dip is measured.** The first day of Eid against the same weekday one week before and one week after.
- **Why both sides.** In spring demand climbs every week. A "before" alone makes the dip look smaller.

#### [day1#drop-weather-does-not-explain] Look for the drop the weather does not explain, then ask someone

*Outliers*

- In Al-Rimal a cool week can cut demand by a quarter. A drop alone proves nothing.
- Keep the days with no calendar event that were **not cooler** than their neighbours.
- One day stands out twice as far as any other.

**-30.5%** 14 March 2023, no holiday, no cold spell

> **Do not delete it yet.** Ask operations what happened that day.

#### [day1#check-wrong-eid-days] Why does last year's pattern miss Eid?

*Checkpoint*

**Question:** A model with a 365-day season forecasts Al-Rimal for spring 2026. It is far too high on a few days in March and too low on a few in early April. What is the likely cause?

- A. Eid al-Fitr moved about 11 days earlier than in 2025
- B. Spring weather is more variable than summer weather
- C. The weekend is Friday and Saturday, not Saturday and Sunday
- D. Demand grew faster in 2026 than the model assumed

#### [day1#lab-1-2] Three time scales

Lab 1.2 · 25 min

Plot Al-Rimal's demand over nine years, one year and four weeks, shade Ramadan and Eid, and measure the weekday profile and the Eid dip of every year.

[D1_2_plot_one_region.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D1_2_plot_one_region.ipynb)

- [ ] Plot nine years and name the trend and the annual shape (5 min)
- [ ] Plot one year against temperature (5 min)
- [ ] Plot four weeks and compute the weekday profile (6 min)
- [ ] Shade Ramadan and Eid and measure the Eid dip per year (6 min)
- [ ] Find the dip that is not a holiday (3 min)
- [ ] ✱ Stretch: Repeat for Al-Jabal, a cool mountain region, and compare

### Session 3: Decomposition

#### [day1#s3] Decomposition

*Session 3*

- Decompose a series with weekly and annual seasons
- Compute the strength of trend and of seasonality
- Read the remainder for holidays, outages and breaks

#### [day1#series-as-parts] Split the series into parts you can name, and one you cannot

*Decomposition*

1. **Trend** the slow level: growth, new feeders
2. **Weekly season** Friday and Saturday
3. **Annual season** the summer
4. **Remainder** everything else

> **On the log scale.** Summer swings are larger in MWh when the level is higher, so the parts multiply. Taking the logarithm turns the product into this sum.

#### [day1#stl-with-two-periods] A decomposition with two seasonal periods, robust to odd days

*The tool*

```python
# file: Lab 1.3 · step 1
from statsmodels.tsa.seasonal import MSTL

res = MSTL(np.log(y), periods=(7, 365),
           stl_kwargs={"robust": True}).fit()

res.trend
res.seasonal["seasonal_7"]
res.seasonal["seasonal_365"]
res.resid
```

- Each season is smoothed over its own cycles and may change slowly over the years.
- **Robust** fitting gives an odd day little weight, so an outage stays in the remainder and does not bend the season.
- The four parts add up to the log of demand exactly. Check it.

#### [day1#strength-of-a-component] Strength compares a component with the noise around it

*A number per part*

```python
# file: Lab 1.3 · step 2
def strength(component, remainder):
    ratio = remainder.var() / (
        component + remainder).var()
    return max(0.0, 1 - ratio)
```

| Al-Rimal, 2017 to 2025 | Strength |
| --- | --- |
| Annual season | 0.96 |
| Trend | 0.50 |
| Weekly season | 0.16 |

> **Read it as.** Near 1: the component dominates the remainder. Near 0: it is lost in it.

#### [day1#twelve-regions-differ] The regions are not alike, so one model setting will not suit all

*One table for 12 regions*

| Region, 2021 to 2025 | Trend | Annual | Weekly | What it says |
| --- | --- | --- | --- | --- |
| Al-Rimal (R01) | 0.19 | 0.96 | 0.17 | hot, residential: the summer rules |
| Al-Sahil (R02) | 0.27 | 0.85 | 0.51 | coastal, more commerce: the week counts |
| Al-Jabal (R03) | 0.40 | 0.71 | 0.34 | cool mountain: a weaker summer |
| Al-Mina (R11) | 0.89 | 0.83 | 0.56 | industrial port: a trend that is too strong |

Trend strength depends on the window: Al-Rimal's is 0.50 over nine years and 0.19 over five. Compare regions within one table.

#### [day1#what-the-remainder-holds] The remainder is the to-do list of the next model

*Read what is left*

- **-13%** mean remainder on the 54 main Eid days of nine years
- **2 of 12** largest remainders are Eid days; most are weather swings
- **-0.35** log remainder on 14 March 2023, the outage

> **What goes wrong.** Expecting holidays to top the list. In a hot region a week that is warmer or cooler than normal moves demand more than Eid does.

#### [day1#a-step-is-not-a-trend] Al-Mina grew 52% in two years: a step, not a trend

*Structural breaks*

- On 1 July 2022 six industrial feeders joined the region.
- A decomposition smooths the step into a steep trend.
- A model trained across the step reads it as fast growth, or averages two levels.

- **Options.** Train after the break. Or tell the model: a step regressor, or forecast the new feeders apart.
- **How you find out.** The yearly means, then a question to network operations.

#### [day1#check-remainder-reading] What does this remainder tell you?

*Checkpoint*

**Question:** You decompose a region's log demand with weekly and annual periods. The remainder is near zero on average, but about -0.12 on the same ten days every year: the days around the two Eids. What do you conclude?

- A. The annual season needs a longer smoothing window
- B. The fit should be made robust so these days are ignored
- C. The series should be decomposed without the logarithm
- D. The effect is real and regular, and needs a regressor of its own

#### [day1#lab-1-3] Decompose and measure strength

Lab 1.3 · 25 min

Decompose regional demand into trend, weekly season, annual season and remainder, compute the strength of each for the 12 regions and read what the remainder holds.

[D1_3_decompose_regional_demand.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D1_3_decompose_regional_demand.ipynb)

- [ ] Decompose log demand with weekly and annual periods (6 min)
- [ ] Compute the strength of trend and of each season (6 min)
- [ ] Rank the 12 regions by seasonal strength (5 min)
- [ ] Read the largest remainder days and name them (5 min)
- [ ] Find the region whose trend breaks (3 min)
- [ ] ✱ Stretch: Decompose one feeder and compare its strengths with its region

### Session 4: The benchmark floor

#### [day1#s4] The benchmark floor

*Session 4*

- Write naive and seasonal-naive forecasts
- Run a rolling-origin backtest over 28-day horizons
- Report scaled errors by horizon and name the floor

#### [day1#a-floor-before-any-model] Before any model, find out how good "no model" is

*The benchmark floor*

- **It costs nothing.** No fitting, no tuning, no upkeep. It cannot break in production.
- **It is often good.** On strongly seasonal series, repeating the right past period is hard to beat.
- **It sets the price.** Every later method reports its distance to the floor, not its own error alone.

> **What goes wrong.** "Our model has a 4% error" says nothing until you know that repeating last year gives 5%, or 3%.

#### [day1#naive-and-seasonal-naive] Naive repeats the last day; seasonal naive repeats the last cycle

*Three ways to repeat the past*

```python
# file: Lab 1.4 · step 2
def naive(train, h):
    return np.repeat(train.iloc[-1], h)

def seasonal_naive(train, h, m):
    last = train.to_numpy()[-m:]
    return np.resize(last, h)
```

| Method | Next Friday equals |
| --- | --- |
| Naive | the last day seen |
| Seasonal naive, m = 7 | last Friday |
| Seasonal naive, m = 364 | the Friday 52 weeks ago |

> **364, not 365.** Fifty-two whole weeks keep a Friday on a Friday.

#### [day1#rolling-origins] Move the origin through the last year and forecast 28 days each time

*The instrument*

Figure: One fold. Solid: what happened. Dashed coral: last week repeated. Teal: the same weekdays a year earlier.

- The **origin** is the last day the forecaster has seen.
- 13 origins, 28 days apart: every day of 2025 is tested exactly once.
- Each fold trains on the 1,096 days before its origin.
- Nothing after the origin may reach the forecast.

#### [day1#scaled-error] A scaled error lets a region and a feeder share one table

*The measure*

```python
# file: Lab 1.4 · step 3
def scaled_error(actual, forecast, train, m=7):
    v = train.to_numpy()
    scale = np.mean(np.abs(v[m:] - v[:-m]))
    err = np.mean(np.abs(actual - forecast))
    return err / scale
```

- The numerator: mean absolute error on the test days.
- The denominator: the error of "last week" **inside the training data**, one week ahead.
- No unit. Below 1 beats that in-sample error.
- Above 1 at 28 days is normal: the yardstick is one week ahead.

#### [day1#the-floor-table] The floor is "the same weekday last year", at 1.06

*Lab 1.4 · Al-Rimal, 13 origins, 364 test days*

| Method | Scaled error | Week 1 | Week 2 | Week 3 | Week 4 |
| --- | --- | --- | --- | --- | --- |
| Naive (last day) | 1.64 | 1.09 | 1.87 | 1.55 | 2.07 |
| Last week (m = 7) | 1.62 | 1.18 | 1.77 | 1.46 | 2.05 |
| Same weekday last year (m = 364) | 1.06 | 0.92 | 0.96 | 1.08 | 1.28 |

> **Report by horizon.** One average hides that "last week" is close in week 1 and twice as bad in week 4.

#### [day1#where-the-floor-fails] On Eid days the floor's error doubles

*Know your floor's weak days*

- **2.17** scaled error of the floor on the 11 Eid days of 2025
- **1.03** on the other 353 days

- Last year's Eid was about 11 days later: the floor puts the dip on the wrong days.
- Eleven days are 3% of the year, so the overall score barely moves.
- They are the days the purchasing lead remembers.

#### [day1#check-which-floor] Which backtest is a fair floor?

*Checkpoint*

**Question:** A colleague reports a floor for a 28-day daily forecast: seasonal naive with m = 7, tested on the 28 days after one origin in January, error 0.28 after scaling. What is the main weakness?

- A. The scale should come from the test days, not the training days
- B. One origin in a calm month says little about the other eleven
- C. A seasonal naive forecast is too simple to count as a floor
- D. The horizon should be shortened to 7 days to match m = 7

#### [day1#lab-1-4] The benchmark floor (AI-off)

Lab 1.4 · 25 min

Write the rolling origins, the naive forecasts and the scaled error yourself, without AI, and backtest three naive methods over 13 origins of 28 days to set the floor.

[D1_4_backtest_naive_seasonal.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D1_4_backtest_naive_seasonal.ipynb)

- [ ] Write the rolling origins (5 min)
- [ ] Write the naive and seasonal-naive forecasts (5 min)
- [ ] Write the scaled error (5 min)
- [ ] Backtest three naive methods over 13 origins (6 min)
- [ ] Read the error by horizon week and name the floor (4 min)
- [ ] ✱ Stretch: Score the Eid days separately from the other days

#### [day1#wrap-up] Three things to keep

*Day 1 · wrap-up*

- **The frame comes first.** The decision fixes the horizon, the level and the quantile. For Shams Power: 28 days, daily, by region, the 80th percentile.
- **Look, then decompose.** Summer is temperature, the weekend is Friday and Saturday, Eid moves. What the decomposition cannot hold sits in the remainder.
- **The floor is 1.06.** "The same weekday last year", over 13 rolling origins. It fails on Eid days. Tomorrow's methods must beat it.

#### [day1#next] Day 2: Statistical forecasters

*Next*

Exponential smoothing and ARIMA against the floor, then temperature and Hijri regressors for the Eid days, then the question of whether an 80% interval holds 80% of the time. Unfinished labs can be finished at home: none is needed tomorrow.

## Day 2: Statistical forecasters

Exponential smoothing, ARIMA and regression with weather and Hijri holidays, each scored against the floor

#### [day2#title] Statistical forecasters

*SDAIA Academy · Forecasting and Time Series*

Day 2: Exponential smoothing, ARIMA and regression with weather and Hijri holidays, each scored against the floor

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-304

#### [day2#agenda] Statistical forecasters

*Today*

- **Session 1 · Exponential smoothing**: Exponential smoothing forecasts from weighted averages that forget; it needs help to see the annual season
- **Session 2 · ARIMA-family models**: ARIMA models the correlation left after differencing; the residuals say whether it has finished
- **Session 3 · Weather and moving holidays**: Regression with ARIMA errors brings in what you know about the future: the calendar always, the weather only in part
- **Session 4 · Intervals that hold**: An interval is a promise about frequency: measure whether 80% intervals hold 80% of the time

#### [day2#review-day-1] What does the scaled error compare?

*Review · Day 1*

**Question:** A method scores 1.3 on a region in the course's backtest. The floor scores 1.06. A colleague says: "Above 1, so it is worse than a naive forecast and useless." What is the right reading?

- A. Right: any score above 1 loses to the naive forecast
- B. It loses to the floor, because 1.3 is above 1.06
- C. It is wrong by 130% of demand on an average day
- D. Nothing can be said: the measure does not work at 28 days

### Session 1: Exponential smoothing

#### [day2#s1] Exponential smoothing

*Session 1*

- Compute simple exponential smoothing by hand
- Fit ETS models with a weekly season
- Combine a decomposition with ETS and score it against the floor

#### [day2#averages-that-forget] A weighted average that forgets the past at a rate you choose

*Simple exponential smoothing*

```python
# file: Lab 2.1 · step 1
def ses_level(values, alpha):
    level = values[0]
    for x in values[1:]:
        level = alpha * x + (1 - alpha) * level
    return level      # the forecast, for every h
```

- The value k days back has the weight `alpha * (1 - alpha) ** k`.
- **alpha near 0:** a long memory, a smooth level.
- **alpha = 1:** the level is the last value. That is the naive forecast.
- The forecast is flat: the same level for every day ahead.

#### [day2#alpha-close-to-one] On daily demand the best alpha is at the top of the grid

*Lab 2.1 · step 1*

| alpha | One-step error, MWh |
| --- | --- |
| 0.1 | 251.0 |
| 0.3 | 187.5 |
| 0.5 | 170.7 |
| 0.7 | 161.4 |
| 0.9 | 157.3 |
| 1.0 | 156.3 |

- **Why.** Demand moves in weather spells that last days. Yesterday is a better guide to today than any average of the past.
- **What it means.** Smoothing has nothing to smooth here. The model is close to "tomorrow equals today".

#### [day2#error-trend-season] Level plus an optional trend plus an optional season

*The ETS family*

- **Error.** Additive or multiplicative. It decides the shape of the intervals.
- **Trend.** None, additive, or damped: a slope that fades with the horizon.
- **Season.** None, additive or multiplicative, with one period: here 7 days.

```python
# file: Lab 2.1 · step 2
from statsforecast.models import AutoETS
U.sf(AutoETS(season_length=7, model="ANA"), train, h)   # additive error, no trend, additive season
```

#### [day2#weekly-ets-loses] A fitted model with intervals loses to the floor by half

*Lab 2.1 · step 2*

- **1.59** ETS with a weekly season, 13 origins
- **1.06** the floor: the same weekday last year
- **-0.50** skill: it adds half again to the floor's error

> **Why.** It forecasts a flat level with a weekday pattern. Over four weeks the temperature moves the level, and this model has no summer.

#### [day2#decompose-then-smooth] Decompose, smooth what is left, put the seasons back

*Give it the summer*

1. **1 · Log** seasons become additions
2. **2 · Decompose** take out the weekly and annual seasons
3. **3 · ETS** forecast the rest: level and trend
4. **4 · Restore** add the seasons, take the exponential

```python
# file: Lab 2.1 · step 3
model = MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"))
np.exp(U.sf(model, np.log(train), h))
```

#### [day2#ets-against-the-floor] With the annual season, exponential smoothing beats the floor

*Lab 2.1 · Al-Rimal, 13 origins*

| Method | Scaled error | Week 1 | Week 2 | Week 3 | Week 4 | Skill |
| --- | --- | --- | --- | --- | --- | --- |
| Floor | 1.06 | 0.92 | 0.96 | 1.08 | 1.28 | 0 |
| ETS, weekly season | 1.59 | 1.02 | 1.80 | 1.48 | 2.04 | -0.50 |
| Decompose, then ETS | 0.98 | 0.97 | 0.97 | 0.85 | 1.15 | 0.07 |

> **Read the first week.** There the floor still wins, 0.92 to 0.97. The gain comes in weeks 3 and 4.

#### [day2#check-ets-season] Which change helps this model most?

*Checkpoint*

**Question:** An ETS model with a weekly season forecasts a hot region's daily demand 28 days ahead. Its fitted alpha is 0.99 and it scores 1.6 against a floor of 1.1. Which change is most likely to bring it under the floor?

- A. Set alpha to 0.3 by hand for a smoother level
- B. Add a damped trend to the weekly model
- C. Train on nine years, not three
- D. Remove the annual season by decomposition before smoothing

#### [day2#lab-2-1] Exponential smoothing against the floor

Lab 2.1 · 25 min

Write simple exponential smoothing, fit ETS with a weekly season, add the annual season through a decomposition and score all of them against the floor.

[D2_1_fit_exponential_smoothing.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D2_1_fit_exponential_smoothing.ipynb)

- [ ] Write simple exponential smoothing by hand (6 min)
- [ ] Fit ETS with a weekly season and score it (6 min)
- [ ] Add the annual season: decompose, then smooth (6 min)
- [ ] Compare with the floor by horizon week (4 min)
- [ ] Read the smoothing weights (3 min)
- [ ] ✱ Stretch: Compare a damped trend with an undamped one at 28 days

### Session 2: ARIMA-family models

#### [day2#s2] ARIMA-family models

*Session 2*

- Difference a series and read its autocorrelation
- Fit a seasonal ARIMA model
- Check residuals with a portmanteau test before trusting intervals

#### [day2#arima-three-parts] Difference until the level stops wandering, then model the memory

*ARIMA*

- **I · differencing.** Work on changes, not levels. A seasonal difference subtracts the same weekday of last week.
- **AR · autoregression.** Today depends on the last few days, each with a coefficient.
- **MA · moving average.** Today depends on the last few forecast errors.

> **Written as.** ARIMA(p, d, q)(P, D, Q) with a period m: the first bracket for day-to-day memory, the second for week-to-week.

#### [day2#autocorrelation-before-after] The autocorrelation says whether differencing has done its work

*Lab 2.2 · step 1*

| Lag in days | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Demand | 0.98 | 0.97 | 0.96 | 0.95 | 0.94 | 0.94 | 0.94 |
| After a seasonal difference | 0.83 | 0.59 | 0.40 | 0.24 | 0.08 | -0.09 | -0.21 |

> **Before.** Correlations that stay near 1: the level wanders. Nothing to model yet.

> **After.** Memory that dies out in five days, and a dip at lag 7: a short AR part and a seasonal MA term.

#### [day2#weekly-arima-loses-too] The weekly ARIMA has the same blindness as the weekly ETS

*Lab 2.2 · step 2*

```python
# file: Lab 2.2
from statsforecast.models import ARIMA

def weekly_arima():
    return ARIMA(order=(1, 0, 1),
                 seasonal_order=(0, 1, 1),
                 season_length=7)

U.sf(weekly_arima(), train, h)
```

**1.62** scaled error, against 1.59 for the weekly ETS and 1.06 for the floor

> **Two families, one miss.** Neither can see the summer through a period of 7.

#### [day2#residuals-must-be-noise] If the residuals still hold a pattern, the model has not finished

*Diagnostics*

- **Residuals** are the one-step errors inside the training data.
- A finished model leaves noise: no autocorrelation.
- The **Ljung-Box test** asks whether the first 14 autocorrelations together are larger than noise gives.
- A small p-value: a pattern is left, and the model's intervals, which assume noise, cannot be trusted.

**0.0001** p-value of the weekly ARIMA: not noise

```python
# file: Lab 2.2 · step 3
acorr_ljungbox(residuals, lags=[14],
               model_df=3)["lb_pvalue"]
```

#### [day2#annual-fourier-terms] Two pairs of annual waves give ARIMA the summer

*Lab 2.2 · step 4*

```python
# file: Lab 2.2
def fourier(dates, K=2):
    t = days_since_2017(dates)
    return DataFrame({
        f"{f}{k}": getattr(np, f)(
            2 * np.pi * k * t / 365.25)
        for k in (1, 2)
        for f in ("sin", "cos")})

U.sf(weekly_arima(), train, h,
     fourier(train.index), fourier(future))
```

- **0.83** scaled error: 22% under the floor
- **0.62** Ljung-Box p-value: the residuals pass

#### [day2#arima-record] Score and diagnostics, side by side, against the floor

*Lab 2.2 · the record*

| Model | Scaled error | Week 1 | Week 4 | Ljung-Box p |
| --- | --- | --- | --- | --- |
| Floor | 1.06 | 0.92 | 1.28 |  |
| ARIMA, weekly | 1.62 | 1.06 | 2.10 | 0.0001 |
| ARIMA, weekly + annual Fourier | 0.83 | 0.71 | 0.87 | 0.62 |

> **Automatic order search.** On two folds it scored 0.82 against 0.85 for the fixed orders, and took 53 seconds. A small gain for a large cost: know what the search buys before you run it 400 times.

#### [day2#check-residual-test] What does this diagnostic tell you?

*Checkpoint*

**Question:** A seasonal ARIMA is fitted to three years of daily demand. The Ljung-Box p-value of its residuals is 0.0003 and its backtest score is close to the floor. What do you do next?

- A. Look for the pattern left in the residuals and give the model a term for it
- B. Report the model: a p-value this small shows the fit is significant
- C. Rerun the test with more lags until it passes
- D. Drop ARIMA and move to another model family

#### [day2#lab-2-2] A seasonal ARIMA with checked residuals

Lab 2.2 · 25 min

Difference the series, fit a seasonal ARIMA model, test its residuals, add annual Fourier terms and score each step against the floor.

[D2_2_fit_seasonal_arima.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D2_2_fit_seasonal_arima.ipynb)

- [ ] Difference the series and read the autocorrelation (5 min)
- [ ] Fit a seasonal ARIMA and score it (6 min)
- [ ] Test the residuals for leftover correlation (5 min)
- [ ] Add annual Fourier terms and score again (6 min)
- [ ] Record both scores against the floor (3 min)
- [ ] ✱ Stretch: Let an automatic search choose the orders and compare time and score

### Session 3: Weather and moving holidays

#### [day2#s3] Weather and moving holidays

*Session 3*

- Build cooling-degree and Hijri holiday regressors
- Fit a regression with ARIMA errors
- Separate what is known at the origin from what is measured later

#### [day2#regression-with-arima-errors] A regression explains what it can; ARIMA models what is left

*Dynamic regression*

1. **Demand** MWh per day
2. **Coefficients × regressors** temperature, Ramadan, Eid, national days
3. **An ARIMA series** the week, the memory, the noise

> **The condition.** Every regressor must be available for the future days, at the origin. That one sentence is this session.

#### [day2#cooling-degrees-and-calendar] Five columns carry the weather and the calendar

*Lab 2.3 · step 1*

| Date | Cooling degrees | Heating degrees | Ramadan | Eid | National |
| --- | --- | --- | --- | --- | --- |
| 28 March 2025 | 2.1 | 0 | 1 | 0 | 0 |
| 29 March 2025 | 0 | 0 | 1 | 0.4 | 0 |
| 30 March 2025, first day of Eid | 0 | 0 | 0 | 1.0 | 0 |
| 31 March 2025 | 0.9 | 0 | 0 | 0.9 | 0 |
| 1 April 2025 | 1.1 | 0 | 0 | 0.7 | 0 |

Cooling degrees: `max(0, temp - 22)`. Heating degrees: `max(0, 16 - temp)`. The Eid column is a weight: 1 on the first day, lower on the eve and the days after.

#### [day2#known-at-the-origin] The calendar is known for the future. The temperature is not.

*The rule*

- **Known at the origin.** Dates, weekdays, Ramadan, both Eids, national days. Use the true future values.
- **Not known at the origin.** Any measured quantity: temperature, prices, the demand of another region. Only a forecast of it may be used.

> **In the labs.** The future temperature is the normal for the date plus the anomaly of the origin day, fading by a fifth each day. That is what a desk without a weather service can know.

#### [day2#gain-on-eid-days] The calendar moves the Eid days from 1.34 to 0.80

*Lab 2.3 · step 3 · 13 origins, 11 Eid days*

| Model | All days | Eid days | Other days |
| --- | --- | --- | --- |
| Floor | 1.06 | 2.17 | 1.03 |
| Regression, weather only | 0.90 | 1.34 | 0.89 |
| Regression, weather + calendar | 0.77 | 0.80 | 0.77 |

> **Score the days that matter apart.** The overall gain from the calendar includes Ramadan and national days; on Eid days alone the error falls by 40%.

#### [day2#the-eid-fold] Only the model that was given the calendar follows the dip

*The fold that holds Eid al-Fitr 2025*

Figure: Al-Rimal, origin 26 March 2025. Navy: demand. Grey dashes: the floor. Coral dots: weather only. Teal: weather and calendar. Shaded: Eid al-Fitr.

#### [day2#honest-against-oracle] With the temperature measured later, the score flatters by 0.22

*Lab 2.3 · step 4*

| Future temperature | Scaled error | Week 1 | Week 4 |
| --- | --- | --- | --- |
| Known at the origin | 0.77 | 0.62 | 0.81 |
| Measured later | 0.55 | 0.39 | 0.66 |

- **0.55 is not a score.** No forecaster has next week's measured temperature. Report 0.77.
- **The gap is useful.** It is what a perfect weather forecast would be worth to the desk.

#### [day2#coefficients-in-mwh] The coefficients are sentences a planner can check

*Lab 2.3 · step 5*

- **+147** MWh per cooling degree: 4.2% of a mean day
- **-599** MWh on the first day of Eid: 17% of a mean day
- **-296** MWh on a national day

> **One Eid regressor or two?** With separate regressors for the two Eids the overall score is 0.75, and the Eid days get worse: 1.00 against 0.80. Half the examples per coefficient.

#### [day2#check-future-regressor] Which regressor may use its true future values?

*Checkpoint*

**Question:** You forecast a region's demand 28 days ahead with a regression with ARIMA errors. In the backtest, which of these regressors may be filled with its true values for the 28 test days?

- A. The measured daily temperature
- B. The demand of the neighbouring region
- C. The first day of Eid al-Adha in the official calendar
- D. The spot price of energy on each day

#### [day2#lab-2-3] Weather and Hijri regressors

Lab 2.3 · 25 min

Add temperature and Ramadan, Eid and national-day regressors to the ARIMA model, measure the gain on Eid days, and compare an honest temperature with the one measured later.

[D2_3_add_temperature_hijri.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D2_3_add_temperature_hijri.ipynb)

- [ ] Build cooling degrees and holiday regressors (5 min)
- [ ] Fit the regression with weather only (5 min)
- [ ] Add Ramadan, Eid and national days; measure the gain on Eid days (6 min)
- [ ] Compare the honest temperature with the one measured later (6 min)
- [ ] Read the coefficients in MWh (3 min)
- [ ] ✱ Stretch: Give Eid al-Fitr and Eid al-Adha separate regressors

### Session 4: Intervals that hold

#### [day2#s4] Intervals that hold

*Session 4*

- Measure the coverage of 80% and 95% intervals in a backtest
- Score quantiles with pinball loss
- Rebuild intervals from backtest errors when the model's are too narrow

#### [day2#interval-is-a-promise] An 80% interval is a promise about frequency, and it can be checked

*Probabilistic forecasts*

- **Quantile.** The 0.9 quantile is a value demand stays under on 9 days in 10.
- **Interval.** The 80% interval runs from the 0.1 to the 0.9 quantile.
- **Coverage.** The share of actual values that fell inside. For an 80% interval it should be near 80%.

> **What goes wrong.** Intervals are printed by every library and checked by almost nobody. Too narrow, and the planner is surprised twice as often as promised.

#### [day2#coverage-of-three-models] The most accurate model keeps its promise least

*Lab 2.4 · step 1 · 364 test days*

| Model | Scaled error | 80% interval covers | 95% interval covers |
| --- | --- | --- | --- |
| Decompose, then ETS | 0.98 | 83.5% | 93.1% |
| ARIMA + annual Fourier | 0.83 | 78.8% | 93.1% |
| Regression: weather + calendar | 0.77 | 63.5% | 76.9% |

> **Read the last row.** Its "95%" interval is missed on almost one day in four.

#### [day2#narrow-band-wide-band] The model's own band is thin; the rebuilt band admits what is unknown

*One summer fold, origin 16 July 2025*

Figure: The model's own 80% interval. Demand (navy) is inside on 16 of 28 days.

Figure: Rebuilt from the errors of earlier folds. Inside on 21 of 28 days.

#### [day2#pinball-loss] Pinball loss charges a quantile unequally for its two kinds of miss

*The score for a quantile*

```python
# file: Lab 2.4 · step 3
def pinball(y, q, tau):
    return np.mean(np.where(
        y >= q,
        tau * (y - q),          # too low
        (1 - tau) * (q - y)))   # too high

pinball([10], [8], 0.9)    # 1.8
pinball([10], [12], 0.9)   # 0.2
```

- A 0.9 quantile pays nine times as much for being too low as for being too high.
- That is the cost structure of Day 1's planner: the quantile **is** the cost ratio.
- At 0.5 it is half the absolute error. Lower is better.
- Averaged over several quantiles it scores the whole forecast distribution.

#### [day2#pinball-changes-the-ranking] By pinball loss the ranking of the three models changes

*Lab 2.4 · step 3 · scaled pinball loss*

| Model | q 0.025 | q 0.1 | q 0.5 | q 0.9 | q 0.975 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Decompose, then ETS | 0.102 | 0.269 | 0.492 | 0.255 | 0.097 | 0.243 |
| ARIMA + annual Fourier | 0.065 | 0.209 | 0.413 | 0.193 | 0.069 | 0.190 |
| Regression: weather + calendar | 0.134 | 0.246 | 0.383 | 0.197 | 0.083 | 0.209 |

> **Read by column.** The regression wins at the median and loses at both ends: its extreme quantiles sit too close to the middle.

#### [day2#intervals-from-backtest-errors] Rebuild the interval from the errors the backtest already made

*Lab 2.4 · step 4*

1. Collect the scaled errors of all **earlier** folds, by week of the horizon.
2. Take their quantiles: 0.025, 0.1, 0.5, 0.9, 0.975.
3. New quantile = point forecast + error quantile × scale.
4. Use at least a year of folds: summer errors are larger than winter errors.

| Regression, 13 folds of 2025 | Own | Rebuilt |
| --- | --- | --- |
| 80% interval covers | 63.5% | 78.3% |
| 95% interval covers | 76.9% | 94.5% |
| Scaled pinball loss | 0.208 | 0.192 |
| Mean width of the 80% interval | 436 MWh | 677 MWh |

#### [day2#report-coverage-as-measured] Report the coverage you measured, not the one on the label

*Honest reporting*

| Rebuilt 80% interval | Covers |
| --- | --- |
| Horizon week 1 | 72.5% |
| Horizon week 2 | 69.2% |
| Horizon week 3 | 82.4% |
| Horizon week 4 | 89.0% |

- Overall 78.3%: close to the promise. By week it still moves from 69% to 89%.
- 364 days can show that an interval is broken, and repair most of it. They cannot tune it to a decimal.
- The purchasing quantile, 0.8: demand stayed under it on 81.0% of days.

#### [day2#check-coverage-reading] Which forecast would you hand to the planner?

*Checkpoint*

**Question:** Model P: scaled error 0.75, its 80% interval covers 62% of days. Model Q: scaled error 0.80, its 80% interval covers 79%. The planner buys on the 0.8 quantile. What do you deliver?

- A. Model P as it is: it has the lower error
- B. Model Q, or model P with intervals rebuilt from its backtest errors
- C. Model P with every interval doubled in width
- D. Neither: average the two scaled errors and report 0.775

#### [day2#lab-2-4] Interval coverage in the backtest

Lab 2.4 · 25 min

Measure the 80% and 95% coverage of three supplied backtests by horizon week, score their quantiles with pinball loss and rebuild the intervals from the errors of earlier folds.

[D2_4_measure_80_95.ipynb](https://sdaiaacademy.github.io/ds-304/labs/D2_4_measure_80_95.ipynb)

- [ ] Compute 80% and 95% coverage for three models (5 min)
- [ ] Break coverage down by horizon week (4 min)
- [ ] Score each quantile with pinball loss (5 min)
- [ ] Rebuild the intervals from the errors of earlier folds (8 min)
- [ ] Report coverage before and after (3 min)
- [ ] ✱ Stretch: Check the single quantile the purchasing lead needs

#### [day2#wrap-up] Three things to keep

*Day 2 · wrap-up*

- **What a model can see.** Weekly ETS and weekly ARIMA both score about 1.6. Given the annual season they score 0.98 and 0.83. The input mattered more than the family.
- **Known at the origin.** The Hijri calendar is; the temperature is not. The calendar took Eid days from 1.34 to 0.80. The measured temperature would have flattered the score by 0.22.
- **Intervals are measured.** The best point forecast covered 63.5% with its 80% interval. Rebuilt from a year of backtest errors: 78.3%.

#### [day2#next] Day 3: Many series and pretrained models

*Next*

From one region to 400 feeders: forecasts that add up, one boosted model trained across all series, a pretrained forecaster that has never seen your data, and a scorecard to choose among them. Lab 3.3 runs best in Colab: have a Google account ready.

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

## Capstone: Capstone project presentations

An eight-minute defence each, then four minutes of questions in which the assessor moves the forecast origin or the horizon

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Forecasting and Time Series*

Capstone: An eight-minute defence each, then four minutes of questions in which the assessor moves the forecast origin or the horizon

Data Science Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · DS-304

#### [capstone#agenda] How the day runs

*Capstone*

- **Expectations and final prep**: the rubric, the format of the defence, a last run from a clean copy
- **Defences**: eight minutes each, then four minutes of questions
- **Review and closing**: peer feedback, the trainer's review, the follow-up task

#### [capstone#rubric] The rubric

*How the work is scored*

| Item | Value |
| --- | --- |
| Framing and floor | 15% |
| Methods | 25% |
| Backtest design | 25% |
| Uncertainty and communication | 25% |
| Reproducibility | 10% |

> **Every claim is read against the floor.** A score without the floor beside it, or an interval without its measured coverage, earns nothing.

#### [capstone#format] Eight minutes to present, four minutes of questions

*Format of the defence*

- **1 · The decision.** Who reads the forecast, what they decide, the horizon, the floor.
- **2 · What you built.** The backtest and three method families: what each can see.
- **3 · What you measured.** The scorecard: point and probabilistic accuracy against the floor.
- **4 · The note.** The fan chart, coverage in counts, when not to use it.

> **Peer feedback.** After each defence, everyone writes one strength, one question and one suggestion.

#### [capstone#the-question] The assessor moves the origin or the horizon: predict, then show

*The four minutes of questions*

- **The origin moves.** "Forecast from the day before the last holiday. What happens to the error and to the note?"
- **The horizon moves.** "The planner needs twice the horizon. What changes in the features, the floor and the bands?"
- **The history shrinks.** "This series is six weeks old. Which of your families still works?"

> **First say what you expect and why. Then run it.** A prediction that turns out wrong and is explained scores above a run without a prediction.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Dataset and decision |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first defence*

Run your analysis once from a clean copy. Open the note. Try one other origin.

#### [capstone#now-presenting] Defence

*Now presenting*

##### Questions

#### [capstone#review] What the defences showed

*Trainer review*

- **Strongest patterns.** What several analyses got right: floors, backtests, notes.
- **Common gaps.** What most would fix next. Usual ones: coverage never measured, a choice made on the test folds.
- **One surprise.** A result nobody expected, and what explained it.

#### [capstone#what-next] Keep going

*After the course*

- **The follow-up task:** backtest one forecast from your workplace, or a public one, against a seasonal-naive floor, and send the table.
- **Your repository stays yours:** keep it running from a clean copy and add the next family you named today.
- **At work:** archive what was known on each origin day, starting now. An honest backtest next year depends on it.
- **Read on:** Forecasting: Principles and Practice, chapters 11 and 13, for weighted reconciliation and practical issues.

#### [capstone#thanks] Thank you

*SDAIA Academy*

Forecasting and Time Series
