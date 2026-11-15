---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/slides/day1.html"
---

# Forecasting and Time Series

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
