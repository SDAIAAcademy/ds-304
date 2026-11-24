---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.2.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/slides/day2.html"
---

# Forecasting and Time Series

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
