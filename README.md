<!-- academy:title -->
# Forecasting and Time Series

التنبؤ والسلاسل الزمنية

Data Science Track · SDAIA Academy · DS-304 · Level 300

REPLACE: one line under the title: what participants can do after the course

Participants forecast with the discipline of a backtest. On synthetic daily electricity demand from Shams Power, a fictional distributor with 12 regions and 400 feeders, they decompose series, set a seasonal-naive benchmark floor, fit exponential smoothing and ARIMA-family models with weather and Hijri holidays, train one boosted model across many series, reconcile feeder and regional forecasts and run pretrained zero-shot forecasters as baselines. Every method is scored by rolling-origin cross-validation with point and probabilistic measures and reaches the planner as a fan chart.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/forecasting/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Apply decomposition and a seasonal-naive benchmark floor scored by rolling-origin cross-validation, given regional daily demand with weekly and annual seasonality | Applied | a backtest table of scaled errors by horizon |
| LO2 | Implement exponential smoothing, ARIMA-family and dynamic regression models with temperature and moving-holiday terms, given the regional series and a Hijri and Gregorian calendar | Applied | residual diagnostics and scores against the floor |
| LO3 | Measure point and quantile accuracy of a global boosted model and a pretrained zero-shot forecaster against statistical models, given 400 feeder series with eight weeks held out | Applied | a scorecard of scaled error, pinball loss, coverage, run time and licence terms |
| LO4 | Review a forecast evaluation for leakage across time and selection on the test window, and correct it, given a notebook with three seeded faults | Performance | a review memo and a corrected backtest |
| LO5 | Build a forecasting analysis for new series and defend method, horizon and uncertainty to a planner, given a supplied dataset and a planner's request | Performance | a repository and a two-page note with fan charts, defended live |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Look, frame and set the floor | What is forecast, and why; Seeing structure; Decomposition; The benchmark floor | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Statistical forecasters | Exponential smoothing; ARIMA-family models; Weather and moving holidays; Intervals that hold | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Many series and pretrained models | Many series and hierarchies; Global boosted forecasters; Pretrained zero-shot forecasters; Choosing among families | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Review and communicate | Leakage across time; New and short series; Combining forecasts; Fan charts and forecast notes | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Write horizon, error measure and decision for three planner requests | [`D1_1_write_horizon_error.ipynb`](labs/D1_1_write_horizon_error.ipynb) |
| 1.2 | Day 1 · Session 2 | Plot one region at three time scales and mark Ramadan and Eid | [`D1_2_plot_one_region.ipynb`](labs/D1_2_plot_one_region.ipynb) |
| 1.3 | Day 1 · Session 3 | Decompose regional demand and quantify trend and seasonal strength | [`D1_3_decompose_regional_demand.ipynb`](labs/D1_3_decompose_regional_demand.ipynb) |
| 1.4 | Day 1 · Session 4 | Backtest naive and seasonal-naive forecasts over 28-day horizons | [`D1_4_backtest_naive_seasonal.ipynb`](labs/D1_4_backtest_naive_seasonal.ipynb) |
| 2.1 | Day 2 · Session 1 | Fit exponential smoothing models and score them against the floor | [`D2_1_fit_exponential_smoothing.ipynb`](labs/D2_1_fit_exponential_smoothing.ipynb) |
| 2.2 | Day 2 · Session 2 | Fit a seasonal ARIMA model, check residuals and score it | [`D2_2_fit_seasonal_arima.ipynb`](labs/D2_2_fit_seasonal_arima.ipynb) |
| 2.3 | Day 2 · Session 3 | Add temperature and Hijri holiday regressors and measure the gain around Eid | [`D2_3_add_temperature_hijri.ipynb`](labs/D2_3_add_temperature_hijri.ipynb) |
| 2.4 | Day 2 · Session 4 | Measure 80% and 95% interval coverage in the backtest | [`D2_4_measure_80_95.ipynb`](labs/D2_4_measure_80_95.ipynb) |
| 3.1 | Day 3 · Session 1 | Fit per-feeder models, reconcile them with regional totals and compare errors at both levels | [`D3_1_fit_per_feeder.ipynb`](labs/D3_1_fit_per_feeder.ipynb) |
| 3.2 | Day 3 · Session 2 | Backtest one boosted model on lag and calendar features across feeders | [`D3_2_backtest_one_boosted.ipynb`](labs/D3_2_backtest_one_boosted.ipynb) |
| 3.3 | Day 3 · Session 3 | Run a pretrained forecaster zero-shot and record accuracy, run time and licence | [`D3_3_run_pretrained_forecaster.ipynb`](labs/D3_3_run_pretrained_forecaster.ipynb) |
| 3.4 | Day 3 · Session 4 | Fill a scorecard of accuracy, cost and licence per method family | [`D3_4_fill_scorecard_accuracy.ipynb`](labs/D3_4_fill_scorecard_accuracy.ipynb) |
| 4.1 | Day 4 · Session 1 | Find three seeded leaks in a notebook and correct the backtest | [`D4_1_find_three_seeded.ipynb`](labs/D4_1_find_three_seeded.ipynb) |
| 4.2 | Day 4 · Session 2 | Forecast a feeder with eight weeks of history using pooled and pretrained methods | [`D4_2_forecast_feeder_eight.ipynb`](labs/D4_2_forecast_feeder_eight.ipynb) |
| 4.3 | Day 4 · Session 3 | Test whether an average of three methods beats the best single one | [`D4_3_test_whether_average.ipynb`](labs/D4_3_test_whether_average.ipynb) |
| 4.4 | Day 4 · Session 4 | Draw a fan chart, write a planner's note and choose the capstone dataset | [`D4_4_draw_fan_chart.ipynb`](labs/D4_4_draw_fan_chart.ipynb) |
<!-- /academy:labs -->

## Capstone

A forecasting analysis on a supplied or public dataset with three method families backtested against the floor, in a public repository whose README reports point and probabilistic accuracy, plus a two-page planner's note; eight-minute defence plus questions

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- DS-201 Statistical Inference by Simulation
- DS-204 Machine Learning for Tabular Data: Baselines and Validation
- Resamples and reshapes time-indexed data in a dataframe

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Analysis language and notebooks:** Python with Jupyter or marimo, R with Quarto
- **Statistical forecasting:** statsforecast, fable for R, statsmodels
- **Machine learning forecasting:** LightGBM with lag features, sktime, Darts
- **Pretrained forecaster as a baseline:** the Chronos family of open-weight forecasters, TimesFM (non-commercial weights), AutoGluon time series
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- NDMO Open Data Policy: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
