---
course: "Forecasting and Time Series"
code: "DS-304"
version: "2027.2.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-304/index.html"
---

# Forecasting and Time Series

التنبؤ والسلاسل الزمنية

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 300

Forecast daily demand with a backtest behind every number, and hand the planner a fan chart they can act on

Participants forecast with the discipline of a backtest. On synthetic daily electricity demand from Shams Power, a fictional distributor with 12 regions and 400 feeders, they decompose series, set a seasonal-naive benchmark floor, fit exponential smoothing and ARIMA-family models with weather and Hijri holidays, train one boosted model across many series, reconcile feeder and regional forecasts and run pretrained zero-shot forecasters as baselines. Every method is scored by rolling-origin cross-validation with point and probabilistic measures and reaches the planner as a fan chart.

**Who it is for:** Specialists. Analysts and data scientists who have completed the level 200 set or can show equivalent work: an analysis or model they built, validated and can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Shams Power.** Shams Power is a fictional electricity distributor with 12 regions and 400 feeders. Its forecast desk, Marqab, owes the load planners a 28-day daily forecast per region for energy purchasing, with honest intervals. Participants work on nine years of synthetic daily demand with temperature and a Hijri and Gregorian calendar: Eid dips that move through the year, a level shift, an outage, twelve new feeders and a notebook with three seeded leaks. All data is synthetic and generated with a fixed seed.

## The course

### Day 1: Look, frame and set the floor

Slides: https://sdaiaacademy.github.io/ds-304/slides/day1.md

- **Session 1: What is forecast, and why**
  - Separate a forecast from a plan and a target
  - Fix horizon, granularity and level from the decision
  - Turn the cost of over- and under-forecasting into a target quantile
  - [Lab 1.1: Frame three requests (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D1_1_write_horizon_error.ipynb)
- **Session 2: Seeing structure**
  - Plot a series at three time scales and name what each shows
  - Measure the weekday profile and the Eid dip
  - Explain why a Hijri holiday is not a seasonal pattern
  - [Lab 1.2: Three time scales (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D1_2_plot_one_region.ipynb)
- **Session 3: Decomposition**
  - Decompose a series with weekly and annual seasons
  - Compute the strength of trend and of seasonality
  - Read the remainder for holidays, outages and breaks
  - [Lab 1.3: Decompose and measure strength (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D1_3_decompose_regional_demand.ipynb)
- **Session 4: The benchmark floor**
  - Write naive and seasonal-naive forecasts
  - Run a rolling-origin backtest over 28-day horizons
  - Report scaled errors by horizon and name the floor
  - [Lab 1.4: The benchmark floor (AI-off) (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D1_4_backtest_naive_seasonal.ipynb)

### Day 2: Statistical forecasters

Slides: https://sdaiaacademy.github.io/ds-304/slides/day2.md

- **Session 1: Exponential smoothing**
  - Compute simple exponential smoothing by hand
  - Fit ETS models with a weekly season
  - Combine a decomposition with ETS and score it against the floor
  - [Lab 2.1: Exponential smoothing against the floor (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D2_1_fit_exponential_smoothing.ipynb)
- **Session 2: ARIMA-family models**
  - Difference a series and read its autocorrelation
  - Fit a seasonal ARIMA model
  - Check residuals with a portmanteau test before trusting intervals
  - [Lab 2.2: A seasonal ARIMA with checked residuals (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D2_2_fit_seasonal_arima.ipynb)
- **Session 3: Weather and moving holidays**
  - Build cooling-degree and Hijri holiday regressors
  - Fit a regression with ARIMA errors
  - Separate what is known at the origin from what is measured later
  - [Lab 2.3: Weather and Hijri regressors (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D2_3_add_temperature_hijri.ipynb)
- **Session 4: Intervals that hold**
  - Measure the coverage of 80% and 95% intervals in a backtest
  - Score quantiles with pinball loss
  - Rebuild intervals from backtest errors when the model's are too narrow
  - [Lab 2.4: Interval coverage in the backtest (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D2_4_measure_80_95.ipynb)

### Day 3: Many series and pretrained models

Slides: https://sdaiaacademy.github.io/ds-304/slides/day3.md

- **Session 1: Many series and hierarchies**
  - Forecast every series of a two-level hierarchy
  - Reconcile bottom-up, top-down and by least squares
  - Compare errors at the feeder and the regional level
  - [Lab 3.1: Feeders that add up to their region (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D3_1_fit_per_feeder.ipynb)
- **Session 2: Global boosted forecasters**
  - Build lag and calendar features that use nothing after the origin
  - Train one boosted model on scaled targets across series
  - Backtest it with point and quantile scores on held-out weeks
  - [Lab 3.2: One boosted model across feeders (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D3_2_backtest_one_boosted.ipynb)
- **Session 3: Pretrained zero-shot forecasters**
  - Run a pretrained forecaster zero-shot on held-out weeks
  - Record accuracy, quantile scores and run time
  - Read a model card for licence and permitted use
  - [Lab 3.3: A pretrained forecaster, zero-shot (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D3_3_run_pretrained_forecaster.ipynb)
- **Session 4: Choosing among families**
  - Compute skill against the floor for every method
  - Weigh accuracy against run time, upkeep and licence
  - Defend the simplest method within noise of the best
  - Compare how forecast error grows under different retraining schedules and choose one
  - [Lab 3.4: The scorecard (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D3_4_fill_scorecard_accuracy.ipynb)

### Day 4: Review and communicate

Slides: https://sdaiaacademy.github.io/ds-304/slides/day4.md

- **Session 1: Leakage across time**
  - Find features and regressors that use data after the origin
  - Recognise selection on the test window
  - Correct a backtest and report the score before and after
  - [Lab 4.1: Three leaks in a backtest (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D4_1_find_three_seeded.ipynb)
- **Session 2: New and short series**
  - Forecast a short series from its own history
  - Pool information from similar series
  - Compare pooled, global and pretrained forecasts on a new series
  - [Lab 4.2: Feeders with eight weeks of history (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D4_2_forecast_feeder_eight.ipynb)
- **Session 3: Combining forecasts**
  - Average forecasts from different method families
  - Choose the best single method without looking at the test folds
  - Combine quantiles and check coverage
  - [Lab 4.3: Does the average beat the best? (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D4_3_test_whether_average.ipynb)
- **Session 4: Fan charts and forecast notes**
  - Draw a fan chart from quantile forecasts
  - State interval coverage in plain words
  - Write a forecast note that says when not to use the forecast
  - [Lab 4.4: Fan chart and planner's note (25 min)](https://sdaiaacademy.github.io/ds-304/labs/D4_4_draw_fan_chart.ipynb)

### Capstone: Capstone project presentations

Slides: https://sdaiaacademy.github.io/ds-304/slides/capstone.md

## Capstone

A forecasting analysis on a supplied or public dataset with three method families backtested against the floor, in a public repository whose README reports point and probabilistic accuracy, plus a two-page planner's note; eight-minute defence plus questions

**What you submit:** A link to a public GitHub repository holding the analysis, a README that reports point and probabilistic accuracy against the floor, the run logs and the two-page planner's note

**Rubric:**

- Framing and floor: 15%
- Methods: 25%
- Backtest design: 25%
- Uncertainty and communication: 25%
- Reproducibility: 10%

## Outcomes

- **LO1** Apply decomposition and a seasonal-naive benchmark floor scored by rolling-origin cross-validation, given regional daily demand with weekly and annual seasonality (Applied) Evidence: a backtest table of scaled errors by horizon
- **LO2** Implement exponential smoothing, ARIMA-family and dynamic regression models with temperature and moving-holiday terms, given the regional series and a Hijri and Gregorian calendar (Applied) Evidence: residual diagnostics and scores against the floor
- **LO3** Measure point and quantile accuracy of a global boosted model and a pretrained zero-shot forecaster against statistical models, given 400 feeder series with eight weeks held out (Applied) Evidence: a scorecard of scaled error, pinball loss, coverage, run time and licence terms
- **LO4** Review a forecast evaluation for leakage across time and selection on the test window, and correct it, given a notebook with three seeded faults (Performance) Evidence: a review memo and a corrected backtest
- **LO5** Build a forecasting analysis for new series and defend method, horizon and uncertainty to a planner, given a supplied dataset and a planner's request (Performance) Evidence: a repository and a two-page note with fan charts, defended live

## Tools

- **Analysis language and notebooks:** Python with Jupyter or marimo, R with Quarto
- **Statistical forecasting:** statsforecast, fable for R, statsmodels
- **Machine learning forecasting:** LightGBM with lag features, sktime, Darts
- **Pretrained forecaster as a baseline:** the Chronos family of open-weight forecasters, TimesFM, AutoGluon time series

## Prerequisites

- DS-201 Statistical Inference by Simulation
- DS-204 Machine Learning for Tabular Data: Baselines and Validation
- Resamples and reshapes time-indexed data in a dataframe

## Sources

### Textbook

- [Hyndman and Athanasopoulos: Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/)
- [FPP3: evaluating point forecast accuracy (scaled errors)](https://otexts.com/fpp3/accuracy.html)
- [FPP3: time series cross-validation](https://otexts.com/fpp3/tscv.html)
- [FPP3: STL features (strength of trend and seasonality)](https://otexts.com/fpp3/stlfeatures.html)
- [FPP3: dynamic regression models](https://otexts.com/fpp3/dynamic.html)
- [FPP3: forecast reconciliation](https://otexts.com/fpp3/reconciliation.html)
- [FPP3: forecast combinations](https://otexts.com/fpp3/combinations.html)

### Libraries

- [statsforecast documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html)
- [statsmodels: MSTL decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html)
- [LightGBM parameters (quantile objective)](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [AutoGluon time series](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)

### Pretrained forecasters and their licences

- [Chronos-Bolt tiny: model card (Apache-2.0)](https://huggingface.co/amazon/chronos-bolt-tiny)
- [Chronos-2: model card (Apache-2.0)](https://huggingface.co/amazon/chronos-2)
- [TimesFM 2.5: model card (Apache-2.0)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Moirai 2.0 small: model card (CC-BY-NC-4.0)](https://huggingface.co/Salesforce/moirai-2.0-R-small)
- [Ansari and others: Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Ansari and others: Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821)
- [Das and others: A decoder-only foundation model for time-series forecasting (TimesFM)](https://arxiv.org/abs/2310.10688)
- [Woo and others: Unified Training of Universal Time Series Forecasting Transformers (Moirai)](https://arxiv.org/abs/2402.02592)

### Research on evaluation, combination and communication

- [Bates and Granger (1969): The Combination of Forecasts](https://doi.org/10.1057/jors.1969.103)
- [Kaufman, Rosset, Perlich and Stitelman (2012): Leakage in data mining](https://doi.org/10.1145/2382577.2382579)
- [Wickramasuriya, Athanasopoulos and Hyndman (2019): Optimal forecast reconciliation through trace minimization](https://doi.org/10.1080/01621459.2018.1448825)
- [Makridakis, Spiliotis and Assimakopoulos (2020): The M4 Competition](https://doi.org/10.1016/j.ijforecast.2019.04.014)
- [Makridakis, Spiliotis and Assimakopoulos (2022): M5 accuracy competition](https://doi.org/10.1016/j.ijforecast.2021.11.013)
- [Gigerenzer and Edwards (2003): Simple tools for understanding risks](https://doi.org/10.1136/bmj.327.7417.741)

### National references

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Open Data Policy (National Data Management Office)](https://sdaia.gov.sa/en/SDAIA/about/Documents/OpenDataPolicy.pdf)
- [Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)

---

SDAIA Academy · Data Science Track · Forecasting and Time Series

Course version 2027.2.0 · engine 1.0.0 · built 2026-11-24 · commit 6daea74
