<!-- academy:title -->
# Forecasting and Time Series

التنبؤ والسلاسل الزمنية

Data Science Track · SDAIA Academy · DS-304 · Level 300

Forecast daily demand with a backtest behind every number, and hand the planner a fan chart they can act on

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

**Shams Power** is a fictional electricity distributor with 12 regions and 400 feeders. Participants are its forecast desk, Marqab, which owes the load planners a daily forecast for each of the next 28 days, by region, with intervals that hold. All data is synthetic and generated with a fixed seed by `labs/prepare_data.py`: nine years of daily demand (2017 to 2025), the day's temperature and its normal, and a Gregorian and Hijri calendar with the weekend, Ramadan, both Eids, National Day and Founding Day.

Problems are planted in the data and each is found by a named lab: Eid dips that move through the year, a level shift when six industrial feeders join one region, a two-day outage, model intervals that are too narrow, twelve feeders with eight weeks of history, and a backtest with three seeded leaks. The trainer's key is `labs/solutions/answer_key.md`.

The people: Shahad Al-Tamimi (load planning manager, the planner the forecast note is written for), Omar Al-Khuzaim (energy purchasing lead), Hamad Al-Dawsari (network operations engineer), Thamer Al-Anzi (data protection officer) and Nada Al-Jasser (the forecast analyst, the role participants play).

## How the course is organised

- **One instrument for every method.** A rolling-origin backtest: 13 origins 28 days apart over 2025 for the regions, two origins over the last eight weeks for the feeders, a 28-day horizon, a scaled error whose yardstick is the in-sample weekly seasonal-naive error, pinball loss and interval coverage at 80% and 95%.
- **A floor first.** Day 1 sets the seasonal-naive benchmark floor. Every later method reports its distance to it.
- **Every lab starts from a supplied reference state**: the data files and, in `labs/data/reference/`, the results earlier labs would have produced. No lab depends on a participant's earlier run, and every exercise has a safety net.
- **Pre-work and follow-up.** Before Day 1: the Academy's air-passengers starter worked to a seasonal-naive benchmark, with an entry quiz. After the course: backtest one workplace or public forecast against that floor.
- **AI policy.** AI off for the supervised checks and for writing the first backtest (Lab 1.4). In the other labs an assistant may explain output without writing the backtest. In the capstone any use of AI is allowed and defended live.
- **Responsible use.** Meter-level demand can reveal how a household lives, so the labs use aggregated synthetic series and discuss when linked consumption is personal data. Every forecast note states its interval coverage and when not to use the forecast; licences of pretrained models are read from their model cards; public capstone data is used under its stated open-data terms.
- **No model is called over an API and none is needed on a laptop.** Lab 3.3 runs a small open-weight pretrained forecaster inside a Colab runtime; where its package is absent, a labelled stand-in keeps the notebook running.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Frame three requests | [`D1_1_write_horizon_error.ipynb`](labs/D1_1_write_horizon_error.ipynb) |
| 1.2 | Day 1 · Session 2 | Three time scales | [`D1_2_plot_one_region.ipynb`](labs/D1_2_plot_one_region.ipynb) |
| 1.3 | Day 1 · Session 3 | Decompose and measure strength | [`D1_3_decompose_regional_demand.ipynb`](labs/D1_3_decompose_regional_demand.ipynb) |
| 1.4 | Day 1 · Session 4 | The benchmark floor (AI-off) | [`D1_4_backtest_naive_seasonal.ipynb`](labs/D1_4_backtest_naive_seasonal.ipynb) |
| 2.1 | Day 2 · Session 1 | Exponential smoothing against the floor | [`D2_1_fit_exponential_smoothing.ipynb`](labs/D2_1_fit_exponential_smoothing.ipynb) |
| 2.2 | Day 2 · Session 2 | A seasonal ARIMA with checked residuals | [`D2_2_fit_seasonal_arima.ipynb`](labs/D2_2_fit_seasonal_arima.ipynb) |
| 2.3 | Day 2 · Session 3 | Weather and Hijri regressors | [`D2_3_add_temperature_hijri.ipynb`](labs/D2_3_add_temperature_hijri.ipynb) |
| 2.4 | Day 2 · Session 4 | Interval coverage in the backtest | [`D2_4_measure_80_95.ipynb`](labs/D2_4_measure_80_95.ipynb) |
| 3.1 | Day 3 · Session 1 | Feeders that add up to their region | [`D3_1_fit_per_feeder.ipynb`](labs/D3_1_fit_per_feeder.ipynb) |
| 3.2 | Day 3 · Session 2 | One boosted model across feeders | [`D3_2_backtest_one_boosted.ipynb`](labs/D3_2_backtest_one_boosted.ipynb) |
| 3.3 | Day 3 · Session 3 | A pretrained forecaster, zero-shot | [`D3_3_run_pretrained_forecaster.ipynb`](labs/D3_3_run_pretrained_forecaster.ipynb) |
| 3.4 | Day 3 · Session 4 | The scorecard | [`D3_4_fill_scorecard_accuracy.ipynb`](labs/D3_4_fill_scorecard_accuracy.ipynb) |
| 4.1 | Day 4 · Session 1 | Three leaks in a backtest | [`D4_1_find_three_seeded.ipynb`](labs/D4_1_find_three_seeded.ipynb) |
| 4.2 | Day 4 · Session 2 | Feeders with eight weeks of history | [`D4_2_forecast_feeder_eight.ipynb`](labs/D4_2_forecast_feeder_eight.ipynb) |
| 4.3 | Day 4 · Session 3 | Does the average beat the best? | [`D4_3_test_whether_average.ipynb`](labs/D4_3_test_whether_average.ipynb) |
| 4.4 | Day 4 · Session 4 | Fan chart and planner's note | [`D4_4_draw_fan_chart.ipynb`](labs/D4_4_draw_fan_chart.ipynb) |
<!-- /academy:labs -->

## Capstone

A forecasting analysis on a supplied or public dataset with three method families backtested against the floor, in a public repository whose README reports point and probabilistic accuracy, plus a two-page planner's note; eight-minute defence plus questions

**Submission:** A link to a public GitHub repository holding the analysis, a README that reports point and probabilistic accuracy against the floor, the run logs and the two-page planner's note.

Two synthetic datasets are supplied in `labs/capstone/data/`: 36 electric-vehicle charging hubs (`ev_hubs.csv`) and eight district-cooling plants (`cooling_plants.csv`). A public dataset may be used under its stated terms. The defence is eight minutes, then four minutes of questions in which the assessor moves the forecast origin or the horizon and the participant predicts, then shows, the effect.

| Criterion | Weight |
|---|---|
| Framing and floor | 15 |
| Methods | 25 |
| Backtest design | 25 |
| Uncertainty and communication | 25 |
| Reproducibility | 10 |

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
- **Pretrained forecaster as a baseline:** the Chronos family of open-weight forecasters, TimesFM, AutoGluon time series
<!-- /academy:tools -->

<!-- academy:sources -->
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

### National references

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Open Data Policy (National Data Management Office)](https://sdaia.gov.sa/en/SDAIA/about/Documents/OpenDataPolicy.pdf)
- [Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
<!-- /academy:sources -->

### National references

- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf, and its implementing regulation: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- NDMO Open Data Policy: https://sdaia.gov.sa/en/SDAIA/about/Documents/OpenDataPolicy.pdf
- SDAIA AI Ethics Principles: https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
