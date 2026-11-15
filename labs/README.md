# Forecasting and Time Series: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · What is forecast, and why | [`D1_1_write_horizon_error.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D1_1_write_horizon_error.ipynb) |
| Day 1 | Session 2 · Seeing structure | [`D1_2_plot_one_region.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D1_2_plot_one_region.ipynb) |
| Day 1 | Session 3 · Decomposition | [`D1_3_decompose_regional_demand.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D1_3_decompose_regional_demand.ipynb) |
| Day 1 | Session 4 · The benchmark floor | [`D1_4_backtest_naive_seasonal.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D1_4_backtest_naive_seasonal.ipynb) |
| Day 2 | Session 1 · Exponential smoothing | [`D2_1_fit_exponential_smoothing.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D2_1_fit_exponential_smoothing.ipynb) |
| Day 2 | Session 2 · ARIMA-family models | [`D2_2_fit_seasonal_arima.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D2_2_fit_seasonal_arima.ipynb) |
| Day 2 | Session 3 · Weather and moving holidays | [`D2_3_add_temperature_hijri.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D2_3_add_temperature_hijri.ipynb) |
| Day 2 | Session 4 · Intervals that hold | [`D2_4_measure_80_95.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D2_4_measure_80_95.ipynb) |
| Day 3 | Session 1 · Many series and hierarchies | [`D3_1_fit_per_feeder.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D3_1_fit_per_feeder.ipynb) |
| Day 3 | Session 2 · Global boosted forecasters | [`D3_2_backtest_one_boosted.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D3_2_backtest_one_boosted.ipynb) |
| Day 3 | Session 3 · Pretrained zero-shot forecasters | [`D3_3_run_pretrained_forecaster.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D3_3_run_pretrained_forecaster.ipynb) |
| Day 3 | Session 4 · Choosing among families | [`D3_4_fill_scorecard_accuracy.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D3_4_fill_scorecard_accuracy.ipynb) |
| Day 4 | Session 1 · Leakage across time | [`D4_1_find_three_seeded.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D4_1_find_three_seeded.ipynb) |
| Day 4 | Session 2 · New and short series | [`D4_2_forecast_feeder_eight.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D4_2_forecast_feeder_eight.ipynb) |
| Day 4 | Session 3 · Combining forecasts | [`D4_3_test_whether_average.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D4_3_test_whether_average.ipynb) |
| Day 4 | Session 4 · Fan charts and forecast notes | [`D4_4_draw_fan_chart.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/forecasting/blob/main/labs/D4_4_draw_fan_chart.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.

## What else is in this folder

| Path | What it is |
|---|---|
| `00_setup_check.ipynb` | Run once before Day 1: packages, data, one fitted model, one boosted model. |
| `shams.py` | The course helper, imported as `U`: loading the data, the backtest, the scores, the plots. Read it; nothing in it is hidden. |
| `prepare_data.py` | Writes every data file from a fixed seed. The files are committed, so participants do not run it. |
| `data/` | Regions, feeders, the calendar, the three requests, and the model cards read for Lab 3.3. |
| `data/reference/` | The reference state: the results each lab starts from, so that no lab depends on a participant's earlier run. |
| `_tools/make_reference.py` | Writes the reference results that no single lab produces (about half an hour on a laptop). The other reference files are written by the solution runs with `SHAMS_WRITE_REF=1`. |
| `capstone/` | The capstone brief and its two datasets. |
| `solutions/` | The solution notebooks and `answer_key.md`, the trainer's key. |

## Running the labs

- Every lab runs on a laptop processor or on a free Colab runtime. No lab calls a language model and none needs an API key.
- `SHAMS_QUICK=1` runs every notebook with fewer folds and regions, for a fast check that the code works. The numbers in the slides and in the answer key come from full runs.
- **Lab 3.3** uses a small open-weight pretrained forecaster. In Colab its setup cell installs the package and the weights load into the Colab runtime. Anywhere the package is absent, a stand-in answers and labels every output `STAND-IN`: its numbers are not a pretrained model's result and are left out of the scorecard of Lab 3.4 and the comparison of Lab 4.2.
- On Windows, keep the repository in a short path and run `git config core.longpaths true` before cloning into a deep folder.
- Lab outputs are written to `labs/out/`, which is not committed.
