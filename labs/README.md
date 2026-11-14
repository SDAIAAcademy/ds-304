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
