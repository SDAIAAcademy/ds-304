"""Write the reference backtest that Lab 2.4 calibrates on: the regression with weather and calendar
regressors of Lab 2.3, over the 13 origins of 2024 (the year before the test year).

    uv run python _tools/make_reference.py

The other reference files are written by the lab solutions themselves (SHAMS_WRITE_REF=1).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shams as U  # noqa: E402
from statsforecast.models import ARIMA  # noqa: E402

RID = "R01"
y = U.region(RID)["demand"]


def dyn_full(train, h, origin):
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
    model = ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)
    return U.sf(model, train, h, U.exog(RID, train.index), U.exog(RID, future, origin=origin))


origins = U.origins(13, last=U.origins()[0])          # 13 folds that end where the 2025 folds begin
bt = U.backtest(y, dyn_full, origins)
U.REF.mkdir(parents=True, exist_ok=True)
bt.round(2).to_csv(U.REF / "bt_dynreg_2024.csv", index=False, lineterminator="\n")
print("bt_dynreg_2024.csv:", len(bt), "rows,", bt.date.min().date(), "to", bt.date.max().date(), U.scores(bt))
