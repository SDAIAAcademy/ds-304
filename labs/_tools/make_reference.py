"""Write the reference results that no lab solution produces by itself.

    uv run python _tools/make_reference.py            # all three parts
    uv run python _tools/make_reference.py regional   # bt_dynreg_2024.csv, about half a minute
    uv run python _tools/make_reference.py windows    # bt_windows.csv, about three minutes
    uv run python _tools/make_reference.py feeders    # fd_stats.json, about half an hour on a laptop

1. bt_dynreg_2024.csv: the regression with weather and calendar regressors of Lab 2.3 over the 13 origins of
   2024, the year before the test year. Lab 2.4 calibrates intervals on it.
2. bt_windows.csv: the corrected regression of Lab 4.1 with five lengths of training window, over the folds of
   2024 and of 2025. Lab 4.1 uses it for the fault of selection on the test window.
3. fd_stats.json: three per-feeder methods on the 388 established feeders over the eight held-out weeks
   (two folds), with their scores and run times: the floor, ETS with a weekly season, and decompose-then-ETS.
   Lab 3.4 fills its scorecard from it. The run times are those of the machine that ran this script.

The other reference files are written by the lab solutions themselves (SHAMS_WRITE_REF=1).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shams as U  # noqa: E402
from statsforecast.models import MSTL, AutoETS  # noqa: E402

part = sys.argv[1] if len(sys.argv) > 1 else "all"
U.REF.mkdir(parents=True, exist_ok=True)

if part in ("all", "regional"):
    y = U.region("R01")["demand"]
    origins = U.origins(13, last=U.origins()[0])          # 13 folds that end where the 2025 folds begin
    bt = U.backtest(y, lambda train, h, origin: U.dynreg("R01", train, h, origin), origins)
    bt.round(2).to_csv(U.REF / "bt_dynreg_2024.csv", index=False, lineterminator="\n")
    print("bt_dynreg_2024.csv:", len(bt), "rows,", bt.date.min().date(), "to", bt.date.max().date(), U.scores(bt))

if part in ("all", "windows"):
    # Lab 4.1, third fault: the corrected regression of that lab with five lengths of training window,
    # backtested over the 13 folds of 2024 and the 13 folds of 2025.
    from statsforecast.models import ARIMA  # noqa: E402

    data, cal = U.region("R01"), U.calendar()
    y = data["demand"]

    def features(dates, origin):
        dates = pd.DatetimeIndex(dates)
        X = pd.DataFrame(index=dates)
        temp = data.temp_c.reindex(dates).where(dates <= origin, U.temp_known("R01", origin, dates))
        X["cdd"] = (temp - 22).clip(lower=0).to_numpy()
        X["smooth"] = data.demand.rolling(7).mean().shift(28).reindex(dates).to_numpy()
        X["ramadan"] = cal.ramadan.reindex(dates).astype(float).to_numpy()
        X["eid"] = cal.eid_weight.reindex(dates).astype(float).to_numpy()
        return X

    def forecast(train, h, origin):
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
        model = ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)
        return U.sf(model, train, h, features(train.index, origin), features(future, origin))

    parts = []
    for year, origins in ((2024, U.origins(13, last=U.origins()[0])), (2025, U.origins(13))):
        for window in (365, 548, 730, 1096, 1461):
            b = U.backtest(y, forecast, origins, window=window)[["origin", "date", "h", "y", "yhat", "scale"]]
            b.insert(0, "window", window)
            b.insert(0, "year", year)
            parts.append(b)
    allw = pd.concat(parts, ignore_index=True)
    # one yardstick for every window: the scale of the 1,096-day window of the same fold
    ref = allw[allw.window == 1096].groupby("origin")["scale"].first()
    allw["scale"] = allw.origin.map(ref)
    print(allw.assign(e=(allw.y - allw.yhat).abs() / allw["scale"]).groupby(["year", "window"]).e.mean().round(3).unstack().to_string())
    allw.round(2).to_csv(U.REF / "bt_windows.csv", index=False, lineterminator="\n")

if part in ("all", "feeders"):
    W = U.feeders()[U.established()]
    out = {}

    def run(name, fn):
        t0 = time.perf_counter()
        folds = []
        for o in U.FEEDER_ORIGINS:
            points, q = fn(W.loc[:o].iloc[-1096:])
            folds.append(U.feeder_long(W, o, points, q))
        seconds = time.perf_counter() - t0
        long = pd.concat(folds, ignore_index=True)
        out[name] = dict(U.scores(long, qs=U.PQS), seconds=round(seconds, 1), fits=2 * W.shape[1],
                         by_fold=[round(U.mase(f), 3) for f in folds])
        print(name, out[name], flush=True)

    run("floor", lambda train: (U.feeder_floor(W, train.index[-1]), None))
    run("ets_weekly", lambda train: U.sf_many(AutoETS(season_length=7, model="ANA"), train, quantiles=True))
    run("stl_ets", lambda train: U.sf_many(MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN")), train,
                                           quantiles=True, log=True))
    (U.REF / "fd_stats.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
