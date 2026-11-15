"""Helpers shared by every lab of Forecasting and Time Series (DS-304).

The case: Shams Power, a fictional electricity distributor with 12 regions and 400 feeders. All data is
synthetic (prepare_data.py, seed 304).

    import shams as U
    y = U.region("R01")["demand"]              # one regional series, indexed by day
    bt = U.backtest(y, U.seasonal_naive)       # rolling-origin backtest, 13 origins of 28 days
    U.mase(bt), U.by_week(bt)                  # scaled error, overall and by week of the horizon
    U.check(cond, "what is true")              # prints a tick, or stops with what went wrong
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
OUT = LABS / "out"
QUICK = os.getenv("SHAMS_QUICK", "0") == "1"
WRITE_REF = os.getenv("SHAMS_WRITE_REF", "0") == "1"

H = 28                                   # the horizon every lab uses, in days
M = 7                                    # the weekly season
QS = (0.025, 0.1, 0.5, 0.9, 0.975)       # the quantiles every probabilistic forecast reports
QCOLS = [f"q{q}" for q in QS]
LAST = pd.Timestamp("2025-12-31")        # the last day of data
HOME = "R01"                             # Al-Rimal, the region most labs work on


def pick(full, quick):
    """The small value in quick mode (SHAMS_QUICK=1), the full one otherwise."""
    return quick if QUICK else full


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError("✗ " + msg)
    print("✓", msg)


# ------------------------------------------------------------------ data
_cache: dict = {}


def regions() -> pd.DataFrame:
    """Long table: date, region, name, demand, temp_c, temp_normal."""
    if "regions" not in _cache:
        df = pd.read_csv(DATA / "regions.csv", parse_dates=["date"])
        _cache["regions"] = df.rename(columns={"demand_mwh": "demand"})
    return _cache["regions"].copy()


def region(rid: str = HOME) -> pd.DataFrame:
    """One region, indexed by day: demand (MWh), temp_c, temp_normal."""
    df = regions()
    return df[df.region == rid].set_index("date")[["demand", "temp_c", "temp_normal"]].asfreq("D")


def region_names() -> dict:
    df = regions()
    return dict(df.drop_duplicates("region")[["region", "name"]].values)


def calendar() -> pd.DataFrame:
    """One row per day to 2026-03-31: weekend, ramadan, eid_fitr, eid_adha, eid_weight, national days."""
    if "cal" not in _cache:
        _cache["cal"] = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"], keep_default_na=False
                                    ).set_index("date")
    return _cache["cal"].copy()


def meta() -> pd.DataFrame:
    return pd.read_csv(DATA / "feeders_meta.csv", parse_dates=["commissioned"])


def feeders(rid: str | None = None) -> pd.DataFrame:
    """Wide table of feeder demand (MWh): one row per day, one column per feeder; NaN before commissioning."""
    if "feeders" not in _cache:
        z = np.load(DATA / "feeders.npz")
        idx = pd.date_range(str(z["start"][0]), periods=z["mwh_x100"].shape[0], freq="D")
        wide = pd.DataFrame(z["mwh_x100"].astype("float64") / 100, index=idx, columns=[str(c) for c in z["feeders"]])
        m = meta().set_index("feeder")["commissioned"]
        for f in m.index[m > idx[0]]:
            wide.loc[wide.index < m[f], f] = np.nan
        _cache["feeders"] = wide
    wide = _cache["feeders"]
    if rid is not None:
        m = meta()
        wide = wide[list(m.feeder[m.region == rid])]
    return wide.copy()


def js(name: str):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def reference(name: str):
    """The reference result of an exercise, written by the solution run: the safety net of every lab."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def keep(name: str, value):
    """In the solution run that writes the references (SHAMS_WRITE_REF=1), save one; always return the value."""
    if WRITE_REF:
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(_plain(value), indent=1) + "\n", encoding="utf-8", newline="\n")
    return value


def keep_bt(name: str, bt: pd.DataFrame) -> pd.DataFrame:
    """Save a backtest table for later labs (only in the run that writes the references)."""
    if WRITE_REF:
        REF.mkdir(parents=True, exist_ok=True)
        bt.round(2).to_csv(REF / f"bt_{name}.csv", index=False, lineterminator="\n")
    return bt


def bt(name: str) -> pd.DataFrame:
    """A supplied backtest table: the checkpoint later labs start from."""
    return pd.read_csv(REF / f"bt_{name}.csv", parse_dates=["origin", "date"])


def save(name: str, text: str) -> Path:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------ rolling origins and scores
def origins(n: int = 13, h: int = H, step: int = H, last=LAST) -> list:
    """The last training day of each of n folds; fold k is tested on the h days after its origin."""
    last = pd.Timestamp(last)
    return [last - pd.Timedelta(days=h + step * k) for k in range(n - 1, -1, -1)]


def seasonal_naive(train: pd.Series, h: int = H, m: int = M) -> np.ndarray:
    """Repeat the last m observed values over the horizon (m=1 is the naive forecast)."""
    last = train.to_numpy()[-m:]
    return np.resize(last, h)


def scale(train: pd.Series, m: int = M) -> float:
    """The denominator of the scaled error: the in-sample mean absolute error of the seasonal naive forecast."""
    v = train.dropna().to_numpy()
    return float(np.mean(np.abs(v[m:] - v[:-m])))


def backtest(y: pd.Series, fn, origin_list=None, h: int = H, window: int | None = 1096, m: int = M) -> pd.DataFrame:
    """Rolling-origin backtest of one series.

    fn(train, h) or fn(train, h, origin), the third argument named "origin" or "o", returns h point forecasts, or a DataFrame with a column "yhat" and
    optionally the quantile columns q0.025 ... q0.975. Each fold is trained on the `window` days up to its
    origin and tested on the next h days. Returns one row per fold and day.
    """
    import inspect
    names = list(inspect.signature(fn).parameters)
    takes_origin = len(names) >= 3 and names[2] in ("origin", "o")
    rows = []
    for o in (origin_list or origins(h=h)):
        train = y.loc[:o].dropna()
        if window:
            train = train.iloc[-window:]
        test = y.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=h)]
        f = fn(train, h, o) if takes_origin else fn(train, h)
        f = pd.DataFrame({"yhat": np.asarray(f, float)}) if not isinstance(f, pd.DataFrame) else f.reset_index(drop=True)
        f = f.iloc[: len(test)].copy()
        f.insert(0, "origin", o)
        f.insert(1, "date", test.index)
        f.insert(2, "h", np.arange(1, len(test) + 1))
        f.insert(3, "y", test.to_numpy())
        f["scale"] = scale(train, m)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def mase(bt: pd.DataFrame, col: str = "yhat") -> float:
    """Mean absolute scaled error: below 1 beats the in-sample seasonal naive error."""
    return float((np.abs(bt.y - bt[col]) / bt["scale"]).mean())


def by_week(bt: pd.DataFrame, col: str = "yhat") -> pd.Series:
    """Scaled error by week of the horizon (days 1-7, 8-14, 15-21, 22-28)."""
    wk = ((bt.h - 1) // 7 + 1).rename("week")
    return (np.abs(bt.y - bt[col]) / bt["scale"]).groupby(wk).mean().round(3)


def pinball(y, q, tau: float) -> float:
    """Pinball (quantile) loss of the forecast q of quantile tau."""
    y, q = np.asarray(y, float), np.asarray(q, float)
    return float(np.mean(np.where(y >= q, tau * (y - q), (1 - tau) * (q - y))))


def scaled_pinball(bt: pd.DataFrame, qs=QS) -> float:
    """Pinball loss averaged over the quantiles, divided by the same scale as the scaled error."""
    tot = 0.0
    for q in qs:
        d = bt.y - bt[f"q{q}"]
        tot += float((np.where(d >= 0, q * d, (q - 1) * d) / bt["scale"]).mean())
    return tot / len(qs)


def coverage(bt: pd.DataFrame, level: int = 80) -> float:
    """Share of actual values inside the central interval of the given level (80 or 95)."""
    lo, hi = {80: ("q0.1", "q0.9"), 95: ("q0.025", "q0.975")}[level]
    return float(((bt.y >= bt[lo]) & (bt.y <= bt[hi])).mean())


PQS = (0.1, 0.5, 0.9)                    # the quantiles every method of Day 3 can deliver (see Lab 3.3)


def scores(bt: pd.DataFrame, qs=None) -> dict:
    """Scaled error and, when the table has quantile columns, scaled pinball loss and coverage.
    qs=U.PQS scores the three quantiles 0.1, 0.5, 0.9 and the 80% interval only."""
    out = {"mase": round(mase(bt), 3)}
    if "q0.1" in bt:
        five = qs is None and "q0.025" in bt
        out.update(pinball=round(scaled_pinball(bt, QS if five else (qs or PQS)), 3), cover80=round(coverage(bt, 80), 3))
        if five:
            out["cover95"] = round(coverage(bt, 95), 3)
    return out


# ------------------------------------------------------------------ statistical models (statsforecast)
def sf(model, train: pd.Series, h: int = H, X: pd.DataFrame | None = None, Xf: pd.DataFrame | None = None
       ) -> pd.DataFrame:
    """Fit one statsforecast model on one series and forecast h days with 80% and 95% intervals.

    X and Xf are the regressors over the training days and over the horizon (same columns).
    Returns yhat and the five quantile columns.
    """
    from statsforecast import StatsForecast
    df = pd.DataFrame({"unique_id": "s", "ds": train.index, "y": train.to_numpy()})
    kw = {}
    if X is not None:
        df = pd.concat([df, X.reset_index(drop=True)], axis=1)
        kw["X_df"] = pd.concat([pd.DataFrame({"unique_id": "s", "ds": Xf.index}), Xf.reset_index(drop=True)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = StatsForecast(models=[model], freq="D").forecast(df=df, h=h, level=[80, 95], **kw)
    a = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
    out = pd.DataFrame({"yhat": fc[a].to_numpy()})
    if f"{a}-lo-80" in fc:
        out["q0.025"], out["q0.1"], out["q0.5"] = fc[f"{a}-lo-95"].to_numpy(), fc[f"{a}-lo-80"].to_numpy(), out.yhat
        out["q0.9"], out["q0.975"] = fc[f"{a}-hi-80"].to_numpy(), fc[f"{a}-hi-95"].to_numpy()
    return out


def sf_many(model, wide_train: pd.DataFrame, h: int = H, quantiles: bool = False, log: bool = False):
    """Fit the same statsforecast model on every column of a wide table.

    Returns the point forecasts as a wide table (h rows, one column per series). With quantiles=True returns
    (points, {"q0.025": wide, ..., "q0.975": wide}). With log=True the model is fitted on the logarithm and
    everything is turned back with the exponential.
    """
    from statsforecast import StatsForecast
    cols = list(wide_train.columns)
    src = np.log(wide_train) if log else wide_train
    long = src.reset_index(names="ds").melt("ds", var_name="unique_id", value_name="y").dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = StatsForecast(models=[model], freq="D", n_jobs=1).forecast(df=long, h=h, level=[80, 95] if quantiles else None)
    a = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
    back = np.exp if log else (lambda t: t)
    wide = lambda c: back(fc.pivot(index="ds", columns="unique_id", values=c)[cols])
    points = wide(a)
    if not quantiles:
        return points
    return points, {"q0.025": wide(f"{a}-lo-95"), "q0.1": wide(f"{a}-lo-80"), "q0.5": points,
                    "q0.9": wide(f"{a}-hi-80"), "q0.975": wide(f"{a}-hi-95")}


def dynreg(rid: str, train: pd.Series, h: int, origin) -> pd.DataFrame:
    """The regression with ARIMA errors of Lab 2.3 (weather known at the origin, plus the calendar) for a
    series of region `rid`: forecast and quantiles for the h days after the origin."""
    from statsforecast.models import ARIMA
    future = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=h, freq="D")
    model = ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7)
    return sf(model, train, h, exog(rid, train.index), exog(rid, future, origin=origin))


# ------------------------------------------------------------------ regressors
def temp_known(rid: str, origin, dates) -> pd.Series:
    """The temperature a forecaster can use for future days: the normal for the day, plus the anomaly seen
    on the origin day fading by a fifth each day. Never the temperature that was measured later."""
    r = region(rid)
    origin = pd.Timestamp(origin)
    anomaly = r.temp_c[origin] - r.temp_normal[origin]
    doy_normal = r.temp_normal.reindex(dates)
    if doy_normal.isna().any():                      # days after the data ends: last year's normal
        last_year = r.temp_normal.reindex(pd.DatetimeIndex(dates) - pd.Timedelta(days=364)).to_numpy()
        doy_normal = pd.Series(np.where(doy_normal.isna(), last_year, doy_normal), index=dates)
    hsteps = (pd.DatetimeIndex(dates) - origin).days
    return doy_normal + anomaly * 0.8 ** np.clip(hsteps, 0, None)


def exog(rid: str, dates, origin=None, oracle: bool = False, holidays: bool = True, weather: bool = True
         ) -> pd.DataFrame:
    """Regressors for the given days: cooling and heating degrees, Ramadan, Eid, national days.

    For past days the measured temperature is used. For days after `origin` the temperature is temp_known
    (honest) unless oracle=True, which uses the temperature measured later: a leak, kept for comparison.
    """
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=dates)
    if weather:
        r = region(rid)
        t = r.temp_c.reindex(dates)
        if origin is not None and not oracle:
            fut = dates > pd.Timestamp(origin)
            t = t.where(~fut, temp_known(rid, origin, dates))
        out["cdd"] = np.clip(t - 22, 0, None)
        out["hdd"] = np.clip(16 - t, 0, None)
    if holidays:
        c = calendar().reindex(dates)
        out["ramadan"] = c.ramadan.astype(float).to_numpy()
        out["eid"] = c.eid_weight.astype(float).to_numpy()
        out["national"] = (c.national_day + c.founding_day).astype(float).to_numpy()
    return out


# ------------------------------------------------------------------ a frame for one model across many series
FEATURES = ["h", "dow", "weekend", "ramadan", "eid", "national", "last7", "dow_profile", "year_ago",
            "cdd_change", "kind", "region"]


def make_frame(wide: pd.DataFrame, origin_list, h: int = H, min_history: int = 28) -> pd.DataFrame:
    """One row per feeder, origin and horizon day, for a model trained across series.

    Every feature uses only what is known on the origin day: the last 28 days of the series (its level, its
    last week, its weekday profile), the same day a year earlier, the calendar of the target day and the
    temperature known at the origin. The target is demand divided by the level, so large and small feeders
    look alike. `y` is NaN when the target day is after the data ends.
    """
    mt = meta().set_index("feeder").loc[list(wide.columns)]
    cal = calendar()
    kinds = {"residential": 0, "commercial": 1, "industrial": 2}
    Y = wide.to_numpy()
    idx = wide.index
    pos = {d: i for i, d in enumerate(idx)}
    reg_ids = sorted(mt.region.unique())
    temps = {r: region(r) for r in reg_ids}
    parts = []
    for o in origin_list:
        o = pd.Timestamp(o)
        i = pos[o]
        tdates = pd.date_range(o + pd.Timedelta(days=1), periods=h, freq="D")
        c = cal.reindex(tdates)
        block = Y[i - 27: i + 1]                                   # the last 28 days, per feeder
        ok = (~np.isnan(block)).sum(axis=0) >= min_history
        level = np.nanmean(block, axis=0)
        sc = feeder_scale(wide, o).to_numpy()
        last7 = block[-7:].mean(axis=0) / level
        prof = block.reshape(4, 7, -1).mean(axis=0) / level        # weekday profile; row k is day o-6+k
        prof_dow = np.array([(o - pd.Timedelta(days=6 - k)).dayofweek for k in range(7)])
        for r in reg_ids:
            cols = np.where((mt.region.to_numpy() == r) & ok)[0]
            if not len(cols):
                continue
            tr = temps[r]
            cdd_known = np.clip(temp_known(r, o, tdates).to_numpy() - 22, 0, None)
            cdd_recent = np.clip(tr.temp_c.loc[o - pd.Timedelta(days=27): o].to_numpy() - 22, 0, None).mean()
            for k, td in enumerate(tdates):
                j = i + k + 1
                y = Y[j, cols] / level[cols] if j < len(Y) else np.full(len(cols), np.nan)
                if j - 364 >= 0 and i - 364 - 27 >= 0:
                    ya = Y[j - 364, cols] / np.nanmean(Y[i - 364 - 27: i - 364 + 1, cols], axis=0)
                else:
                    ya = np.full(len(cols), np.nan)
                p = prof[np.where(prof_dow == td.dayofweek)[0][0], cols]
                parts.append(pd.DataFrame({
                    "feeder": wide.columns[cols], "origin": o, "date": td, "h": k + 1,
                    "dow": td.dayofweek, "weekend": int(c.weekend.iloc[k]), "ramadan": float(c.ramadan.iloc[k]),
                    "eid": float(c.eid_weight.iloc[k]), "national": float(c.national_day.iloc[k] + c.founding_day.iloc[k]),
                    "last7": last7[cols], "dow_profile": p, "year_ago": ya,
                    "cdd_change": cdd_known[k] - cdd_recent,
                    "kind": mt.kind.map(kinds).to_numpy()[cols], "region": int(r[1:]),
                    "level": level[cols], "scale": sc[cols], "y": y}))
    return pd.concat(parts, ignore_index=True)


def feeder_scale(wide: pd.DataFrame, origin, m: int = M, window: int = 364) -> pd.Series:
    """The scaled-error denominator of every feeder: in-sample weekly seasonal-naive error up to the origin."""
    tr = wide.loc[:origin].iloc[-window:]
    return (tr - tr.shift(m)).abs().mean()


def score_wide(actual: pd.DataFrame, fc: pd.DataFrame, sc: pd.Series) -> float:
    """Mean scaled error over every feeder and day of one fold (wide tables with the same shape)."""
    return float(((actual - fc.to_numpy()).abs() / sc).mean().mean())


FEEDER_ORIGINS = origins(2)              # the eight held-out weeks of the feeders: two folds of 28 days
LGB = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1,
           feature_fraction=0.8, verbose=-1, seed=304, num_threads=4)   # settings of the boosted model


def established() -> list:
    """The 388 feeders commissioned before 2025: three years of history or more."""
    m = meta()
    return list(m.feeder[m.commissioned < "2025-01-01"])


def new_feeders() -> list:
    """The 12 feeders commissioned on 2025-09-11: eight weeks of history at the first held-out origin."""
    m = meta()
    return list(m.feeder[m.commissioned >= "2025-01-01"])


def feeder_floor(wide: pd.DataFrame, origin, h: int = H) -> pd.DataFrame:
    """The floor for every feeder: the same weekdays 52 weeks before each of the h days after the origin."""
    days = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=h, freq="D")
    out = wide.reindex(days - pd.Timedelta(days=364))
    out.index = days
    return out


def feeder_long(wide: pd.DataFrame, origin, yhat, quantiles: dict | None = None, h: int = H) -> pd.DataFrame:
    """One fold of feeder forecasts as a long table with the columns of a regional backtest plus `feeder`:
    feeder, origin, date, h, y, yhat, the quantile columns when given, scale. `yhat` and each quantile are
    h rows by one column per feeder (a wide table or an array, columns in the order of `wide`).
    U.mase, U.scores, U.coverage and U.by_week work on the result."""
    origin = pd.Timestamp(origin)
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
    n = wide.shape[1]
    out = pd.DataFrame({"feeder": np.tile(wide.columns, h), "origin": origin, "date": np.repeat(days, n),
                        "h": np.repeat(np.arange(1, h + 1), n), "y": wide.reindex(days).to_numpy().ravel(),
                        "yhat": np.asarray(yhat, float).ravel()})
    for q, table in (quantiles or {}).items():
        out[q] = np.asarray(table, float).ravel()
    out["scale"] = np.tile(feeder_scale(wide, origin).to_numpy(), h)
    return out


# ------------------------------------------------------------------ a pretrained forecaster, or its stand-in
PRETRAINED = "amazon/chronos-bolt-tiny"


def pretrained(contexts: list, h: int = H, qs=PQS) -> tuple:
    """Zero-shot quantile forecasts for a list of series (each a 1-D array of history).

    Uses the pretrained forecaster named in PRETRAINED when the package `chronos-forecasting` is installed
    and SHAMS_QUICK is not set (Colab: the setup cell of Lab 3.3 installs it and the weights download into
    the Colab runtime). Otherwise a STAND-IN answers, so the notebook runs anywhere: it repeats each series'
    weekday profile and widens with the horizon. The stand-in is not a pretrained model, and the info it
    returns says so. Returns (array of shape [series, quantiles, h], info).

    The default quantiles are 0.1, 0.5 and 0.9: this model family was trained on the quantiles 0.1 to 0.9,
    and its package returns the 0.1 and 0.9 values for anything further out.
    """
    t0 = time.perf_counter()
    if not QUICK and os.getenv("SHAMS_PRETRAINED", "on") != "off":
        try:
            import torch
            from chronos import BaseChronosPipeline
            pipe = _cache.get("pipe") or BaseChronosPipeline.from_pretrained(PRETRAINED, device_map="cpu")
            _cache["pipe"] = pipe
            out = []
            for a in range(0, len(contexts), 64):
                ctx = [torch.tensor(np.asarray(c, dtype="float32")[-2048:]) for c in contexts[a: a + 64]]
                q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=list(qs))
                out.append(q.numpy().transpose(0, 2, 1))
            return np.concatenate(out), {"model": PRETRAINED, "live": True, "seconds": time.perf_counter() - t0}
        except ImportError:
            pass
    z = {0.025: -1.96, 0.1: -1.2816, 0.5: 0.0, 0.9: 1.2816, 0.975: 1.96}
    out = np.empty((len(contexts), len(qs), h))
    for n, c in enumerate(contexts):
        c = np.asarray(c, float)
        c = c[~np.isnan(c)]
        k = min(len(c) // 7, 4) * 7
        med = np.resize(c[-k:].reshape(-1, 7).mean(axis=0), h)
        sd = np.std(c[-k:] - np.resize(c[-k:].reshape(-1, 7).mean(axis=0), k)) * np.sqrt(1 + np.arange(h) / 7)
        for j, q in enumerate(qs):
            out[n, j] = med + z[q] * sd
    return out, {"model": "STAND-IN (weekday profile, not a pretrained model)", "live": False,
                 "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------ plots
def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (10, 3.6), "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})
    return plt


def shade_events(ax, start, end) -> None:
    """Shade Ramadan (light) and the two Eids (darker) on a time axis."""
    c = calendar().loc[start:end]
    for col, colour, alpha in (("ramadan", "#7b61ff", 0.10), ("eid_fitr", "#e4572e", 0.30), ("eid_adha", "#e4572e", 0.30)):
        on = c[col].astype(int)
        edges = on.diff().fillna(on.iloc[0])
        starts, ends = list(c.index[edges == 1]), list(c.index[edges == -1])
        if len(ends) < len(starts):
            ends.append(c.index[-1])
        for a, b in zip(starts, ends):
            ax.axvspan(a, b, color=colour, alpha=alpha, lw=0)


def fan(ax, history: pd.Series, fc: pd.DataFrame, title: str = "") -> None:
    """A fan chart: the history, the median, and the 80% and 95% bands (fc has date and the quantile columns)."""
    ax.plot(history.index, history.to_numpy(), color="#16324f", lw=1.2, label="measured")
    if "q0.025" in fc:
        ax.fill_between(fc.date, fc["q0.025"], fc["q0.975"], color="#00a19a", alpha=0.18, lw=0, label="95 in 100")
    ax.fill_between(fc.date, fc["q0.1"], fc["q0.9"], color="#00a19a", alpha=0.35, lw=0, label="80 in 100")
    ax.plot(fc.date, fc["q0.5"], color="#00776f", lw=1.6, label="median forecast")
    if "y" in fc and fc.y.notna().any():
        ax.plot(fc.date, fc.y, color="#e4572e", lw=1.0, ls="--", label="what happened")
    ax.set_title(title, loc="left")
    ax.set_ylabel("MWh per day")
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
