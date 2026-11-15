# %% [markdown]
# # Lab 0 · Setup check
#
# Run this notebook **before Day 1** (about 3 minutes). It checks the packages, the course data and the
# supplied reference results, fits one small model, and trains one small boosted model.
#
# No lab of this course calls a language model and none needs an API key. Everything runs on a laptop
# processor or on a free Colab runtime. One lab, 3.3, loads a small open-weight forecaster into the Colab
# runtime; on a laptop that lab runs with a labelled stand-in.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-304.git"
    if not Path("/content/forecasting").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/forecasting"], check=True)
    os.chdir("/content/forecasting/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("shams.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import shams as U

plt = U.style()

# %% [markdown]
# ## 1 · Packages

# %%
import platform
from importlib.metadata import PackageNotFoundError, version

rows = []
for name in ["numpy", "pandas", "matplotlib", "statsmodels", "statsforecast", "lightgbm"]:
    try:
        rows.append((name, version(name)))
    except PackageNotFoundError:
        rows.append((name, "MISSING"))
print("python", platform.python_version(), "| Colab" if IN_COLAB else "| local")
print(pd.DataFrame(rows, columns=["package", "version"]).to_string(index=False))
U.check(all(v != "MISSING" for _, v in rows), "every package of the course is installed")

# %% [markdown]
# ## 2 · The data and the reference results

# %%
regions, cal, meta, W = U.regions(), U.calendar(), U.meta(), U.feeders()
print(f"regions.csv: {regions.region.nunique()} regions | calendar.csv: {len(cal)} days | "
      f"feeders.npz: {W.shape[1]} feeders, {W.index.min().date()} to {W.index.max().date()}")
U.check(regions.region.nunique() == 12 and W.shape[1] == 400, "12 regions and 400 feeders")
U.check(W.index.max() == U.LAST, "the data end on 31 December 2025")
needed = ["bt_floor.csv", "bt_stl_ets.csv", "bt_arima_fourier.csv", "bt_dynreg.csv", "bt_dynreg_2024.csv", "bt_windows.csv",
          "fd_stats.json", "d3_2.json", "d3_3_row.json"]
missing = [f for f in needed if not (U.REF / f).exists()]
print("reference results:", len(list(U.REF.glob("*"))), "files | missing:", missing or "none")
U.check(not missing, "the reference results that the labs start from are present")

# %% [markdown]
# ## 3 · One statistical model

# %%
import time

from statsforecast.models import AutoETS

y = U.region()["demand"]
t0 = time.perf_counter()
fc = U.sf(AutoETS(season_length=7, model="ANA"), y.iloc[-365:], 28)
print(f"ETS with a weekly season on one year of Al-Rimal: {time.perf_counter() - t0:.1f} s; first forecast day {fc.yhat.iloc[0]:,.0f} MWh")
U.check(len(fc) == 28 and fc.yhat.notna().all(), "statsforecast fits and forecasts 28 days")

# %% [markdown]
# ## 4 · One boosted model

# %%
import lightgbm as lgb

rng = np.random.default_rng(304)
X = rng.normal(size=(2000, 4))
target = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=2000)
model = lgb.train({**U.LGB, "objective": "l1"}, lgb.Dataset(X[:1500], target[:1500]), num_boost_round=50)
error = float(np.mean(np.abs(model.predict(X[1500:]) - target[1500:])))
print(f"LightGBM trained 50 trees; error on held-out rows {error:.2f}")
U.check(error < 0.6, "LightGBM trains on the processor")

# %% [markdown]
# ## 5 · A plot

# %%
fig, ax = plt.subplots(figsize=(9, 3))
y.iloc[-365:].plot(ax=ax)
ax.set_title("Al-Rimal, daily demand in 2025, MWh")
plt.show()
print("Setup complete. Open Lab 1.1.")
