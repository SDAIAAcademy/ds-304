# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Generate the Shams Power case study: synthetic daily electricity demand.

    uv run prepare_data.py

Shams Power is a fictional distributor with 12 regions and 400 feeders. Everything here is synthetic and
deterministic (seed 304): the same bytes on every run. Files written into data/:

    calendar.csv       one row per day, 2017-01-01 to 2026-03-31: weekend, Ramadan, the two Eids, national days
    regions.csv        one row per region and day, 2017-01-01 to 2025-12-31: demand, temperature, its normal
    feeders_meta.csv   one row per feeder: region, kind, the day it was commissioned
    feeders.npz        the 400 feeder series (days x feeders, MWh x 100 as integers; 0 before commissioning)
    requests.json      three planner requests (Lab 1.1)
    capstone/          two more datasets from the same generator, for the capstone

A regional series is the exact sum of its feeders. The planted problems are listed in
solutions/answer_key.md.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SEED = 304
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
START, END, CAL_END = date(2017, 1, 1), date(2025, 12, 31), date(2026, 3, 31)

# The first day of Ramadan, Eid al-Fitr and Eid al-Adha as announced in Saudi Arabia, 2017 to 2026.
# The 2026 dates are the calendar's computed dates. Check against the Umm al-Qura calendar before reuse.
HIJRI = {
    2017: ("2017-05-27", "2017-06-25", "2017-09-01"),
    2018: ("2018-05-17", "2018-06-15", "2018-08-21"),
    2019: ("2019-05-06", "2019-06-04", "2019-08-11"),
    2020: ("2020-04-24", "2020-05-24", "2020-07-31"),
    2021: ("2021-04-13", "2021-05-13", "2021-07-20"),
    2022: ("2022-04-02", "2022-05-02", "2022-07-09"),
    2023: ("2023-03-23", "2023-04-21", "2023-06-28"),
    2024: ("2024-03-11", "2024-04-10", "2024-06-16"),
    2025: ("2025-03-01", "2025-03-30", "2025-06-06"),
    2026: ("2026-02-18", "2026-03-20", "2026-05-27"),
}

# id, name, climate mean (deg C), annual amplitude, number of feeders, share of each kind (res, com, ind)
REGIONS = [
    ("R01", "Al-Rimal", 27.5, 11.0, 46, (0.60, 0.28, 0.12)),
    ("R02", "Al-Sahil", 28.5, 6.0, 40, (0.50, 0.35, 0.15)),
    ("R03", "Al-Jabal", 18.5, 7.0, 24, (0.70, 0.22, 0.08)),
    ("R04", "Al-Wadi", 26.0, 10.0, 34, (0.62, 0.26, 0.12)),
    ("R05", "Al-Dana", 27.5, 8.5, 38, (0.45, 0.35, 0.20)),
    ("R06", "Al-Ghadeer", 25.5, 10.5, 30, (0.66, 0.24, 0.10)),
    ("R07", "Al-Safa", 29.0, 6.5, 36, (0.55, 0.33, 0.12)),
    ("R08", "Al-Badia", 26.5, 12.0, 22, (0.72, 0.20, 0.08)),
    ("R09", "Al-Marjan", 28.0, 7.0, 32, (0.48, 0.40, 0.12)),
    ("R10", "Al-Qimma", 19.5, 7.5, 20, (0.74, 0.20, 0.06)),
    ("R11", "Al-Mina", 27.0, 8.0, 44, (0.35, 0.25, 0.40)),
    ("R12", "Al-Hazm", 26.0, 11.5, 34, (0.64, 0.26, 0.10)),
]
KINDS = ("residential", "commercial", "industrial")
# per kind: cooling response per 10 degrees above 22, heating response per 10 degrees below 16,
# Friday and Saturday factors, Ramadan factor, Eid factor on the deepest day, noise standard deviation
KIND = {
    "residential": dict(cool=0.95, heat=0.25, fri=0.97, sat=0.99, ram=1.05, eid=0.90, sd=0.050),
    "commercial": dict(cool=0.60, heat=0.10, fri=0.78, sat=0.90, ram=0.93, eid=0.66, sd=0.060),
    "industrial": dict(cool=0.15, heat=0.02, fri=0.74, sat=0.86, ram=0.95, eid=0.58, sd=0.075),
}
EID_SHAPE_FITR = [0.35, 1.0, 0.9, 0.7, 0.4]      # the eve, then four days
EID_SHAPE_ADHA = [0.5, 1.0, 0.9, 0.75, 0.5, 0.2]  # Arafah day, then five days


def days(a: date, b: date) -> list[date]:
    return [a + timedelta(n) for n in range((b - a).days + 1)]


def build_calendar() -> list[dict]:
    rows = []
    ram, fitr, adha = {}, {}, {}
    for y, (r, f, a) in HIJRI.items():
        r, f, a = date.fromisoformat(r), date.fromisoformat(f), date.fromisoformat(a)
        for n in range((f - r).days):
            ram[r + timedelta(n)] = 1
        for k, w in enumerate(EID_SHAPE_FITR):
            fitr[f + timedelta(k - 1)] = w
        for k, w in enumerate(EID_SHAPE_ADHA):
            adha[a + timedelta(k - 1)] = w
    for d in days(START, CAL_END):
        nat = 1 if (d.month, d.day) == (9, 23) else 0
        found = 1 if (d.month, d.day) == (2, 22) and d.year >= 2022 else 0
        event = ("eid_fitr" if d in fitr else "eid_adha" if d in adha else "ramadan" if d in ram
                 else "national_day" if nat else "founding_day" if found else "")
        rows.append(dict(date=d.isoformat(), dow=d.strftime("%a"), weekend=int(d.weekday() in (4, 5)),
                         ramadan=ram.get(d, 0), eid_fitr=int(d in fitr), eid_adha=int(d in adha),
                         eid_weight=max(fitr.get(d, 0.0), adha.get(d, 0.0)),
                         national_day=nat, founding_day=found, event=event))
    return rows


def ar1(rng, n, phi, sd):
    e = rng.normal(0, sd * np.sqrt(1 - phi * phi), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def temperature(rng, n, mean, amp):
    doy = np.array([d.timetuple().tm_yday for d in days(START, END)])
    normal = mean + amp * np.cos(2 * np.pi * (doy - 205) / 365.25)
    warm = 0.04 * np.arange(n) / 365.25           # a slow warming, part of the normal
    anomaly = ar1(rng, n, 0.80, 1.9)
    return normal + warm + anomaly, normal + warm


def series_block(rng, cal, temp, kinds, start_idx, base, growth, reg_shock):
    """Daily MWh for a set of feeders of one region. Returns an array (days, feeders)."""
    n, m = len(temp), len(kinds)
    t = np.arange(n)
    dow = np.array([date.fromisoformat(r["date"]).weekday() for r in cal[:n]])
    ram = np.array([r["ramadan"] for r in cal[:n]], float)
    eid = np.array([r["eid_weight"] for r in cal[:n]], float)
    nat = np.array([r["national_day"] + r["founding_day"] for r in cal[:n]], float)
    felt = 0.7 * temp + 0.3 * np.concatenate([[temp[0]], temp[:-1]])  # buildings keep yesterday's heat
    cdd, hdd = np.clip(felt - 22, 0, None) / 10, np.clip(16 - felt, 0, None) / 10
    out = np.zeros((n, m))
    for j, kind in enumerate(kinds):
        k = KIND[kind]
        cool = k["cool"] * rng.uniform(0.8, 1.2)
        level = base[j] * (1 + growth[j]) ** (t / 365.25)
        weather = 1 + cool * cdd + k["heat"] * hdd
        week = np.where(dow == 4, k["fri"], np.where(dow == 5, k["sat"], 1.0))
        hol = (1 + (k["ram"] - 1) * ram) * (1 - (1 - k["eid"]) * eid) * (1 - 0.5 * (1 - k["eid"]) * nat)
        noise = np.exp(ar1(rng, n, 0.55, k["sd"]) + reg_shock)
        y = level * weather * week * hol * noise
        y[: start_idx[j]] = 0.0
        out[:, j] = y
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    cal = build_calendar()
    dts = days(START, END)
    n = len(dts)
    idx = {d: i for i, d in enumerate(dts)}

    meta, blocks, reg_rows = [], [], []
    fid = 0
    for rid, name, mean, amp, nf, shares in REGIONS:
        temp, normal = temperature(rng, n, mean, amp)
        kinds = list(rng.choice(KINDS, size=nf, p=shares))
        base = np.clip(rng.lognormal(np.log(38), 0.45, nf), 12, 110)
        base = np.where(np.array(kinds) == "industrial", base * 1.5, base)
        growth = rng.normal(0.025, 0.012, nf)
        start = np.zeros(nf, int)
        if rid == "R11":                      # planted: six industrial feeders join on 2022-07-01
            for j in range(nf - 6, nf):
                kinds[j], start[j], base[j] = "industrial", idx[date(2022, 7, 1)], 150.0 + 10 * (j % 3)
        new = []                              # planted: one new feeder per region on 2025-09-11
        j_new = int(rng.integers(0, nf - 6))
        start[j_new] = idx[date(2025, 9, 11)]
        new.append(j_new)
        shock = ar1(rng, n, 0.6, 0.016)
        y = series_block(rng, cal, temp, kinds, start, base, growth, shock)
        if rid == "R01":                      # planted: a sandstorm outage on 2023-03-14 and 15
            hit = rng.choice(nf, size=nf // 2, replace=False)
            for d, f in ((date(2023, 3, 14), 0.35), (date(2023, 3, 15), 0.60)):
                y[idx[d], hit] *= f
        y = np.round(np.clip(y, 0, 650) * 100).astype(np.uint16)
        blocks.append(y)
        for j in range(nf):
            fid += 1
            meta.append(dict(feeder=f"F{fid:03d}", region=rid, kind=kinds[j],
                             commissioned=dts[start[j]].isoformat()))
        total = y.astype(np.int64).sum(axis=1) / 100
        for i, d in enumerate(dts):
            reg_rows.append((d.isoformat(), rid, name, f"{total[i]:.2f}", f"{temp[i]:.1f}", f"{normal[i]:.1f}"))

    feeders = np.concatenate(blocks, axis=1)
    assert feeders.shape == (n, 400), feeders.shape

    with open(DATA / "calendar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cal[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(cal)
    with open(DATA / "regions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "region", "name", "demand_mwh", "temp_c", "temp_normal"])
        w.writerows(reg_rows)
    with open(DATA / "feeders_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(meta)
    np.savez_compressed(DATA / "feeders.npz", mwh_x100=feeders,
                        feeders=np.array([m["feeder"] for m in meta]), start=np.array([START.isoformat()]))

    requests = [
        dict(id="A", who="Omar Al-Khuzaim, energy purchasing lead",
             text="Every Thursday I nominate the energy we buy for each of the next 28 days, per region. If we are "
                  "short on a day I buy the gap on the spot market at about three times the contract price. If we "
                  "are long, I resell the surplus at about half price. What will demand be?",
             cost_under=2.0, cost_over=0.5, data_delay_days=2),
        dict(id="B", who="Hamad Al-Dawsari, network operations engineer",
             text="I can take a feeder out for maintenance next week only on a day when its neighbours can carry "
                  "its load. For feeder F017, how high could daily demand go on each of the next 7 days?",
             cost_under=9.0, cost_over=1.0, data_delay_days=1),
        dict(id="C", who="Shahad Al-Tamimi, load planning manager",
             text="The board decides in March on new transformer capacity for Al-Rimal. How high will the peak "
                  "summer day be in each of the next three years?",
             cost_under=4.0, cost_over=1.0, data_delay_days=30),
    ]
    (DATA / "requests.json").write_text(json.dumps(requests, indent=1) + "\n", encoding="utf-8")

    # Capstone datasets: new series from the same world, never used in a lab.
    cap = HERE / "capstone" / "data"
    cap.mkdir(parents=True, exist_ok=True)
    crng = np.random.default_rng(SEED + 1)
    c_start = date(2023, 7, 1)
    cdts = days(c_start, END)
    off = idx[c_start]
    ccal = cal[off:]
    rows = []
    for h in range(36):                       # electric-vehicle charging hubs
        rid, name, mean, amp, _, _ = REGIONS[h % 12]
        temp, _ = temperature(crng, n, mean, amp)
        temp = temp[off:]
        m = len(cdts)
        open_at = int(crng.integers(0, m - 200)) if h % 3 else 0
        t = np.arange(m)
        adoption = 1 / (1 + np.exp(-(t - open_at - 240) / 110))          # an S-curve of adoption
        dow = np.array([d.weekday() for d in cdts])
        week = np.where(dow == 4, 1.25, np.where(dow == 5, 1.12, np.where(dow == 3, 1.10, 1.0)))
        eidw = np.array([r["eid_weight"] for r in ccal[:m]])
        ramw = np.array([r["ramadan"] for r in ccal[:m]])
        y = (crng.uniform(2.5, 9.0) * adoption * week * (1 + 0.55 * eidw) * (1 - 0.12 * ramw)
             * (1 + 0.012 * np.clip(temp - 30, 0, None)) * np.exp(ar1(crng, m, 0.4, 0.13)))
        y[:open_at] = np.nan
        for i, d in enumerate(cdts):
            if not np.isnan(y[i]):
                rows.append((d.isoformat(), f"H{h + 1:02d}", rid, f"{y[i]:.3f}", f"{temp[i]:.1f}"))
    with open(cap / "ev_hubs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "hub", "region", "energy_mwh", "temp_c"])
        w.writerows(rows)

    rows = []
    c2 = date(2021, 1, 1)
    off2 = idx[c2]
    for p in range(8):                        # district-cooling plants
        rid, name, mean, amp, _, _ = REGIONS[(p * 5) % 12]
        temp, _ = temperature(crng, n, mean, amp)
        temp = temp[off2:]
        m = len(temp)
        cap_mw = crng.uniform(40, 120)
        load = cap_mw * 24 * 0.011 * np.clip(temp - 17, 0.5, None) ** 1.25 / 10
        dow = np.array([d.weekday() for d in days(c2, END)])
        load = load * np.where(dow == 4, 0.93, 1.0) * (1 + 0.03 * np.arange(m) / 365.25)
        load = load * np.exp(ar1(crng, m, 0.5, 0.045))
        if p == 5:                            # a chiller refit: three weeks at 60%
            a = (date(2024, 2, 4) - c2).days
            load[a:a + 21] *= 0.6
        for i, d in enumerate(days(c2, END)):
            rows.append((d.isoformat(), f"P{p + 1}", rid, f"{load[i]:.2f}", f"{temp[i]:.1f}"))
    with open(cap / "cooling_plants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "plant", "region", "cooling_mwh", "temp_c"])
        w.writerows(rows)

    print(f"calendar.csv {len(cal)} days , regions.csv {len(reg_rows)} rows, feeders.npz {feeders.shape} "
          f", capstone: ev_hubs.csv, cooling_plants.csv")


if __name__ == "__main__":
    main()
