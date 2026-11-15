# Capstone · A forecasting analysis, defended to a planner

The capstone is individual. You build a **forecasting analysis** on one of the two supplied datasets, or on a
public dataset whose terms permit your use. You publish it in a **public repository**. Its README reports point
and probabilistic accuracy for three method families against a floor. You write a **two-page planner's note**
with fan charts, and you defend the work in an **eight-minute live defence** followed by four minutes of
questions, in which the assessor moves the forecast origin or the horizon and asks what changes.

## The two supplied datasets

Both are synthetic and belong to the course's case, Shams Power. They are written by `labs/prepare_data.py`
with the course's fixed seed.

| File | Series | Columns | History |
|---|---|---|---|
| `data/ev_hubs.csv` | 36 electric-vehicle charging hubs | `date`, `hub`, `region`, `energy_mwh`, `temp_c` | daily; the oldest hub opened on 1 July 2023, the newest has 226 days |
| `data/cooling_plants.csv` | 8 district-cooling plants | `date`, `plant`, `region`, `cooling_mwh`, `temp_c` | daily, 1 January 2021 to 31 December 2025 |

The calendar of the course, `labs/data/calendar.csv`, applies to both.

### The planner's request: charging hubs

> **Hamad Al-Dawsari, network operations.** "We opened 36 charging hubs in two and a half years and we keep
> opening them. Every second Thursday I reserve feeder capacity for each hub for the 14 days that follow. A
> hub that draws more than I reserved trips a limit; capacity I reserve and do not use is refused to someone
> else. Some hubs are a few months old. What will each hub draw, and how sure are you?"

What makes it interesting: series of very different lengths, growth after opening, a weekly pattern unlike a
feeder's, busy holidays, and no "same weekday last year" for the youngest hubs.

### The planner's request: district-cooling plants

> **Shahad Al-Tamimi, load planning.** "Eight district-cooling plants send us their electricity demand
> a month ahead. I plan the summer reserve on their upper range, not on their average. Give me each plant's
> daily energy for the next 28 days with a range I can plan on, and tell me how often that range held in the
> past."

What makes it interesting: demand driven by temperature in a way that is not a straight line, five years of
history, slow growth, and one plant with three weeks that should not be learned from.

### A public dataset

Allowed. State in your README where it comes from, its licence or open-data terms, the date you downloaded
it, and why those terms permit your use. Do not use personal data or internal data of your entity.

## Minimum requirements

1. **Framing and floor.** The decision the forecast serves, the horizon, the cost of an error in each
   direction, and a floor: the best naive method for your data, scored by rolling-origin backtesting.
2. **Three method families** in the same backtest as the floor: for example a statistical model per series,
   one boosted model across series, and a pretrained forecaster, a pooled forecast or a combination. Three
   variants of one family count as one.
3. **Backtest design.** Several origins; nothing after an origin reaches the forecast made at it; any choice
   of model or setting made on folds before the test folds. Say how you checked.
4. **Uncertainty and communication.** Quantile forecasts, pinball loss and measured coverage for every family
   that gives intervals; fan charts; a two-page note with the six parts of Lab 4.4, limits included.
5. **Reproducibility.** The repository runs from a clean copy with one command. Every number in the README is
   backed by a committed file or a run log. Licences of the software, the weights and the data are stated.

## What to hand in

A link to your public GitHub repository. It holds:

- the analysis (notebooks or scripts) and the command that runs it;
- a `README.md` with the table of results: scaled error, pinball loss, coverage, run time and licence for the
  floor and the three families, as in Lab 3.4;
- the run logs or result files behind that table;
- the two-page planner's note with its fan charts (PDF or Markdown).

## How it is assessed

| Criterion | Weight | Full marks look like |
|---|---|---|
| Framing and floor | 15% | the decision, horizon and costs are stated; the floor is the strongest naive method for the data, and every result is read against it |
| Methods | 25% | three families that differ in what they can see; each choice is argued from the structure of the data |
| Backtest design | 25% | rolling origins; a written argument, or a test, that nothing after the origin is used; no selection on the test folds |
| Uncertainty and communication | 25% | coverage measured and stated in counts; a note a planner can act on, with its limits |
| Reproducibility | 10% | runs from a clean copy; numbers traceable to files; licences stated |

## The defence

Eight minutes: the decision and the floor, what you built, what you measured, the note. Then four minutes of
questions. Expect one of these:

- "Move the origin to the day before the last holiday. What does your forecast do, and what does the note say?"
- "The planner now needs twice the horizon. What changes in the features, the floor and the bands?"
- "This series is new and has six weeks of history. Which of your three families still works, and how do you know?"

A backtest written as a function of the origin and the horizon lets you answer by running it.

## Suggested path

1. Plot before you model: Lab 1.2 on your data.
2. Write the backtest and score the floor: Lab 1.4.
3. Add one family at a time and keep every score: Labs 2.1 to 3.3.
4. Review your own backtest for the three faults of Lab 4.1.
5. Build the intervals from backtest errors and measure them: Labs 2.4 and 4.4.
6. Fill the scorecard and write the choice: Lab 3.4.
7. Write the note last, for the planner whose request you answered.
