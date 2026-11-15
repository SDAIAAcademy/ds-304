# Trainer's answer key

What is planted in the data, which lab finds it, and the numbers the solution notebooks print in a full run
(not in quick mode, which uses fewer folds and regions). All data is synthetic, written by
`labs/prepare_data.py` with seed 304. A number that differs in the second or third decimal on another
machine is normal; a number that differs in the first decimal is a fault to look into.

The frame of every backtest: a 28-day horizon; for the regions, 13 origins 28 days apart over 2025 with a
training window of 1,096 days; for the feeders, two origins (5 November and 3 December 2025). The scaled error
divides by the in-sample error of "the same weekday last week".

## What is planted, and where it is found

| Planted | Where | Found in |
|---|---|---|
| Eid dips that move about 11 days earlier each year | every region; Eid al-Fitr days fall 8% to 28% below the same weekdays around them | Labs 1.2, 1.3, 1.4, 2.3 |
| A two-day outage | Al-Rimal, 14 and 15 March 2023, about 30% below its neighbours | Lab 1.2 |
| A level shift | Al-Mina (R11), when six industrial feeders join: about +52% | Lab 1.3 |
| Stale data | requests with a delay of 1 to 30 days | Lab 1.1 |
| Intervals that are too narrow | the regression's own 80% interval covers 63.5% | Labs 2.4, 4.3 |
| Twelve new feeders | one per region, commissioned 11 September 2025 | Lab 4.2 |
| Three leaks in a backtest | a centred average, measured temperature, a window chosen per test fold | Lab 4.1 |
| A chiller refit | capstone, cooling plant P6, three weeks at 60% from 4 February 2024 | capstone |

## Day 1

| Lab | Expected results |
|---|---|
| 1.1 | Quantile to forecast: request A 0.8, B 0.9, C 0.8. Error grows with the age of the data: 4.3%, 6.2%, 8.0%, 15.0%, 16.5%. |
| 1.2 | Growth about 2.7% a year; July is 2.3 times January; correlation of demand with cooling degrees 0.97; Friday 0.936 and Saturday 0.984 of the weekly mean; outage on 14 March 2023 at -30.5%. |
| 1.3 | Strengths for Al-Rimal: trend 0.50, annual season 0.96, weekly season 0.16. Mean Eid remainder -0.137 on the log scale. Al-Mina level shift +51.8%. The decomposition of 12 regions takes about a minute. |
| 1.4 | Naive 1.642; seasonal naive, m = 7: 1.616; same weekday last year (m = 364), the floor: 1.061. By week of the horizon: 0.92, 0.96, 1.08, 1.28. On Eid days the floor scores 2.17 against 1.03 elsewhere. |

## Day 2

| Lab | Expected results |
|---|---|
| 2.1 | Best alpha 1.0 on the grid. ETS with a weekly season 1.587; decompose, then ETS 0.984. Damped trend leaves week 4 unchanged. |
| 2.2 | Weekly seasonal ARIMA 1.617, Ljung-Box p = 0.0001. With annual Fourier terms 0.826, p = 0.623. Automatic order selection 0.822 against 0.850, in about 53 s. |
| 2.3 | Weather only 0.901; weather and calendar 0.766 (Eid days 0.796 against 1.339, floor 2.166). With the measured temperature (the oracle) 0.547: not a result. Coefficients: 147 MWh per cooling degree, Eid -599, national days -296. |
| 2.4 | Coverage of the own 80% and 95% intervals: decompose-then-ETS 83.5 and 93.1; ARIMA with Fourier 78.8 and 93.1; regression 63.5 and 76.9. Rebuilt from 2024 errors: 78.3 and 94.5; pinball 0.208 to 0.192; mean width 436 to 677 MWh. |

## Day 3

| Lab | Expected results |
|---|---|
| 3.1 | Al-Dana, 37 feeders. Gap between levels 3.0% and 2.4%. Feeders and region: base 0.722 and 0.463; bottom-up 0.722 and 0.634; top-down 0.650 and 0.463; least squares 0.651 and 0.465. Stretch: top-down lowers the feeder error in 8 regions of 12. |
| 3.2 | 315,056 rows; 287,896 training rows; 5,432 rows would leak with a split by origin. Floor 0.753 (0.878, 0.628); boosted model 0.622 (0.730, 0.515); skill 0.17; pinball 0.196; 80% coverage 72.1%; no crossing. Top features: cdd_change 0.50, year_ago 0.12, dow_profile 0.11. Without scaling: 3.749. |
| 3.3 | **No expected numbers are given for the pretrained model.** The course's own build ran this lab with the stand-in only. In Colab the first output must name `amazon/chronos-bolt-tiny`; three participants' rows should agree to the second decimal. The stand-in prints 1.17 and is labelled STAND-IN: it is not a model result. Licences on 15 November 2026: Chronos-Bolt, Chronos-2 and TimesFM 2.5 Apache-2.0; Moirai 2.0 CC BY-NC 4.0. |
| 3.4 | Skill: weekly ETS -0.19, decompose-then-ETS 0.04, boosted 0.17. Seconds per 1,000 forecasts on the build machine: 181, 1,884 and 63. Ranking for purchasing: boosted model first (0.83). Ranking for operations: the floor first (0.63), boosted second (0.57). Stretch: at an upkeep weight of 0.25 the boosted model takes first place for operations. The written choice is a judgment: accept any entry with numbers and a stated condition for changing it. |

## Day 4

| Lab | Expected results |
|---|---|
| 4.1 | As received 0.244. After fix 1 (feature shifted by 28 days) 0.414. After fix 2 (temperature known at the origin) 0.711. As reported with the best window per fold 0.651; one window chosen on 2024 (365 days) scores 0.683 on 2025. Leaking columns: cdd and smooth, then cdd, then none. |
| 4.2 | Last week repeated 1.103; ETS on 56 days 0.949; pooled shape 0.919; global boosted model 0.747 with 80% coverage of 60.4%. The pretrained row: from Colab only, as in Lab 3.3. Stretch, four weeks of history: ETS 0.846 (noise, and the lab says so). |
| 4.3 | Whole year: 0.984, 0.826, 0.766; average 0.793. Late folds: the method chosen on the early folds (the regression) 0.913, the average 0.914: a tie. Averaged quantiles: pinball 0.184, coverage 81.6% and 94.0%. Stretch weights 0.26, 0.34, 0.39 give 0.906. |
| 4.4 | January 2026: median between 2,257 and 2,554 MWh a day; upper edge of the dark band peaks at 3,025 MWh on 25 January; dark band 525 MWh wide in week 1 and 869 in week 4. Coverage sentence: 78 of 100 and 95 of 100. The note is the participant's own text: check the six parts, the two coverage numbers and at least two limits. |

## Checkpoint answers in the decks

| Deck | Review | Session 1 | Session 2 | Session 3 | Session 4 |
|---|---|---|---|---|---|
| Day 1 | none | C | A | D | B |
| Day 2 | B | D | A | C | B |
| Day 3 | C | A | D | B | C |
| Day 4 | B | C | A | D | B |

The answer and its reason are in the speaker notes of every checkpoint slide; the notes are the authority.
