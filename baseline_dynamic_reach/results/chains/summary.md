# Step 9: consecutive goals (chained goals, evaluation only)

Model O3_c1w0t4_n256, seeds 7, 10, 13, 16, 19, 22, 25, 28; the scripted driver as yardstick. 500 chains per condition (chain seeds 70,000–70,499), up to 5 goals, 25 s per goal. Score per goal: +1 reached, −1 crash, 0 timeout or lost anchor. Every chained goal passed the feasibility check (time needed ≤ 15.72 s, the 95th percentile of goals drawn at reset).

**Goal 1 against the ordinary episode** (first 50 chains of every policy and condition): identical outcome and length in all.

## Both drivers: goal 1 against goals 2–5

Added after the results; it is not part of the bar. Each driver's later goals start where its own previous goal ended, so the two drivers face different later goals, and the scripted driver gets to goal 2 less often (a goal-1 timeout ends its chain). The scripted driver reads the true state, so 'origin kept' is identical to 'new car' for it.

| Condition | Driver | Goal 1: reached / crash / timeout | Goal 1 score | Goals 2–5 (n) | Goals 2–5: reached / crash / timeout | Goals 2–5 score | Drop |
|---|---|---|---|---|---|---|---|
| new car | model (8 seeds pooled) | 88.8% / 10.9% / 0.2% | +0.778 | 7638 | 68.6% / 31.1% / 0.2% | +0.375 | +0.404 |
| new car | scripted | 74.4% / 1.8% / 23.8% | +0.726 | 647 | 50.4% / 2.5% / 47.1% | +0.479 | +0.247 |
| same car | model (8 seeds pooled) | 88.8% / 10.9% / 0.2% | +0.778 | 4912 | 55.9% / 26.8% / 17.3% | +0.291 | +0.487 |
| same car | scripted | 74.4% / 1.8% / 23.8% | +0.726 | 651 | 88.5% / 0.2% / 11.4% | +0.883 | -0.157 |
| origin kept | model (8 seeds pooled) | 88.8% / 10.9% / 0.2% | +0.778 | 7609 | 67.1% / 32.1% / 0.9% | +0.350 | +0.428 |
| origin kept | scripted | 74.4% / 1.8% / 23.8% | +0.726 | 647 | 50.4% / 2.5% / 47.1% | +0.479 | +0.247 |

- **New car and origin kept:** the scripted driver drops too, by timing out more (24% → 47%), with almost no crashes. The model drops more, by crashing more (11% → 31–32%). Part of the model's drop is the later goals being harder for any driver; the rest (0.16 with a new car, 0.19 with the origin kept) is its own.
- **Same car:** the scripted driver does better on later goals than on goal 1, and the model drops further than in the other conditions, so the whole drop is the model's.

## Condition: new car

| Seed | Goal 1 score | Goals 2–5 score | Goals 2–5 (n) | Drop |
|---:|---:|---:|---:|---:|
| 7 | +0.810 | +0.401 | 961 | +0.409 |
| 10 | +0.856 | +0.386 | 976 | +0.470 |
| 13 | +0.818 | +0.428 | 1009 | +0.390 |
| 16 | +0.844 | +0.439 | 1028 | +0.405 |
| 19 | +0.760 | +0.317 | 925 | +0.443 |
| 22 | +0.804 | +0.424 | 1000 | +0.380 |
| 25 | +0.696 | +0.302 | 891 | +0.394 |
| 28 | +0.640 | +0.275 | 848 | +0.365 |

**Drop** (goal 1 − goals 2–5, paired over 8 seeds): +0.407 (95% interval +0.379 to +0.436, p 0.000). **Verdict: breaks.**

**Against the scripted driver's drop** (added after the results, not part of the bar): the scripted driver's drop is +0.247, so the model's extra drop is +0.160 (95% interval +0.132 to +0.189, p 0.000; one-sample t over the 8 seeds with the scripted drop held fixed, so the scripted driver's own sampling noise is not included).

Pairs the scripted driver reaches in under 20% of goals (flagged as possibly infeasible): none. Without them: +0.407 (95% +0.379 to +0.436, p 0.000), breaks.

**By goal** (model: all 8 seeds pooled; a goal counts if its chain got to it):

| Goal | Model goals | Reached | Crash | Timeout | Score | Scripted goals | Scripted score |
|---|---|---|---|---|---|---|---|
| 1 | 4000 | 88.8% | 10.9% | 0.2% | +0.778 | 500 | +0.726 |
| 2 | 3552 | 73.2% | 26.6% | 0.1% | +0.466 | 371 | +0.456 |
| 3 | 2563 | 62.4% | 37.5% | 0.2% | +0.249 | 180 | +0.483 |
| 4 | 1244 | 67.8% | 31.8% | 0.4% | +0.359 | 77 | +0.532 |
| 5 | 279 | 71.7% | 27.6% | 0.7% | +0.441 | 19 | +0.684 |

**How chains end** (share of chains):

| End | Model | Scripted |
|---|---|---|
| all goals reached | 5.0% | 2.6% |
| crash | 70.4% | 5.0% |
| no car in range | 23.8% | 7.6% |
| no feasible goal | 0.1% | 0.0% |
| timeout | 0.7% | 84.8% |

Goals reached per chain: model 2.20 (seeds 2.25, 2.28, 2.35, 2.40, 2.10, 2.33, 2.01, 1.88), scripted 1.40.

Crashes in a goal's first 3 s: goals 2–5 1427 of 7638 goals (18.7%); goal 1 106 of 4000 (2.6%).

Feasibility check at the model's switches: 2606 of 57847 candidate goals removed (4.5%).

**From slot → to slot** (goals 2–5; model pooled over seeds):

| Pair | Model goals | Reached | Crash | Timeout | Score | Time needed (median s) | Scripted goals | Scripted reached |  |
|---|---|---|---|---|---|---|---|---|---|
| back → back | 862 | 73.8% | 26.1% | 0.1% | +0.477 | 8.4 | 96 | 64.6% |  |
| back → front | 660 | 43.0% | 56.7% | 0.3% | -0.136 | 12.2 | 58 | 24.1% |  |
| back → left | 710 | 66.6% | 33.4% | 0.0% | +0.332 | 8.9 | 69 | 27.5% |  |
| back → right | 552 | 60.9% | 39.1% | 0.0% | +0.217 | 9.1 | 47 | 34.0% |  |
| front → back | 587 | 85.9% | 14.0% | 0.2% | +0.719 | 7.7 | 49 | 79.6% |  |
| front → front | 464 | 51.7% | 47.8% | 0.4% | +0.039 | 11.1 | 44 | 34.1% |  |
| front → left | 366 | 64.2% | 35.8% | 0.0% | +0.284 | 9.2 | 28 | 39.3% |  |
| front → right | 368 | 61.7% | 38.0% | 0.3% | +0.236 | 9.0 | 32 | 50.0% |  |
| left → back | 516 | 83.7% | 15.9% | 0.4% | +0.678 | 7.4 | 34 | 85.3% |  |
| left → front | 430 | 54.7% | 44.4% | 0.9% | +0.102 | 10.9 | 33 | 60.6% |  |
| left → left | 266 | 77.4% | 22.6% | 0.0% | +0.549 | 7.0 | 20 | 55.0% |  |
| left → right | 341 | 80.1% | 19.6% | 0.3% | +0.604 | 9.1 | 21 | 42.9% |  |
| right → back | 565 | 78.9% | 20.9% | 0.2% | +0.581 | 7.5 | 46 | 73.9% |  |
| right → front | 349 | 71.3% | 28.7% | 0.0% | +0.427 | 11.6 | 25 | 36.0% |  |
| right → left | 257 | 80.9% | 18.7% | 0.4% | +0.623 | 8.7 | 18 | 44.4% |  |
| right → right | 345 | 75.1% | 24.9% | 0.0% | +0.501 | 8.0 | 27 | 51.9% |  |

## Condition: same car

| Seed | Goal 1 score | Goals 2–5 score | Goals 2–5 (n) | Drop |
|---:|---:|---:|---:|---:|
| 7 | +0.810 | +0.372 | 648 | +0.438 |
| 10 | +0.856 | +0.198 | 621 | +0.658 |
| 13 | +0.818 | +0.302 | 629 | +0.516 |
| 16 | +0.844 | +0.416 | 690 | +0.428 |
| 19 | +0.760 | -0.115 | 506 | +0.875 |
| 22 | +0.804 | +0.389 | 700 | +0.415 |
| 25 | +0.696 | +0.184 | 515 | +0.512 |
| 28 | +0.640 | +0.464 | 603 | +0.176 |

**Drop** (goal 1 − goals 2–5, paired over 8 seeds): +0.502 (95% interval +0.333 to +0.671, p 0.000). **Verdict: breaks.**

**Against the scripted driver's drop** (added after the results, not part of the bar): the scripted driver's drop is -0.157, so the model's extra drop is +0.659 (95% interval +0.490 to +0.829, p 0.000; one-sample t over the 8 seeds with the scripted drop held fixed, so the scripted driver's own sampling noise is not included).

Pairs the scripted driver reaches in under 20% of goals (flagged as possibly infeasible): front→back, left→right, right→left. Without them: +0.366 (95% +0.197 to +0.535, p 0.001), breaks.

**By goal** (model: all 8 seeds pooled; a goal counts if its chain got to it):

| Goal | Model goals | Reached | Crash | Timeout | Score | Scripted goals | Scripted score |
|---|---|---|---|---|---|---|---|
| 1 | 4000 | 88.8% | 10.9% | 0.2% | +0.778 | 500 | +0.726 |
| 2 | 3038 | 60.9% | 25.8% | 13.2% | +0.351 | 297 | +0.889 |
| 3 | 1279 | 45.7% | 31.7% | 22.7% | +0.140 | 170 | +0.876 |
| 4 | 413 | 50.6% | 21.1% | 28.3% | +0.295 | 110 | +0.873 |
| 5 | 182 | 56.6% | 22.0% | 21.4% | +0.346 | 74 | +0.892 |

**How chains end** (share of chains):

| End | Model | Scripted |
|---|---|---|
| all goals reached | 2.6% | 13.2% |
| crash | 43.9% | 2.0% |
| no feasible goal | 32.1% | 46.2% |
| timeout | 21.4% | 38.6% |

Goals reached per chain: model 1.57 (seeds 1.64, 1.58, 1.61, 1.77, 1.32, 1.76, 1.33, 1.58), scripted 1.90.

Crashes in a goal's first 3 s: goals 2–5 945 of 4912 goals (19.2%); goal 1 106 of 4000 (2.6%).

Feasibility check at the model's switches: 352 of 11345 candidate goals removed (3.1%).

**From slot → to slot** (goals 2–5; model pooled over seeds):

| Pair | Model goals | Reached | Crash | Timeout | Score | Time needed (median s) | Scripted goals | Scripted reached |  |
|---|---|---|---|---|---|---|---|---|---|
| back → front | 747 | 71.4% | 24.6% | 4.0% | +0.467 | 4.1 | 117 | 96.6% |  |
| back → left | 428 | 96.7% | 1.9% | 1.4% | +0.949 | 3.4 | 58 | 91.4% |  |
| back → right | 422 | 91.5% | 6.6% | 1.9% | +0.848 | 3.4 | 67 | 98.5% |  |
| front → back | 450 | 10.2% | 35.3% | 54.4% | -0.251 | 12.1 | 27 | 0.0% | flagged |
| front → left | 435 | 14.9% | 34.0% | 51.0% | -0.191 | 11.1 | 47 | 100.0% |  |
| front → right | 257 | 9.3% | 38.9% | 51.8% | -0.296 | 11.9 | 32 | 100.0% |  |
| left → back | 445 | 45.2% | 30.8% | 24.0% | +0.144 | 6.2 | 55 | 100.0% |  |
| left → front | 596 | 86.9% | 11.7% | 1.3% | +0.752 | 4.9 | 87 | 95.4% |  |
| left → right | 122 | 0.0% | 100.0% | 0.0% | -1.000 | 4.2 | 13 | 0.0% | flagged |
| right → back | 463 | 41.5% | 39.5% | 19.0% | +0.019 | 6.8 | 68 | 100.0% |  |
| right → front | 414 | 88.9% | 10.9% | 0.2% | +0.780 | 4.9 | 63 | 92.1% |  |
| right → left | 133 | 0.0% | 100.0% | 0.0% | -1.000 | 4.2 | 17 | 5.9% | flagged |

## Condition: origin kept

| Seed | Goal 1 score | Goals 2–5 score | Goals 2–5 (n) | Drop |
|---:|---:|---:|---:|---:|
| 7 | +0.810 | +0.259 | 931 | +0.551 |
| 10 | +0.856 | +0.426 | 1009 | +0.430 |
| 13 | +0.818 | +0.384 | 997 | +0.434 |
| 16 | +0.844 | +0.421 | 1040 | +0.423 |
| 19 | +0.760 | +0.359 | 942 | +0.401 |
| 22 | +0.804 | +0.393 | 983 | +0.411 |
| 25 | +0.696 | +0.381 | 904 | +0.315 |
| 28 | +0.640 | +0.131 | 803 | +0.509 |

**Drop** (goal 1 − goals 2–5, paired over 8 seeds): +0.434 (95% interval +0.375 to +0.494, p 0.000). **Verdict: breaks.**

**Against the scripted driver's drop** (added after the results, not part of the bar): the scripted driver's drop is +0.247, so the model's extra drop is +0.187 (95% interval +0.128 to +0.247, p 0.000; one-sample t over the 8 seeds with the scripted drop held fixed, so the scripted driver's own sampling noise is not included).

Pairs the scripted driver reaches in under 20% of goals (flagged as possibly infeasible): none. Without them: +0.434 (95% +0.375 to +0.494, p 0.000), breaks.

**By goal** (model: all 8 seeds pooled; a goal counts if its chain got to it):

| Goal | Model goals | Reached | Crash | Timeout | Score | Scripted goals | Scripted score |
|---|---|---|---|---|---|---|---|
| 1 | 4000 | 88.8% | 10.9% | 0.2% | +0.778 | 500 | +0.726 |
| 2 | 3552 | 73.5% | 26.3% | 0.3% | +0.472 | 371 | +0.456 |
| 3 | 2572 | 60.1% | 39.0% | 0.8% | +0.211 | 180 | +0.483 |
| 4 | 1211 | 62.9% | 35.0% | 2.1% | +0.279 | 77 | +0.532 |
| 5 | 274 | 67.9% | 28.5% | 3.6% | +0.394 | 19 | +0.684 |

**How chains end** (share of chains):

| End | Model | Scripted |
|---|---|---|
| all goals reached | 4.7% | 2.6% |
| crash | 71.9% | 5.0% |
| no car in range | 21.3% | 7.6% |
| no feasible goal | 0.2% | 0.0% |
| timeout | 1.9% | 84.8% |

Goals reached per chain: model 2.16 (seeds 2.06, 2.36, 2.27, 2.40, 2.16, 2.27, 2.10, 1.70), scripted 1.40.

Crashes in a goal's first 3 s: goals 2–5 1378 of 7609 goals (18.1%); goal 1 106 of 4000 (2.6%).

Feasibility check at the model's switches: 2610 of 57682 candidate goals removed (4.5%).

**From slot → to slot** (goals 2–5; model pooled over seeds):

| Pair | Model goals | Reached | Crash | Timeout | Score | Time needed (median s) | Scripted goals | Scripted reached |  |
|---|---|---|---|---|---|---|---|---|---|
| back → back | 856 | 73.8% | 25.9% | 0.2% | +0.479 | 8.5 | 96 | 64.6% |  |
| back → front | 669 | 45.1% | 54.4% | 0.4% | -0.093 | 12.2 | 58 | 24.1% |  |
| back → left | 714 | 65.7% | 33.3% | 1.0% | +0.324 | 8.9 | 69 | 27.5% |  |
| back → right | 558 | 54.8% | 44.8% | 0.4% | +0.100 | 9.4 | 47 | 34.0% |  |
| front → back | 597 | 82.1% | 17.3% | 0.7% | +0.648 | 7.7 | 49 | 79.6% |  |
| front → front | 476 | 52.7% | 44.7% | 2.5% | +0.080 | 11.3 | 44 | 34.1% |  |
| front → left | 362 | 61.9% | 36.5% | 1.7% | +0.254 | 9.0 | 28 | 39.3% |  |
| front → right | 370 | 57.0% | 41.6% | 1.4% | +0.154 | 8.9 | 32 | 50.0% |  |
| left → back | 525 | 82.7% | 16.4% | 1.0% | +0.663 | 7.5 | 34 | 85.3% |  |
| left → front | 421 | 59.4% | 39.7% | 1.0% | +0.197 | 10.9 | 33 | 60.6% |  |
| left → left | 264 | 74.6% | 24.2% | 1.1% | +0.504 | 7.1 | 20 | 55.0% |  |
| left → right | 337 | 73.9% | 25.8% | 0.3% | +0.481 | 9.1 | 21 | 42.9% |  |
| right → back | 546 | 80.2% | 19.6% | 0.2% | +0.606 | 7.5 | 46 | 73.9% |  |
| right → front | 334 | 68.3% | 29.9% | 1.8% | +0.383 | 11.6 | 25 | 36.0% |  |
| right → left | 247 | 74.9% | 23.5% | 1.6% | +0.514 | 8.7 | 18 | 44.4% |  |
| right → right | 333 | 71.5% | 28.2% | 0.3% | +0.432 | 8.0 | 27 | 51.9% |  |

## Verdicts (bar written before the run)

| Condition | Drop | 95% interval | p | Verdict | Without flagged pairs | Scripted driver's drop (added after) | Model's extra drop (added after) |
|---|---|---|---|---|---|---|---|
| new car | +0.407 | +0.379 to +0.436 | 0.000 | **breaks** | +0.407, breaks | +0.247 | +0.160 (+0.132 to +0.189) |
| same car | +0.502 | +0.333 to +0.671 | 0.000 | **breaks** | +0.366, breaks | -0.157 | +0.659 (+0.490 to +0.829) |
| origin kept | +0.434 | +0.375 to +0.494 | 0.000 | **breaks** | +0.434, breaks | +0.247 | +0.187 (+0.128 to +0.247) |
