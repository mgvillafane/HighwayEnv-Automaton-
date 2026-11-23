# Learning curves of the Step 8 recipes (6M transitions, 8 seeds)

Deterministic score = reached − crash on the 400 selection episodes (full task), every 200k checkpoint. Window means are per seed, then paired over the 8 seeds (t-test). Practice rounds (order O, rounds 1–2) are trained on practice episodes but scored on the full task.

| Recipe | End of round 1 (2M) | End of round 2 (4M) | Round 3, 4.2–5.0M | Round 3, 5.2–6.0M | Last 1M − previous 1M (p, seeds up) | Slope over 5.0–6.0M per 1M (p) | Peak snapshot (mean) |
|---|---:|---:|---:|---:|---|---|---:|
| N_c05w0t1 | +0.380 | +0.528 | +0.548 | +0.605 | +0.057 (p 0.06, 6 of 8) | +0.042 (p 0.01) | +0.626 at 5.8M |
| O_c1w0t4 | +0.169 | +0.446 | +0.618 | +0.635 | +0.016 (p 0.44, 6 of 8) | +0.056 (p 0.14) | +0.689 at 6.0M |
| O_c1w01 | +0.169 | +0.407 | +0.582 | +0.650 | +0.068 (p 0.11, 5 of 8) | +0.010 (p 0.83) | +0.669 at 5.2M |
| N_c1w0t4 | +0.380 | +0.553 | +0.558 | +0.582 | +0.024 (p 0.09, 6 of 8) | -0.004 (p 0.84) | +0.597 at 5.0M |
| N_c1w01 | +0.380 | +0.486 | +0.555 | +0.581 | +0.026 (p 0.21, 7 of 8) | +0.038 (p 0.02) | +0.614 at 6.0M |
| N_c0w0 | +0.380 | +0.447 | +0.481 | +0.493 | +0.011 (p 0.51, 3 of 8) | +0.008 (p 0.78) | +0.522 at 4.4M |
| O_c0w0 | +0.169 | +0.316 | +0.510 | +0.557 | +0.048 (p 0.15, 6 of 8) | +0.048 (p 0.26) | +0.588 at 5.8M |

**Order N, round 3 against round 2.** Rounds 2 and 3 of order N have the same task and reward, so round 3 is 2M more transitions of the same training. Score = mean of the last 3 snapshots of each round.

| Recipe | Round 2 (3.6–4.0M) | Round 3 (5.6–6.0M) | Difference (95% interval, p, seeds up) | Crash change | Timeout change |
|---|---:|---:|---|---:|---:|
| N_c05w0t1 | +0.547 | +0.617 | +0.071 ([+0.026, +0.116], p 0.01, 7 of 8) | -3.5 pts | -0.0 pts |
| N_c1w0t4 | +0.521 | +0.585 | +0.064 ([+0.025, +0.103], p 0.01, 8 of 8) | -3.1 pts | -0.2 pts |
| N_c1w01 | +0.500 | +0.591 | +0.091 ([+0.022, +0.160], p 0.02, 7 of 8) | -4.1 pts | -0.9 pts |
| N_c0w0 | +0.424 | +0.494 | +0.071 ([-0.003, +0.144], p 0.06, 7 of 8) | -3.3 pts | -0.5 pts |

**FRONT give-up over round 3** (share of the 133 FRONT selection episodes that time out; mean of seeds):

| Recipe | 4.2M | 4.6M | 5.0M | 5.4M | 5.8M | 6.0M | Seeds with ≥ 30% FRONT timeouts at 6.0M |
|---|---:|---:|---:|---:|---:|---:|---|
| N_c05w0t1 | 43% | 51% | 41% | 39% | 41% | 41% | 10, 19, 22, 25, 28 |
| O_c1w0t4 | 36% | 37% | 31% | 35% | 37% | 37% | 10, 16, 28 |
| O_c1w01 | 37% | 38% | 34% | 37% | 35% | 37% | 10, 16, 28 |
| N_c1w0t4 | 49% | 50% | 54% | 45% | 36% | 45% | 10, 13, 19, 22, 25, 28 |
| N_c1w01 | 61% | 71% | 62% | 59% | 57% | 56% | 10, 13, 16, 19, 22, 25, 28 |
| N_c0w0 | 1% | 3% | 0% | 0% | 1% | 0% | none |
| O_c0w0 | 1% | 0% | 0% | 0% | 0% | 0% | none |
