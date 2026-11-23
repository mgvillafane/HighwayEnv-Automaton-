# Learning curves of Step 8e (9M transitions, 8 seeds, 64 and 256 wide)

Deterministic score = reached − crash on the 400 selection episodes (full task), every 200k checkpoint. Round 4 continues round 3 for 3M more transitions (same task and reward). End of a round = mean of its last 3 snapshots (the grid's scored snapshots). Per-seed values, paired over the 8 seeds (t-test).

| Recipe | Width | End of round 2 (4M) | End of round 3 (6M) | End of round 4 (9M) | Round 4 − round 3 (95% interval, p, seeds up) | Slope over the last 1M, per 1M (p) | Peak snapshot (mean) |
|---|---|---:|---:|---:|---|---|---:|
| N_c05w0t1 | 64 | +0.547 | +0.617 | +0.661 | +0.044 ([-0.046, +0.134], p 0.28, 5 of 8) | +0.044 (p 0.23) | +0.695 at 8.8M |
| N_c05w0t1 | 256 | +0.554 | +0.609 | +0.725 | +0.116 ([+0.030, +0.201], p 0.02, 8 of 8) | -0.010 (p 0.62) | +0.772 at 8.2M |
| N_c1w0t4 | 64 | +0.521 | +0.585 | +0.624 | +0.039 ([-0.004, +0.081], p 0.07, 6 of 8) | -0.042 (p 0.10) | +0.677 at 8.2M |
| N_c1w0t4 | 256 | +0.630 | +0.682 | +0.723 | +0.042 ([-0.035, +0.119], p 0.24, 5 of 8) | -0.022 (p 0.43) | +0.789 at 8.2M |
| O_c1w0t4 | 64 | +0.437 | +0.652 | +0.672 | +0.020 ([-0.074, +0.114], p 0.63, 5 of 8) | -0.005 (p 0.90) | +0.707 at 9.0M |
| O_c1w0t4 | 256 | +0.795 | +0.768 | +0.797 | +0.029 ([-0.018, +0.075], p 0.19, 6 of 8) | +0.010 (p 0.66) | +0.852 at 8.4M |

**FRONT give-up** (share of the 133 FRONT selection episodes that time out; mean of seeds; end of a round = mean of its last 3 snapshots):

| Recipe | Width | End of round 2 | End of round 3 | End of round 4 | Seeds with ≥ 30% FRONT timeouts at the end of round 4 | FRONT reached, end of round 4 | FRONT crash, end of round 4 |
|---|---|---:|---:|---:|---|---:|---:|
| N_c05w0t1 | 64 | 46% | 43% | 34% | 10, 13, 22, 25, 28 | 51% | 15% |
| N_c05w0t1 | 256 | 23% | 8% | 5% | 19 | 71% | 24% |
| N_c1w0t4 | 64 | 45% | 43% | 39% | 13, 19, 22, 25, 28 | 45% | 17% |
| N_c1w0t4 | 256 | 18% | 12% | 5% | 22 | 71% | 24% |
| O_c1w0t4 | 64 | 31% | 37% | 26% | 10, 16, 28 | 54% | 20% |
| O_c1w0t4 | 256 | 0% | 0% | 0% | none | 80% | 19% |
