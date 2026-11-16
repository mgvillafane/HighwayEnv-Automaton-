| Policy | Success [95% CI] | Collision | Timeout | Anchor lost | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO B4, seed 7 | 56.5% [50%, 63%] | 15.0% | 28.5% | 0.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B4, seed 8 | 55.0% [48%, 62%] | 14.0% | 31.0% | 0.0% | 0/62 | 57/61 | 29/40 | 24/37 |
| PPO B4, seed 9 | 44.5% [38%, 51%] | 17.5% | 38.0% | 0.0% | 0/62 | 56/61 | 25/40 | 8/37 |
| PPO B4, mean ± sd over seeds | 52.0% ± 6.5% | | | | | | | |
| Uniform random | 11.5% [8%, 17%] | 79.0% | 9.5% | 0.0% | 2/62 | 12/61 | 5/40 | 4/37 |
| Scripted (reads true state) | 76.5% [70%, 82%] | 2.5% | 21.0% | 0.0% | 40/62 | 58/61 | 32/40 | 23/37 |

Difference from B3m (ivcheck):
  environment.reward_weights.collision: 50.0 -> 150.0
