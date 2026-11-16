| Policy | Success [95% CI] | Collision | Timeout | Anchor lost | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO B5, seed 7 | 56.5% [50%, 63%] | 14.5% | 29.0% | 0.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B5, seed 8 | 56.5% [50%, 63%] | 13.0% | 30.5% | 0.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B5, seed 9 | 56.5% [50%, 63%] | 17.0% | 26.5% | 0.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B5, mean ± sd over seeds | 56.5% ± 0.0% | | | | | | | |
| Uniform random | 11.5% [8%, 17%] | 79.0% | 9.5% | 0.0% | 2/62 | 12/61 | 5/40 | 4/37 |
| Scripted (reads true state) | 76.5% [70%, 82%] | 2.5% | 21.0% | 0.0% | 40/62 | 58/61 | 32/40 | 23/37 |

Difference from B3m (ivcheck):
  environment.reward_scale: '<missing>' -> 0.02
