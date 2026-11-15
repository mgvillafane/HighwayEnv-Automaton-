| Policy | Success [95% CI] | Collision | Timeout | Anchor lost | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO B1, seed 7 | 0.5% [0%, 3%] | 1.0% | 98.5% | 0.0% | 0/62 | 1/61 | 0/40 | 0/37 |
| PPO B1, seed 8 | 3.5% [2%, 7%] | 1.5% | 95.0% | 0.0% | 0/62 | 7/61 | 0/40 | 0/37 |
| PPO B1, seed 9 | 0.0% [0%, 2%] | 0.5% | 99.5% | 0.0% | 0/62 | 0/61 | 0/40 | 0/37 |
| PPO B1, mean ± sd over seeds | 1.3% ± 1.9% | | | | | | | |
| Uniform random | 11.5% [8%, 17%] | 79.0% | 9.5% | 0.0% | 2/62 | 12/61 | 5/40 | 4/37 |
| Scripted (reads true state) | 76.5% [70%, 82%] | 2.5% | 21.0% | 0.0% | 40/62 | 58/61 | 32/40 | 23/37 |

Difference from B0 (ivcheck):
  environment.reward_weights.collision: 0.0 -> 50.0
