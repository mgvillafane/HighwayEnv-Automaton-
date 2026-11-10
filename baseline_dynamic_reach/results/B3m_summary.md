| Policy | Success [95% CI] | Collision | Timeout | Anchor lost | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO B3m, seed 7 | 55.5% [49%, 62%] | 13.5% | 31.0% | 0.0% | 0/62 | 57/61 | 32/40 | 22/37 |
| PPO B3m, seed 8 | 56.5% [50%, 63%] | 14.0% | 29.5% | 0.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B3m, seed 9 | 55.0% [48%, 62%] | 15.5% | 29.5% | 0.0% | 0/62 | 55/61 | 31/40 | 24/37 |
| PPO B3m, mean ± sd over seeds | 55.7% ± 0.8% | | | | | | | |
| Uniform random | 11.5% [8%, 17%] | 79.0% | 9.5% | 0.0% | 2/62 | 12/61 | 5/40 | 4/37 |
| Scripted (reads true state) | 76.5% [70%, 82%] | 2.5% | 21.0% | 0.0% | 40/62 | 58/61 | 32/40 | 23/37 |

Difference from B3 (ivcheck):
  platform.machine: 'AMD64' -> 'arm64'
  platform.node: 'Catherine-PC' -> 'wifi-10-44-147-178.wifi.berkeley.edu'
  platform.processor: 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel' -> 'arm'
  platform.python: '3.13.1' -> '3.12.13'
  platform.release: '11' -> '25.6.0'
  platform.system: 'Windows' -> 'Darwin'
