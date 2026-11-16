# Update probe, trained runs

Medians over the minibatches of one rollout (seed 123); KL: one full PPO update in memory. Clip factor 1 = the gradient is not clipped.

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_seed7 | 0.332 | 11.396 | 99.9% | 0.044 | 8.5 | 0.0047 | 0.0183 |
| B3m_seed7 | 0.341 | 16.048 | 100.0% | 0.031 | 2.46 | 0.0020 | 0.0109 |
| B4_seed7 | 0.404 | 47.127 | 100.0% | 0.011 | 5.81 | 0.0030 | 0.0185 |
