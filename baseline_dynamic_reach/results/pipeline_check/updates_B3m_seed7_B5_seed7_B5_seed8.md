# Update probe, trained runs

Medians over the minibatches of one rollout (seed 123); KL: one full PPO update in memory. Clip factor 1 = the gradient is not clipped.

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3m_seed7 | 0.341 | 16.048 | 100.0% | 0.031 | 2.46 | 0.0020 | 0.0109 |
| B5_seed7 | 0.311 | 0.051 | 2.6% | 1.000 | 0.00097 | 0.0037 | 0.0077 |
| B5_seed8 | 0.589 | 0.355 | 18.7% | 0.824 | 0.0536 | 0.0046 | 0.0085 |
