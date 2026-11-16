# Update probe, starting point of B5

As for trained runs, but with the arm's settings and its initial weights, before any training (fresh Adam state, as in train.py).

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| B5 start (seed 7; B0_seed7.zip, value output x 0.02) | 0.377 | 0.315 | 44.2% | 1.000 | 0.0431 | 0.0089 | 0.0078 |
