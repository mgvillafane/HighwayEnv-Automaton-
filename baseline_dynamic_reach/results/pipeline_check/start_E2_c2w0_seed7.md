# Update probe, starting point of E2_c2w0

As for trained runs, but with the arm's settings and its initial weights, before any training (fresh Adam state, as in train.py).

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2_c2w0 start (seed 7; E1_seed7.zip, value output x 1) | 0.440 | 1.037 | 82.6% | 0.444 | 0.291 | 0.0091 | 0.0085 |
