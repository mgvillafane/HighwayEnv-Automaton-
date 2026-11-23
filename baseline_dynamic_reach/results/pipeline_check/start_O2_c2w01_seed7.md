# Update probe, starting point of O2_c2w01

As for trained runs, but with the arm's settings and its initial weights, before any training (fresh Adam state, as in train.py).

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| O2_c2w01 start (seed 7; E1_seed7.zip, value output x 1) | 0.582 | 0.231 | 7.1% | 0.828 | 0.116 | 0.0107 | 0.0080 |
