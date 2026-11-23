# Update probe, starting point of E2_c1w01

As for trained runs, but with the arm's settings and its initial weights, before any training (fresh Adam state, as in train.py).

| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2_c1w01 start (seed 7; E1_seed7.zip, value output x 1) | 0.446 | 0.236 | 21.9% | 0.992 | 0.0896 | 0.0095 | 0.0087 |
