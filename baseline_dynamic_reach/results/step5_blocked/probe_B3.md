# Probe of B3 on the evaluation episodes

Deterministic replay, as in evaluate.py. "First actions" are the first 2 s.
"Value" is the critic at reset with the shaping constant removed: its estimate of the
discounted 50 x success - 50 x crash. BACK/LEFT/RIGHT episodes only.

## B3_seed7

Replay matches the saved evaluation: True 

| Tag | Episodes | Reached | Crashed | Mean value | LANE_LEFT | IDLE | LANE_RIGHT | FASTER | SLOWER | First lane change (median s) | Ever SLOWER before the end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 90 | 3 | 26.6 | 7% | 4% | 24% | 62% | 2% | 0.0 (78/93) | 35/93 |
| blocked | 45 | 23 | 22 | 0.8 | 4% | 10% | 9% | 78% | 0% | 0.0 (22/45) | 12/45 |

- Value separates clear from blocked episodes with AUC 0.88, and reached from failed episodes with AUC 0.94 (0.5 = no information).
- Mean value: reached 24.1, failed -8.5.
- Crashes in blocked episodes that hit a car tagged as a blocker at reset: 20/22.
- All 25 crashes by kind: {'other car ahead': 25}.

## B3_seed8

Replay matches the saved evaluation: True 

| Tag | Episodes | Reached | Crashed | Mean value | LANE_LEFT | IDLE | LANE_RIGHT | FASTER | SLOWER | First lane change (median s) | Ever SLOWER before the end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 89 | 4 | 21.8 | 7% | 0% | 6% | 87% | 0% | 0.0 (79/93) | 0/93 |
| blocked | 45 | 23 | 22 | -7.6 | 5% | 0% | 4% | 91% | 0% | 0.0 (26/45) | 0/45 |

- Value separates clear from blocked episodes with AUC 0.85, and reached from failed episodes with AUC 0.93 (0.5 = no information).
- Mean value: reached 19.3, failed -18.4.
- Crashes in blocked episodes that hit a car tagged as a blocker at reset: 20/22.
- All 26 crashes by kind: {'other car ahead': 21, 'anchor ahead': 2, 'other car ahead during lane change': 2, 'other car beside during lane change': 1}.

## B3_seed9

Replay matches the saved evaluation: True 

| Tag | Episodes | Reached | Crashed | Mean value | LANE_LEFT | IDLE | LANE_RIGHT | FASTER | SLOWER | First lane change (median s) | Ever SLOWER before the end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 90 | 3 | 28.2 | 6% | 0% | 4% | 89% | 0% | 0.0 (78/93) | 1/93 |
| blocked | 45 | 22 | 23 | 3.2 | 1% | 0% | 1% | 97% | 0% | 0.0 (24/45) | 2/45 |

- Value separates clear from blocked episodes with AUC 0.85, and reached from failed episodes with AUC 0.93 (0.5 = no information).
- Mean value: reached 26.2, failed -6.5.
- Crashes in blocked episodes that hit a car tagged as a blocker at reset: 21/23.
- All 26 crashes by kind: {'other car ahead': 25, 'other car beside': 1}.
