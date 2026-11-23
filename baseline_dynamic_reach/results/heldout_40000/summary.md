# Step 7h: P6 and B5 on held-out episodes (seeds 40,000–40,199)

## Usual episodes

Slots: FRONT 62, BACK 61, LEFT 40, RIGHT 37. BACK/LEFT/RIGHT: 138, of which 45 blocked at reset.

Cells are reached / crash / timeout (success rate).

| Policy | BACK/LEFT/RIGHT | clear | blocked | FRONT | All 200 |
|---|---|---|---|---|---|
| P6 seed 7 | 134/4/0 (97.1%) | 92/1/0 (98.9%) | 42/3/0 (93.3%) | 48/14/0 (77.4%) | 182/18/0 (91.0%) |
| P6 seed 8 | 125/13/0 (90.6%) | 92/1/0 (98.9%) | 33/12/0 (73.3%) | 41/21/0 (66.1%) | 166/34/0 (83.0%) |
| P6 seed 9 | 123/15/0 (89.1%) | 90/3/0 (96.8%) | 33/12/0 (73.3%) | 20/42/0 (32.3%) | 143/57/0 (71.5%) |
| B5 seed 7 | 113/25/0 (81.9%) | 90/3/0 (96.8%) | 23/22/0 (51.1%) | 0/4/58 (0.0%) | 113/29/58 (56.5%) |
| B5 seed 8 | 113/25/0 (81.9%) | 90/3/0 (96.8%) | 23/22/0 (51.1%) | 0/1/61 (0.0%) | 113/26/61 (56.5%) |
| B5 seed 9 | 113/25/0 (81.9%) | 90/3/0 (96.8%) | 23/22/0 (51.1%) | 0/9/53 (0.0%) | 113/34/53 (56.5%) |
| Scripted | 113/3/22 (81.9%) | 87/1/5 (93.5%) | 26/2/17 (57.8%) | 40/2/20 (64.5%) | 153/5/42 (76.5%) |
| Random | 21/105/12 (15.2%) | 17/68/8 (18.3%) | 4/37/4 (8.9%) | 2/53/7 (3.2%) | 23/158/19 (11.5%) |

## Held-out episodes

Slots: FRONT 71, BACK 56, LEFT 39, RIGHT 34. BACK/LEFT/RIGHT: 129, of which 39 blocked at reset.

Cells are reached / crash / timeout (success rate).

| Policy | BACK/LEFT/RIGHT | clear | blocked | FRONT | All 200 |
|---|---|---|---|---|---|
| P6 seed 7 | 127/2/0 (98.4%) | 88/2/0 (97.8%) | 39/0/0 (100.0%) | 51/20/0 (71.8%) | 178/22/0 (89.0%) |
| P6 seed 8 | 116/13/0 (89.9%) | 85/5/0 (94.4%) | 31/8/0 (79.5%) | 45/26/0 (63.4%) | 161/39/0 (80.5%) |
| P6 seed 9 | 117/12/0 (90.7%) | 89/1/0 (98.9%) | 28/11/0 (71.8%) | 29/40/2 (40.8%) | 146/52/2 (73.0%) |
| B5 seed 7 | 108/21/0 (83.7%) | 89/1/0 (98.9%) | 19/20/0 (48.7%) | 0/2/69 (0.0%) | 108/23/69 (54.0%) |
| B5 seed 8 | 108/21/0 (83.7%) | 89/1/0 (98.9%) | 19/20/0 (48.7%) | 0/0/71 (0.0%) | 108/21/71 (54.0%) |
| B5 seed 9 | 107/22/0 (82.9%) | 88/2/0 (97.8%) | 19/20/0 (48.7%) | 0/12/59 (0.0%) | 107/34/59 (53.5%) |
| Scripted | 111/4/14 (86.0%) | 88/1/1 (97.8%) | 23/3/13 (59.0%) | 39/4/28 (54.9%) | 150/8/42 (75.0%) |
| Random | 25/98/6 (19.4%) | 22/62/6 (24.4%) | 3/36/0 (7.7%) | 7/61/3 (9.9%) | 32/159/9 (16.0%) |

## Predictions (written before the evaluation)

| Prediction | Held | Numbers |
|---|---|---|
| 1. P6 beats the scripted driver on BACK/LEFT/RIGHT by >= 5 points, every seed | **no** | leads +12.4, +3.9, +4.7 |
| 2. P6 beats B5 on BACK/LEFT/RIGHT by >= 5 points, every seed | yes | leads +14.7, +6.2, +7.8 |
| 3. The lead over the scripted driver shrinks by <= 7 points on at least 2 of 3 seeds | yes | usual +15.2, +8.7, +7.2; held-out +12.4, +3.9, +4.7; shrink +2.8, +4.8, +2.6 |
| 4. FRONT: P6 seed 7 at least matches the scripted driver | yes | 71.8% vs 54.9% |
| 5. P6 crashes more often than the scripted driver, every seed | yes | crashes 22, 39, 52 vs 8 |
