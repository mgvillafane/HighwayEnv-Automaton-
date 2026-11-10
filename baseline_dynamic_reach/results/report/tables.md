# Tables for REPORT.md (written by report.py; do not edit by hand)

## Outcomes per arm

Overall: 200 evaluation episodes per seed, mean over seeds 7, 8, 9. BACK/LEFT/RIGHT: reached / crash / timeout of 138 episodes, per seed. FRONT: reached of 62, per seed.

| Arm | Change from parent | Success (overall) | Crash | Timeout | BACK/LEFT/RIGHT, seeds 7 · 8 · 9 | FRONT |
|---|---|---:|---:|---:|---|---|
| B0 | 50 x success | 55.2% ± 2.0 | 44.3% | 0.5% | 114/24/0 · 105/33/0 · 111/27/0 | 0 · 1 · 0 |
| B1 | + crash -50, from scratch | 1.3% ± 1.9 | 1.0% | 97.7% | 1/2/135 · 7/2/129 · 0/1/137 | 0 · 0 · 0 |
| B2 | B1, starting from B0's policy | 52.7% ± 4.0 | 16.3% | 31.0% | 106/32/0 · 113/25/0 · 97/41/0 | 0 · 0 · 0 |
| B3 | + potential-based progress | 56.2% ± 0.3 | 14.3% | 29.5% | 113/25/0 · 112/26/0 · 112/26/0 | 0 · 0 · 0 |
| B3m | B3 trained again on the Mac | 55.7% ± 0.8 | 14.3% | 30.0% | 111/27/0 · 113/25/0 · 110/28/0 | 0 · 0 · 0 |
| B4 | crash -50 -> -150 | 52.0% ± 6.5 | 15.5% | 32.5% | 113/25/0 · 110/28/0 · 89/29/20 | 0 · 0 · 0 |
| Scripted | rule-based, reads the true state | 76.5% | 2.5% | 21.0% | 113/3/22 | 40 |
| Random | uniform random actions | 11.5% | 79.0% | 9.5% | 21/105/12 | 2 |

## BACK/LEFT/RIGHT by tag

Blocked: at reset a car is in the slot's lane, between ego and the slot (45 episodes); clear: the other 93. Reached / crash / timeout, summed over the three seeds (so out of 3 × 45 = 135 and 3 × 93 = 279).

| Arm | Clear: reached / crash / timeout (of 279) | Blocked: reached / crash / timeout (of 135) |
|---|---|---|
| B0 | 263/16/0 | 67/68/0 |
| B1 | 8/4/267 | 0/1/134 |
| B2 | 247/32/0 | 69/66/0 |
| B3 | 269/10/0 | 68/67/0 |
| B3m | 266/13/0 | 68/67/0 |
| B4 | 251/16/12 | 61/66/8 |
| Scripted | 87/1/5 (one run, of 93) | 26/2/17 (one run, of 45) |
