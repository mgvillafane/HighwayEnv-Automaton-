# Blocked and clear evaluation episodes

Blocked: at reset, a car is in the slot's lane, ahead of ego and behind the slot center.
Arm cells add up the three seeds. Columns: success / crashes / timeouts.

## B0

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 137/138 | 1 | 0 | 45/46 | 0 | 1 |
| back | blocked | 15 | 36/45 | 9 | 0 | 13/15 | 1 | 1 |
| left | clear | 25 | 69/75 | 6 | 0 | 23/25 | 0 | 2 |
| left | blocked | 15 | 19/45 | 26 | 0 | 9/15 | 0 | 6 |
| right | clear | 22 | 57/66 | 9 | 0 | 19/22 | 1 | 2 |
| right | blocked | 15 | 12/45 | 33 | 0 | 4/15 | 1 | 10 |
| front | blocked | 62 | 1/186 | 182 | 3 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 263/279 | 16 | 0 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 67/135 | 68 | 0 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 24, of which blocked: 22.

## B1

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 8/138 | 2 | 128 | 45/46 | 0 | 1 |
| back | blocked | 15 | 0/45 | 0 | 45 | 13/15 | 1 | 1 |
| left | clear | 25 | 0/75 | 2 | 73 | 23/25 | 0 | 2 |
| left | blocked | 15 | 0/45 | 1 | 44 | 9/15 | 0 | 6 |
| right | clear | 22 | 0/66 | 0 | 66 | 19/22 | 1 | 2 |
| right | blocked | 15 | 0/45 | 0 | 45 | 4/15 | 1 | 10 |
| front | blocked | 62 | 0/186 | 1 | 185 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 8/279 | 4 | 267 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 0/135 | 1 | 134 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 131, of which blocked: 45.

## B2

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 125/138 | 13 | 0 | 45/46 | 0 | 1 |
| back | blocked | 15 | 35/45 | 10 | 0 | 13/15 | 1 | 1 |
| left | clear | 25 | 66/75 | 9 | 0 | 23/25 | 0 | 2 |
| left | blocked | 15 | 22/45 | 23 | 0 | 9/15 | 0 | 6 |
| right | clear | 22 | 56/66 | 10 | 0 | 19/22 | 1 | 2 |
| right | blocked | 15 | 12/45 | 33 | 0 | 4/15 | 1 | 10 |
| front | blocked | 62 | 0/186 | 0 | 186 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 247/279 | 32 | 0 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 69/135 | 66 | 0 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 23, of which blocked: 21.

## B3

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 138/138 | 0 | 0 | 45/46 | 0 | 1 |
| back | blocked | 15 | 36/45 | 9 | 0 | 13/15 | 1 | 1 |
| left | clear | 25 | 73/75 | 2 | 0 | 23/25 | 0 | 2 |
| left | blocked | 15 | 21/45 | 24 | 0 | 9/15 | 0 | 6 |
| right | clear | 22 | 58/66 | 8 | 0 | 19/22 | 1 | 2 |
| right | blocked | 15 | 11/45 | 34 | 0 | 4/15 | 1 | 10 |
| front | blocked | 62 | 0/186 | 9 | 177 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 269/279 | 10 | 0 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 68/135 | 67 | 0 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 24, of which blocked: 22.

## B3m

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 135/138 | 3 | 0 | 45/46 | 0 | 1 |
| back | blocked | 15 | 35/45 | 10 | 0 | 13/15 | 1 | 1 |
| left | clear | 25 | 73/75 | 2 | 0 | 23/25 | 0 | 2 |
| left | blocked | 15 | 21/45 | 24 | 0 | 9/15 | 0 | 6 |
| right | clear | 22 | 58/66 | 8 | 0 | 19/22 | 1 | 2 |
| right | blocked | 15 | 12/45 | 33 | 0 | 4/15 | 1 | 10 |
| front | blocked | 62 | 0/186 | 6 | 180 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 266/279 | 13 | 0 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 68/135 | 67 | 0 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 24, of which blocked: 22.

## B4

| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| back | clear | 46 | 136/138 | 2 | 0 | 45/46 | 0 | 1 |
| back | blocked | 15 | 35/45 | 10 | 0 | 13/15 | 1 | 1 |
| left | clear | 25 | 67/75 | 4 | 4 | 23/25 | 0 | 2 |
| left | blocked | 15 | 18/45 | 25 | 2 | 9/15 | 0 | 6 |
| right | clear | 22 | 48/66 | 10 | 8 | 19/22 | 1 | 2 |
| right | blocked | 15 | 8/45 | 31 | 6 | 4/15 | 1 | 10 |
| front | blocked | 62 | 0/186 | 11 | 175 | 40/62 | 2 | 20 |
| back+left+right | clear | 93 | 251/279 | 16 | 12 | 87/93 | 1 | 5 |
| back+left+right | blocked | 45 | 61/135 | 66 | 8 | 26/45 | 2 | 17 |

Episodes of back+left+right that fail on all three seeds: 23, of which blocked: 20.
