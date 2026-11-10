# What-if: B3 plus the scripted driver's following rule

Nothing is trained. The policy is replayed deterministically on the BACK/LEFT/RIGHT episodes, and one
hand-written rule is laid over its action: when the gap to the car ahead in ego's lane is below
12 m + 0.8 s x ego speed, the target speed may not exceed the anchor's speed (10 m/s below a 9 m gap).
Then SLOWER replaces the action while the target speed is above that, and IDLE replaces FASTER at it.
Lane changes are the policy's own. "Without the rule" is the saved evaluation.

## B3_seed7

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 90 | 3 | 0 | 87 | 1 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 2 | 13 |
| blocked, blocking car leaves by itself | 27 | 21 | 6 | 0 | 23 | 0 | 4 |
| all BACK/LEFT/RIGHT | 138 | 113 | 25 | 0 | 113 | 3 | 22 |

- The rule changed 4,897 of 12,841 decisions, in 92 of 138 episodes.
- Episodes gained: [10002, 10071, 10117].
- Episodes lost: [10039, 10053, 10098].
- Still crashing: [10030, 10088, 10177].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3.md: a blocking car has more time to leave when ego waits behind it.

## B3_seed8

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 89 | 4 | 0 | 87 | 1 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 3 | 12 |
| blocked, blocking car leaves by itself | 27 | 21 | 6 | 0 | 22 | 2 | 3 |
| all BACK/LEFT/RIGHT | 138 | 112 | 26 | 0 | 112 | 6 | 20 |

- The rule changed 4,913 of 12,196 decisions, in 105 of 138 episodes.
- Episodes gained: [10002, 10018, 10071, 10117].
- Episodes lost: [10053, 10098, 10136, 10195].
- Still crashing: [10030, 10088, 10104, 10120, 10136, 10177].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3.md: a blocking car has more time to leave when ego waits behind it.

## B3_seed9

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 90 | 3 | 0 | 87 | 1 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 3 | 12 |
| blocked, blocking car leaves by itself | 27 | 20 | 7 | 0 | 22 | 1 | 4 |
| all BACK/LEFT/RIGHT | 138 | 112 | 26 | 0 | 112 | 5 | 21 |

- The rule changed 5,028 of 12,324 decisions, in 94 of 138 episodes.
- Episodes gained: [10002, 10071, 10117].
- Episodes lost: [10039, 10053, 10098].
- Still crashing: [10030, 10088, 10104, 10136, 10177].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3.md: a blocking car has more time to leave when ego waits behind it.
