# What-if: B3m plus the scripted driver's following rule

Nothing is trained. The policy is replayed deterministically on the BACK/LEFT/RIGHT episodes, and one
hand-written rule is laid over its action: when the gap to the car ahead in ego's lane is below
12 m + 0.8 s x ego speed, the target speed may not exceed the anchor's speed (10 m/s below a 9 m gap).
Then SLOWER replaces the action while the target speed is above that, and IDLE replaces FASTER at it.
Lane changes are the policy's own. "Without the rule" is the saved evaluation.

## B3m_seed7

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 88 | 5 | 0 | 85 | 3 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 2 | 13 |
| blocked, blocking car leaves by itself | 27 | 21 | 6 | 0 | 23 | 0 | 4 |
| all BACK/LEFT/RIGHT | 138 | 111 | 27 | 0 | 111 | 5 | 22 |

- The rule changed 5,024 of 12,660 decisions, in 96 of 138 episodes.
- Episodes gained: [10002, 10071, 10094, 10117].
- Episodes lost: [10039, 10053, 10072, 10098].
- Still crashing: [10030, 10064, 10088, 10177, 10194].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3m.md: a blocking car has more time to leave when ego waits behind it.

## B3m_seed8

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 90 | 3 | 0 | 87 | 1 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 2 | 13 |
| blocked, blocking car leaves by itself | 27 | 21 | 6 | 0 | 22 | 1 | 4 |
| all BACK/LEFT/RIGHT | 138 | 113 | 25 | 0 | 112 | 4 | 22 |

- The rule changed 5,097 of 12,422 decisions, in 95 of 138 episodes.
- Episodes gained: [10002, 10071, 10117].
- Episodes lost: [10039, 10053, 10098, 10136].
- Still crashing: [10030, 10088, 10136, 10177].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3m.md: a blocking car has more time to leave when ego waits behind it.

## B3m_seed9

| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 93 | 88 | 5 | 0 | 87 | 1 | 5 |
| blocked, blocking car stays | 18 | 2 | 16 | 0 | 3 | 2 | 13 |
| blocked, blocking car leaves by itself | 27 | 20 | 7 | 0 | 23 | 0 | 4 |
| all BACK/LEFT/RIGHT | 138 | 110 | 28 | 0 | 113 | 3 | 22 |

- The rule changed 5,169 of 12,530 decisions, in 99 of 138 episodes.
- Episodes gained: [10002, 10034, 10071, 10106, 10117, 10119].
- Episodes lost: [10039, 10053, 10098].
- Still crashing: [10030, 10088, 10177].
- "Stays" and "leaves" are measured in this replay, so they can differ from passing_B3m.md: a blocking car has more time to leave when ego waits behind it.
