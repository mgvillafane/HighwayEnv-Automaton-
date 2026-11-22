# Reward, scenario sampling and RIGHT-action audit

## Current reward

Per 0.1-second physics tick:

`r = 50*success - 50*collision + 0.1*(previous_distance-current_distance) - 0.05 + 0.2*inside_valid_slot`

`distance` is Euclidean distance to the moving goal. The lane, speed-match and overtake weights are all zero. Committed lane actions return the sum of their internal physics rewards. Time costs 0.5 reward per simulated second; valid-slot occupancy earns 2 reward per simulated second before termination.

## Actual completed training scenarios

1236 completed episodes; the sampler chooses each goal with probability 25%. The last unfinished episodes at the 200,000-decision limit are excluded from this table.

| Goal | Episodes | Episode share | Recorded PPO decisions | Decision share | Training successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FRONT | 323 | 26.13% | 81,891 | 41.07% | 5 |
| BACK | 291 | 23.54% | 12,733 | 6.39% | 238 |
| LEFT | 305 | 24.68% | 20,971 | 10.52% | 238 |
| RIGHT | 317 | 25.65% | 83,822 | 42.03% | 0 |

FRONT and RIGHT together account for 51.78% of completed episodes and 83.10% of recorded PPO decisions. They receive many decisions because their episodes usually reach the deadline. They are not undersampled.

## Current deterministic policy: 100 evaluation episodes

| Goal | Evaluation episodes | LEFT commands | RIGHT commands |
| --- | ---: | ---: | ---: |
| FRONT | 18 | 180 | 0 |
| BACK | 27 | 102 | 13 |
| LEFT | 33 | 15 | 8 |
| RIGHT | 22 | 3 | 0 |

The policy executes RIGHT in BACK and LEFT tasks, but selects no RIGHT actions in the inspected FRONT or RIGHT tasks. This is goal-conditioned behavior, not a general inability to execute a rightward lane change.
For RIGHT-task states where the goal lane is above the requested lane index, the mean RIGHT-action probability is 0.0337% across 4,500 decisions. The corresponding mean probabilities are 44.37% in BACK-task states and 58.02% in LEFT-task states that require a higher lane index.
A direct RIGHT command in the recorded RIGHT scenario completed a lane change from 0 to 1 in 1.9 seconds without collision, confirming the action mechanism works.

The saved policy therefore has a strong goal-dependent action bias. The observations and mechanics permit RIGHT, and goal exposure is balanced, but the RIGHT task has zero successful training episodes. The current reward has no separate lateral correction term, so early lateral feedback can be weak. This audit identifies the behavior; it does not establish one exclusive learning cause.

[Full counts, probabilities and controller check](reward_sampling_audit.json)
