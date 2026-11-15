# Lane-control diagnosis

The recorded failure is caused by the policy selecting or retaining the wrong requested lane. The low-level steering controller completed a lane change in the isolated check. No environment, reward or trained weights were changed for this audit.

## Observed FRONT case: seed 10002

| Time before action | Physical lane | Requested lane before | PPO action | Requested lane after | Goal lane |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0.5 s | 2 | 2 | LEFT | 1 | 1 |
| 0.6 s | 2 | 1 | LEFT | 0 | 1 |

Each LEFT/RIGHT action changes the previously requested lane by one. It is not a command to keep moving in that direction toward one fixed lane. IDLE, FASTER and SLOWER retain the requested lane while steering continues.
The car reaches lane 0 and stays there. The FRONT episode has 53 LEFT commands and no RIGHT commands. At the end, longitudinal slot error is only -0.167 m, but lateral error is 4 m: one complete lane away.

## Other observed cases

- RIGHT, seed 10000: no lane-change command at all. The policy issues SLOWER and IDLE and stays in lane 0 although the goal is in lane 1.
- LEFT, seed 10004: repeated LEFT requests are followed by RIGHT requests 0.1-0.2 seconds later. This redirects the steering target during the maneuver and produces lateral wobble. It eventually succeeds.

## Controller and intervention checks

- One LEFT command followed by IDLE moved the car from lane 2 to lane 1. After three seconds, lateral error from that lane center was 0.0128 m.
- Replayed FRONT to t=25 s with unchanged PPO actions. At that point the car was in lane 0, with goal lane 1 and longitudinal error 0.935 m. Overriding one action to RIGHT, then using IDLE, produced GOAL REACHED at t=25.9 s.
- The intervention is a diagnostic, not a learned-policy result or an evaluation-rate improvement.

## Likely learning contributors

- The observation includes physical lateral position and velocity plus goal position/error, but omits the controller state `ego.target_lane_index`. The policy cannot directly observe the lane already requested by its previous actions.
- Lane changes may be retargeted every 0.1 s even though physical completion takes longer. Repeated requests can select another lane; opposite requests can reverse an unfinished maneuver.
- The saved run has lane reward weight zero. The Euclidean distance reward supplies weak lateral feedback far from the goal. At 35 m longitudinal error, correcting 4 m laterally changes distance by only sqrt(35^2+4^2)-35 = 0.228 m, corresponding to 0.0228 reward at progress weight 0.1.
- The policy exhibits a strong LEFT preference in FRONT and fails to request RIGHT in the inspected RIGHT episode. These are observed action biases; this audit does not establish a single exclusive cause of the learning failure.

Useful next changes: expose actual/requested/goal lane state explicitly, prevent repeated retargeting during an unfinished maneuver, and provide lateral-goal feedback with a pass-then-merge target for FRONT. Observation changes require a compatible policy update and training; the existing checkpoint was kept unchanged.

[Full lane targets, positions and action probabilities](lane_control_audit.json)
