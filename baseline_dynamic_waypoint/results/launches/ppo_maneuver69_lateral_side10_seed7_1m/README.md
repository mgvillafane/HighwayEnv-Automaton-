# Lateral-only route reward experiment

Fresh PPO training for exactly 1,000,000 transitions, seed 7, compared with
`ppo_maneuver69_notrack_side10_seed7_1p5m`.

The new `maneuver_lateral_side10` profile uses:

```
ey = goal_lane_center_y - ego_y
route_cost = 2 * abs(ey)
route_progress = 0.5 * (previous_route_cost - current_route_cost)
```

Equivalently, route reward is the decrease in absolute lateral error in meters.
The implementation uses the goal lane's local lateral coordinate, so it does
not depend on global axis assumptions. FRONT uses the same fixed goal-lane
target as every other task; no passing-lane selection or lead-dependent blend
enters the lateral error. The longitudinal and preferred-speed route-cost
weights are both zero. Speed and longitudinal observations remain available.

The original distance-progress term (weight 0.1), FRONT-only one-time alongside
milestone (+10), goal (+50), collision (-50), timeout (-10), time (-0.05/tick),
hold (+0.2/tick), success checks, and 69 input layout are retained. Tracking is zero.
Moving one 4 m lane away from the FRONT goal lane costs 4 route reward;
the alongside milestone supplies +10, and returning recovers 4 route reward.
This tradeoff deliberately replaces dense passing guidance with the milestone.

Other settings match the comparison: 4 environments, 250 steps/environment per
rollout, learning rate 0.0001, gamma 0.99, dropout 0.05, all four goals, 100 held-out
evaluation episodes starting at seed 10000 before/after training and every
250,000 transitions. Checkpoints are saved every 25,000 transitions.

`launch.ps1` saves the source snapshot, launches hidden Python, and records PID,
arguments, and paths in `launch.json`. Logs are `training_stdout.log` and
`training_stderr.log`; live training artifacts are in
`../../ppo_maneuver69_lateral_side10_seed7_1m/`.
