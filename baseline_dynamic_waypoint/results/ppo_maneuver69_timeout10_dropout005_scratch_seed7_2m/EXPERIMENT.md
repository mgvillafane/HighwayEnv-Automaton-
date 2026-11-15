# FRONT/RIGHT navigation guidance, fresh 2M-step PPO

Run: `ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m`.

The previous fresh 53-input run had no deterministic FRONT/RIGHT successes at
any monitored evaluation through 1.25M steps; it was stopped with a saved 1.45M
checkpoint. This experiment supplies local navigation features and denser
geometric feedback rather than reusing its failed policy.

## Observation changes

Keep the original 53 legacy inputs and append 16 normalized `navigation`
features, for **69 scalar inputs**:

1. Anchor lane minus ego lane.
2. Goal lane minus ego lane.
3. Requested lane minus ego lane.
4. Lateral error to the requested lane center.
5. Anchor longitudinal gap.
6. Anchor minus ego speed.
7. Signed longitudinal slot error on a 60 m scale, wider than the existing 10 m local scale.
8. Lateral error to the navigation target.
9. Preferred speed minus ego speed.
10. Preferred speed.
11. FRONT merge fraction.
12. FRONT pass fraction.
13. Clearance beyond the FRONT passing margin.
14. Whether a left lane exists.
15. Whether a right lane exists.
16. Closing collision risk from bumper gap, relative speed, and lateral overlap.

Lane differences are divided by two; requested-lane lateral error by 4 m;
navigation lateral error by 8 m; gaps by 60 m; speed errors by 10 m/s;
preferred speed by 25 m/s. Features are clipped to [-1, 1].

The preferred speed is `traffic_speed + clip(0.25 * longitudinal_slot_error,
-5, 5)`, clipped to the existing 10–25 m/s speed range. This exposes the
catch-up/braking target as an observation; PPO still selects every native action.

For FRONT, the navigation target is initially the nearest lane beside the
traffic car. Once ego leads it by 10 m, the target blends toward the FRONT goal
lane, reaching that lane at a 20 m lead. For other goals, it is the actual goal
lane. No policy reflection, forced steering, or action masking is applied.

## Reward changes

Retain the previous reward:

`base = 50*success - 50*collision + 0.1*(previous_distance-distance) - 0.05 + 0.2*inside_valid_slot - 10*timeout`

Add, per 0.1-second tick:

`route_progress = 0.5*(previous_route_cost-route_cost)`

`route_cost = abs(longitudinal_slot_error) + 2*abs(navigation_lateral_error) + abs(preferred_speed-ego_speed)`

`tracking = -0.05*(0.4*L + 0.25*D + 0.25*S + 0.1*risk)`

Here `L`, `D`, and `S` are absolute navigation lateral, longitudinal, and speed
errors divided by 8 m, 60 m, and 10 m/s respectively, each capped at one.
`risk` is in [0, 1], reaching one for zero bumper clearance while closing in
the same path, and zero when separated laterally or not closing. Its horizon
is five seconds. The extra tracking cost is always between -0.05 and zero;
remaining near a subgoal does not earn a repeated positive tracking bonus.

Route progress is a state-error difference, not a guaranteed policy-invariant
shaping term under PPO's discount. Its shaped return is not directly comparable
with the previous run's return. Compare success and collision rates per goal.

## Preserved experimental settings

- Fresh actor, critic, and optimizer; seed 7; exactly 2,000,000 transitions.
- Four environments; 250 steps each; 1,000 transitions per rollout.
- Learning rate 1e-4; gamma .99; GAE lambda .95; batch size 250; five epochs.
- Actor and critic [256, 256], Tanh, dropout .05; entropy .01; value coefficient
  .5; clip .2; maximum gradient norm .5; CPU.
- Ego plus one constant-speed traffic car; three lanes; randomized lanes;
  traffic speed 12–18 m/s; ego initial speed 20 m/s; 35 m initial anchor gap.
- All four goals sampled as before, ordinary per-tick lane decisions, 30 s
  episodes, 10 Hz physics and decisions.
- Original success box, speed tolerance, 0.5 s hold, and 0.3 s grace remain.
- Timeout penalty 10; collision penalty 50; success bonus 50.

Checkpoint every 25k steps. Deterministic monitoring uses the same 100 seeds
(10000–10099) before training, every 250k, and at completion. These reused
monitoring cases are not an independent test set. The best checkpoint is ranked
by success, then fewer collisions, then reward. No improvement is assumed in
advance; FRONT and RIGHT evaluation rates determine whether this helps.

The new guidance is opt-in through reward profile `maneuver10`. Checkpoint
sidecars store its feature settings and reward weights. Other profiles and old
checkpoints retain their existing environment and observations.

## Validation

25 targeted tests passed across `test_maneuver_guidance`, `test_single_car`,
`test_dropout_timeout`, and `test_training`: unchanged original physics and
success, mirrored LEFT/RIGHT feedback, FRONT pass/merge staging, risk features,
bounded costs, timeout accounting, training CSV logging, dropout, save/reload,
and TensorBoard integration. These checks validate implementation, not learning
performance.

Two older multi-car tests in `test_environment` fail with the existing legacy
settings: positive lane-shaping reward while its default weight is zero, and
the scripted driver's FRONT five-second hold. The new single-car profile does
not alter that legacy environment.

Live comparison dashboard: http://127.0.0.1:6018/live
