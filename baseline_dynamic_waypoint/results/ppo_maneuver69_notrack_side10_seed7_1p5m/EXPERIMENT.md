# No tracking cost, FRONT side milestone: fresh 1.5M steps

Run: `ppo_maneuver69_notrack_side10_seed7_1p5m`.
Comparison: `ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m`.
Opt-in reward profile: `maneuver_side10`.

## Requested changes

The old `tracking` reward weight is **0.0**, so it contributes zero at every
step. The original 69 observations, collision-risk feature, preferred speed,
staged FRONT navigation, and route-progress reward are preserved.

A new **+10 intermediate milestone** is paid only in a FRONT episode, the
first time ego reaches the physical LEFT or RIGHT side slot of the anchor:

- Ego's actual lane is directly adjacent to the anchor lane, on either side.
- Absolute longitudinal center-to-center separation is strictly less than 5 m.
- Absolute lateral offset from that adjacent lane's center is less than 1.8 m.
- There is no ego collision or anchor loss on that step.

There is no speed-match or hold requirement for the intermediate milestone.
Requesting a lane change, or being in a side lane while still far behind, is
insufficient. Either side qualifies, but there is **only one +10 payment per
episode total**. Leaving, switching sides, or revisiting cannot earn another.
The flag resets with each episode. LEFT/BACK/RIGHT objectives never receive it.

Final FRONT success is unchanged: reach the FRONT slot 20 m ahead of the anchor,
within 5 m along and 1.8 m across, match speed within 2 m/s, and accumulate
0.5 s of valid holding with the existing 0.3 s exit grace.

## Full per-tick reward

```text
r = 50*success - 50*collision - 10*timeout - 0.05
    + 0.2*inside_valid_slot
    + 0.1*(previous_distance - current_distance)
    + 0.5*(previous_route_cost - current_route_cost)
    + 10*new_front_side_milestone

route_cost = abs(longitudinal_slot_error)
             + 2*abs(navigation_lateral_error)
             + abs(preferred_speed - ego_speed)
```

The component `tracking` remains in the logs with zero contribution. The new
component is `front_side`; `reward_front_side` appears in training CSV rows,
TensorBoard reward components, and evaluation reward summaries. Step info also
records `front_side_reached` and `front_side_new`. The once-per-episode flag is
reward bookkeeping in `info`, not an added observation; the feature count stays
69. This means reward history is not fully represented by the instantaneous
observation if ego revisits a side slot.

## Training and monitoring

- Fresh policy, critic, and optimizer; seed 7; exactly 1,500,000 transitions.
- Four environments, 250 steps each, 1,000 transitions per rollout.
- Same PPO settings: LR 1e-4, gamma .99, GAE lambda .95, batch 250, five epochs,
  clip .2, entropy .01, value coefficient .5, max gradient norm .5.
- Actor/critic [256, 256], Tanh, dropout .05, CPU.
- All four objectives sampled as before; three lanes; ego plus one traffic car.
- Ordinary 10 Hz decisions, 30-second episodes, constant traffic speed 12-18 m/s.
- Timeout -10, collision -50, final success +50, time cost -0.05 per 0.1 s tick.
- Checkpoints every 25k transitions. Evaluate 100 fixed seeds 10000-10099 before
  training, every 250k steps, and at 1.5M completion.

The previous experiment is longer (2M), so compare its saved **1.5M evaluation**
with this run's final 1.5M evaluation for an equal-budget comparison. Compare
success, collisions, and timeouts per goal; shaped returns use different rewards.
The reused monitoring seeds are not an independent test set. The new reward
changes are combined in one experiment and do not isolate their individual effects.

## Validation

30 targeted tests passed across `test_maneuver_guidance`, `test_single_car`,
`test_dropout_timeout`, and `test_training`. New checks cover zero tracking,
unchanged observations/physics, payouts on either side, one payment across
revisits/switches, reset behavior, actual alignment, FRONT-only eligibility,
collision/anchor-loss exclusion, logging, and checkpoint restoration.

Live comparison dashboard: http://127.0.0.1:6019/live
