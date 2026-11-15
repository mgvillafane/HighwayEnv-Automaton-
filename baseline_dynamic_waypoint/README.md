# Dynamic waypoints: reach and hold a slot of a moving car

Latest replacement: `ppo_targetlanes57_lr4e5_time045_dropout005_gamma097_seed7_1m`
completed 1M steps from scratch with LR 0.00004 and per-tick time cost -0.045.
Final evaluation: 25% success, 9% collisions, 66% timeouts; best monitored
checkpoint: 28% success at 750k (21% collisions). FRONT and RIGHT finished at
0%; BACK 40.7%, LEFT 42.4%. No lane-change-required case succeeded.
[TensorBoard, port 6016](http://127.0.0.1:6016/live) |
[Report](results/ppo_targetlanes57_lr4e5_time045_dropout005_gamma097_seed7_1m/TRAINING_REPORT.md).
The preceding stability-diagnostics run was cancelled with its 650k checkpoint preserved.

The [stability audit](STABILITY_AUDIT.md) records actual run settings, optimizer
metrics, reward/termination conditions, and conservative diagnostic changes.
Single-car evaluation now retains `_best.zip` and an environment sidecar,
ranked by success, fewer collisions, then reward. Existing periodic evaluation
uses `--eval-every 250000`; it does not run additional evaluation passes.

## Fresh 1M run with 57 inputs and explicit target lanes

`--observation-schema legacy_target_lanes` retains the legacy committed-lane
layout while replacing `achieved_goal` (2 redundant ego-position values) with
`target_lanes` (2 raw lane IDs: `[target_car_lane, target_goal_lane]`). Lane IDs
are 0, 1, 2 from left to right and are recomputed from the simulator state at
every observation, including after committed lane actions. `vehicles`, `anchor`,
`desired_goal`, `goal_delta`, `slot`, `slot_error`, `hold_progress`,
`requested_lane`, and `lane_change_in_progress` retain their legacy meanings.
There are exactly **57 inputs**. Historical `legacy` checkpoints are unchanged.

`ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m` uses fresh policy and
optimizer weights, all four goals, timeout -10, and dropout 0.1 in actor and
critic. It evaluates the same 100 held-out seeds before training, every 250k
steps, and at the final 1M steps; checkpoints are saved every 25k steps.

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema legacy_target_lanes --enabled-slots front back left right --reward-profile timeout10 --dropout-rate 0.1 --timesteps 1000000 --num-envs 4 --rollout-steps 250 --seed 7 --eval-episodes 100 --eval-seed 10000 --eval-every 250000 --run-name ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m --quiet
```

[TensorBoard](http://127.0.0.1:6015/live) |
[Configuration](results/ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m/run.json)

Completed exactly 1M steps from scratch in 24.15 minutes. On 100 fixed evaluation
cases, success rose from 0% to 30%: FRONT 0/18, BACK 11/27 (40.7%), LEFT 15/33
(45.5%), and RIGHT 4/22 (18.2%). Collisions were 21% and timeouts 49%. All
successes started in the goal lane: 30/36 such cases succeeded, while 0/64 cases
requiring a lane change succeeded. The explicit lane inputs were verified for all
100 evaluation resets. All 21 observation/dropout/timeout/sampling checks passed.

[Report](results/ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m/TRAINING_REPORT.md) |
[Video](results/ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m/videos/all_goals.mp4) |
[Lane audit](results/ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m/lane_observation_audit.json)

## Restored legacy observation and timeout -10

`GOAL_CONFIG["observation_schema"]` now defaults to `legacy`. With committed
lane actions this restores the 57 inputs used by the earlier BACK/LEFT policy:
`vehicles` (7x5), `achieved_goal` (2), `desired_goal` (2), `goal_delta` (2),
`anchor` (4), `slot` (4), `slot_error` (3), `hold_progress` (1),
`requested_lane` (3), and `lane_change_in_progress` (1).

The `ppo_legacy57_timeout10_dropout01_seed7_200k` experiment starts from
`ppo_singlecar_committed57_seed7_200k.zip`, retains all its learned input
weights, enables all four goals, and adds dropout 0.1 in actor/critic. Adding
dropout resets the optimizer. The `timeout10` profile charges -10 once on timeout.

Completed 200,000 additional steps: held-out success changed from 55% to 39%.
BACK changed from 81.5% to 59.3%, LEFT from 100% to 69.7%, and FRONT/RIGHT
remained at 0%. Collisions changed from 7% to 11% on the same 100 cases.

[TensorBoard](http://127.0.0.1:6014/live) |
[Run report](results/ppo_legacy57_timeout10_dropout01_seed7_200k/TRAINING_REPORT.md) |
[Configuration](results/ppo_legacy57_timeout10_dropout01_seed7_200k/run.json)

## Continue from 2 million to 4 million steps

The `ppo_allgoals_timeout25_dropout01_seed7_4m` run continues the final 2M
checkpoint for **2 million additional PPO decisions** with its saved optimizer.
All four goals, the timeout -25 cost, dropout 0.1, the 54-input observation,
and the remaining settings match the source run.

[Original 2M video](results/ppo_allgoals_timeout25_dropout01_seed7_2m/videos/all_goals.mp4) |
[TensorBoard comparison](http://127.0.0.1:6013/live) |
[Continuation configuration](results/ppo_allgoals_timeout25_dropout01_seed7_4m/run.json)

The continuation's TensorBoard step counter measures additional steps:
0 corresponds to 2M lifetime steps, and 2M corresponds to 4M lifetime steps.

## All goals, terminal timeout cost, and dropout

The `ppo_allgoals_timeout25_dropout01_seed7_2m` run starts from scratch for
2,000,000 PPO decisions with FRONT, LEFT, BACK, and RIGHT enabled. It retains
the 54-input observation and committed lane actions. The `timeout25` reward
profile adds -25 once when an episode times out; success, collision, and anchor
loss do not receive that cost.

`DropoutMultiInputPolicy` adds dropout 0.1 after each of the two hidden Tanh
activations in both actor and critic. Dropout is active during gradient updates
and disabled for rollout collection and evaluation. Linear parameter names are
preserved so compatible checkpoint weight transfer continues to work.

[Live TensorBoard](http://127.0.0.1:6012/live) | [Run configuration](results/ppo_allgoals_timeout25_dropout01_seed7_2m/run.json)

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema lane_goal_v2 --enabled-slots front left back right --reward-profile timeout25 --dropout-rate 0.1 --timesteps 2000000 --num-envs 4 --rollout-steps 250 --seed 7 --eval-episodes 100 --eval-seed 10000 --eval-every 250000 --run-name ppo_allgoals_timeout25_dropout01_seed7_2m --quiet
```

Use a fresh run name to repeat it. Checkpoints are saved every 25,000 steps;
100 fixed held-out seeds are evaluated before training, every 250,000 steps,
and after training. Terminal timeout costs are recorded in `reward_timeout`
and `reward_components/timeout`.

## LEFT/BACK-only comparison

The [LEFT/BACK run](results/ppo_singlecar_left_back_only_scratch_seed7_200k/TRAINING_REPORT.md)
uses the same settings as the FRONT/RIGHT-only experiment, with fresh weights,
seed 7, 54 inputs, and exactly 200,000 PPO decisions. Only the enabled goal pair
changes (`--enabled-slots left back`); FRONT/RIGHT are rejected in training and
evaluation. On 100 held-out cases, LEFT scored 0/45 and BACK 0/55, with zero
collisions and 100% timeouts. Both runs are displayed in
[TensorBoard](http://127.0.0.1:6011/live).

## FRONT/RIGHT enabled, BACK/LEFT disabled

`train_single_car --enabled-slots right front` restricts both training and
evaluation to those goals. Explicit resets requesting BACK or LEFT are rejected.
This differs from `--training-slots`, which restricts training sampling while
allowing all four goals during evaluation.

The fresh 200k run is available in [TensorBoard](http://127.0.0.1:6010/live),
with per-goal training/evaluation curves and sampled episode counts. See its
[report](results/ppo_singlecar_front_right_only_scratch_seed7_200k/TRAINING_REPORT.md)
and [saved configuration](results/ppo_singlecar_front_right_only_scratch_seed7_200k/run.json).

## Optional lane-aware observation: lane_goal_v2

Select `--observation-schema lane_goal_v2` to use the alternative layout.
It contains six dictionary fields, totaling **54 inputs** with seven vehicle rows:

| Field | Shape | Encoding |
| --- | --- | --- |
| `vehicles` | `(7, 6)` | `[presence, x, y, vx, vy, lane_id]`; original five normalized features plus raw lane ID |
| `goal_delta` | `(2,)` | `(goal_position - ego_position) / [200, 12]` |
| `anchor` | `(4,)` | Original normalized/clipped anchor-minus-ego x, y, vx, vy |
| `lane_difference` | `(2,)` | `[anchor_lane - ego_lane, goal_lane - ego_lane]`; signed lane counts |
| `desired_goal` | `(3,)` | `[target_car_id, relative_x / 200, relative_y / 12]`; x/y relative to the target car |
| `hold_progress` | `(1,)` | Valid hold ticks / required hold ticks, capped at 1 |

Lane IDs start at zero; a positive lane difference means right. Vehicle rows keep
the native nearest-car sorting, with missing rows entirely zero. In the single-car
task there are two occupied rows and five padded rows. The target car ID is its
stable index in `road.vehicles` during the episode, matching `info["anchor_id"]`;
it is independent of nearest-car row ordering. The explicit `anchor` field identifies
the target's state even when the target is outside the nearest six traffic rows.

Before coordinate scaling, target-relative goal offsets are FRONT `[20, 0]`,
BACK `[-15, 0]`, LEFT `[0, -4]`, and RIGHT `[0, 4]` meters on this straight road.
`desired_goal` and `goal_delta` are scaled without clipping. Lane IDs, lane
differences, and the target car ID are raw integer-valued floats. This layout
replaces `achieved_goal`, `slot`, `slot_error`, `requested_lane`, and
`lane_change_in_progress` inputs. Committed lane actions still settle before the
next policy decision, and goal success uses the configured position, speed, and
hold requirements.

Historical sidecars without `observation_schema` rebuild the legacy layout.
Both training entry points accept `--observation-schema legacy` for reproducing
historical runs. Continuing an old model into the new layout copies the retained
vehicle, anchor, goal-delta, and hold input weights, resets the optimizer, and
starts the added/redefined inputs at zero. Dropping old inputs changes the policy's
outputs; this is a partial transfer rather than an exact policy-preserving expansion.

Completed 54-input experiment: [report and comparison](results/ppo_singlecar_lane_goal_v2_front_right_seed7_200k/TRAINING_REPORT.md)
and [all-goal replays](results/ppo_singlecar_lane_goal_v2_front_right_seed7_200k/videos/all_goals.mp4).
After exactly 200,000 additional decisions on FRONT/RIGHT only, held-out success
rose from 14% immediately after transfer to 28%, with 0% collisions. FRONT and
RIGHT each remained at 0%; BACK reached 37.0% and LEFT 54.5%. The original
57-input source checkpoint had 55% overall success on the same 100 scenarios.

The [from-scratch 200k experiment](results/ppo_singlecar_lane_goal_v2_front_right_scratch_seed7_200k/TRAINING_REPORT.md)
used random weights, the same 54 inputs, and FRONT/RIGHT-only sampling.
It recorded 39 RIGHT and six FRONT training successes, but deterministic
held-out success remained 0% on both trained goals. Collisions on the 40
FRONT/RIGHT evaluation cases rose from 0% in the transferred run to 40% from
scratch. Overall success across all four goals was 3%, versus 28% after transfer.
[Watch the from-scratch replays](results/ppo_singlecar_lane_goal_v2_front_right_scratch_seed7_200k/videos/all_goals.mp4).

## Front/right-only continuation

`train_single_car --training-slots right front` samples those two goals uniformly
during training. Its before/after evaluation still samples all four goals on the
same held-out seeds, so BACK/LEFT retention is measured as well. The allowed
training goals are stored in checkpoint sidecars as `goal_config.sampled_slots`.
Explicit `reset(options={"slot": ...})` calls can inspect any valid goal.

This experiment resumes the original committed-lane checkpoint with its saved
policy and optimizer, baseline rewards, and 57 observation inputs:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --observation-schema legacy --committed-lanes --training-slots right front --timesteps 200000 --num-envs 4 --rollout-steps 250 --seed 7 --init-model baseline_dynamic_waypoint/models/ppo_singlecar_committed57_seed7_200k.zip --reward-profile baseline --run-name ppo_singlecar_front_right_seed7_200k --eval-episodes 100 --eval-seed 10000 --quiet
```

Use a fresh run name to repeat it. See the [experiment report](results/ppo_singlecar_front_right_seed7_200k/TRAINING_REPORT.md)
and [all-goal videos](results/ppo_singlecar_front_right_seed7_200k/videos/all_goals.mp4).
The completed 200,000-decision continuation produced no FRONT/RIGHT successes.
On the same 100 held-out scenarios, overall success fell from 55% to 28%;
BACK fell from 81.5% to 51.9% and LEFT from 100% to 42.4%.

## Historical committed LEFT/RIGHT actions and 57-input PPO

The `ppo_singlecar_committed57_seed7_200k` experiment uses `CommittedLaneEnv`
in `committed_lane.py`. LEFT or RIGHT changes the requested lane once. The
environment then continues automatic IDLE control at 10 Hz until the car is
within 0.2 m of that lane's center and lateral speed is at most 0.2 m/s.
PPO receives its next decision only then. No additional lane or speed command
is applied during the maneuver. Collisions, anchor loss or the 30-second
episode deadline can interrupt it. Success waits for the lane maneuver to settle.

The observation adds `requested_lane` (three-value one-hot encoding) and
`lane_change_in_progress` (one flag), for 57 inputs total. The flag is set on
intermediate physics observations and interrupted terminal observations; at
normal PPO decision boundaries the maneuver is finished and the flag is zero.
The relative LEFT/RIGHT action meanings are preserved.

Reward weights and per-physics-step terms are unchanged. A committed action
returns the sum of its internal physics rewards. The 200,000-step budget counts
PPO decisions, with physics ticks counted separately. Gamma remains 0.99 per
PPO decision, so the physical time covered by one transition varies.

The latest flexible-hold policy is transferred by copying all existing weights
and remapping the actor/critic input columns by observation key. The four new
input connections start at zero, preserving the original policy probabilities
and value estimates before training. The optimizer is reset for the expanded
layers. Old environments and checkpoints keep their original observation layout.

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --observation-schema legacy --committed-lanes --timesteps 200000 --num-envs 4 --rollout-steps 250 --init-model baseline_dynamic_waypoint/models/ppo_singlecar_flexible30s_seed7_200k.zip --run-name ppo_singlecar_committed57_seed7_200k --eval-episodes 100 --quiet
```

Live TensorBoard: **http://127.0.0.1:6008/live**. It includes
`lane_changes/started`, `lane_changes/completed`, `lane_changes/interrupted`,
and `time/physics_steps` alongside the usual success and collision charts.
The earlier flexible experiment remains on port 6007.

Completed: [training report](results/ppo_singlecar_committed57_seed7_200k/TRAINING_REPORT.md)
and [full-physics PPO videos](results/ppo_singlecar_committed57_seed7_200k/videos/SCENARIOS.md).
The run completed 200,000 PPO decisions and 252,891 physics ticks. All 2,813
started lane changes completed (2,737) or ended in a terminal interruption (76).
On 100 paired scenarios, success rose from 46% to 55% in the new interface,
with 7% collisions after training. FRONT and RIGHT remain unsuccessful.

The NumPy kinematics encoder is enabled only for this profile; exact observation,
reward and physics equivalence to the DataFrame encoder is tested. All rewards
and observations are still updated on every 0.1-second internal tick.

## Randomized one-car experiment with flexible holding

The 30-second rerun is implemented and stored entirely under
`baseline_dynamic_waypoint/`: `single_car.py`, `train_single_car.py`, and the
`ppo_singlecar_flexible30s_seed7_200k` model/results. The existing multi-car
configuration and older saved sidecars retain their original rules. The simple
environment's entry point remains as a compatibility wrapper for its old presets.

| Setting | Previous two-car run | Flexible rerun |
| --- | ---: | ---: |
| Episode limit | 15 s | 30 s |
| Longitudinal goal tolerance | +/-3 m | +/-5 m |
| Lateral tolerance | +/-1.5 m | +/-1.8 m |
| Speed difference allowed | 0.5 m/s | 2.0 m/s |
| Valid time required | 1.0 s | 0.5 s |
| Brief exit grace | None | 0.3 s |

Grace pauses the counter for up to three consecutive invalid decisions at
10 Hz. It awards no hold reward or success outside the target. A fourth invalid
decision resets the count. Collisions and anchor loss reset it immediately.
The car must be in the valid target region when success is declared. Old goal
configurations without `hold_grace_seconds` still reset on the first missed step.

The experiment samples all four slots, traffic speeds of 12-18 m/s, and both
starting lanes while keeping side goals on the road. Ego's maximum target speed
is 25 m/s. It repeats 200,000 additional transitions from the same original
gamma-0.99 dynamic-waypoint checkpoint used for the preceding experiment.

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --observation-schema legacy --timesteps 200000 --num-envs 4 --rollout-steps 250 --init-model baseline_dynamic_waypoint/models/ppo_front20m_gamma099_seed7_1m.zip --run-name ppo_singlecar_flexible30s_seed7_200k --eval-episodes 100 --quiet
```

Use a new run name to repeat the command; existing runs are protected.
The profile settings are `SINGLE_CAR_ENV_CONFIG` and `SINGLE_CAR_GOAL_CONFIG`
in `config.py`. Before/after evaluations share 100 seeds under the new rules.
Success rates from this run and the previous stricter run use different success
definitions and episode limits.

Live TensorBoard for this experiment: **http://127.0.0.1:6007/live**.
The monitor reloads events every five seconds and browser charts every thirty
seconds. Useful charts include `episodes/success_mean_100`,
`episodes/collision_mean_100`, `episodes/max_hold_steps_mean_100`,
`evaluation/*`, and `setup/*`.

Completed rerun: [training report](results/ppo_singlecar_flexible30s_seed7_200k/TRAINING_REPORT.md)
and [trained-policy videos](results/ppo_singlecar_flexible30s_seed7_200k/videos/SCENARIOS.md).
Under the new criteria, evaluation success increased from 26% to 60% and
collisions decreased from 26% to 6% on the same 100 scenarios. BACK and LEFT
succeeded in every evaluated case; FRONT and RIGHT still had no successes.

This extends Marcos's [`baseline_waypoint/`](../baseline_waypoint/README.md) PPO baseline.
There, the goal is a fixed world point. Here, the goal is a **slot relative to a moving
traffic car** (the *anchor*): its FRONT, BACK, LEFT or RIGHT. Success means staying in
that slot for a configurable hold time (1 s, 3 s or 5 s). Nothing in `baseline_waypoint/`
or the installed simulator is modified: `DynamicGoalHighwayEnv` subclasses
`GoalHighwayEnv`, and the viewer, plots and CSV helpers are imported from it.

## Start here

1. `config.py`: all local environment, goal, reward, PPO, seed and training defaults.
   Edit this file to tune the dynamic task independently of `baseline_waypoint/config.py`.
2. `environment.py`: anchor/slot sampling -> per-step slot update -> hold counter -> reward.
3. `scripted.py`: a rule-based reference driver, used as a feasibility test and yardstick.
4. `train.py`, `evaluate.py`, `test.py`: same structure and flags as Marcos's scripts,
   plus `--hold-seconds`.

## How the dynamic waypoint works

At each reset:

- The **anchor** is drawn uniformly from the traffic cars 30-120 m ahead of ego.
- A **slot** is drawn uniformly from those that lie on the road: no LEFT in lane 0,
  no RIGHT in the last lane.
- The anchor's MOBIL lane changes are switched off (`enable_lane_change = False`), so
  its LEFT/RIGHT slot can never move off the road. IDM still controls its speed and
  following, including reacting to ego.

After every simulator step, the slot is **recomputed in the anchor's lane frame**:

```python
lane = road.network.get_lane(anchor.lane_index)
s, _ = lane.local_coordinates(anchor.position)
FRONT: lane.position(s + 20, 0)        BACK: lane.position(s - 15, 0)
LEFT/RIGHT: adjacent_lane.position(s, 0)   # lane ID -1 / +1, the same as LANE_LEFT / LANE_RIGHT
```

The same `goal_position` attribute as the static task is updated in place, so the
distance, progress reward and drawing code all keep working.

**Success box.** Ego's center must be within ±3 m along the slot lane and ±1.5 m across
it. It must stay there for `hold_seconds × 10` consecutive decisions, and leaving the box
resets the count. A collision on the same step takes priority. If the anchor crashes or
leaves the road, the episode is **truncated** as `ANCHOR LOST`; this is not counted as
ego's failure or as a timeout.

### Current settings and historical physics measurements

The current FRONT offset is 20 m. Earlier measurements with `scripted.py` used
longer hold times and several offsets; their results below describe those earlier tasks.

| Setting | Value | Why |
| --- | --- | --- |
| Target speeds | 10-25 m/s in **2.5 m/s steps** | `MDPVehicle` sets FASTER/SLOWER to one grid step from the *current* speed, so the step size also limits acceleration (about step / 0.6 s). With 1 m/s steps ego brakes at only ~1.7 m/s²: the scripted driver crashed in 41-46% of episodes and never held FRONT. Marcos's 5 m/s steps cannot match 12-17 m/s traffic closely enough for 5 s holds (33% vs 45% scripted success over all slots, measured with a 15 m FRONT offset). |
| FRONT offset | **20 m** center-to-center (BACK: 15 m) | Current experiment setting. The anchor's IDM driver follows ego after the pass. Historical scripted FRONT success with a 5 s hold, by offset: 15 m 0%, 25 m 20%, 30 m 33%, 35 m 65%. These are not measurements of the current 20 m, 1 s hold task. |

The episode lasts 25 s for every hold time, so the variants differ only in the hold requirement.

## Legacy observation layout (current default)

The following describes `observation_schema = "legacy"`, the current default,
and older sidecars. The optional six-field lane-aware layout is documented above.

All of Marcos's keys are kept. `desired_goal` and `goal_delta` now move with the anchor.

| Key | Shape | Meaning |
| --- | --- | --- |
| `vehicles` | `(7, 5)` | Unchanged Kinematics traffic rows |
| `achieved_goal`, `desired_goal`, `goal_delta` | `(2,)` each | Ego, slot, and slot minus ego, in the episode frame |
| `anchor` | `(4,)` | Anchor x, y, vx, vy **relative to ego**, normalized with the Kinematics ranges |
| `slot` | `(4,)` | One-hot of FRONT, BACK, LEFT, RIGHT |
| `hold_progress` | `(1,)` | Consecutive in-slot decisions ÷ required, in [0, 1] |
| `slot_error` | `(3,)` | Slot minus ego along the slot lane ÷ 10 m, across it ÷ 4 m, anchor minus ego speed ÷ 10 m/s, clipped to [-1, 1] |

Without the anchor's velocity, the policy would see a moving target only as noise in
`goal_delta`. With it, the policy can anticipate where the slot is going.

`slot_error` was added after the first runs failed. With `goal_delta`'s 200 m scale, the
±3 m success box spans only ±0.015 of the input. With `slot_error`'s local scale it spans
±0.3, and the speed mismatch is given directly. It is on by default (`slot_error_features`
in `config.py`); `train.py --no-slot-error` leaves it out. Model sidecars written before
the flag existed load without it, so the first 300k checkpoints still evaluate. The
policy is still SB3's default `MultiInputPolicy` (53 inputs, or 50 without
`slot_error`, with the configured 256-256 actor and critic MLPs), with no custom network.

## Reward

Set the eight weights in `config.py`'s `REWARD_WEIGHTS`. `DynamicGoalHighwayEnv.step()`
in `environment.py` computes their weighted components and returns their sum; it
discards the simulator's native driving reward. `PPO_CONFIG` controls the learning
algorithm separately. All defaults are defined locally, including the base weights.

```text
reward = 50 * success - 50 * collision + 0.1 * (previous_distance - distance)
         - 0.05 + 0.2 * in_slot
         + lane_weight * (previous_lateral_error - lateral_error)
         + speed_match_weight * (previous_speed_error - speed_error)
         + overtake_weight * (lead - previous_lead)  # FRONT only
```

The current configuration sets `lane`, `speed_match`, and `overtake` weights to
zero, so those three terms do not contribute to this run. Their behavior when
enabled is described below for interpreting earlier experiments.

The first four terms use the same formula as the static waypoint task. Distance is
measured to the *current* slot position, with progress using the previous step's
distance. `hold` pays 0.2 per decision spent in the box; an uninterrupted 5 s hold
adds 10 points, in addition to the 50-point goal bonus and other reward terms.

These are per-decision amounts. At the default 10 decisions per second, the time
penalty is 0.5 points per simulated second and uninterrupted holding earns 2 points
per second, in addition to the other terms. The default hold requirement is 1 s
(`GOAL_CONFIG["hold_seconds"]`); success requires remaining strictly within 3 m
longitudinally and 1.5 m laterally. Leaving the box resets the consecutive hold
counter. Collision prevents success, and anchor loss truncates the episode without
an extra terminal penalty. The time penalty and shaping terms still apply to the
terminal decision.

The `lane` and `speed_match` terms were added after the first runs: PPO never
moved sideways into a LEFT/RIGHT slot lane.
- `lateral_error` is ego's distance in meters from the slot lane's center, so each lane
  closed earns +4.
- `speed_error` is `|anchor speed - ego speed|`, weighted by a proximity factor that fades
  from 1 at the slot to 0 at 30 m (`shaping_range`). Far from the slot, ego should catch
  up, not match speed.

- `overtake` (FRONT only, weight 0.3) pays per meter of ego's lead over the anchor along
  its lane, capped at the 35 m slot offset. With this term enabled, a FRONT slot's lane
  target is the nearest lane *beside* the anchor until ego leads it by 10 m
  (`pass_margin`). Without that, the lane term charges -4 for pulling out to pass and
  pays it back only after the merge, which discourages exactly the maneuver FRONT needs.

The progress and shaping terms use consecutive differences. Their undiscounted
episode totals telescope to the change between the initial and final values.
PPO applies its configured discount to rewards, so this does not guarantee that
shaping preserves the optimal policy. `train.py --no-shaping` drops the lane,
speed_match and overtake terms, and
`--no-overtake` drops only the FRONT term. Checkpoints trained earlier keep
their five-term reward from their sidecar. Every term is in `info["reward_components"]`,
the CSVs and TensorBoard.

## Commands

The current run uses FRONT=20 m, gamma=0.99, seed 7, four simulators, and a target
of one million transitions, with the other settings from `config.py`. The earlier
gamma-0.9 experiment retains its saved configuration. From PowerShell:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B baseline_dynamic_waypoint/train.py --timesteps 1000000 --num-envs 4 --seed 7 --run-name ppo_front20m_gamma099_seed7_1m --quiet
```

Use a fresh run name if that experiment already exists. Training saves its initial
configuration immediately and writes reloadable checkpoints every 50,000
transitions. TensorBoard receives new points after each PPO rollout. The final
rollout may take the transition count slightly beyond one million.

To see all historical and new PPO and DQN runs together, start
`monitor_training.py` from the repository root and open
<http://127.0.0.1:6006/live>. See [the live-monitoring instructions](../TENSORBOARD.md).

Run from the repository root with the project's Python environment (`.venv`):

```bash
# Environment contract, slot geometry, motion, hold logic, scripted feasibility, rendering.
.venv/bin/python -B -m unittest baseline_dynamic_waypoint.test_environment -v

# Train (run name defaults to ppo_hold<seconds>s_seed<seed>) and evaluate on 50 paired seeds.
.venv/bin/python -B baseline_dynamic_waypoint/train.py --hold-seconds 3 --timesteps 300000 --quiet

# Re-evaluate: PPO vs random vs scripted on identical seeds.
.venv/bin/python -B baseline_dynamic_waypoint/evaluate.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --output-name eval_hold3s

# Watch or record; --cycle-slots shows front, back, left, right in turn.
.venv/bin/python -B baseline_dynamic_waypoint/test.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --cycle-slots
.venv/bin/python -B baseline_dynamic_waypoint/test.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --cycle-slots --headless --record --output-name videos_hold3s
```

The hold time is saved in each model's JSON sidecar, and `evaluate.py`/`test.py` rebuild
the task from it. Debug resets accept `options={"slot": "left", "anchor_index": 0}`
(eligible anchors are numbered nearest first). In the viewer, the anchor has a magenta
outline and the slot is a green box at the actual success tolerance; it thickens while
ego is holding. The HUD shows the slot, the relative speed and a hold-progress bar.

## Results

See [RESULTS.md](RESULTS.md).
