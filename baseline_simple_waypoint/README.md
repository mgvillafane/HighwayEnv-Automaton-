# One traffic car: FRONT, BACK, LEFT and RIGHT goals

The latest relaxed-hold, 30-second rerun is in
[`baseline_dynamic_waypoint`](../baseline_dynamic_waypoint/README.md).
This folder preserves the earlier strict presets and results.

This is a separate environment in `baseline_simple_waypoint/`. The previous
`baseline_dynamic_waypoint/`, `baseline_waypoint/` and `baseline_DQN/` settings,
training scripts and saved models stay in their original folders.

There are **two vehicles total**: your blue controlled car and one amber traffic
car outlined in magenta. Both start in the middle of three lanes. Ego starts at
20 m/s, 35 m behind the traffic car, which holds 15 m/s and stays in its lane.
Each episode samples one of four moving goals uniformly:

| Goal | Target position relative to the traffic car |
| --- | --- |
| BACK | 15 m behind, in the same lane |
| FRONT | 20 m ahead, in the same lane; pass in an adjacent lane first |
| LEFT | Beside the car, centered in the lane to its left |
| RIGHT | Beside the car, centered in the lane to its right |

The traffic car starts in the middle lane so all four goals are on the road.
It uses a constant-speed lane controller; it does not brake in response to ego
after a pass. Collision detection remains active for both cars.

The task is to reach the selected green box and hold it for one second.
The episode lasts at most 15 seconds. Success requires remaining
within 3 m along the lane and 1.5 m laterally, with speed within 0.5 m/s of
the traffic car throughout the hold; a collision ends the episode. Requiring
speed matching prevents a fast pass through the box from counting as success.

Actions and timing match the previous PPO task: LEFT, IDLE, RIGHT, FASTER and
SLOWER, every 0.1 seconds, with target speeds 10–25 m/s in 2.5 m/s steps.
The dictionary observation also retains its 53 inputs. It contains ego plus
the one traffic car, with five unused traffic rows padded with zeroes, along
with the moving goal, anchor state, slot, hold progress and local slot error.
The reward uses the previous formula; lane/speed/overtake shaping remains zero.
For all four goals the hold condition includes speed matching. Goal coordinates
and the slot one-hot input update for the selected slot.

## Preview all four scenarios

From the repository root in PowerShell:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B -m baseline_simple_waypoint.demo --headless --cycle-slots
```

The demo uses a scripted controller. **It does not train PPO or represent a
trained result.** It passes and merges ahead for FRONT, follows for BACK, and
changes lanes to pull alongside for LEFT/RIGHT.

- [Watch all four goals](results/demo_all_slots_seed7/all_scenarios.mp4)
- [Individual videos, screenshots and outcomes](results/demo_all_slots_seed7/SCENARIOS.md)
- [BACK](results/demo_all_slots_seed7/back/scenario.mp4)
- [FRONT](results/demo_all_slots_seed7/front/scenario.mp4)
- [LEFT](results/demo_all_slots_seed7/left/scenario.mp4)
- [RIGHT](results/demo_all_slots_seed7/right/scenario.mp4)

The recorded seed-7 scenarios succeed without collisions: BACK in 5.4 seconds,
FRONT in 11.6 seconds, LEFT and RIGHT in 6.8 seconds each. Each includes the
full one-second hold with speed matching. The earlier BACK-only preview stays
available in `results/demo_seed7/`.

Omit `--headless` for an interactive simulation window. Use `--output-name`
to choose a different preview folder. To choose just one goal:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B -m baseline_simple_waypoint.demo --headless --slot front
```

With no slot option the demo uses the environment's seeded random goal.

## Randomized training from the dynamic-waypoint model

`--randomize` samples traffic velocity uniformly from 12-18 m/s and ego's
initial lane from all three lanes. It samples the goal first, then a valid
traffic lane, so FRONT/BACK/LEFT/RIGHT remain equally likely and side goals
stay on the road. Traffic can start in any of the three lanes; its lane
distribution depends on which goal is selected. Ego still starts at 20 m/s
and has a maximum target speed of 25 m/s, above the traffic maximum of 18 m/s.
The traffic car holds its sampled speed for that episode.

The fixed demo configuration remains available without `--randomize`.
To fine-tune the saved dynamic-waypoint policy and optimizer for exactly
200,000 additional transitions:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_simple_waypoint.train --timesteps 200000 --randomize --slot random --num-envs 4 --rollout-steps 250 --init-model baseline_dynamic_waypoint/models/ppo_front20m_gamma099_seed7_1m.zip --run-name ppo_randomized_dynamic099_seed7_200k --eval-episodes 100 --quiet
```

The actor/critic, discount and other PPO settings come from the source model.
Rollout length and minibatch size are 250 so each four-environment rollout
contains 1,000 transitions and the requested run finishes at exactly 200,000.
The model's step counter records the new transitions; metadata also records
the source model's prior transitions and the lifetime total. The original model
file is read only, with its SHA-256 recorded in the new run.

Outputs in `results/ppo_randomized_dynamic099_seed7_200k/` include the saved
configuration, per-episode rewards/outcomes, actual sampled speeds/lanes in
`sampled_scenarios.csv`, and evaluations before and after training using the
same 100 held-out seeds (10000-10099). Models and 25,000-step checkpoints live
in this environment's `models/` folder.

Completed run: [training report](results/ppo_randomized_dynamic099_seed7_200k/TRAINING_REPORT.md).
On the same 100 evaluation scenarios, success increased from 10% to 27% and
collisions decreased from 25% to 6%. BACK and LEFT improved; FRONT and RIGHT
still had no successes. [Watch learned successes](results/ppo_randomized_dynamic099_seed7_200k/videos/successful_examples.mp4)
or [inspect the first scenario for every goal](results/ppo_randomized_dynamic099_seed7_200k/videos/SCENARIOS.md).
These new videos use trained PPO, with no scripted actions.

## Train the fixed setup

Training starts only when you explicitly run:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B -m baseline_simple_waypoint.train --timesteps 100000 --run-name ppo_slots_seed7
```

The trainer writes only to this environment's `models/` and `results/` folders
and refuses to overwrite an existing run. It saves the environment configuration
beside each model, episode metrics, TensorBoard logs, and checkpoints every
25,000 transitions. For an exact budget, the requested transitions must be a
multiple of `--num-envs` times `--rollout-steps` (1,000 with the defaults).
Training samples all four goals by default. Add `--slot back`, `--slot front`,
`--slot left` or `--slot right` to train only one; `--slot random` samples all four.
The selected setting is saved beside the model and restored during playback.
This task has fixed initial relative geometry and one traffic car; success here
does not establish generalization to the full traffic environment.

To preview a model after training:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B -m baseline_simple_waypoint.demo --headless --cycle-slots --model baseline_simple_waypoint/models/ppo_slots_seed7.zip --output-name trained_slots_seed7
```

## Go back to the previous environment

There is no global environment switch to undo. Run its original entry point:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_dynamic_waypoint/test.py --model baseline_dynamic_waypoint/models/ppo_front20m_gamma099_seed7_1m.zip --episodes 1
```

That reloads the previous model and its original eight-traffic-car configuration
from its saved sidecar. Its original training command also continues to work;
use a fresh run name for a new training run. A shared observation shape does
not by itself guarantee that a policy trained with one traffic car will handle
the full traffic task.

## Checks

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B -m unittest baseline_simple_waypoint.test_environment -v
```
