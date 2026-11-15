# Live PPO and DQN training

Current 1.5M-step reward comparison: **http://127.0.0.1:6019/live**.
Run: `ppo_maneuver69_notrack_side10_seed7_1p5m`. Fresh policy; tracking reward
disabled; FRONT receives +10 once per episode on reaching either actual side
slot (within 5 m along and 1.8 m across, without collision/anchor loss).
The 69 observations, timeout -10, dropout .05, and PPO settings are preserved.
Watch `reward_components/front_side`, `reward_components/tracking`, and
`evaluation/front/success_rate`. Compare at 1.5M steps with the previous run.

Current navigation-guidance experiment and its stopped baseline comparison:
**http://127.0.0.1:6018/live**.
Run: `ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m`.
Fresh 2M steps, 69 inputs, staged FRONT passing/merging and route-progress
feedback, timeout -10, dropout .05, LR 1e-4, gamma .99, all four goals.
Watch `evaluation/front/success_rate` and `evaluation/right/success_rate`
for fixed-seed monitoring, and `episodes/front/success_mean_100` /
`episodes/right/success_mean_100` for training outcomes.

Latest replacement run on **http://127.0.0.1:6016/live**:
`ppo_targetlanes57_lr4e5_time045_dropout005_gamma097_seed7_1m`.
Fresh 1M steps: learning rate 0.00004 (20% lower), time cost -0.045 per
physics tick (10% lower), timeout -10, gamma 0.97, dropout 0.05, all four goals,
57 inputs. The preceding stability-diagnostics run was cancelled by the user;
its last saved checkpoint is 650k.


Current experiment on **http://127.0.0.1:6016/live**:
`ppo_stability_diagnostics_targetlanes57_seed7_1m`. Fresh 1M steps with
57 inputs, dropout 0.05, LR 0.00005, gamma 0.97, all goals, timeout -10.
Includes aggregate timeout/success-length diagnostics and automatic best-policy saving.

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6016 --ppo-logdir baseline_dynamic_waypoint/results/ppo_stability_diagnostics_targetlanes57_seed7_1m --latest 1
```


The lower-dropout comparison is at **http://127.0.0.1:6016/live**.
Run: `ppo_targetlanes57_timeout10_dropout005_lr5e5_gamma097_scratch_seed7_1m`;
fresh 1M steps, dropout 0.05, learning rate 0.00005, gamma 0.97,
57 inputs with explicit target lanes, all four goals, timeout -10.


The fresh **1,000,000-step, 57-input** experiment is at
**http://127.0.0.1:6015/live**. Run:
`ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m`; all four goals,
timeout -10, actor/critic dropout 0.1. The `legacy_target_lanes` schema replaces
the two redundant `achieved_goal` values with explicit
`[target_car_lane, target_goal_lane]` IDs, keeping 57 inputs. Policy and optimizer
start from scratch. Fixed-seed evaluation runs every 250,000 steps.

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6015 --ppo-logdir baseline_dynamic_waypoint/results/ppo_targetlanes57_timeout10_dropout01_scratch_seed7_1m --latest 1
```

The restored 57-input experiment is at **http://127.0.0.1:6014/live**.
Run: `ppo_legacy57_timeout10_dropout01_seed7_200k`; all four goals, timeout -10,
actor/critic dropout 0.1, 200,000 additional steps from the earlier BACK/LEFT
checkpoint. See the [report](baseline_dynamic_waypoint/results/ppo_legacy57_timeout10_dropout01_seed7_200k/TRAINING_REPORT.md).

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6014 --ppo-logdir baseline_dynamic_waypoint/results/ppo_legacy57_timeout10_dropout01_seed7_200k --latest 1
```

The 2M-to-4M continuation is at **http://127.0.0.1:6013/live**, alongside its
original 2M source run. The continuation retains the saved policy and optimizer,
all four goals, timeout -25, and actor/critic dropout 0.1. Its native step counter
is additional steps 0-2M, corresponding to lifetime steps 2M-4M.

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6013 --ppo-logdir baseline_dynamic_waypoint/results --latest 2
```

The all-goal 2-million-step experiment is at **http://127.0.0.1:6012/live**.
Run: `ppo_allgoals_timeout25_dropout01_seed7_2m`. It enables all four goals,
uses actor/critic dropout 0.1, and charges -25 on timeout. Per-goal training
curves use `episodes/<goal>/*`; held-out results use `evaluation/<goal>/*`
at 250,000-step intervals. Setup values include `setup/dropout_rate` and
`setup/reward_weight_timeout`.

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6012 --ppo-logdir baseline_dynamic_waypoint/results/ppo_allgoals_timeout25_dropout01_seed7_2m --latest 1
```

The latest comparison is **http://127.0.0.1:6011/live**. It displays the
LEFT/BACK-only and FRONT/RIGHT-only experiments together, each trained from
scratch for exactly 200,000 steps with the same 54 inputs, PPO settings,
rewards, physics, and seed. The LEFT/BACK run disables FRONT/RIGHT in both
training and evaluation. Its goal curves use `episodes/left/*`,
`episodes/back/*`, `evaluation/left/*`, and `evaluation/back/*`.
See the [LEFT/BACK report](baseline_dynamic_waypoint/results/ppo_singlecar_left_back_only_scratch_seed7_200k/TRAINING_REPORT.md).

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6011 --ppo-logdir baseline_dynamic_waypoint/results --latest 2
```

The FRONT/RIGHT-only 200k experiment is open at
**http://127.0.0.1:6010/live**. It shows only
`ppo_singlecar_front_right_only_scratch_seed7_200k`, with BACK/LEFT disabled
in both training and evaluation. Goal curves include
`episodes/front/success_mean_100`, `episodes/right/success_mean_100`, and
`evaluation/front/success_rate` / `evaluation/right/success_rate`.

Restart that dashboard with its isolated TensorBoard environment:

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6010 --ppo-logdir baseline_dynamic_waypoint/results/ppo_singlecar_front_right_only_scratch_seed7_200k --latest 1
```

See the [run report](baseline_dynamic_waypoint/results/ppo_singlecar_front_right_only_scratch_seed7_200k/TRAINING_REPORT.md)
for exact goal counts, evaluation results, and event verification.

From the repository root in PowerShell:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B monitor_training.py
```

Open **http://127.0.0.1:6006/live**. This enables TensorBoard's browser chart
refresh and opens its Time Series dashboard with **only the four newest training
runs** across these two results roots:

- `PPO`: `baseline_dynamic_waypoint/results`
- `DQN`: `baseline_DQN/results`

At startup, the launcher scans both roots and selects four run folders using
the `created_utc` value in each folder's `run.json`. It requires TensorBoard event
files in the folder. Directory modification times and ongoing checkpoint writes
do not affect the selection. Video-only and evaluation-only folders are excluded.
All new points from the selected four runs continue to update live. Restart the
monitor to select a newer set after creating another experiment.

The current four experiments are:

- `DQN/dqn_quickstart_gamma09_seed2_200k`
- `DQN/dqn_quickstart_gamma095_seed2_200k_fast`
- `PPO/ppo_front20m_current_seed7_1m`
- `PPO/ppo_front20m_gamma099_seed7_1m`

Run labels include the algorithm family and folder name.

The server reloads event files every **5 seconds**. Browser charts refresh every
**30 seconds**, TensorBoard's minimum chart refresh interval. Training writes new
points at its logging intervals, so the charts advance after those points flush.
The launcher reads existing event files independently of training processes.

Select the runs to compare in the left sidebar, or filter names with `gamma09`,
`gamma095`, `gamma099` or `front20m`. Useful tags include
`rollout/ep_rew_mean`, `episodes/reward_mean_100`, and
`episodes/collision_mean_100`. PPO also logs `episodes/success_mean_100`,
`episodes/final_distance_mean_100`, hold metrics, and `reward_components/*`.
DQN also logs `episodes/completed_mean_100`, exploration rate, and Q-network loss.
PPO and DQN have different rewards and decision frequencies; reward magnitudes
and transition counts should be interpreted within each task.

To narrow either group to an individual run, watch other result roots, or use
another port:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B monitor_training.py --port 6007 --ppo-logdir baseline_dynamic_waypoint/results/MY_PPO_RUN --dqn-logdir baseline_DQN/results/MY_DQN_RUN
```

Use `--latest 2` for the newest two runs. Use `--all-runs` to restore the full
historical view and recursively discover future runs as their events appear:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B monitor_training.py --all-runs
```

Use `--check` to validate the paths and TensorBoard flags without starting a server.
The `/live` entry point preserves existing browser display settings while enabling
refresh. The TensorBoard service listens on localhost by default.
