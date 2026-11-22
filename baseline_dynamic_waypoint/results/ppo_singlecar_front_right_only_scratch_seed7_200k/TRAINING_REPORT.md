# FRONT/RIGHT only: training and evaluation

Fresh PPO weights and optimizer; 54-input observation; exactly 200,000 training decisions.

BACK and LEFT are disabled in the run's environment, including explicit reset requests. Training and evaluation both sample only FRONT and RIGHT.

Training time: 4.06 minutes; physics ticks: 256,739.

| Goal | Training episodes | Training successes | Evaluation episodes | Evaluation success | Evaluation collisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| FRONT | 519 | 6 | 55 | 0.0% | 21.8% |
| RIGHT | 528 | 39 | 45 | 0.0% | 31.1% |

Evaluation success: 0.0%; collisions: 26.0%; timeouts: 74.0%. Evaluation uses deterministic actions on 100 held-out seeds (10000-10099). Training actions are sampled.

## TensorBoard

[Open the dashboard](http://127.0.0.1:6010/live). Only this run is displayed. Event reload: 5 seconds; browser refresh: 30 seconds.

Training charts: `episodes/front/success_mean_100`, `episodes/right/success_mean_100`, the corresponding `reward_mean_100` and `collision_mean_100` tags, and `sampling/front/episodes` / `sampling/right/episodes`.

Held-out charts: `evaluation/front/success_rate` and `evaluation/right/success_rate`, with before/after points at steps 0 and 200,000. All goal charts were verified through the running server's scalar API.

The server uses an isolated Python environment with TensorBoard 2.21.0 and grpcio 1.76.0. The training environment's grpcio build was blocked by Windows Application Control; no system policy was changed.

Restart the dashboard from the repository root:

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6010 --ppo-logdir baseline_dynamic_waypoint/results/ppo_singlecar_front_right_only_scratch_seed7_200k --latest 1
```

Repeat training with a fresh run name:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema lane_goal_v2 --enabled-slots right front --timesteps 200000 --num-envs 4 --rollout-steps 250 --seed 7 --reward-profile baseline --run-name YOUR_NEW_RUN --eval-episodes 100 --eval-seed 10000 --quiet
```

[Run settings](run.json) | [Evaluation](evaluation.json) | [Verification](verification.json) | [TensorBoard values](tensorboard_metrics.json) | [Training curves](training_curves.png)
