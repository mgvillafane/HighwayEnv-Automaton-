# BACK/LEFT only: training and evaluation

Fresh PPO weights and optimizer; 54-input observation; exactly 200,000 training decisions.

FRONT and RIGHT are disabled in the run's environment, including explicit reset requests. Training and evaluation both sample only BACK and LEFT.

Training time: 4.07 minutes; physics ticks: 256,102.

| Goal | Training episodes | Training successes | Evaluation episodes | Evaluation success | Evaluation collisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| BACK | 512 | 40 | 55 | 0.0% | 0.0% |
| LEFT | 526 | 51 | 45 | 0.0% | 0.0% |

Evaluation success: 0.0%; collisions: 0.0%; timeouts: 100.0%. Evaluation uses deterministic actions on 100 held-out seeds (10000-10099). Training actions are sampled.

## TensorBoard

[Open the dashboard](http://127.0.0.1:6011/live). The LEFT/BACK and prior FRONT/RIGHT runs are displayed together. Event reload: 5 seconds; browser refresh: 30 seconds.

Training charts: `episodes/back/success_mean_100`, `episodes/left/success_mean_100`, the corresponding `reward_mean_100` and `collision_mean_100` tags, and `sampling/back/episodes` / `sampling/left/episodes`.

Held-out charts: `evaluation/back/success_rate` and `evaluation/left/success_rate`, with before/after points at steps 0 and 200,000. All goal charts were verified through the running server's scalar API.

The server uses an isolated Python environment with TensorBoard 2.21.0 and grpcio 1.76.0. The training environment's grpcio build was blocked by Windows Application Control; no system policy was changed.

Restart the dashboard from the repository root:

```powershell
.\.tensorboard-venv\Scripts\python.exe -u -B monitor_training.py --port 6011 --ppo-logdir baseline_dynamic_waypoint/results --latest 2
```

Repeat training with a fresh run name:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema lane_goal_v2 --enabled-slots left back --timesteps 200000 --num-envs 4 --rollout-steps 250 --seed 7 --reward-profile baseline --run-name YOUR_NEW_RUN --eval-episodes 100 --eval-seed 10000 --quiet
```

[Run settings](run.json) | [Evaluation](evaluation.json) | [Verification](verification.json) | [TensorBoard values](tensorboard_metrics.json) | [Training curves](training_curves.png)


Reference FRONT/RIGHT run: success 0.0%, collisions 26.0%. The same training settings and seed were used; goal assignments and therefore scenarios differ.

[Configuration comparison and results](comparison.json)
