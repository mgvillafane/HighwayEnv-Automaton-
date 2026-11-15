# Four single-car reward experiments

Each task has one controlled ego vehicle and one traffic vehicle. The comparison
below uses the four runs discussed in the reward table. Their models, environment
sidecars, checkpoints, training records, evaluations, videos, and reports are
included alongside the scenario/training implementation.

The earlier fixed-scenario generator lives in `baseline_simple_waypoint/`.
These four randomized single-car experiments use
`baseline_dynamic_waypoint/train_single_car.py` and `single_car.py`.

## Models and results

Training success percentages are pooled across all completed episodes logged in
each run, including exploration and policies changing during training. They are
not the final policy's deterministic evaluation rates.

| Experiment | Training LEFT / RIGHT / BACK / FRONT | Saved model and configuration | Training results | Evaluation results |
| --- | --- | --- | --- | --- |
| Simple reward; pretrained +200k | 50.2% / 0% / 79.3% / 0% | [Model](../../models/ppo_singlecar_flexible30s_seed7_200k.zip), [sidecar](../../models/ppo_singlecar_flexible30s_seed7_200k.json) | [Episodes](../ppo_singlecar_flexible30s_seed7_200k/training_episodes.csv), [run](../ppo_singlecar_flexible30s_seed7_200k/run.json) | [Final evaluation](../ppo_singlecar_flexible30s_seed7_200k/evaluation.json) |
| Timeout + dropout; interrupted | 52.0% / 0.056% / 64.5% / 0.057% | [Preserved 1.45M model](../../models/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m.zip), [sidecar](../../models/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m.json) | [Episodes](../ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/training_episodes.csv), [run](../ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/run.json) | [Latest recorded evaluation at 1.25M](../ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/evaluation_1250000.json), [matching checkpoint](../../models/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m_1250000.zip) |
| Route progress + tracking; fresh 2M | 96.3% / 96.3% / 99.1% / 62.5% | [Final model](../../models/ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m.zip), [sidecar](../../models/ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m.json) | [Episodes](../ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m/training_episodes.csv), [run](../ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m/run.json) | [Final evaluation](../ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m/evaluation.json) |
| No tracking + FRONT side milestone; fresh 1.5M | 96.5% / 96.7% / 99.2% / 62.6% | [Final model](../../models/ppo_maneuver69_notrack_side10_seed7_1p5m.zip), [sidecar](../../models/ppo_maneuver69_notrack_side10_seed7_1p5m.json) | [Episodes](../ppo_maneuver69_notrack_side10_seed7_1p5m/training_episodes.csv), [run](../ppo_maneuver69_notrack_side10_seed7_1p5m/run.json) | [Final evaluation](../ppo_maneuver69_notrack_side10_seed7_1p5m/evaluation.json) |

The simple run's warm-start source is included:
[1,000,448-step source model](../../models/ppo_front20m_gamma099_seed7_1m.zip)
and [source configuration](../../models/ppo_front20m_gamma099_seed7_1m.json).
Its training history in this table covers only the additional 200k steps.
The interrupted timeout/dropout run logged 1,466,000 transitions; its last saved
checkpoint is 1.45M and its last recorded evaluation is 1.25M.

## Reward table

All per-tick terms are computed at 10 Hz. Terminal and milestone bonuses/costs
occur only when triggered.

| Component | Simple | Timeout + dropout | Route + tracking | No tracking + FRONT side milestone |
| --- | ---: | ---: | ---: | ---: |
| Final success | +50 | +50 | +50 | +50 |
| Ego collision | -50 | -50 | -50 | -50 |
| Timeout | 0 | -10 | -10 | -10 |
| Time per tick | -0.05 | -0.05 | -0.05 | -0.05 |
| Valid final-slot occupancy per tick | +0.2 | +0.2 | +0.2 | +0.2 |
| Moving-goal progress | 0.1 * delta_distance | 0.1 * delta_distance | 0.1 * delta_distance | 0.1 * delta_distance |
| Route progress | 0 | 0 | 0.5 * delta_route_error | 0.5 * delta_route_error |
| Tracking cost | 0 | 0 | -0.05 * tracking_error | 0 |
| First side-slot arrival in a FRONT task | 0 | 0 | 0 | +10 once per episode |

`delta_distance` and `delta_route_error` mean previous minus current error.
The route score is absolute longitudinal slot error plus twice absolute
navigation lateral error plus absolute preferred-speed error. Tracking combines
normalized lateral, longitudinal, speed errors, and collision risk.
The FRONT milestone accepts either adjacent side, within 5 m along and 1.8 m
of its lane center, without collision/anchor loss. It does not require holding
or speed matching; returning to a side slot cannot earn a second payment.

The first two experiments have 53 inputs; the last two have 69. Dropout is zero
for the simple run and 0.05 for the other three. The observation layout and full
reward equations are described in the
[14-page technical PDF](../ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m/reports/Rewards_and_Observations_Maneuver69_PPO.pdf).

## Graphs and reproducibility

- [Training-only graph (PDF)](training_history_success.pdf), [PNG](training_history_success.png), [CSV](training_history_success.csv).
- [Training and evaluation separately (PDF)](success_training_vs_evaluation.pdf), [CSV](success_training_vs_evaluation.csv).
- [Broader timeout/dropout audit](timeout_dropout_audit.csv).
- [First route experiment details](../ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m/EXPERIMENT.md).
- [Side-milestone experiment details](../ppo_maneuver69_notrack_side10_seed7_1p5m/EXPERIMENT.md).

Regenerate the training-only graph from the repository root:

```powershell
.\.venv\Scripts\python.exe -B baseline_dynamic_waypoint/results/reward_experiment_comparison/plot_training_history.py
```

Use `plot_verified_success_rates.py` for the two-panel graph. The evaluation-only
view can be regenerated with `plot_success_rates.py`. Training and evaluation
measure different policies and action sampling; they must remain separate.
For an equal-budget evaluation comparison, use the route experiment's saved
`evaluation_1500000.json` against the side-milestone run's final evaluation.
The repeated monitoring seeds are not an independent test set.
