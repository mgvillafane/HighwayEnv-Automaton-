# Flexible 53-input PPO: timeout 10, dropout 0.05, fresh 2M steps

**Stopped at the user's request.** The final model preserves the latest saved
checkpoint at **1,450,000 steps**. Training logs reached **1,466,000 steps**;
the later unsaved transitions were discarded. See `training_stop.json`.
RIGHT and FRONT recordings are in [videos](videos/README.md).

Copies the environment, goal settings, baseline reward, and PPO settings from
`ppo_singlecar_flexible30s_seed7_200k`, whose saved deterministic evaluation
achieved 100% BACK and LEFT success (60% overall).

This run starts with fresh actor, critic, and optimizer, as requested. The source
run had pretrained weights, so this is a configuration replication with a
different initialization and training budget.

## Configuration

- Budget: exactly 2,000,000 PPO transitions, seed 7, four environments.
- Observation: legacy dictionary, 53 scalar inputs; seven five-feature vehicle
  rows plus positional goals, anchor state, goal one-hot, hold progress, and
  three local slot/speed errors. Unused vehicle rows are zero padded.
- Goals: all four of FRONT, BACK, LEFT, RIGHT, sampled using the source defaults.
- Dynamics: ego plus one traffic car, three lanes, constant traffic speed
  sampled in 12–18 m/s, ego initial speed 20 m/s, 30-second episodes, 10 Hz.
- Lane changes use the source's ordinary per-tick decisions.
- Success: longitudinal error <5 m, lateral error <1.8 m, speed mismatch
  <=2 m/s, valid holding for 0.5 s, exit grace 0.3 s.
- Reward per 0.1-second tick:
  `50*success - 50*collision + 0.1*(previous_distance-distance) - 0.05 + 0.2*inside - 10*timeout`.
- Actor and critic: [256, 256], Tanh, dropout 0.05 after each hidden activation.
- PPO: learning rate 0.0001, gamma 0.99, GAE lambda 0.95, n_steps 250 per
  environment (1,000 transitions per rollout), batch size 250, five epochs,
  clip range 0.2, entropy coefficient 0.01, value coefficient 0.5,
  gradient norm limit 0.5, CPU.

## Outputs

`run.json` records the actual configuration and checkpoint progress.
`configuration_verification.json` checks it against the source configuration.
`progress.csv`, `training_episodes.csv`, `sampled_scenarios.csv`, and TensorBoard
events record learning and scenario sampling.

Checkpoints are saved every 25,000 transitions. Deterministic evaluations use
100 fixed scenarios, seeds 10000–10099, before training, every 250,000
transitions, and after all 2,000,000 transitions. `best_evaluation.json` records
the best monitored checkpoint, ranked by success, fewer collisions, then reward.
These repeated monitoring seeds are not an independent final test set.

The final model is saved under `baseline_dynamic_waypoint/models/` with this run
name; its final evaluation is `evaluation.json` in this directory.

Launch command and process information:
[launch.json](../launches/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/launch.json).
Logs: [stdout](../launches/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/training_stdout.log)
and [stderr](../launches/ppo_flexible53_timeout10_dropout005_scratch_seed7_2m/training_stderr.log).
