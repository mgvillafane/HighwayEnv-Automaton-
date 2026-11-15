# Flexible holding, 30-second two-car PPO experiment

**Completed exactly 200,000 additional transitions under `baseline_dynamic_waypoint/`.**

[Open live TensorBoard](http://127.0.0.1:6007/live) | [Watch trained PPO for all four goals](videos/all_goals.mp4)

## Hold rules

| Rule | Previous strict profile | This profile |
| --- | ---: | ---: |
| Episode limit | 15 s | 30 s |
| Longitudinal tolerance | +/-3 m | +/-5 m |
| Lateral tolerance | +/-1.5 m | +/-1.8 m |
| Speed tolerance | 0.5 m/s | 2 m/s |
| Valid hold time | 1 s | 0.5 s |
| Brief exit grace | 0 s | 0.3 s |

At 10 Hz, five valid decisions complete the hold. Up to three consecutive invalid decisions pause the counter, with no hold reward or success. A fourth clears it. Re-entry is required for success; collisions and anchor loss clear it immediately. Legacy profiles without the grace setting keep their original immediate reset.

## Training and sampling

Started again from `ppo_front20m_gamma099_seed7_1m.zip` (1,000,448 prior transitions), preserving its weights and optimizer. This is an independent rerun from the same initial checkpoint used for the previous experiment.
Training: 200,000 new transitions, four environments, rollout length 250, minibatch size 250, gamma 0.99. Training time: 11.27 minutes.

Exactly one traffic car and ego. Traffic speed is sampled uniformly from 12-18 m/s, below ego maximum 25 m/s. Ego lane and a valid traffic lane are sampled, with all four goals sampled equally. The moving-slot geometry and PPO input dimensions remain unchanged.
Completed training episodes: 1041. Goal counts: BACK 261, FRONT 269, LEFT 251, RIGHT 260.

## Evaluation with the new criteria

Both policies use the same 100 seeds (10000-10099) and identical sampled initial states. Deterministic actions are used for evaluation.

| Outcome | Before retraining | After 200,000 steps |
| --- | ---: | ---: |
| Success | 26% | 60% |
| Collision | 26% | 6% |
| Timeout | 48% | 34% |

| Goal | Episodes | Successes | Collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| BACK | 27 | 27 | 0 | 0 |
| FRONT | 18 | 0 | 3 | 15 |
| LEFT | 33 | 33 | 0 | 0 |
| RIGHT | 22 | 0 | 3 | 19 |

BACK and LEFT succeeded in every evaluated case. FRONT and RIGHT remain unlearned. The previous strict experiment had 27% overall success, but that number uses a different success definition and shorter episode limit; it is not a controlled measure of policy improvement. The 26%-to-60% comparison above uses the same new rules.

## Checks and artifacts

Nineteen tests passed: relaxed position/speed tolerances, grace pausing/resuming, longer exits, collisions/anchor loss, the 30-second timeout, checkpoint dispatch, and compatibility with the old simple and multi-car hold behavior. The source checkpoint is unchanged, the saved model was reloaded, its counter is exactly 200,000, and the four video replays reproduce their evaluation outcomes/rewards/durations. TensorBoard serves the final evaluation point at step 200,000.

- [Final model](../../models/ppo_singlecar_flexible30s_seed7_200k.zip)
- [Run configuration](run.json)
- [Evaluation](evaluation.json)
- [Per-episode evaluation](evaluation_episodes.csv)
- [Training curves](training_curves.png)
- [Sampled training scenarios](sampled_scenarios.csv)
- [Verification](verification.json)
- [All-goal video reel](videos/all_goals.mp4)
