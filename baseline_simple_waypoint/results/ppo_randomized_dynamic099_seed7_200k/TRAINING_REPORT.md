# Randomized two-car PPO fine-tuning

**Completed exactly 200,000 additional training transitions.**

Initialized policy and optimizer from `ppo_front20m_gamma099_seed7_1m.zip` (1,000,448 prior transitions). Lifetime total: 1,200,448. The source checkpoint hash is unchanged.

Training took 12.83 minutes using four environments. Actor and critic use the source model's 256-256 MLPs, gamma 0.99, learning rate 0.0001, and five epochs. Rollout length and minibatch size were 250 for an exact 1,000-transition rollout.

## Sampled scenarios

- Exactly one controlled ego car plus one constant-speed traffic car.
- Traffic speed sampled uniformly from 12-18 m/s each reset; ego starts at 20 m/s and has a maximum target speed of 25 m/s.
- Ego starting lane sampled from all three lanes. The goal is sampled first, then a valid traffic lane, keeping side goals on the road.
- All four moving goals are sampled. Success requires the slot tolerance and speed matching within 0.5 m/s for one uninterrupted second.
- 1,409 completed training episodes, covering all nine starting-lane pairs; unfinished episodes at the step limit are not included in this count.
- Actual sampled traffic speeds: 12.006744-17.997766 m/s.
- Completed-episode goal counts: BACK 334, FRONT 370, LEFT 349, RIGHT 356.

## Held-out evaluation

Before and after evaluations use exactly the same 100 seeds (10000-10099). Their initial slots, lanes, speeds and distances were checked for equality. Deterministic PPO actions are used for evaluation. Training actions were sampled from the PPO policy.

| Outcome | Before fine-tuning | After 200,000 transitions |
| --- | ---: | ---: |
| Success | 10% | 27% |
| Collision | 25% | 6% |
| Timeout | 65% | 67% |

| Goal | Episodes | Success before | Success after | Collisions after |
| --- | ---: | ---: | ---: | ---: |
| BACK | 27 | 0 | 11 | 0 |
| FRONT | 18 | 0 | 0 | 6 |
| LEFT | 33 | 10 | 16 | 0 |
| RIGHT | 22 | 0 | 0 | 0 |

Success improved and collisions decreased on this evaluation set, but goal reaching is incomplete. FRONT has six collisions and twelve timeouts; RIGHT has twenty-two timeouts. No FRONT or RIGHT goals succeeded. These results apply to the one-traffic-car task, not the original eight-car environment.

## Artifacts

- [Final trained PPO](../../models/ppo_randomized_dynamic099_seed7_200k.zip)
- [Learned successful examples](videos/successful_examples.mp4)
- [First held-out scenario for each goal](videos/all_goals.mp4)
- [All video seeds, initial conditions and outcomes](videos/SCENARIOS.md)
- [Training curves](training_curves.png)
- [Evaluation](evaluation.json) and [per-episode results](evaluation_episodes.csv)
- [Before evaluation](evaluation_before.json)
- [Actual training samples](sampled_scenarios.csv)
- [Run configuration](run.json)
- [Model and source-checkpoint verification](model_verification.json)

Ten environment tests passed before training, including randomized observations, valid goal geometry, speed bounds, checkpoint loading and the original environment. The final model was reloaded, its counter checked at 200,000, and twelve policy parameter tensors were confirmed to differ from the starting model. Video replays are checked against the held-out evaluation.
