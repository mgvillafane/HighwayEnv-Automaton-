# Latest completed dynamic-waypoint run

`ppo_maneuver69_lateral_side10_seed7_1m` completed on November 16, 2026.
It trained from scratch for exactly 1,000,000 PPO decisions using seed 7,
four environments, dropout 0.05, and the `maneuver_lateral_side10` reward profile.
Training took 4,993.2 seconds (83.2 minutes).

The task uses one ego and one constant-speed traffic car on three lanes.
All four slots are enabled. The moving slot must be held for 0.5 seconds,
with a 0.3-second hold grace period and a 30-second episode deadline.

The route cost is `2 * abs(goal_lane_center - ego_lateral_position)`;
longitudinal and speed errors have zero route weight. The original
distance-progress reward remains. Tracking is disabled, the timeout penalty
is -10, and the first qualifying side-slot arrival during a FRONT task earns +10.

Final deterministic evaluation: 100 scenarios, seeds 10000-10099.

| Goal | Successes | Success rate | Collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| FRONT | 0/18 | 0% | 2 | 16 |
| BACK | 27/27 | 100% | 0 | 0 |
| LEFT | 31/33 | 93.9% | 0 | 2 |
| RIGHT | 22/22 | 100% | 0 | 0 |
| Overall | 80/100 | 80% | 2 | 18 |

Mean episode return: 39.2046. Mean time to reach among successes: 4.4975 seconds.
The selected best checkpoint was at 250,000 decisions: 82% success,
2% collisions, and 16% timeouts. It reached every BACK/LEFT/RIGHT case,
and no FRONT case. Best-checkpoint selection used these same monitoring
scenarios and is not an independent test.

The preceding no-tracking + side-bonus run reached 96% overall and 14/18 FRONT
goals after 1.5M decisions, with 4% collisions. The full route-progress + tracking
run reached 95% overall and 14/18 FRONT goals after 2M decisions, with zero
collisions. The latest run has a smaller budget, so these results do not isolate
the effect of its reward change. The observed FRONT failure remains unresolved.

Sources: [saved configuration](run.json), [final evaluation](evaluation.json),
[episode outcomes](evaluation_episodes.csv), [best checkpoint selection](best_evaluation.json),
and [final model sidecar](../../models/ppo_maneuver69_lateral_side10_seed7_1m.json).
The `configuration_verification.json` status of `running` is a launch-time snapshot;
the final sidecar records `status: trained` and `actual_timesteps: 1000000`.
