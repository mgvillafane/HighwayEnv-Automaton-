# Committed lane-change PPO, 57 inputs

**Completed exactly 200,000 new PPO decisions.** Internal simulation advanced 252,891 ticks of 0.1 s.

[Open TensorBoard](http://127.0.0.1:6008/live) | [Watch all four goals](videos/all_goals.mp4)

## Execution and observations

- LEFT or RIGHT sets the target once. Automatic IDLE steering continues without another PPO prediction until lateral position is within 0.2 m of the requested lane center and lateral speed is at most 0.2 m/s.
- The target speed is held throughout the maneuver. A new lane or speed decision is accepted after completion. Collisions, anchor loss and the 30-second deadline can interrupt it.
- Goal success is accepted only once the maneuver has settled, so an episode cannot stop halfway through a successful lane change.
- Added `requested_lane` (three one-hot values) and `lane_change_in_progress` (one flag): 57 inputs total. At normal decision boundaries the flag is zero; internal physics observations and interrupted terminal states expose the active flag.
- Reward weights and per-physics-step terms are unchanged. Returned action reward sums the internal tick rewards. Gamma 0.99 is applied per PPO decision, so one transition can cover varying physical time.

Training executed 2,813 lane changes: 2,737 completed, 76 interrupted by terminal events. These counts account for every started maneuver.

## Initialization and scenario sampling

Initialized from the latest flexible-hold PPO. Existing actor/critic weights were copied with input columns remapped by observation key. The four new input connections started at zero; tests verified unchanged policy probabilities and value estimates before learning. The optimizer was reset for the expanded layers. The source checkpoint is unchanged.
Training time: 6.03 minutes. Four environments, 250 rollout steps each, minibatches of 250. Completed training episodes: 1236.
The same randomized two-car task is retained: all four slots, valid randomized starting lanes, traffic velocity 12-18 m/s, ego maximum 25 m/s, 30-second episodes, the flexible 0.5 s hold and 0.3 s grace.

## Paired evaluation

Both evaluations use the same 100 seeds, initial lanes and traffic speeds under committed action execution. The previous policy scored 60% under the earlier action interface; transferring its unchanged network into this interface gave a 46% baseline. Those different execution modes are not identical evaluations.

| Outcome | Before retraining in new interface | After 200,000 decisions |
| --- | ---: | ---: |
| Success | 46% | 55% |
| Collision | 9% | 7% |
| Timeout | 45% | 38% |

| Goal | Episodes | Successes | Collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| BACK | 27 | 22 | 5 | 0 |
| FRONT | 18 | 0 | 2 | 16 |
| LEFT | 33 | 33 | 0 | 0 |
| RIGHT | 22 | 0 | 0 | 22 |

Lane execution now completes before the next decision. Goal selection remains incomplete: FRONT and RIGHT still have no successes. Automatic completion does not choose the correct lane for the policy, and target-speed changes are unavailable while the maneuver is executing.

## Validation and artifacts

23 tests passed, covering completion, command blocking, terminal interruption, success deferral, 57-input checkpoint loading, output-preserving transfer, legacy compatibility, and exact fast/standard encoder equivalence. Recorded PPO trajectories reproduce their evaluation results; each completed recorded maneuver has a stable requested lane and only automatic IDLE ticks between policy decisions. Videos preserve every physics frame.

- [Final model](../../models/ppo_singlecar_committed57_seed7_200k.zip)
- [Configuration and counters](run.json)
- [Verification](verification.json)
- [Evaluation](evaluation.json)
- [Episode metrics](evaluation_episodes.csv)
- [Training curves](training_curves.png)
- [Scenario samples](sampled_scenarios.csv)
- [All-goal videos](videos/all_goals.mp4)
- [Full recorded action traces](videos/episodes.json)
