# PPO observation and goal audit

Audited the final `ppo_front20m_current_seed7_1m` (gamma 0.9) and
`ppo_front20m_gamma099_seed7_1m` (gamma 0.99) checkpoints using the eight
recorded evaluation episodes from each run. No training or environment behavior
was changed.

The traffic and goal inputs reach both saved policies with the expected values.
The observed crashes include unsafe decisions in response to visible traffic.
This audit found no missing-car or goal-wiring error in the sampled episodes;
it does not establish that the observation is a complete state representation
or that the policies reason correctly about traffic.

## Inputs that reach the actor

Both saved policies use `MultiInputActorCriticPolicy` and `CombinedExtractor`.
The extractor flattens and concatenates all dictionary keys into 53 features.
The extracted tensor was checked against the complete input dictionary on every
decision; no traffic or goal keys were dropped.

| Input | Shape | Meaning |
| --- | --- | --- |
| `vehicles` | 7 x 5 | Ego plus up to six nearby traffic cars; presence, x, y, vx, vy |
| `achieved_goal` | 2 | Ego position relative to the episode origin, scaled by 200 m and 12 m |
| `desired_goal` | 2 | Current moving slot position in the same frame and scale |
| `goal_delta` | 2 | Desired position minus ego position |
| `anchor` | 4 | Anchor position and velocity relative to ego |
| `slot` | 4 | FRONT/BACK/LEFT/RIGHT one-hot encoding |
| `hold_progress` | 1 | Consecutive time in the slot divided by the hold requirement |
| `slot_error` | 3 | Local along-lane error, lateral error and anchor-minus-ego speed |

Traffic rows use relative position and velocity. Ego's row uses episode-relative
x progress, absolute y and absolute velocity. Neighbors include cars behind ego
and are sorted by absolute longitudinal distance. The observation has at most
six traffic rows, even though the scenario contains eight traffic cars. Cars
outside the simulator's perception distance can also be absent; absent rows
are zero padded and have presence zero.

## Measurements

| Check | Gamma 0.9 | Gamma 0.99 |
| --- | ---: | ---: |
| Audited episodes | 8 | 8 |
| Checked observation states, including terminal states | 896 | 1,088 |
| Traffic rows checked against simulator positions and velocities | 5,085 | 6,350 |
| Recorded collisions whose nearby crashed traffic car was present before impact | 4/4 | 4/4 |
| Decisions changed when traffic rows were removed, keeping ego/anchor/goal fixed | 113/888 (12.7%) | 564/1,080 (52.2%) |
| Decisions changed when the slot was changed while traffic and anchor stayed identical | 4/18 | 7/19 |

Every checked observation matched its declared space, contained finite float32
values, and had correctly scaled traffic rows, anchor state, goal coordinates,
goal delta and slot error. The slot intervention used valid alternative slots
at reset with the same seed, anchor and traffic.

Removing traffic is an input sensitivity diagnostic. These perturbed inputs
do not constitute a driving evaluation, and the action-change percentages do
not measure safety. Together with the verified feature tensor, they show that
traffic reaches the actor and affects its decisions.

The existing SB3 contract, slot geometry and slot-error input tests also passed
(three tests). SB3 warns about the 7 x 5 non-image observation shape; inspection
of both saved extractors confirmed that it is flattened into 35 traffic/ego
features and included in the 53-feature actor input.

## Concrete failure

Gamma 0.9, seed 10000, at 3.6 seconds, immediately before the recorded crash:

- The first traffic row contained a car 5.631 m ahead in the same lane.
- Its relative longitudinal velocity was -8.214 m/s: ego was closing rapidly.
- The row was `[1, 0.02815546, 0, -0.27380437, 0]`, which decodes to those values
  using the configured 200 m and 30 m/s scales.
- The deterministic policy chose IDLE, with probabilities 48.3% IDLE, 32.5%
  LANE_LEFT and 18.9% SLOWER.
- The next step ended in collision. IDLE maintains the existing target speed;
  the controlled ego does not automatically apply traffic-following brakes.

This is evidence of an unsafe learned response to an observed car. It does
not identify one exclusive cause of the training failure.

## Representation and learning limitations

- The network receives a distance-sorted flattened list. Row identities can
  change as cars change order; it has no explicit per-lane front/rear gaps.
- Distance uses a 200 m normalization scale. A five-meter center gap is only
  0.025 in the traffic x feature. This is numerically valid, but there is no
  dedicated bumper-clearance or time-to-collision feature.
- Both new runs apply the collision penalty at contact and have no explicit
  unsafe-headway or time-to-collision penalty. Their lane, speed-match and
  overtake reward weights are zero.
- Gamma 0.9 at 10 decisions/second gives a reward three seconds ahead a
  discount factor of 0.9^30 = 0.0424. Gamma 0.99 gives 0.99^30 = 0.7397.
  These are discount weights, not hard planning horizons.

The evidence supports examining safety features, early safety feedback and
learned braking behavior before concluding that PPO cannot see traffic.
Changing the observation schema requires a new compatible policy; existing
checkpoints retain their original input dimensions.

Full numerical results and the pre-impact input snapshots:

- [Gamma 0.9](ppo_front20m_current_seed7_1m/observation_audit.json)
- [Gamma 0.99](ppo_front20m_gamma099_seed7_1m/observation_audit.json)
