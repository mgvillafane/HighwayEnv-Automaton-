# B3 on the blocked BACK/LEFT/RIGHT episodes: who gets out of whose way?

Deterministic replay. The *blocking car* is the nearest car between ego and the slot, in the slot's
lane, at reset. It *leaves* when it starts a lane change of its own (its MOBIL model decides this;
the anchor never changes lane, other cars may). Ego *passes it in its lane* when ego gets ahead of it
while it is still in its starting lane.

## B3_seed7

| Blocking car | Episodes | Reached | Crashed into it | Crashed into another car | Timeout |
|---|---:|---:|---:|---:|---:|
| stays in its lane | 19 | 2 | 17 | 0 | 0 |
| leaves its lane by itself | 26 | 21 | 1 | 4 | 0 |

- Ego passed a blocking car that was still in its lane: 0/45 episodes.
- Slot in ego's own lane (27 episodes): ego started a lane change while behind the blocking car in 0, and chose SLOWER while behind it in 1.
- Reached although the blocking car stayed: [10082, 10158].
- The 18 crashes into the blocking car. Critic value (shaping removed; a crash is -50): -7.6 at reset, -8.6 at the last decision, 0.1 s before the crash (4 of them above 0).
- Same crashes, sampling probabilities per decision while behind the blocking car (what training explores): LANE_LEFT 2.7%, IDLE 28.6%, LANE_RIGHT 11.9%, FASTER 46.1%, SLOWER 10.7%.

## B3_seed8

| Blocking car | Episodes | Reached | Crashed into it | Crashed into another car | Timeout |
|---|---:|---:|---:|---:|---:|
| stays in its lane | 19 | 2 | 17 | 0 | 0 |
| leaves its lane by itself | 26 | 21 | 1 | 4 | 0 |

- Ego passed a blocking car that was still in its lane: 0/45 episodes.
- Slot in ego's own lane (27 episodes): ego started a lane change while behind the blocking car in 6, and chose SLOWER while behind it in 0.
- Reached although the blocking car stayed: [10082, 10158].
- The 18 crashes into the blocking car. Critic value (shaping removed; a crash is -50): -17.2 at reset, -22.0 at the last decision, 0.1 s before the crash (2 of them above 0).
- Same crashes, sampling probabilities per decision while behind the blocking car (what training explores): LANE_LEFT 6.4%, IDLE 6.4%, LANE_RIGHT 5.7%, FASTER 80.6%, SLOWER 1.0%.

## B3_seed9

| Blocking car | Episodes | Reached | Crashed into it | Crashed into another car | Timeout |
|---|---:|---:|---:|---:|---:|
| stays in its lane | 19 | 2 | 17 | 0 | 0 |
| leaves its lane by itself | 26 | 20 | 2 | 4 | 0 |

- Ego passed a blocking car that was still in its lane: 0/45 episodes.
- Slot in ego's own lane (27 episodes): ego started a lane change while behind the blocking car in 0, and chose SLOWER while behind it in 0.
- Reached although the blocking car stayed: [10082, 10158].
- The 19 crashes into the blocking car. Critic value (shaping removed; a crash is -50): -5.0 at reset, -9.2 at the last decision, 0.1 s before the crash (6 of them above 0).
- Same crashes, sampling probabilities per decision while behind the blocking car (what training explores): LANE_LEFT 8.8%, IDLE 3.1%, LANE_RIGHT 1.7%, FASTER 86.2%, SLOWER 0.2%.
