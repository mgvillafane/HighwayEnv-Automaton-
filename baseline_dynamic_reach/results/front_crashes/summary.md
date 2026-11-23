# Step 8g: how O3_c1w0t4_n256 crashes on FRONT (diagnosis, no training)

133 FRONT episodes of the selection sets (10,000-10,199 and 40,000-40,199), final models of seeds 7, 10, 13, 16, 19, 22, 25, 28, deterministic actions. Replayed outcomes and lengths match the saved evaluations: yes, all.

## Outcomes

| Run | Reached | Crash | Timeout |
|---|---|---|---|
| O3_c1w0t4_n256_seed7 | 105 (79%) | 26 (20%) | 2 |
| O3_c1w0t4_n256_seed10 | 109 (82%) | 24 (18%) | 0 |
| O3_c1w0t4_n256_seed13 | 104 (78%) | 29 (22%) | 0 |
| O3_c1w0t4_n256_seed16 | 120 (90%) | 13 (10%) | 0 |
| O3_c1w0t4_n256_seed19 | 102 (77%) | 31 (23%) | 0 |
| O3_c1w0t4_n256_seed22 | 106 (80%) | 27 (20%) | 0 |
| O3_c1w0t4_n256_seed25 | 94 (71%) | 39 (29%) | 0 |
| O3_c1w0t4_n256_seed28 | 89 (67%) | 44 (33%) | 0 |
| scripted | 79 (59%) | 6 (5%) | 48 |
| all 8 seeds | 829 (77.9%) | 233 (21.9%) | 2 |

## The 233 crashes of the 8 seeds

**Car hit and where it was:**

| Crash | Count | Share |
|---|---|---|
| other car, ahead | 76 | 33% |
| lead car, ahead | 47 | 20% |
| other car, ahead, ego changing lanes | 46 | 20% |
| anchor, ahead | 21 | 9% |
| anchor, ahead, ego changing lanes | 21 | 9% |
| anchor, beside, ego changing lanes | 6 | 3% |
| other car, beside, ego changing lanes | 5 | 2% |
| anchor, beside | 5 | 2% |
| anchor, behind | 3 | 1% |
| lead car, ahead, ego changing lanes | 2 | 1% |
| other car, behind, ego changing lanes | 1 | 0% |

**Phase of the FRONT maneuver at the crash:**

| Phase | Count | Share |
|---|---|---|
| behind the anchor | 93 | 40% |
| beside the anchor | 48 | 21% |
| past the anchor, merging into its lane | 42 | 18% |
| past the anchor, in its lane | 36 | 15% |
| past the anchor, other lane | 14 | 6% |

**Phase × car hit:**

| Phase, car hit | Count | Share |
|---|---|---|
| behind the anchor: lead car ahead | 49 | 21% |
| past the anchor, in its lane: other car ahead | 36 | 15% |
| past the anchor, merging into its lane: other car ahead | 36 | 15% |
| behind the anchor: anchor ahead | 32 | 14% |
| beside the anchor: other car ahead | 24 | 10% |
| past the anchor, other lane: other car ahead | 14 | 6% |
| behind the anchor: other car ahead | 12 | 5% |
| beside the anchor: anchor beside | 11 | 5% |
| beside the anchor: anchor ahead | 10 | 4% |
| past the anchor, merging into its lane: other car beside | 5 | 2% |
| beside the anchor: anchor behind | 3 | 1% |
| past the anchor, merging into its lane: other car behind | 1 | 0% |

**Numbers at the crash (medians):**

| Group | n | Time (s) | Ego speed | Anchor speed | Closing speed | Ego − anchor (m) | Slot − ego (m) | SLOWER in last 2 s | FASTER in last 2 s |
|---|---|---|---|---|---|---|---|---|---|
| all crashes | 233 | 8.1 | 25.0 | 13.6 | 12.1 | -0.8 | 34.5 | 2% | 46% |
| behind the anchor | 93 | 3.8 | 24.9 | 13.6 | 11.4 | -40.4 | 75.4 | 0% | 44% |
| beside the anchor | 48 | 8.1 | 25.0 | 15.1 | 11.7 | -0.6 | 34.5 | 0% | 56% |
| past the anchor, merging into its lane | 42 | 8.8 | 25.0 | 13.5 | 12.3 | 28.8 | 6.2 | 0% | 50% |
| past the anchor, in its lane | 36 | 8.0 | 25.0 | 12.5 | 12.8 | 30.8 | 4.2 | 11% | 39% |
| past the anchor, other lane | 14 | 11.2 | 25.0 | 14.2 | 12.1 | 12.2 | 22.8 | 0% | 36% |

## Are the crashes tied to episodes or to seeds?

Number of the 8 seeds that crash on an episode, over the 133 FRONT episodes:

| Seeds crashing | Episodes |
|---|---|
| 0 | 49 |
| 1 | 23 |
| 2 | 30 |
| 3 | 13 |
| 4 | 7 |
| 6 | 1 |
| 7 | 3 |
| 8 | 7 |

Episodes where at least half the seeds crash: 18, with 111 of the 233 crashes. The scripted driver on them: 4 reached, 4 crash, 10 timeout.

| Episode | Seeds crashing | Lead car is the anchor | Scripted | Typical crash |
|---|---:|---|---|---|
| 10017 | 4 | no | T | past the anchor, in its lane: other car ahead |
| 10054 | 8 | no | T | behind the anchor: anchor ahead |
| 10061 | 8 | no | T | beside the anchor: other car ahead |
| 10089 | 8 | no | T | past the anchor, in its lane: other car ahead |
| 10107 | 4 | no | S | past the anchor, other lane: other car ahead |
| 10151 | 8 | no | S | behind the anchor: anchor ahead |
| 10181 | 8 | no | S | beside the anchor: other car ahead |
| 10184 | 7 | no | T | beside the anchor: other car ahead |
| 10190 | 4 | no | C | behind the anchor: anchor ahead |
| 40005 | 6 | no | T | past the anchor, other lane: other car ahead |
| 40007 | 4 | yes | S | past the anchor, merging into its lane: other car ahead |
| 40014 | 7 | yes | C | past the anchor, in its lane: other car ahead |
| 40037 | 4 | yes | C | past the anchor, in its lane: other car ahead |
| 40052 | 8 | no | T | past the anchor, in its lane: other car ahead |
| 40054 | 8 | no | C | past the anchor, merging into its lane: other car ahead |
| 40092 | 7 | no | T | behind the anchor: anchor ahead |
| 40152 | 4 | no | T | past the anchor, merging into its lane: other car ahead |
| 40191 | 4 | no | T | behind the anchor: lead car ahead |

## Scripted driver's outcome on each PPO outcome

Each PPO episode (seed × episode) against the scripted driver's outcome on the same episode:

| PPO \ scripted | S | C | T |
|---|---|---|---|
| S | 542 | 21 | 266 |
| C | 89 | 27 | 117 |
| T | 1 | 0 | 1 |
