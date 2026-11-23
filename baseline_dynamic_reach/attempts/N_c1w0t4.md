# N_c1w0t4: no practice, c1w0t4

*Step 8c2, trained 2026-11-18 to 2026-11-20. Seeds: 7, 10, 13, 16, 19, 22, 25, 28. [All attempts](README.md) · [Experiment notes](../EXPERIMENT_NOTES.md)*

**Status:** 8d finalist: confirmed on 5 new seeds (8 in all); not chosen.

## Idea

**Order N, no practice.** All three rounds are on the full task. Round 1 is success only (B0's lesson: learn to reach before any crash cost); rounds 2 and 3 use the reward under test. It asks whether practice is needed at all.

**Crash cost 1 × goal**, P6's level: a crash costs as much as a goal.

**Time cost 0.004 per decision:** a whole timed-out episode costs 1 goal, the largest level that keeps waiting cheaper than the smallest crash cost tested (0.5). Aimed at the FRONT give-up.

Step 8c2 added a time cost, because 8c measured that every crash cost makes runs give up on FRONT (timeouts of 10% or more).

## Approach

Three rounds of 2M transitions on the Windows PC, each starting from the previous round's trained policy (same seed); round 1 from fresh weights, success only. A fourth round of 3M (9M in all) continued round 3 in Step 8e.

| Run | Arm | Task | Reward | Transitions | Starts from | Trained on |
|---|---|---|---:|---:|---|---|
| 1 | `N1` | full task | +1 reach, crash 0 | 2M | fresh weights | PC |
| 2 | `N2_c1w0t4` | full task | +1 reach, −1 crash, −0.004/decision | 2M | `N1`'s trained policy (same seed) | PC |
| 3 | `N3_c1w0t4` | full task | +1 reach, −1 crash, −0.004/decision | 2M | `N2_c1w0t4`'s trained policy (same seed) | PC |
| 4 | `N4_c1w0t4` | full task | +1 reach, −1 crash, −0.004/decision | 3M | `N3_c1w0t4`'s trained policy (same seed) | PC |

## Reward function

Of the last run, `N3_c1w0t4` (per decision, in goal units after the reward scale):

- +1 when ego reaches the slot (the episode ends)
- −1 on a crash (the episode ends)
- −0.004 per decision (0.1 s); a whole 25 s episode costs 1
- 0 on a timeout (25 s) or a lost anchor

Written out: `r = 0.02 × (50·[reached] − 50·[crash] − 0.2 per decision)`, where [reached] and [crash] are 1 on the decision the episode ends that way and 0 otherwise.

Earlier runs of the pipeline used:

- `N1`: +1 reach, crash 0 (`r = 0.02 × (50·[reached])`)

## Hyperparameters

From `N3_c1w0t4`'s saved configuration (identical for every run of the pipeline except the task, reward and starting weights shown above).

| Setting | Value |
|---|---|
| Algorithm | PPO, Stable-Baselines3 2.9.2 |
| Policy network | MLP 64-64 (tanh), separate from the value network |
| Value network | MLP 64-64 |
| Learning rate | 0.0003 (constant) |
| Environments in parallel | 3 |
| Steps per environment per update (n_steps) | 256 |
| Transitions per update | 768 |
| Minibatch size | 256 |
| Epochs per update | 5 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Gradient-norm clip (policy and value together) | 0.5 |
| Device | cpu |

<details><summary>Environment and task settings</summary>

| Setting | Value |
|---|---|
| Simulator | highway-env 1.12.1, highway-v0, 3 lanes |
| Traffic | 8 cars, density 0.7, preferred speeds 12–17 m/s (IDM and MOBIL) |
| Start | ego at 20 m/s; a car 35 m ahead in ego's lane |
| Decisions | 10 per second; episode limit 25 s (250 decisions) |
| Actions | LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER; target speeds 10, 12.5, 15, 17.5, 20, 22.5, 25 m/s |
| Observation | 7 cars × 5 (presence, x, y, vx, vy), x ±200 m, plus goal, anchor, slot type, slot error and hold progress (53 numbers) |
| Anchor and slot | anchor: a random car 30–120 m ahead, held in its lane; FRONT 35 m ahead of it, BACK 15 m behind, LEFT/RIGHT beside |
| Success | ego's center within ±3 m along and ±1.5 m across the slot for 0.1 s (one decision) |
| Reward scale | 0.02 |
| Task | the full task: all four slots, natural start (ego at 20 m/s, a 12 m/s car 35 m ahead), traffic free to change lanes |

</details>

Software: highway-env 1.12.1, gymnasium 1.4.0, Stable-Baselines3 2.9.2, PyTorch 2.15.0, NumPy 2.5.4, Python 3.13.1.

## Results

Score = reached − crash per episode (+1, −1, 0 for a timeout), the study's measure since Step 8.

### Full task: 200 fixed episodes (10,000–10,199), final model of every seed

|  | Reached | Crash | Timeout | Score | B/L/R reached | B/L/R crash | FRONT reached | FRONT crash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 7 | 78.5% | 21.5% | 0.0% | +0.570 | 78.3% | 21.7% | 79.0% | 21.0% |
| seed 10 | 64.0% | 24.0% | 12.0% | +0.400 | 78.3% | 21.7% | 32.3% | 29.0% |
| seed 13 | 68.5% | 8.0% | 23.5% | +0.605 | 83.3% | 7.2% | 35.5% | 9.7% |
| seed 16 | 87.5% | 12.5% | 0.0% | +0.750 | 92.0% | 8.0% | 77.4% | 22.6% |
| seed 19 | 58.0% | 11.0% | 31.0% | +0.470 | 84.1% | 15.9% | 0.0% | 0.0% |
| seed 22 | 70.0% | 7.0% | 23.0% | +0.630 | 87.0% | 7.2% | 32.3% | 6.5% |
| seed 25 | 71.0% | 16.0% | 13.0% | +0.550 | 89.1% | 10.9% | 30.6% | 27.4% |
| seed 28 | 67.5% | 9.5% | 23.0% | +0.580 | 83.3% | 11.6% | 32.3% | 4.8% |
| **mean of 8** | **70.6%** | **13.7%** | 15.7% | **+0.569** | 84.4% | 13.0% | 39.9% | 15.1% |
| scripted driver | 76.5% | 2.5% | 21.0% | +0.740 | 81.9% | 2.2% | 64.5% | 3.2% |
| random driver | 11.5% | 79.0% | 9.5% | -0.675 | 15.2% | 76.1% | 3.2% | 85.5% |

### Step 8's own measure: 400 selection episodes (10,000–10,199 and 40,000–40,199), full task

Every round's model on the full task. Rounds 3 and 4 are the mean of the final model and the checkpoints 400k and 200k before the end, as the grid scored them; rounds 1–2 are the final model.

| Model | Score | Reached | Crash | Timeout | FRONT reached | Seeds giving up on FRONT (≥ 30% timeouts) | Score by seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `N1` (after 2M) | +0.380 | 68.5% | 30.5% | 1.1% | 34.7% | 0 of 8 | +0.41, +0.26, +0.37, +0.47, +0.31, +0.38, +0.40, +0.43 |
| `N2_c1w0t4` (after 4M) | +0.553 | 68.4% | 13.1% | 18.5% | 30.8% | 6 of 8 | +0.53, +0.50, +0.64, +0.73, +0.48, +0.52, +0.55, +0.49 |
| `N3_c1w0t4` (after 6M) | +0.585 | 71.4% | 12.9% | 15.7% | 40.5% | 6 of 8 | +0.68, +0.49, +0.63, +0.77, +0.41, +0.56, +0.56, +0.58 |
| `N4_c1w0t4` (after 9M) | +0.624 | 73.7% | 11.3% | 15.0% | 44.5% | 5 of 8 | +0.78, +0.45, +0.60, +0.82, +0.47, +0.63, +0.65, +0.61 |

## Training curves

![N_c1w0t4 training curves](figures/N_c1w0t4_training.png)


## Demos

Training seed 7's final model on its own evaluation episodes (the full task), chosen by a fixed rule: the first FRONT episode it reaches (else the first blocked BACK/LEFT/RIGHT episode, else the first it reaches; on practice episodes simply the first), and the first it crashes in; a timeout fills either place if it never happens. Each figure shows four frames, then per decision the slot error, lanes, speeds, gap to the car ahead, the action and the critic's value. In the videos the anchor is outlined in magenta and the slot is the green box.

**Reached: episode 10001, FRONT slot, success at 12.2 s.** [Video](../results/examples/N3_c1w0t4_seed7/front_seed10001.mp4)

![N_c1w0t4 reached, episode 10001](../results/examples/N3_c1w0t4_seed7/front_seed10001.png)

**Crash: episode 10002, BACK slot, collision at 2.6 s.** [Video](../results/examples/N3_c1w0t4_seed7/back_seed10002.mp4)

![N_c1w0t4 crash, episode 10002](../results/examples/N3_c1w0t4_seed7/back_seed10002.png)

## Analysis

- On the screening seeds (7, 10, 13) it ranks 11 of the 26 64-64 pipelines (score +0.600; the best is O_c1w01, +0.697).
- Against success only, same order (`N3_c0w0`): +0.091 (95% interval +0.008 to +0.173, p 0.04, ahead on 7 of 8 seeds; crash rate -12.3 points).
- Against P6's reward, same order (`N3_c1w01`): -0.006 (95% interval -0.086 to +0.074, p 0.87, ahead on 3 of 8 seeds; crash rate +3.4 points).
- Against order O, same reward (`O3_c1w0t4`): -0.067 (95% interval -0.259 to +0.126, p 0.44, ahead on 3 of 8 seeds; crash rate +3.2 points).

Across 8c2: a time cost moves the give-up but does not remove the trade. Where t4 ends it (crash 0.5), part of the timeouts become reaches and part crashes. No time-cost cell is beyond noise of its no-time-cost cell.

**The winner's order with O_c1w0t4's reward:** it links the two by one factor each. An 8d finalist.

