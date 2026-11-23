# N_c0w0: no practice, c0w0

*Step 8c, trained 2026-11-18 to 2026-11-20. Seeds: 7, 10, 13, 16, 19, 22, 25, 28. [All attempts](README.md) · [Experiment notes](../EXPERIMENT_NOTES.md)*

**Status:** 8d finalist: confirmed on 5 new seeds (8 in all); not chosen.

## Idea

**Order N, no practice.** All three rounds are on the full task. Round 1 is success only (B0's lesson: learn to reach before any crash cost); rounds 2 and 3 use the reward under test. It asks whether practice is needed at all.

**Crash cost 0:** success only in every round, the control. Discounting alone pays for arriving early, so it predicts fast driving and many crashes.

Step 8c was the reward grid on orders N and O: crash 0 / 0.5 / 1 / 2 × progress 0 / 0.1, 3 seeds. The rule picks the simplest reward that the best one does not beat beyond noise.

## Approach

Three rounds of 2M transitions on the Windows PC, each starting from the previous round's trained policy (same seed); round 1 from fresh weights, success only.

| Run | Arm | Task | Reward | Transitions | Starts from | Trained on |
|---|---|---|---:|---:|---|---|
| 1 | `N1` | full task | +1 reach, crash 0 | 2M | fresh weights | PC |
| 2 | `N2_c0w0` | full task | +1 reach, crash 0 | 2M | `N1`'s trained policy (same seed) | PC |
| 3 | `N3_c0w0` | full task | +1 reach, crash 0 | 2M | `N2_c0w0`'s trained policy (same seed) | PC |

## Reward function

Of the last run, `N3_c0w0` (per decision, in goal units after the reward scale):

- +1 when ego reaches the slot (the episode ends)
- 0 on a crash: a crash ends the episode with nothing, like a timeout
- 0 on a timeout (25 s) or a lost anchor

Written out: `r = 0.02 × (50·[reached])`, where [reached] and [crash] are 1 on the decision the episode ends that way and 0 otherwise.

## Hyperparameters

From `N3_c0w0`'s saved configuration (identical for every run of the pipeline except the task, reward and starting weights shown above).

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
| seed 7 | 87.0% | 13.0% | 0.0% | +0.740 | 92.0% | 8.0% | 75.8% | 24.2% |
| seed 10 | 63.0% | 37.0% | 0.0% | +0.260 | 80.4% | 19.6% | 24.2% | 75.8% |
| seed 13 | 76.0% | 24.0% | 0.0% | +0.520 | 88.4% | 11.6% | 48.4% | 51.6% |
| seed 16 | 76.0% | 24.0% | 0.0% | +0.520 | 85.5% | 14.5% | 54.8% | 45.2% |
| seed 19 | 68.0% | 32.0% | 0.0% | +0.360 | 80.4% | 19.6% | 40.3% | 59.7% |
| seed 22 | 73.0% | 26.5% | 0.5% | +0.465 | 87.7% | 12.3% | 40.3% | 58.1% |
| seed 25 | 76.5% | 23.5% | 0.0% | +0.530 | 89.9% | 10.1% | 46.8% | 53.2% |
| seed 28 | 72.5% | 27.0% | 0.5% | +0.455 | 84.8% | 15.2% | 45.2% | 53.2% |
| **mean of 8** | **74.0%** | **25.9%** | 0.1% | **+0.481** | 86.1% | 13.9% | 47.0% | 52.6% |
| scripted driver | 76.5% | 2.5% | 21.0% | +0.740 | 81.9% | 2.2% | 64.5% | 3.2% |
| random driver | 11.5% | 79.0% | 9.5% | -0.675 | 15.2% | 76.1% | 3.2% | 85.5% |

### Step 8's own measure: 400 selection episodes (10,000–10,199 and 40,000–40,199), full task

Every round's model on the full task. Rounds 3 and 4 are the mean of the final model and the checkpoints 400k and 200k before the end, as the grid scored them; rounds 1–2 are the final model.

| Model | Score | Reached | Crash | Timeout | FRONT reached | Seeds giving up on FRONT (≥ 30% timeouts) | Score by seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `N1` (after 2M) | +0.380 | 68.5% | 30.5% | 1.1% | 34.7% | 0 of 8 | +0.41, +0.26, +0.37, +0.47, +0.31, +0.38, +0.40, +0.43 |
| `N2_c0w0` (after 4M) | +0.447 | 72.0% | 27.3% | 0.7% | 42.3% | 0 of 8 | +0.57, +0.34, +0.47, +0.62, +0.33, +0.49, +0.31, +0.43 |
| `N3_c0w0` (after 6M) | +0.494 | 74.6% | 25.2% | 0.2% | 48.0% | 0 of 8 | +0.80, +0.28, +0.51, +0.63, +0.36, +0.48, +0.47, +0.41 |

## Training curves

![N_c0w0 training curves](figures/N_c0w0_training.png)


## Demos

Training seed 7's final model on its own evaluation episodes (the full task), chosen by a fixed rule: the first FRONT episode it reaches (else the first blocked BACK/LEFT/RIGHT episode, else the first it reaches; on practice episodes simply the first), and the first it crashes in; a timeout fills either place if it never happens. Each figure shows four frames, then per decision the slot error, lanes, speeds, gap to the car ahead, the action and the critic's value. In the videos the anchor is outlined in magenta and the slot is the green box.

**Reached: episode 10001, FRONT slot, success at 12.2 s.** [Video](../results/examples/N3_c0w0_seed7/front_seed10001.mp4)

![N_c0w0 reached, episode 10001](../results/examples/N3_c0w0_seed7/front_seed10001.png)

**Crash: episode 10009, RIGHT slot, collision at 3.8 s.** [Video](../results/examples/N3_c0w0_seed7/right_seed10009.mp4)

![N_c0w0 crash, episode 10009](../results/examples/N3_c0w0_seed7/right_seed10009.png)

## Analysis

- On the screening seeds (7, 10, 13) it ranks 20 of the 26 64-64 pipelines (score +0.534; the best is O_c1w01, +0.697).
- Against P6's reward, same order (`N3_c1w01`): -0.097 (95% interval -0.150 to -0.043, p 0.00, ahead on 1 of 8 seeds; crash rate +15.8 points).
- Against order O, same reward (`O3_c0w0`): -0.077 (95% interval -0.262 to +0.107, p 0.35, ahead on 2 of 8 seeds; crash rate +3.8 points).

Across 8c: any crash cost brings the FRONT give-up, and success only never does. The score swaps crashes for timeouts almost one for one. Nothing beat success only by the rule with 3 seeds; P6's reward scored highest in both orders.

**Success only, no practice.** The rule's choice after the 3-seed screening (nothing beat it beyond noise), but on
the 5 confirmation seeds every crash-cost reward beat it in order N (+0.09 to +0.16, p 0.002–0.049). It never gives
up and crashes in about 25% of episodes.

