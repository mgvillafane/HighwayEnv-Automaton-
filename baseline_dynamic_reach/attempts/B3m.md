# B3m: B3 trained again on the Mac (the noise floor)

*Step 5b, trained 2026-11-09. Seeds: 7, 8, 9. [All attempts](README.md) · [Experiment notes](../EXPERIMENT_NOTES.md)*

**Status:** Within 1 point of B3; the same 24 episodes fail on all seeds. Sets the noise floor.

## Idea

Work moved to the Mac, and `ivcheck` refuses a Mac child of a PC parent, so B3 was trained again there. It is also a
measurement of how much a result moves when nothing but the machine changes.

*Prediction:* within 5 points of B3.

## Approach

B3's settings exactly, trained on the Mac (declared IV `platform`).

| Run | Arm | Task | Reward | Transitions | Starts from | Trained on |
|---|---|---|---:|---:|---|---|
| 1 | `B0` | full task | +50 reach, crash 0 | 1M | fresh weights | PC |
| 2 | `B3m` | full task | +50 reach, −50 crash, progress 0.1/m | 1M | `B0`'s trained policy (same seed) | Mac |

## Reward function

Of the last run, `B3m` (per decision, in goal units after the reward scale):

- +50 when ego reaches the slot (the episode ends)
- −50 on a crash (the episode ends)
- +0.1 × (γ·Φ(next) − Φ(now)) per decision, Φ = −(distance to the slot in m), γ = 0.99: potential-based progress, which cannot change the best policy
- 0 on a timeout (25 s) or a lost anchor

Written out: `r = 50·[reached] − 50·[crash] + 0.1·(0.99·Φ′ − Φ)`, where [reached] and [crash] are 1 on the decision the episode ends that way and 0 otherwise.

Earlier runs of the pipeline used:

- `B0`: +50 reach, crash 0 (`r = 50·[reached]`)

## Hyperparameters

From `B3m`'s saved configuration (identical for every run of the pipeline except the task, reward and starting weights shown above).

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
| Reward scale | 1 |
| Task | the full task: all four slots, natural start (ego at 20 m/s, a 12 m/s car 35 m ahead), traffic free to change lanes |

</details>

Software: highway-env 1.12.1, gymnasium 1.4.0, Stable-Baselines3 2.9.2, PyTorch 2.15.0, NumPy 2.5.4, Python 3.12.13.

## Results

Score = reached − crash per episode (+1, −1, 0 for a timeout), the study's measure since Step 8.

### Full task: 200 fixed episodes (10,000–10,199), final model of every seed

|  | Reached | Crash | Timeout | Score | B/L/R reached | B/L/R crash | FRONT reached | FRONT crash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 7 | 55.5% | 13.5% | 31.0% | +0.420 | 80.4% | 19.6% | 0.0% | 0.0% |
| seed 8 | 56.5% | 14.0% | 29.5% | +0.425 | 81.9% | 18.1% | 0.0% | 4.8% |
| seed 9 | 55.0% | 15.5% | 29.5% | +0.395 | 79.7% | 20.3% | 0.0% | 4.8% |
| **mean of 3** | **55.7%** | **14.3%** | 30.0% | **+0.413** | 80.7% | 19.3% | 0.0% | 3.2% |
| scripted driver | 76.5% | 2.5% | 21.0% | +0.740 | 81.9% | 2.2% | 64.5% | 3.2% |
| random driver | 11.5% | 79.0% | 9.5% | -0.675 | 15.2% | 76.1% | 3.2% | 85.5% |

## Training curves

![B3m training curves](figures/B3m_training.png)


## Demos

Training seed 7's final model on its own evaluation episodes (the full task), chosen by a fixed rule: the first FRONT episode it reaches (else the first blocked BACK/LEFT/RIGHT episode, else the first it reaches; on practice episodes simply the first), and the first it crashes in; a timeout fills either place if it never happens. Each figure shows four frames, then per decision the slot error, lanes, speeds, gap to the car ahead, the action and the critic's value. In the videos the anchor is outlined in magenta and the slot is the green box.

**Reached: episode 10004, BACK slot, success at 4.3 s.** [Video](../results/examples/B3m_seed7/back_seed10004.mp4)

![B3m reached, episode 10004](../results/examples/B3m_seed7/back_seed10004.png)

**Crash: episode 10002, BACK slot, collision at 2.6 s.** [Video](../results/examples/B3m_seed7/back_seed10002.mp4)

![B3m crash, episode 10002](../results/examples/B3m_seed7/back_seed10002.png)

## Analysis

- **Within 1 point of B3.** The same 24 episodes fail on all three seeds.
- **The noise floor:** repeating B3 changes 3–4 of the 138 BACK/LEFT/RIGHT episodes per seed. An effect is credible
  only if it is larger than about 5 episodes per seed and has the same sign on all three seeds.
- Evaluation is identical on both machines; training differs within this floor.

