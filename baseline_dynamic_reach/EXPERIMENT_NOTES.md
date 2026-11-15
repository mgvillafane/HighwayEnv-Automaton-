# Experiment notes: reaching a slot of a moving car

A living record of the dynamic-waypoint *reach* experiments. Each step is added at the end of
the [experiment log](#9-experiment-log), and every decision gets a line in the
[decision log](#10-decision-log). All numbers come from files in this folder.

**Status (2026-11-08, 16:40).** Steps 0–4 are done.

| Arm | Change | Success | Crash | Main failure |
|---|---|---:|---:|---|
| B0 | Reward R0 = 50 × success | 55% | 44% | Ramming: FRONT hits the anchor; elsewhere the lead car |
| B1 | Crash weight 0 → 50, trained from scratch | 1% | 1% | Collapse: drives at 10 m/s and never reaches anything |
| B2 | B1, but training starts from B0's policy | 53% | 16% | Gives up on FRONT entirely; same lead-car crashes as B0 |
| B3 | B2 + potential-based progress toward the slot | 56% | 14% | Still gives up on FRONT; BACK/LEFT/RIGHT at the scripted driver's level |

Step 5 is proposed at the end of the log and **waiting for review**.

## 1. Aim and working rules

**Aim.** Train PPO to drive ego into a slot (FRONT, BACK, LEFT or RIGHT) of a moving traffic
car. Start from the simplest reward and add to it step by step, so that every part of the
final design is backed by a measured effect.

**Rules.**

1. **One independent variable (IV) per comparison.**
   - An arm is defined in [`config.py`](config.py) as its parent arm plus exactly one changed setting.
   - [`ivcheck.py`](ivcheck.py) compares the saved configurations of two runs and refuses to train or compare them if anything else differs, including package versions and the machine.
2. **Paired replicates.**
   - Every arm trains on the same three seeds (7, 8, 9).
   - Every policy and every arm is measured on the same 200 evaluation episodes.
3. **A prediction before the results.** Each step states what we expect and why, before training starts.
4. **A reason for every decision**, recorded in the decision log.
5. **Failures stay.** No run is deleted or repeated to get a better number. `train.py` refuses to overwrite an existing run.
6. **Simple first.** No reward term is added without a measured failure that it addresses.

**Earlier work is context only.** The earlier study ([`../baseline_dynamic_waypoint/RESULTS.md`](../baseline_dynamic_waypoint/RESULTS.md)) used a different task (hold) and a different reward. Nothing here is concluded from comparing with it.

## 2. The task: reach

| | |
|---|---|
| Road | 3 straight lanes, each 4 m wide. Lane 0 is the leftmost. |
| Ego | Starts at 20 m/s. Makes one decision every 0.1 s (10 Hz). |
| Traffic | 8 IDM cars at 12–17 m/s. A 12 m/s lead car starts 35 m ahead of ego, in ego's lane. |
| Anchor | Chosen uniformly from the traffic cars 30–120 m ahead of ego. Its lane changes are switched off, so its LEFT/RIGHT slot never leaves the road. IDM still sets its speed. |
| Slot | Chosen uniformly from the slots that are on the road for the anchor's lane: no LEFT in lane 0, no RIGHT in lane 2. |
| Slot box | ±3 m along the slot lane, ±1.5 m across it (figure in section 3). The box moves with the anchor. |
| **Success** | **The first decision where ego's center is inside the box, without a crash.** Speed does not matter. |
| Episode end | Success or a crash ends the episode (a crash wins if both happen on the same decision). Reaching 25 s cuts it off as a timeout. If the anchor crashes or leaves the road, the episode is cut off as "anchor lost". |

In code, reach is the inherited hold task with `hold_seconds = 0.1`, which is one decision. The environment code is unchanged.

**Why reach before hold.**
- Holding is reaching plus staying.
- In the earlier study, the scripted driver succeeded equally often with 1 s and 5 s holds (76%), so arriving is the hard part.
- Step 0 confirms this: the same scripted driver reaches the slot in 76.5% of episodes.
- Holding will come back later as a one-IV step: `hold_seconds` 0.1 → 1.

## 3. Goals: sampling, geometry, and samples

**The four slots around one anchor, to scale.** A FRONT slot center is 35 m ahead of the anchor, a BACK slot center is 15 m behind it, and the LEFT and RIGHT slots sit beside it in the next lane:

![Slot geometry](figures/slot_geometry.png)

**Where the goal starts in all 200 evaluation episodes**, relative to ego's starting position ([data](figures/goal_samples.json)):

![Goal samples](figures/goal_samples.png)

**One real frame per slot at reset**, as the viewer draws it: ego in blue, the anchor outlined in magenta, the slot box in green.

![One frame per slot](figures/slot_frames.png)

What the samples show:

- **Slot counts:** FRONT 62, BACK 61, LEFT 40, RIGHT 37. LEFT and RIGHT are rarer because the anchor is drawn first, and an anchor in an edge lane has no slot on that side.
- **Distance:** the median initial distance to the slot center is FRONT 104 m, BACK 55 m, LEFT 70 m and RIGHT 72 m. FRONT is farthest because its slot sits 35 m beyond the anchor.
- **Lane changes needed:**

  | Slot | 0 | 1 | 2 |
  |---|---:|---:|---:|
  | FRONT | 28 | 22 | 12 |
  | BACK | 25 | 24 | 12 |
  | LEFT | 9 | 29 | 2 |
  | RIGHT | 8 | 24 | 5 |

- **The inherited lead car is the anchor in 51 of the 200 episodes (25%).**
  - It is always 12 m/s and always starts 35 m ahead in ego's lane.
  - That is why the top plot has tight clusters at 20 m (its BACK slot) and at 35 m one lane over (its LEFT/RIGHT slots).
  - These episodes are much easier: random actions succeed in 15 of 51 of them, against 8 of the other 149.
  - We keep this unchanged and report it, because it is part of the inherited task. Removing it would be its own IV.

## 4. Observation space (53 inputs, unchanged from the earlier study)

PPO's `MultiInputPolicy` flattens these into one vector of 53 numbers. Every entry, with its real value at reset on evaluation seed 10,000, is labelled in [`figures/observation_sample.md`](figures/observation_sample.md).

| Key | Size | What it holds | Scale | Why it is there |
|---|---:|---|---|---|
| `vehicles` | 7 × 5 | Ego and the 6 nearest cars (cars behind ego included): presence, x, y, vx, vy. Other cars are relative to ego. Ego's x is its progress since reset. | x / 200 m, y / 12 m, vx / 30 m/s, vy / 10 m/s, clipped to [-1, 1] | To see the traffic it must avoid. This is highway-env's standard `Kinematics` observation. |
| `achieved_goal` | 2 | Ego's position in the episode frame (origin: ego's start x, road y = 0) | / [200, 12] m | Inherited from the static-goal task |
| `desired_goal` | 2 | The slot's position in the same frame. It moves with the anchor. | / [200, 12] m | Inherited from the static-goal task |
| `goal_delta` | 2 | Slot minus ego | / [200, 12] m | Where the goal is, at long range |
| `anchor` | 4 | Anchor minus ego: x, y, vx, vy | Same as `vehicles`, clipped | How the goal moves |
| `slot` | 4 | One-hot: FRONT, BACK, LEFT, RIGHT | 0/1 | Which slot is the target |
| `hold_progress` | 1 | Decisions held in the box divided by decisions required | [0, 1] | **Does nothing in reach:** it is always 0 before success. Kept so that reach → hold later changes only one setting. |
| `slot_error` | 3 | Slot minus ego along the slot lane and across it, and anchor speed minus ego speed | / 10 m, / 4 m, / 10 m/s, clipped | Where the goal is, at short range: the ±3 m box spans ±0.3 here, against ±0.015 in `goal_delta` |

Known properties, recorded here, not changed:
- `achieved_goal` and `desired_goal` are redundant with `goal_delta` for a moving goal.
- `slot_error` saturates beyond 10 m along and beyond 4 m across. In the sample above, a slot two lanes away reads +1.0 (+4 m) instead of +8 m.
- Each of these is a candidate IV (section 8).

## 5. Action space

There are 5 discrete actions, one decision every 0.1 s:
- `LANE_LEFT` / `LANE_RIGHT` start a lane change, steered by highway-env's own controller.
- `FASTER` / `SLOWER` move the target speed one step on the grid 10, 12.5, …, 25 m/s.
- `IDLE` keeps the current targets.

## 6. Reward functions

### R0 (Step 1): success only

```text
r_t = 50 · [ego reached the slot box at decision t, without a crash]
```

[`config.py`](config.py) (`R0`) sets the inherited terms `collision`, `progress`, `time` and `hold` to weight 0. The environment is unchanged.

**Why this reward.**
- It is the scored outcome and nothing else, so there is no term to justify yet.
- The value 50 is Marcos's goal bonus. When a later arm turns on one of his terms at his weight, only that term changes, not also the overall scale.
- Within R0 alone the scale barely matters, because PPO normalizes advantages.

**What R0 implies.**
- A crash and a timeout both earn 0. The agent avoids crashing only when a crash would cost it the goal.
- The discount (γ = 0.99) still favors arriving sooner: a goal k decisions away is worth 50 · 0.99^k now.

### Marcos's reward, for reference: not used as a single arm

```text
r_t = 50 · success − 50 · crash + 0.1 · (previous distance − distance) − 0.05
```

- This is **not** highway-env's reward. Marcos wrote it for his fixed-point task, replacing highway-env's own reward, which is speed + right lane + crash for `highway-v0`.
- Compared with R0 it changes three terms at once, so it cannot be one comparison.
- If the evidence calls for them, its terms will be added one per step.

**A note for later shaping terms.**
- Shaping preserves the best policy only in the form `γ·Φ(s′) − Φ(s)` (Ng, Harada and Russell, 1999).
- The earlier study's "telescoping" terms used `Φ(s) − Φ(s′)`, which leaves out γ, so strictly they could change the best policy.
- Any shaping term added here will use the γ form.

## 7. Fixed protocol (the same for every arm)

**Training.**
- Stable-Baselines3 PPO with `MultiInputPolicy`, and separate 64-64 networks for the policy and the value function.
- Learning rate 3e-4; 256 steps × 3 environments = 768 transitions per update; batch 256; 5 epochs; γ 0.99; GAE λ 0.95; clip 0.2; entropy 0.01; value coefficient 0.5; gradient clip 0.5; CPU.
- 1,000,000 requested transitions, which becomes 1,000,704 with whole rollouts.
- Seeds 7, 8 and 9. Each run's 3 workers use seed, seed+1 and seed+2.

**Evaluation.**
- Episodes: seeds 10,000–10,199, with goals sampled exactly as in training (section 3).
- Policies, on identical episodes:
  - PPO with deterministic actions;
  - uniform random actions (a fixed random stream per episode);
  - the scripted driver from [`../baseline_dynamic_waypoint/scripted.py`](../baseline_dynamic_waypoint/scripted.py). It reads the true simulator state, so it is a yardstick, not a competitor.
- Automatic checks:
  - every episode has exactly one outcome;
  - its reward terms add up to its return;
  - all policies start from identical episodes.

**Metrics.**
- Success rate with a 95% Wilson interval.
- Crash, timeout and anchor-lost rates.
- The same broken down by slot.
- Time to reach (successes only) and final distance.

**Machine and packages.**
- Windows 11, Intel (32 logical cores), Python 3.13.1.
- highway-env 1.12.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0, numpy 2.5.4.
- These are the versions of the lock file and of the earlier study's runs.

## 8. Inherited settings and candidate IVs

Everything except the task and the reward is inherited unchanged from the earlier study's last runs.

| Setting | Value | Reason given when it was chosen | Still sound for reach? |
|---|---|---|---|
| Speed grid | 10–25 m/s in 2.5 m/s steps | A step also limits acceleration (about step / 0.6 s). With 1 m/s steps braking was too weak; with 5 m/s steps speeds could not be matched. Measured with the scripted driver. | Braking authority still matters; fine speed matching matters less for reach. |
| FRONT offset | 35 m | Once ego is ahead, the anchor's IDM follows it and pushes a closer slot away. That reason was about **holding**. | To re-check: a closer FRONT slot might be reachable once. |
| BACK offset | 15 m | Keeps the box clear of the anchor's 5 m body | Yes |
| Box | ±3 m along, ±1.5 m across | ±1.5 m excludes the neighbouring lane; ego moves at most about 1.5 m relative per decision, so it cannot skip the box | Yes |
| Episode length | 25 s | The same for every hold time | Yes; the scripted driver times out in 21% of episodes, mostly FRONT and RIGHT |
| Traffic | 12–17 m/s, plus a 12 m/s lead car 35 m ahead | Marcos's slow traffic: ego meets cars early | Yes. The lead car is the anchor in 25% of episodes (section 3). |
| Observation | 53 inputs | Earlier study sections 1–3 | Yes, with the redundancies noted in section 4 |
| PPO settings, 3 envs, 1M | See section 7 | Marcos's PPO settings; the 3 envs and 1M of the last runs | Not under test; it changes only if diagnostics point to optimization |

**Candidate IVs, deliberately not changed yet:**
- dropping redundant or inert inputs;
- drawing the slot before the anchor, which would balance the slot counts;
- the FRONT offset;
- removing the injected lead car;
- 4 environments instead of 3.

## 9. Experiment log

### Step 0: the task before any training

No training. Uniform random actions and the scripted driver were run on the 200 evaluation episodes. Files: [`results/step0_B0/`](results/step0_B0/).

| Policy | Success [95% CI] | Crash | Timeout | FRONT | BACK | LEFT | RIGHT | Time to reach |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform random | 11.5% [7.8, 16.7] | 79.0% | 9.5% | 2/62 | 12/61 | 5/40 | 4/37 | 6.2 s |
| Scripted | 76.5% [70.2, 81.8] | 2.5% | 21.0% | 40/62 | 58/61 | 32/40 | 23/37 | 7.5 s |

What this tells us:
- **R0's reward is seen by chance.** A freshly initialized PPO policy acts almost uniformly at random, so about one episode in nine pays the +50 from the start. The rate varies by slot: BACK 20%, LEFT 13%, RIGHT 11% and FRONT 3%.
- **The task is feasible with these actions**, as the scripted driver's 76.5% shows. FRONT is the hardest slot even for the scripted driver (40/62), and its failures are mostly timeouts.
- **Random actions mostly crash** (79%). Ego starts faster than all traffic and is heading toward a 12 m/s car 35 m ahead.
- **The lead-car episodes are easier**: random 15/51 versus 8/149 otherwise; scripted 51/51 versus 102/149.

### Step 1: B0, the root (reward R0)

- **Parent:** none. **IV:** none.
- **Configuration:** sections 2–7. Compared with the earlier study's last run, B0 differs in `goal_config.hold_seconds` (1.0 → 0.1) and in the reward weights; nothing else differs. A test checks this.
- **Runs:** `B0_seed7`, `B0_seed8`, `B0_seed9`, each 1,000,704 transitions.
- **Motivation:** establish what the simplest possible reward achieves, as the root of every later comparison.

**Prediction (written 2026-11-08, before training):**

1. **R0 learns.**
   - *Expected:* overall success clearly above random (11.5%) and below the scripted driver (76.5%); best guess 35–60%.
   - *Why:* the reward is seen in about 11% of episodes from the start, enough for PPO to find it.
2. **The slots rank in the order of their chance success rates.**
   - *Expected:* BACK highest (70% or more), LEFT and RIGHT in between, FRONT lowest (15% or less).
   - *Why:* random FRONT successes are rare (2/62), and FRONT needs a long overtake before any reward arrives.
3. **Crashes are the main failure overall.**
   - *Expected:* they fall far below random's 79%, but stay the largest failure category.
   - *Why:* under R0 a crash costs nothing beyond the missed goal, which is the same as a timeout. For FRONT, timeouts may dominate instead.
4. **The three seeds agree within about ±10 points** of overall success.

**First attempt, interrupted.**
- The first launch (12:50) ran inside the assistant's session and was killed at about 286k transitions when that session ended. It had saved no models.
- Its logs are kept in [`results/interrupted/`](results/interrupted/).
- The runs were restarted at 13:03 with identical settings, as detached processes.
- **Check of the restart:** the restarted runs reproduced every episode of the interrupted attempt exactly: 5,214, 5,404 and 5,404 episodes for seeds 7, 8 and 9. With the same seed on the same machine, training is fully deterministic.

**Results.** Files:
- per-seed folders: [`results/B0_seed7/`](results/B0_seed7/), [`results/B0_seed8/`](results/B0_seed8/), [`results/B0_seed9/`](results/B0_seed9/);
- summary: [`results/B0_summary.md`](results/B0_summary.md);
- crash analysis: `results/B0_seed*/crash_diagnosis.md`;
- rollouts: [`results/B0_seed7/rollouts/`](results/B0_seed7/rollouts/).

| Policy | Success [95% CI] | Crash | Timeout | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO B0, seed 7 | 57.0% [50, 64] | 42.5% | 0.5% | 0/62 | 58/61 | 32/40 | 24/37 |
| PPO B0, seed 8 | 53.0% [46, 60] | 47.0% | 0.0% | 1/62 | 57/61 | 27/40 | 21/37 |
| PPO B0, seed 9 | 55.5% [49, 62] | 43.5% | 1.0% | 0/62 | 58/61 | 29/40 | 24/37 |
| **PPO B0, mean ± sd** | **55.2% ± 2.0** | | | | | | |
| Uniform random | 11.5% [8, 17] | 79.0% | 9.5% | 2/62 | 12/61 | 5/40 | 4/37 |
| Scripted | 76.5% [70, 82] | 2.5% | 21.0% | 40/62 | 58/61 | 32/40 | 23/37 |

The random and scripted rows reproduce Step 0 exactly.

![B0 learning curves](results/B0_learning_curves.png)

**The prediction, checked:**
1. **Correct:** 55.2% overall, inside the predicted 35–60%.
2. **Correct:** the slot order is BACK 93–95%, then LEFT 68–80% and RIGHT 57–65%, with FRONT at 0–1 of 62.
3. **Partly correct:** crashes are the failure (42.5–47%) and timeouts are near 0. **Wrong for FRONT**, where I expected timeouts: 60–61 of 62 FRONT episodes end in a crash.
4. **Correct:** the seeds agree within 4 points.

**What we learned.**
- **Without FRONT, R0 already matches the scripted driver.** BACK + LEFT + RIGHT successes are 114, 105 and 111 of 138 per seed, against 113 for the scripted driver, which reads the true state.
- **Learning plateaus by about 200k transitions.**
  - Training success is 54–55% between 200k and 400k, and 56–57% between 800k and 1M.
  - The training FRONT success briefly reaches about 10% near 100k, then falls to 0 as the policy commits to speed.
  - A larger budget is not what R0 is missing.
- **The behavior is one simple rule** (rollout traces below): move to the slot's lane at once, accelerate to the top speed (25 m/s), and never brake.
- **Every crash is a rear-end into a slower car directly ahead.** The crash replay matched the evaluation outcome in all 600 episodes.
  - **FRONT:** 58, 56 and 56 of 62 episodes end with ego hitting **the anchor itself**, at a median closing speed of 8–12 m/s, 6–8 s in. The slot is 35 m beyond the anchor in its own lane, and the policy drives at it straight through the car.
  - **BACK, LEFT, RIGHT:** 3–16 crashes per slot, almost all at about 2.6 s with about 13 m/s closing speed. This is ego at 25 m/s hitting the 12 m/s lead car that starts 35 m ahead.
- **Why R0 produces this:**
  - R0 pays only for arriving, and the discount pays more for arriving sooner. A crash costs nothing beyond the missed goal, so flat out is R0's best strategy wherever it usually works.
  - In FRONT episodes the return is 0 whatever the policy does. Nothing in R0 shapes FRONT behavior, so FRONT simply inherits the other slots' rule.

![B0 rollouts, seed 7](results/B0_seed7/rollouts/traces.png)

### Step 2: B1, adding the crash penalty

- **Parent:** B0. **IV:** `environment.reward_weights.collision`, 0 → 50. `ivcheck` confirmed this is the only difference from B0's runs.
- **Reward R1:** `r_t = 50 · [success at t] − 50 · [crash at t]`.
- **Motivation:** B0's only failure is crashing: 42–47% of episodes, nearly all rear-ending a slower car directly ahead, and in FRONT the anchor itself. R0 scores a crash exactly like a timeout, so nothing in the reward teaches the agent to avoid one. A crash penalty is also the first FRONT learning signal of any kind, since every FRONT return under R0 is 0.
- **Why 50:** it is Marcos's inherited crash weight, equal in size to the goal. Any other value would be an untested new choice.

**Alternatives considered, and why not now:**

| Alternative | Why not now |
|---|---|
| Progress shaping toward the slot | FRONT gets no reward signal, but progress toward the FRONT slot points *through* the anchor. Without a crash cost it would reinforce the measured failure. Revisit after B1. |
| Time penalty | There are no timeouts to fix (0–1%), and it would make an early crash relatively cheaper. |
| More training | B0 is flat from 200k to 1M. |
| PPO settings | PPO already equals the scripted driver on three slots. |
| Task changes (FRONT offset, the lead car) | They change the problem rather than how it is learned. They stay candidate IVs. |

**Prediction** (formed at 13:48 before B1 was launched, written here before any B1 result):
1. **Crashes:** the crash rate falls far below B0's 42–47%, to under 20%.
2. **Timeouts:** they appear (B0: 0–1%), mainly in FRONT. Avoiding the anchor without a way past it means following it until time runs out.
3. **BACK/LEFT/RIGHT:** success is at least B0's, i.e. 105 or more of 138 per seed. The early lead-car crashes should become successes, since slowing down or changing lanes still leaves time to reach the slot. Some extra caution may add a few timeouts.
4. **FRONT:** success stays low, 10 or fewer of 62 per seed. The penalty tells ego what not to do, not how to get ahead of the anchor and merge back.
5. **Overall success:** between B0's 55% and B0 + 10 points.

**Runs.**
- The first launch at 13:49 failed after about 25 s on all three seeds. The training workers died without an error message, all at the same moment. That attempt is kept in [`results/interrupted/`](results/interrupted/).
- A smoke run of B1 then trained normally, so the configuration is not the cause.
- The runs were relaunched unchanged at 13:51.

**Interim finding during training: B1 collapses (training logs at 80–260k transitions, all three seeds).**
- From about 20–40k transitions onward, every training episode is a timeout: 0 crashes, 0 successes, and every episode runs the full 250 decisions.
- The agent stopped crashing by stopping driving toward goals altogether. Every slot is ahead of ego, so slowing below traffic speed guarantees a return of 0.
- **This contradicts predictions 3 and 5** (prediction 2, timeouts appearing, held, far more than expected).
- **Why it happens:**
  - The starting policy is close to uniform random, which crashes in 79% of episodes and succeeds in 11.5% (Step 0).
  - Under R1, trying is worth about 0.115 · 50 − 0.79 · 50 ≈ −34, while giving up is worth 0.
  - PPO finds the safe policy long before it can learn to reach safely, and once there, exploration rarely produces a success.
- *Context only:* the earlier study's 62–90% timeouts with Marcos's reward (`RESULTS.md` section 2) look like the same mechanism.
- B1 continues to 1M transitions, to measure whether it ever recovers.

**Results (1M transitions).** Files:
- per-seed folders: `results/B1_seed*/`;
- summary: [`results/B1_summary.md`](results/B1_summary.md);
- rollouts: [`results/B1_seed7/rollouts/`](results/B1_seed7/rollouts/).

| Policy | Success [95% CI] | Crash | Timeout | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO B1, seed 7 | 0.5% [0, 3] | 1.0% | 98.5% | 0/62 | 1/61 | 0/40 | 0/37 |
| PPO B1, seed 8 | 3.5% [2, 7] | 1.5% | 95.0% | 0/62 | 7/61 | 0/40 | 0/37 |
| PPO B1, seed 9 | 0.0% [0, 2] | 0.5% | 99.5% | 0/62 | 0/61 | 0/40 | 0/37 |
| **PPO B1, mean ± sd** | **1.3% ± 1.9** | | | | | | |

**What happened.**
- **B1 never recovers.** Training success is 0 for more than 900k of its 1M transitions.
- **The behavior** (rollouts, seed 7, all four slots): ego slows from 20 m/s to the minimum 10 m/s within 3 s and stays there. All traffic drives at 12–17 m/s, so every slot moves away, and episodes end 60–220 m short.
- **It is not a lack of exploration.** The policy's entropy stays at 0.84–1.04 nats (the maximum is 1.61), but random actions from a 10 m/s policy almost never reach a slot.

**The prediction, checked:**
1. **Correct:** crashes fell to 0.5–1.5%.
2. **Correct, far beyond what was expected:** timeouts appeared, in 95–99.5% of episodes.
3. **Wrong.**
4. **Correct only trivially:** FRONT is 0, like everything else.
5. **Wrong:** overall success was 1.3%, not 55–65%.

**Lesson.** A crash penalty as large as the goal, introduced from scratch, makes giving up the best policy within reach of a random start. The penalty itself is not the problem; the moment it meets the policy is.

### Step 3: B2, starting B1's training from B0's policy

- **Parent:** B1. **IV:** `init_policy`, None (random weights) → `B0`.
  - The policy and value networks start from B0's trained weights on the same seed; the optimizer starts fresh.
  - Reward (R1), task, PPO settings and budget are all B1's.
  - The run checked its configuration against B1's *defined* configuration (`--parent-in-progress`), because B1 had not saved its sidecars yet. It will be re-checked with `ivcheck` against the saved sidecars when B1 finishes.
- **Decided at 13:57 on B1's training logs, before B1 finished.** The collapse was complete and identical on all three seeds for 60–200k transitions.
- **Motivation:**
  - The B0 policy already reaches the slot in 55% of episodes, so under R1 trying is worth about 0.55 · 50 − 0.45 · 50 ≈ +5 from the first update, not −34.
  - The crash penalty can then teach the policy to avoid crashes rather than to stop driving.
  - This builds the policy step by step, the same way the reward is built: learn to reach under R0, then learn not to crash under R1.
  - What this IV includes, stated plainly: B2's starting point carries B0's 1M transitions of experience.

**Alternatives considered, and why not now:**

| Alternative | Why not now |
|---|---|
| Potential-based progress shaping (prepared in [`environment.py`](environment.py) and tested) | For any fixed policy it leaves the advantages unchanged: the shaped value is the old value minus Φ(s), which cancels in the advantage. The incentive to give up stays. It helps only through how the critic generalizes, so it is not a fix for the collapse. Kept for FRONT later. |
| A smaller crash penalty | Trying beats giving up from a random start only if the penalty is below about 7. That value would be picked to make this calculation work, and it keeps most of B0's risky driving. |
| A higher entropy coefficient | More exploration does not change the fact that giving up is worth more than trying. |
| Removing the injected lead car | It causes most early crashes, but removing it changes the task, not how the task is learned. |

**Prediction** (written at 13:59, before any B2 result):
1. **No collapse:** training success stays at or above 40% throughout, starting from B0's level of about 55%.
2. **Crashes:** below 20% in evaluation, against B0's 42–47%.
3. **BACK/LEFT/RIGHT:** success at least B0's, i.e. 105 or more of 138 per seed. Their crashes were mostly early rear-ends into the lead car, which braking avoids with time to spare.
4. **FRONT:** crashes into the anchor turn into timeouts (following the anchor), and FRONT success stays at 10 or fewer of 62 per seed. Under R1 every FRONT episode of the B0 policy costs −50, so the quickest fix is to stop pursuing FRONT.
5. **Overall:** 50–65% success, crashes under 20%, and most other episodes FRONT timeouts. Far above B1 if B1 stays collapsed.

**Runs.**
- A first B2 launch at 13:58 was refused by the one-IV guard, because B1 had no saved sidecars yet. It did not train.
- The runs were relaunched at 13:58 with `--parent-in-progress`.
- **Re-check after B1 finished:** `ivcheck B1_seedN B2_seedN` finds exactly one difference on every seed, `init_policy: <missing> → 'B0'`.

**Results (1M transitions).** Files:
- per-seed folders: `results/B2_seed*/`;
- summary: [`results/B2_summary.md`](results/B2_summary.md);
- crash analysis: `results/B2_seed*/crash_diagnosis.md`.

| Policy | Success [95% CI] | Crash | Timeout | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO B2, seed 7 | 53.0% [46, 60] | 16.0% | 31.0% | 0/62 | 51/61 | 31/40 | 24/37 |
| PPO B2, seed 8 | 56.5% [50, 63] | 12.5% | 31.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B2, seed 9 | 48.5% [42, 55] | 20.5% | 31.0% | 0/62 | 51/61 | 26/40 | 20/37 |
| **PPO B2, mean ± sd** | **52.7% ± 4.0** | | | | | | |
| B0, for reference (same seeds) | 57.0 / 53.0 / 55.5% | 42.5 / 47.0 / 43.5% | 0.5 / 0 / 1.0% | 0 / 1 / 0 | 58 / 57 / 58 | 32 / 27 / 29 | 24 / 21 / 24 |

![B2 learning curves](results/B2_learning_curves.png)

**The prediction, checked:**
1. **Correct:** no collapse. Training success stays between 46% and 60% throughout.
2. **Mostly correct:** crashes are 16.0%, 12.5% and 20.5%, against B0's 42.5%, 47.0% and 43.5%.
3. **Wrong.**
   - BACK + LEFT + RIGHT successes are 106, 113 and 97 of 138, against B0's 114, 105 and 111.
   - Crashes in those slots are 32, 25 and 41, against B0's 24, 33 and 27.
   - No improvement.
4. **Correct:** FRONT is 0/62 on every seed. **All 62 FRONT episodes time out**, where B0 crashed into the anchor in 60–61 of them.
5. **Mostly correct:** 53.0%, 56.5% and 48.5% (one seed just under 50%), far above B1's 1.3%.

**What we learned.**
- **The whole drop in crashes, 44% → 16%, comes from abandoning FRONT.** The curves show it: within the first 60k transitions crashes fall by the size of the FRONT share and timeouts rise by the same amount. FRONT success is 0 throughout.
- **BACK/LEFT/RIGHT crashes are unchanged in kind and in number.** They are still the same early rear-end into the 12 m/s lead car, at about 2.6 s and about 13 m/s closing speed.
- **Why they persist** (a hypothesis that fits the evidence, not yet tested):
  - The policy still explores: its entropy stays at 1.1–1.15 nats throughout.
  - Even so, the BACK/LEFT/RIGHT training crash rate is flat: 23% → 20%, 23% → 17% and 20% → 20% between the first and last 100k transitions.
  - Under R1 a crash costs exactly what a success pays. When the alternative to flooring it is a timeout (0), flooring it is worth taking whenever it succeeds more often than it crashes. **R1 itself tolerates up to about 50% crash risk** in those episodes.
  - So the remaining crashes look like the risk R1 accepts, not a failure to explore.
- **Paired by seed, B2 minus B0:** success −4.0, +3.5 and −7.0 points; crashes −26.5, −34.5 and −23.0 points.

### Step 4: B3, adding potential-based progress (option A below, approved 2026-11-08)

The two candidates are kept below as they were proposed. Option A was approved, and B stays a candidate.

- **Parent:** B2. **IV:** `environment.reward_weights.potential_progress`, absent → `{"weight": 0.1, "gamma": 0.99}`.
- Everything else is B2's, including `init_policy = B0`.
- **Reward R3:**

  ```text
  r_t = 50 · success − 50 · crash + 0.1 · (0.99 · Φ(s_{t+1}) − Φ(s_t))
  Φ(s) = −(distance to the slot center, m);  Φ = 0 once the episode has ended (success or crash)
  ```

- The term is implemented in [`environment.py`](environment.py) and tested:
  - with the term off, the environment is identical to the inherited one;
  - with it on, the discounted sum of the term over a finished episode is exactly 0.1 × the initial distance, whatever the car does.
- **Prediction:** as written for option A below.

**Runs.**
- Launched at 15:12.
- **At 15:43 all three runs received a console interrupt (`^C` in their logs), seconds after Windows logged a session transition (Kernel-Power event 566 at 15:42:50).**
  - Seeds 7 and 9 lost their environment workers at about 727k transitions. Their status is `failed`; they are kept in [`results/interrupted/`](results/interrupted/).
  - Seed 8 had finished training (`trained`, 1,000,704 transitions) but was interrupted during evaluation.
- **The same cause explains B1's failed first launch.** Event 566 was logged at 13:49:21, and those workers died about 10 s later. Other session transitions that day (13:20, 13:30, 14:10, 15:00, 15:31) did no harm.
- **Fix** ([`train.py`](train.py), `ignore_console_interrupts`): the learner and every environment worker now ignore console Ctrl+C and Ctrl+Break. Detached runs are stopped with `taskkill` instead. This does not change anything that is computed.
- **Recovery:**
  - Seed 8's saved model was evaluated with the unchanged measurement into its own folder.
  - Seeds 7 and 9 were relaunched at 15:46. Training is deterministic, so their first 727k transitions should repeat the interrupted attempt exactly; this is checked below.
- **The relaunch failed too, and the cause is not the code.**
  - The relaunched seeds 7 and 9, and seed 8's evaluation, all died about 9 s after starting (15:46:27), with no `^C`.
  - A 60-second test script, also launched detached, died after 4 s (15:49:47).
  - Windows logged no crash, no resource exhaustion and no session event at those moments, so the processes were terminated from outside.
  - Kill times since 15:42 were 15:42:50, 15:46:27 and 15:49:47. Before 15:42, runs survived for 40+ minutes.
  - Another session was training a GPU notebook on this machine at the time; commit memory stood at 56 of 65.8 GB. No transcript shows any session issuing a kill.
  - The cause is still unknown.

- **What works:** a single evaluation launched as a background job *inside* the assistant's session (15:52–15:56) ran to completion.
  - Seed 8's evaluation is that run.
  - Seeds 7 and 9 were relaunched the same way at about 15:57.
  - The second attempt's models and logs are in `results/interrupted/*_attempt2/`. Its result folders, at most about 9 s of logging, were deleted by mistake before being moved.

**Interim evidence from training logs** (sampling policy, not the evaluation): seed 8 to 1M; seeds 7 and 9 to about 727k from the interrupted attempt.
- FRONT training success is **0.00 on all three seeds from 50k transitions onward**: FRONT episodes end in timeouts 81–97% of the time, and the rest are crashes.
- BACK/LEFT/RIGHT training success is 73–82%, with crashes at 18–25%, the same range as B2.
- So by the training logs, progress shaping did **not** get PPO to the front spot within 1M transitions.

**Reproducibility check:** the relaunched seeds 7 and 9 repeated every episode of the interrupted first attempt exactly (7,024 and 6,972 episodes). Ignoring console signals did not change anything that is computed.

**Results (1M transitions, all three seeds).** Files:
- per-seed folders: `results/B3_seed*/`;
- summary: [`results/B3_summary.md`](results/B3_summary.md);
- crash analysis: `results/B3_seed*/crash_diagnosis.md`.

| Policy | Success [95% CI] | Crash | Timeout | FRONT | BACK | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO B3, seed 7 | 56.5% [50, 63] | 13.5% | 30.0% | 0/62 | 58/61 | 31/40 | 24/37 |
| PPO B3, seed 8 | 56.0% [49, 63] | 14.0% | 30.0% | 0/62 | 58/61 | 32/40 | 22/37 |
| PPO B3, seed 9 | 56.0% [49, 63] | 15.5% | 28.5% | 0/62 | 58/61 | 31/40 | 23/37 |
| **PPO B3, mean ± sd** | **56.2% ± 0.3** | | | | | | |
| B2, for reference (same seeds) | 53.0 / 56.5 / 48.5% | 16.0 / 12.5 / 20.5% | 31.0% each | 0 / 0 / 0 | 51 / 58 / 51 | 31 / 31 / 26 | 24 / 24 / 20 |

![B3 learning curves](results/B3_learning_curves.png)

**The prediction, checked:**
1. **FRONT above 0 (at least 5/62 on at least two seeds): wrong.** FRONT is 0/62 on every seed, and training FRONT success is 0 from 50k to 1M.
2. **BACK/LEFT/RIGHT unchanged within about 5: wrong, in the better direction.**
   - B3 reaches 113, 112 and 112 of 138, against B2's 106, 113 and 97.
   - That is B0's level (114, 105, 111) and the scripted driver's (113).
3. **"Possibly more FRONT crashes": correct, but small.** There are 2, 2 and 5 FRONT crashes (B2: 0), slow bumps into traffic about 9–10 s in. The other 57–60 FRONT episodes time out.

**What we learned.**
- **Progress shaping did not open FRONT.** Under every reward so far the car either rams the anchor (R0) or gives up (R1, R1 + shaping). It never once completes the overtake and merge in evaluation. That points at the maneuver itself, which is long and has to happen in traffic, rather than at the reward.
- **Shaping may have made BACK/LEFT/RIGHT learning more consistent.**
  - BACK is 58/61 on every seed (B2: 51, 58, 51), and the spread of overall success across seeds fell from ±4.0 to ±0.3 points.
  - Paired by seed, B3 minus B2: success +3.5, −0.5 and +7.5 points.
  - With three seeds this is a hint, not an established effect.
- **The early lead-car rear-ends remain.** BACK/LEFT/RIGHT crashes are 25, 26 and 26 per seed, the same kind as before. That is consistent with R1's risk tolerance (Step 3), which option B would test.

### Proposed Step 5 (waiting for review)

Three reward variants have not produced a single FRONT success, so the next step should **find out what blocks FRONT** before adding more reward terms. One IV from B3 in each case:

**A (recommended): FRONT slot 35 m → 15 m ahead of the anchor** (`goal_config.slot_offsets.front`).
- **What it tests:** whether the length of the maneuver is the barrier. FRONT would then mirror BACK at 15 m; the pass and the merge are still required, but ego only has to get 15 m ahead rather than 35 m.
- **Why the offset can change:** the 35 m value was chosen for *holding*, because the anchor's driver pushes a close slot away (section 8, already flagged to re-check). It is the same "easier version first" idea as reach before hold.
- **This does change the task for FRONT**, which earlier steps avoided. The reason for it now is diagnosis.
- **Reading the result:**
  - if FRONT becomes learnable, grow the offset back toward 35 m;
  - if it does not, the pass-and-merge itself is the barrier, and the next candidate is starting from a policy that imitates the scripted driver (40/62 on FRONT).
- **Prediction:** FRONT success above 0 (at least 5/62 on at least two seeds); BACK/LEFT/RIGHT unchanged within 5.

**B: crash weight 50 → 150.** Unchanged from Step 4. It targets the BACK/LEFT/RIGHT crashes and tests R1's risk tolerance.

**C: start from a policy that imitates the scripted driver.** It is the most direct route to FRONT, but it adds a new component (collecting demonstrations and supervised pretraining), so it should come after A has shown where the barrier is.

Two candidates. Each is one IV from B2, and both keep `init_policy = B0`, like their parent.

**A (recommended): add potential-based progress toward the slot.**
- Setting: `reward_weights.potential_progress = {"weight": 0.1, "gamma": 0.99}`. The weight is Marcos's progress scale; γ is PPO's.
- **Target: FRONT**, the largest gap. It is 31% of episodes, with 0 successes against the scripted driver's 40/62.
- The term does not change the best policy (section 6), so any change is in what PPO manages to find.
- **To be honest about its odds:** for an exact critic it leaves the advantages unchanged. It acts through the critic and GAE (λ = 0.95), giving immediate per-step credit before the critic has learned.
  - If it does nothing for FRONT, that is informative: FRONT would then need a curriculum or demonstrations rather than reward shaping.
- **Prediction:** FRONT success above 0 (at least 5/62 on at least two seeds); BACK/LEFT/RIGHT unchanged within about 5; possibly more FRONT crashes.

**B: crash weight 50 → 150.**
- A crash would then cost 3× the goal, so a gamble is worth taking only at better than 3:1 odds.
- **Target: the BACK/LEFT/RIGHT crashes.** It directly tests the risk-tolerance explanation above.
- **Prediction:** BACK/LEFT/RIGHT crashes under 10%, with somewhat fewer successes there and more timeouts.

**Why A first:**
- FRONT is the larger loss: about 31 points of episodes, against about 14 for the BACK/LEFT/RIGHT crashes.
- B mostly turns crashes into timeouts, while A is the only candidate aimed at new successes.

## 10. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-08 | Restart the dynamic-waypoint study as a chain with one IV per comparison | Meeting notes; the earlier study changed several settings per step, so effects could not be separated |
| 2026-11-08 | Study reach before hold (`hold_seconds` 0.1) | Arriving is the hard part (scripted: 76% for 1 s and 5 s holds alike) |
| 2026-11-08 | Root reward R0 = 50 × success | Simplest reward equal to the scored outcome; 50 matches Marcos's scale for later one-term additions |
| 2026-11-08 | Marcos's reward is not one arm | It differs from R0 in three terms; each can be added alone later. It is also not highway-env's reward. |
| 2026-11-08 | Keep every other setting inherited, unchanged | Changing them would add IVs to the root. They are listed as candidate IVs (section 8). |
| 2026-11-08 | 3 paired seeds; 200 fixed evaluation episodes with natural goal sampling; Wilson intervals | The earlier study used 1–2 seeds and 50 episodes; per-slot counts of 8 were too small to compare |
| 2026-11-08 | `ivcheck.py` checks every run against its parent and its sibling seeds | Makes the one-IV rule enforced by code rather than by memory |
| 2026-11-08 | Rollout episodes: for each slot, the first evaluation seed with that slot (10,001 FRONT, 10,000 BACK, 10,005 LEFT, 10,009 RIGHT) | Fixed before training, so examples are not picked for their outcome |
| 2026-11-08 | Launch training as detached processes | The first B0 launch died when the assistant's session ended |
| 2026-11-08 | Add `diagnose.py`, a crash replay | Choosing the next IV needs to know *what* fails, not only how often |
| 2026-11-08 | Step 2 = B1: crash weight 0 → 50 | B0 fails only by rear-end crashes, and R0 scores a crash like a timeout. Alternatives are listed under Step 2. |
| 2026-11-08 | Add an optional potential-based progress term (`environment.py`); keep it unused for now | It is ready for FRONT later. Tests show that with the term off, the environment is identical to the inherited one, and that it telescopes exactly when on. `train.py` requires its γ to equal PPO's. |
| 2026-11-08 | Step 3 = B2: `init_policy` None → B0, decided on B1's training logs | B1 collapsed to giving up on all seeds. Starting from a policy that already reaches makes trying worth more than giving up. Alternatives are listed under Step 3. |
| 2026-11-08 | `ivcheck` treats a setting that older sidecars lack as its default (None) | `init_policy` was added after B0 and B1 were saved. Any real value is still a difference. |
| 2026-11-08 | B1 recorded as a collapse (1.3% success at 1M) | It never recovers. Kept as the evidence for why the crash penalty needs a policy that already reaches. |
| 2026-11-08 | B2 recorded: crashes fall only by abandoning FRONT | FRONT 62/62 timeouts; BACK/LEFT/RIGHT crashes unchanged, consistent with R1's ~50% risk tolerance |
| 2026-11-08 | Step 4 not launched; A (progress shaping, for FRONT) recommended over B (crash 150) | B2 finished after the 2:30 PM autonomy window. FRONT is the larger gap. |
| 2026-11-08 | Step 4 = B3 (option A), approved by the user; launched 15:12 | `ivcheck` against B2's saved sidecars: the only difference is the new term |
| 2026-11-08 | Training ignores console Ctrl+C/Ctrl+Break | Two batches died right after Windows session transitions (13:49 and 15:42) delivered a console interrupt. This changes nothing computed. |
| 2026-11-08 | After 15:46, launch runs as background jobs inside the assistant's session | Detached processes kept being terminated from outside (cause unknown); in-session jobs completed |
| 2026-11-08 | B3 recorded: FRONT still 0; BACK/LEFT/RIGHT back to the scripted level | Progress shaping does not open FRONT within 1M transitions |
| 2026-11-08 | Step 5 proposed: A (FRONT offset 35 → 15 m) recommended, as a diagnosis | Three reward variants left FRONT at 0; first find whether the maneuver's length is the barrier |
