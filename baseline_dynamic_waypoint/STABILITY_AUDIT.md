# Conservative PPO stability audit

Inspected project configuration, trainer overrides, installed PPO implementation,
saved run metadata, episode CSVs, and optimizer metrics. No new long training run
was started by this change. The referenced 476k snapshot cannot be uniquely
identified from the pasted description; the latest two completed 57-input runs
provide the evidence below.

## Actual configuration before changes

| Parameter | Latest completed run | Project default / preceding 57-input run |
| --- | --- | --- |
| Learning rate | constant 0.00005 | constant 0.0001 |
| Clip range | constant 0.2 | 0.2 |
| Epochs | 5 | 5 |
| n_steps | 250 per worker, 4 workers: 1000 decisions/rollout | config 256; single-car trainer overrides to 250 |
| Minibatch | 250 | config 256; single-car trainer overrides to 250 |
| Gamma | 0.97 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Entropy coefficient | 0.01 | 0.01 |
| target_kl | None (installed PPO default; not configured) | None |
| Actor / critic | two 256-unit Tanh layers each; dropout 0.05 | preceding dropout 0.1 |
| Value coefficient / max gradient norm | 0.5 / 0.5 | 0.5 / 0.5 |

The current environment uses 57 inputs with explicit target-car and goal lane
IDs, all four goals, randomized starting lanes, and constant-speed traffic drawn
from 12–18 m/s. Committed lane changes contain multiple physics ticks per policy
decision. The rollout budget counts policy decisions.

Rewards: +50 success, -50 collision, +0.1 times distance progress, -0.05 per
physics tick, +0.2 per valid hold tick, and -10 once on deadline timeout.
Lane, speed-match, and overtake reward weights are zero. Success requires
longitudinal error below 5 m, lateral error below 1.8 m, speed mismatch at most
2 m/s, 0.5 seconds valid hold, and a settled lane change. Brief hold gaps up to
0.3 seconds pause progress. Success/collision terminate; deadline at 30 seconds
or target-car loss truncates. Collision, success, and target loss do not incur
the timeout cost.

Existing logging: per-episode CSVs, per-goal rolling success/collision/timeout/
reward, reward components, physics steps, lane changes, and native PPO KL,
clipping fraction, explained variance, losses, entropy, and learning rate.
Checkpoints every 25k decisions. Evaluation before/after training; optional
periodic evaluation was explicitly enabled every 250k in both completed runs.
Each evaluation uses deterministic actions on seeds 10000–10099 in a separate
environment, without changing training traffic sampling. CLI default
`--eval-every 0` is preserved; use the existing explicit flag for periodic runs.

## Evidence and decisions

| Metric | Previous run: LR 1e-4, gamma .99, dropout .1 | Latest: LR 5e-5, gamma .97, dropout .05 |
| --- | ---: | ---: |
| Median approximate KL | 0.002882 | 0.000892 |
| Maximum approximate KL | 0.021725 | 0.018157 |
| Median clip fraction | 0.009 | 0 |
| Maximum clip fraction | 0.1928 | 0.1162 |
| Median explained variance | 0.922 | 0.496 |
| Final held-out success | 30% | 20% |
| Final held-out collision | 21% | 5% |
| Final held-out timeout | 49% | 75% |

These runs changed several parameters together, so they do not isolate causes.
Low median KL and clipping do not support tighter clipping, fewer epochs, or a
new KL threshold. Five epochs is not excessive on this evidence. Further LR
reduction/decay could slow or stall useful learning; preserve the existing
constant LR and all hyperparameters for now (0% change). Keep rollout/batch
sizes, gamma, GAE, entropy, architecture, observation, traffic and rewards.
This does not assert that current hyperparameters are optimal.

In the latest run's 400k–500k training window: success 9.6%, collisions 18.6%,
timeouts 71.8%, mean length 244.8 decisions, mean reward -24.4. Mean time cost
was -12.27, timeout cost -7.18, collision cost -9.31, success bonus +4.78.
Thus duration and outcome mixtures materially affect reward; smoothing reward
alone cannot establish better driving. Existing reward-component tags contain
the latest finished episode, which can also look noisy.

## Implemented changes

| Original | Updated | Relative change | Benefit | Downside |
| --- | --- | --- | --- | --- |
| Aggregate success/collision but no aggregate timeout TensorBoard rate | Add `episodes/timeout_mean_100` | N/A | Distinguish waiting from task completion | Rolling-window lag |
| No steps-to-success metric | Add training `episodes/steps_to_success_mean_100`, success count; evaluation mean steps, seconds and successful count, aggregate and per goal | N/A | Measure completion speed separately from failure duration | Conditional on success; interpret with success count |
| Latest-episode reward components only | Preserve them and add `reward_components_mean_100/*` | N/A | Separate outcome mixtures from individual terminal spikes | Rolling averages hide brief changes |
| Aggregate evaluation reward only in JSON | Also log aggregate reward in TensorBoard | N/A | Compare reward with success and collision rates | More dashboard tags |
| Periodic/final checkpoints; best chosen by offline report only | Save `_best.zip`, matching environment sidecar and `best_evaluation.json` after initial, periodic and final evaluation | N/A | Preserve useful policies when later performance regresses | Extra disk writes; selection can overfit monitoring seeds |

Best selection ranks higher success, fewer collisions, then higher reward;
exact ties preserve the earlier policy. The initial policy can remain best if
all subsequent evaluations are worse. A resumed policy's initial evaluation is
recorded at zero additional steps, preserving source lifetime accounting.
Existing final/checkpoint files and CSV columns are preserved. No additional
evaluation episodes or passes were added. Missing successful episodes produce
JSON null and no mean-success-time scalar, with successful count zero; they are
never represented as zero completion time.

Validation: nine focused tests cover logging, real PPO updates, periodic
evaluation, timeout accounting, best checkpoint reload, selection priorities,
missing-success metrics, and resumed-baseline step accounting. Use a future
controlled fixed-seed experiment to assess policy stability; these changes
improve measurement and checkpoint retention, not demonstrated policy quality.
