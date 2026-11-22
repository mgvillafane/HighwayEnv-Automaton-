# Why the policy rarely changes right

## Conclusion

The demonstrated failure is a learned, goal-dependent left/right bias carried
through the initialization chain. No wrong y sign, clipping asymmetry, RIGHT
action mapping, slower right steering, or shortage of RIGHT goals was found.
The model knows a successful side-slot strategy for LEFT, but does not apply its
reflection to RIGHT. This is broader than issuing the RIGHT action: even RIGHT
episodes that start in the goal lane fail to approach and hold the slot.

No reward, environment, checkpoint, or training settings were changed. Only
diagnostic scripts and results were added.

## Strongest test: reflect the policy, retain the actual scenario

The trained model was evaluated on seeds 10000–10099 twice. In the diagnostic
controller, inputs were reflected across the middle lane (`y -> 8-y` meters,
lateral velocity signs flipped, LEFT/RIGHT goal labels swapped), then LEFT/RIGHT
outputs were swapped back before execution. The physical scenario and reward
remained exactly the same. No model weights were changed.

| Objective | Original policy successes | Reflected policy successes |
| --- | ---: | ---: |
| RIGHT | 0/22 | **22/22** |
| LEFT | **33/33** | 0/33 |
| BACK | 22/27 | 24/27 |
| FRONT | 0/18 | 0/18 |

This reverses the side-task failure: the environment supports the RIGHT task
under the existing reward, but the policy has learned asymmetric behavior.
Applying reflection to every task is a diagnostic, not a proposed universal
replacement: it breaks the previously successful LEFT behavior.

Forcing only the initial RIGHT command when a RIGHT goal requires a higher lane
index still yields 0/22 RIGHT successes (15 episodes receive the intervention).
Changing just one lane command is insufficient; the approach/hold strategy also
needs to transfer.

## When the bias appeared

| Actual training ancestor | RIGHT completed episodes | RIGHT successes |
| --- | ---: | ---: |
| Original gamma .99, 1 million steps | 934 | 0 |
| Flexible 30-second holding, 200k | 260 | 0 |
| Committed lane actions, 200k | 317 | 0 |

The failure predates both committed lane changes and the new 57-input
observation. Both later runs preserve the actor from the preceding model.
Across this initialization chain, there are zero RIGHT-goal successes in 1,511
completed training episodes. Current RIGHT episodes include 148 initially
requiring right, 57 requiring left, and 112 already in the goal lane; none logs
even one valid hold tick.

On the same 450 reference states from failed RIGHT evaluations, all requiring a
higher goal lane index, the mean RIGHT probability evolves as follows:

| Checkpoint | Mean RIGHT probability |
| --- | ---: |
| Original 1m model | 1.0224% |
| Flexible-holding predecessor | 0.8355% |
| Current model | 0.0347% |

These states were sampled at approximately one-second intervals from the current
policy's trajectories; they are a fixed-input comparison, not independent
checkpoint evaluation success rates. The current model selects RIGHT as the
highest-probability action in none of these 450 states. The full earlier audit
of all 4,500 corresponding decision states gives 0.0337% mean probability.

## What was ruled out

- Seven mirrored left/right maneuver pairs, spanning 10–25 m/s, complete in the
  same number of physics ticks with exactly equal reward.
- In 379 physically mirrored states, observation reflection agrees within
  float32 precision (maximum error 5.96e-8); rewards and termination match.
- Fast and native encoders are bitwise identical across 529 tested states.
- In 2,000 fresh scenario resets, RIGHT appears 498 times and LEFT 510 times;
  ego lanes are sampled 670/666/664 times. Actual training also includes 317
  RIGHT episodes, or 25.65% of completed scenarios.
- Signed goal_delta and slot_error match the actual geometry. Positive lateral
  error means right, and action 2 increases the target lane index. Lateral
  slot_error clips two-lane errors, but goal_delta preserves the distance and
  this clipping is symmetric.

## Interpretation and next step without reward changes

The strongest supported explanation is that PPO inherited a failed RIGHT-goal
strategy and fine-tuning further suppressed the rightward choices needed to
escape it. LEFT success does not automatically teach its RIGHT reflection to
this ordinary neural-network policy. Balanced scenario counts alone do not
provide successful RIGHT trajectories. Weak lateral progress feedback at long
range may make discovery harder, but is symmetric and does not explain the
directional preference by itself.

The exact initial cause of the original asymmetry (random learning history,
representation, or optimization settings) is not established by these tests;
that would require controlled retraining. There is no evidence supporting a
claim that changing y values or reward is necessary.

The most directly supported next experiment is to share the successful LEFT
strategy with RIGHT through consistent observation/action reflection, then
evaluate/train under the unchanged reward. A narrowly scoped RIGHT-to-LEFT
canonicalization is preferable to blindly reflecting every task. No such
behavior change has been applied here.

## Reproduction and evidence

Run `python -B -m baseline_dynamic_waypoint.diagnose_right_bias` for the paired
policy tests; `--summarize-only` verifies the cached baseline and rewrites this
report. Supporting results: [paired rollouts and checkpoint probes](right_bias_diagnosis.json),
[training ancestry](training_ancestry_diagnosis.json),
[controller/encoder/sampler checks](mechanics_diagnosis.json), and
[physical observation reflection checks](mechanics_reflection.json).
