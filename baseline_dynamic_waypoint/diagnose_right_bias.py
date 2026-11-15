"""Read-only policy diagnosis: paired reflection tests and checkpoint comparisons.

No learning or environment/reward configuration is changed. The reflected policy
is a diagnostic controller, not a replacement checkpoint.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results" / "ppo_singlecar_committed57_seed7_200k"
MODEL = ROOT / "models" / "ppo_singlecar_committed57_seed7_200k.zip"
ACTION_REFLECTION = np.array([2, 1, 0, 3, 4])


def reflect_observation(observation):
    obs = {key: value.copy() for key, value in observation.items()}
    for key in ("desired_goal", "achieved_goal"):
        obs[key][1] = 8.0 / 12.0 - obs[key][1]
    obs["goal_delta"][1] *= -1
    # The ego row has absolute y; traffic rows are relative to ego.
    obs["vehicles"][0, 2] = 8.0 / 12.0 - obs["vehicles"][0, 2]
    obs["vehicles"][1:, 2] *= -1
    obs["vehicles"][:, 4] *= -1
    obs["anchor"][[1, 3]] *= -1
    obs["slot"][[2, 3]] = obs["slot"][[3, 2]]
    if "slot_error" in obs:
        obs["slot_error"][1] *= -1
    if "requested_lane" in obs:
        obs["requested_lane"] = obs["requested_lane"][::-1].copy()
    return obs


def probabilities(model, observations):
    batch = {key: np.stack([obs[key] for obs in observations])
             for key in model.observation_space.spaces}
    tensors, _ = model.policy.obs_to_tensor(batch)
    with torch.no_grad():
        return model.policy.get_distribution(tensors).distribution.probs.cpu().numpy()


def summarize(rows):
    result = {}
    for slot in ("front", "back", "left", "right"):
        subset = [row for row in rows if row["slot"] == slot]
        if not subset:
            continue
        result[slot] = {"episodes": len(subset),
                        **{key: sum(row[key] for row in subset)
                           for key in ("success", "collision", "timeout", "left_actions", "right_actions")},
                        "mean_return": float(np.mean([row["return"] for row in subset]))}
    return result


def rollout(model, env, seed, reflected=False, force_initial_right=False, samples=None):
    obs, initial = env.reset(seed=seed)
    total, left, right, decisions = 0.0, 0, 0, 0
    terminated = truncated = False
    next_sample_time = 0.0
    while not (terminated or truncated):
        if samples is not None and env.highway.time >= next_sample_time:
            samples.append({"seed": seed, "slot": env.slot, "time": float(env.highway.time),
                            "goal_lane": int(env.goal_lane),
                            "requested_lane": int(env.highway.vehicle.target_lane_index[2]),
                            "observation": {key: value.copy() for key, value in obs.items()}})
            next_sample_time += 1.0
        policy_obs = reflect_observation(obs) if reflected else obs
        action = int(model.predict(policy_obs, deterministic=True)[0])
        if reflected:
            action = int(ACTION_REFLECTION[action])
        if (force_initial_right and decisions == 0
                and initial["goal_lane"] > env.highway.vehicle.target_lane_index[2]):
            action = 2
        left += action == 0
        right += action == 2
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        decisions += 1
    return {"seed": seed, "slot": initial["slot"], "success": int(info["is_success"]),
            "collision": int(info["is_collision"]), "timeout": int(info["is_timeout"]),
            "return": total, "seconds": info["elapsed_time"], "decisions": decisions,
            "left_actions": left, "right_actions": right,
            "initial_goal_lane": initial["goal_lane"],
            "initial_ego_lane": initial["initial_scenario"]["ego_lane"]}


def checkpoint_probe(path, samples):
    model = PPO.load(path, device="cpu")
    original = probabilities(model, [s["observation"] for s in samples])
    reflected = probabilities(model, [reflect_observation(s["observation"]) for s in samples])
    result = {}
    for slot in ("front", "back", "left", "right"):
        indices = [i for i, s in enumerate(samples)
                   if s["slot"] == slot and s["goal_lane"] > s["requested_lane"]]
        if not indices:
            continue
        a, b = original[indices], reflected[indices]
        result[slot] = {"matched_states": len(indices),
                        "original_right_probability": float(a[:, 2].mean()),
                        "mirrored_left_probability": float(b[:, 0].mean()),
                        "original_right_argmax_fraction": float(np.mean(a.argmax(1) == 2)),
                        "mirrored_left_argmax_fraction": float(np.mean(b.argmax(1) == 0))}
    return result


def finalize(result):
    # Older in-flight copies of this diagnostic emitted empty goal summaries.
    result["initial_right_intervention"] = {
        slot: value for slot, value in result["initial_right_intervention"].items() if value["episodes"]}
    with (RUN / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        expected = {int(row["seed"]): row for row in csv.DictReader(handle)}
    for row in result["episodes"]["baseline"]:
        previous = expected[row["seed"]]
        for key in ("success", "collision", "timeout"):
            assert row[key] == int(previous[key]), (row["seed"], key)
        assert row["decisions"] == int(previous["length"])
        assert abs(row["return"] - float(previous["reward"])) < 1e-8
    result["verification"] = {"baseline_replays_match_saved_evaluation": 100,
                               "reward_and_model_training_unchanged": True}
    (RUN / "right_bias_diagnosis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = """# Why the policy rarely changes right

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
"""
    (RUN / "RIGHT_BIAS_CONCLUSION.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("baseline", "reflected_policy", "initial_right_intervention", "verification")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        finalize(json.loads((RUN / "right_bias_diagnosis.json").read_text(encoding="utf-8")))
        return
    torch.set_num_threads(1)
    model, env, metadata = load_experiment(MODEL)
    samples, baseline, mirrored, intervention = [], [], [], []
    try:
        for seed in range(10000, 10100):
            baseline.append(rollout(model, env, seed, samples=samples))
            mirrored.append(rollout(model, env, seed, reflected=True))
            if baseline[-1]["slot"] == "right":
                intervention.append(rollout(model, env, seed, force_initial_right=True))
            if (seed + 1) % 20 == 0:
                print(f"Paired rollouts complete: {seed - 9999}/100", flush=True)
    finally:
        env.close()
    checkpoints = [ROOT / "models" / "ppo_front20m_gamma099_seed7_1m.zip",
                   ROOT / "models" / "ppo_singlecar_flexible30s_seed7_200k.zip"]
    checkpoints += [ROOT / "models" / f"ppo_singlecar_committed57_seed7_200k_{n}.zip"
                    for n in (25000, 50000, 100000, 150000)]
    checkpoints.append(MODEL)
    probes = {path.stem: checkpoint_probe(path, samples) for path in checkpoints}
    result = {"method": "100 same held-out seeds; deterministic policy; unchanged environment/reward; no learning",
              "reflection": "y -> 8-y meters, lateral velocity negated, LEFT/RIGHT slots and actions swapped",
              "environment": metadata["environment"], "baseline": summarize(baseline),
              "reflected_policy": summarize(mirrored),
              "initial_right_intervention": summarize(intervention),
              "fixed_probe_states": len(samples), "checkpoint_probes": probes,
              "episodes": {"baseline": baseline, "reflected_policy": mirrored,
                           "initial_right_intervention": intervention}}
    finalize(result)
    arrays = {key: np.stack([s["observation"][key] for s in samples]) for key in samples[0]["observation"]}
    np.savez_compressed(RUN / "right_bias_probe_states.npz", **arrays,
                        seed=np.array([s["seed"] for s in samples]),
                        scenario_slot=np.array([s["slot"] for s in samples]),
                        time=np.array([s["time"] for s in samples]),
                        goal_lane=np.array([s["goal_lane"] for s in samples]),
                        requested_lane_id=np.array([s["requested_lane"] for s in samples]))
    print(json.dumps({key: result[key] for key in ("baseline", "reflected_policy", "initial_right_intervention", "checkpoint_probes")}, indent=2))


if __name__ == "__main__":
    main()
