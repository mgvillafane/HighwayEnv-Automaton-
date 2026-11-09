"""The fixed measurement: PPO, uniform random and the scripted driver on the same 200 episodes.

Every arm is measured the same way, so differences between arms come from the arm's IV.

    python -B baseline_dynamic_reach/evaluate.py --run B0_seed7
    python -B baseline_dynamic_reach/evaluate.py --references B0     # Step 0: no trained policy
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
import torch

from baseline_dynamic_reach.config import EVAL_EPISODES, EVAL_SEED, MODELS_DIR, RESULTS_DIR, SLOTS, arm_config
from baseline_dynamic_reach.environment import ReachEnv
from baseline_dynamic_waypoint.evaluate import evaluate_policy, summarize
from baseline_waypoint.evaluate import save_rows

OUTCOMES = ("success", "collision", "timeout", "anchor_lost")
PAIRED_KEYS = ("seed", "slot", "goal_lane", "initial_lane", "anchor_initial_gap", "initial_distance")


def wilson(successes, n, z=1.959964):
    """95% Wilson score interval for a binomial rate; sensible at 0 and n, unlike the normal one."""
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def check_rows(rows):
    """Every episode has exactly one outcome, and its reward terms add up to its return."""
    for row in rows:
        outcomes = [name for name in OUTCOMES if row[name]]
        if len(outcomes) != 1:
            raise AssertionError(f"{row['policy']} seed {row['seed']}: outcomes {outcomes}, expected exactly one")
        components = sum(value for key, value in row.items() if key.startswith("reward_"))
        if abs(components - row["reward"]) > 1e-6:
            raise AssertionError(f"{row['policy']} seed {row['seed']}: terms sum to {components}, return {row['reward']}")


def check_paired(row_sets):
    """All policies start from identical episodes: same traffic seed, anchor, slot and distance."""
    reference = [tuple(row[key] for key in PAIRED_KEYS) for row in row_sets[0]]
    for rows in row_sets[1:]:
        if [tuple(row[key] for key in PAIRED_KEYS) for row in rows] != reference:
            raise AssertionError(f"{rows[0]['policy']} did not start from the same episodes as {row_sets[0][0]['policy']}")


def describe(rows):
    """Overall and per-slot outcome rates, each success rate with its 95% interval."""
    summary = summarize(rows)
    n, k = len(rows), sum(row["success"] for row in rows)
    summary["successes"] = k
    summary["success_ci95"] = wilson(k, n)
    summary["outcome_counts"] = {name: int(sum(row[name] for row in rows)) for name in OUTCOMES}
    by_slot = {}
    for slot in SLOTS:
        subset = [row for row in rows if row["slot"] == slot]
        if not subset:
            continue
        wins = sum(row["success"] for row in subset)
        times = [row["elapsed_time"] for row in subset if row["success"]]
        by_slot[slot] = {
            "episodes": len(subset), "successes": wins, "success_rate": wins / len(subset),
            "success_ci95": wilson(wins, len(subset)),
            **{f"{name}_rate": float(np.mean([row[name] for row in subset])) for name in OUTCOMES[1:]},
            "mean_time_to_reach_seconds": float(np.mean(times)) if times else None,
            "mean_final_distance_meters": float(np.mean([row["final_distance"] for row in subset])),
        }
    summary["by_slot"] = by_slot
    return summary


def run_policies(environment, model=None, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """Row lists for PPO (if a model is given), uniform random and scripted, all on the same seeds."""
    torch.set_num_threads(1)
    env = ReachEnv(**deepcopy(environment))
    rows = {}
    try:
        if model is not None:
            rows["ppo"] = evaluate_policy(env, model, episodes, seed, "ppo")
        rows["random"] = evaluate_policy(env, None, episodes, seed, "random")
        rows["scripted"] = evaluate_policy(env, "scripted", episodes, seed, "scripted")
    finally:
        env.close()
    for policy_rows in rows.values():
        check_rows(policy_rows)
    check_paired(list(rows.values()))
    return rows


def write(output_dir, header, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows(output_dir / "evaluation_episodes.csv", [row for policy_rows in rows.values() for row in policy_rows])
    summary = {**header, **{label: describe(policy_rows) for label, policy_rows in rows.items()}}
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for label in rows:
        result = summary[label]
        slots = ", ".join(f"{slot} {info['successes']}/{info['episodes']}" for slot, info in result["by_slot"].items())
        low, high = result["success_ci95"]
        print(f"{label:>8}: success {result['success_rate']:.1%} [{low:.1%}, {high:.1%}], "
              f"collision {result['collision_rate']:.1%}, timeout {result['timeout_rate']:.1%}, "
              f"anchor lost {result['anchor_lost_rate']:.1%} | {slots}", flush=True)
    return summary


def evaluate_run(model_path, output_dir, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    model_path = Path(model_path)
    with model_path.with_suffix(".json").open(encoding="utf-8") as handle:
        sidecar = json.load(handle)
    model = PPO.load(model_path, device="cpu")
    rows = run_policies(sidecar["environment"], model, episodes, seed)
    header = {"run": sidecar.get("run_name"), "arm": sidecar.get("arm"), "model": str(model_path.resolve()),
              "first_seed": seed, "episodes": episodes, "environment": sidecar["environment"]}
    return write(output_dir, header, rows)


def evaluate_references(arm, output_dir, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """Step 0: what the task looks like before any training, for an arm's environment."""
    environment = arm_config(arm)["environment"]
    rows = run_policies(environment, None, episodes, seed)
    header = {"arm": arm, "first_seed": seed, "episodes": episodes, "environment": environment}
    return write(output_dir, header, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Trained run name, e.g. B0_seed7 (model and sidecar in models/).")
    group.add_argument("--references", metavar="ARM", help="Evaluate only random and scripted on this arm's task.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--output-name", help="Results subfolder; defaults to <run>/evaluation_repeat or step0_<arm>.")
    args = parser.parse_args()
    if args.run:
        output = RESULTS_DIR / (args.output_name or f"{args.run}/evaluation_repeat")
        evaluate_run(MODELS_DIR / f"{args.run}.zip", output, args.episodes, args.seed)
    else:
        output = RESULTS_DIR / (args.output_name or f"step0_{args.references}")
        evaluate_references(args.references, output, args.episodes, args.seed)


if __name__ == "__main__":
    main()
