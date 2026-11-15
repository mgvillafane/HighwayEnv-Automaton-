"""Evaluate trained PPO on reproducible sampled two-car scenarios."""
import json
from pathlib import Path

from baseline_dynamic_waypoint.evaluate import evaluate_policy, summarize
from baseline_waypoint.evaluate import save_rows
from baseline_simple_waypoint.environment import SimpleFollowingEnv


def evaluate(model, environment, output, episodes=100, seed=10_000, prefix="evaluation"):
    env = SimpleFollowingEnv(**environment)
    try:
        rows = evaluate_policy(env, model, episodes, seed, "ppo")
        # Recover the exact sampled initial conditions for the per-episode report.
        for row in rows:
            _, info = env.reset(seed=row["seed"])
            row.update({f"initial_{key}": value for key, value in info["initial_scenario"].items()
                        if key != "slot"})
        result = {"first_seed": seed, "environment": environment, "ppo": summarize(rows)}
    finally:
        env.close()
    output = Path(output)
    save_rows(output / f"{prefix}_episodes.csv", rows)
    (output / f"{prefix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    stats = result["ppo"]
    print(f"{prefix}: {episodes} episodes, success {stats['success_rate']:.1%}, "
          f"collision {stats['collision_rate']:.1%}, timeout {stats['timeout_rate']:.1%}, "
          f"success by slot {stats['success_rate_by_slot']}", flush=True)
    return result
