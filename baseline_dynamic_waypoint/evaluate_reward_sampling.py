"""Compare direct stochastic PPO execution before/after reward experiments.

This is an evaluation only. Observations and sampled actions are sent directly
between PPO and the saved environment, with no reflection or policy adapter.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import torch

from baseline_dynamic_waypoint.evaluate import load_experiment


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "reward_guidance_comparison_seed7"
MODELS = {
    "source": ROOT / "models" / "ppo_singlecar_committed57_seed7_200k.zip",
    "approach": ROOT / "models" / "ppo_singlecar_reward_approach_seed7_200k.zip",
}


def summary(rows):
    def aggregate(subset):
        count = len(subset)
        return {"episodes": count,
                **{name: sum(row[name] for row in subset)
                   for name in ("success", "collision", "timeout", "left_commands", "right_commands",
                                "lane_changes_started", "lane_changes_completed", "lane_changes_interrupted")},
                **{name + "_rate": sum(row[name] for row in subset) / count if count else None
                   for name in ("success", "collision", "timeout")}}
    result = aggregate(rows)
    result["by_slot"] = {slot: aggregate([row for row in rows if row["slot"] == slot])
                         for slot in ("front", "back", "left", "right")}
    result["by_slot_and_initial_goal_lane_relation"] = {
        slot: {relation: aggregate([row for row in rows
                                  if row["slot"] == slot and row["initial_relation"] == relation])
               for relation in ("LEFT", "SAME", "RIGHT")}
        for slot in ("front", "back", "left", "right")}
    return result


def evaluate(label, model_path):
    destination = OUTPUT / f"stochastic_{label}_episodes.csv"
    if destination.exists():
        raise FileExistsError(f"Preserve the existing stochastic evaluation: {destination}")
    fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model, env, metadata = load_experiment(model_path)
    rows = []
    fields = ["policy", "seed", "slot", "initial_lane", "goal_lane", "initial_relation",
              "success", "collision", "timeout", "reward", "elapsed_time", "final_distance",
              "policy_decisions", "physics_steps", "left_commands", "right_commands",
              "lane_changes_started", "lane_changes_completed", "lane_changes_interrupted"]
    try:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for seed in range(10000, 10100):
                obs, initial = env.reset(seed=seed)
                # SB3 samples its categorical action using torch's CPU RNG.
                torch.manual_seed(seed + 1_000_000)
                initial_lane = int(env.highway.vehicle.lane_index[2])
                goal_lane = int(initial["goal_lane"])
                counters = {key: 0 for key in ("policy_decisions", "physics_steps", "left_commands",
                            "right_commands", "lane_changes_started", "lane_changes_completed",
                            "lane_changes_interrupted")}
                total_reward = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = int(model.predict(obs, deterministic=False)[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    counters["policy_decisions"] += 1
                    counters["physics_steps"] += info.get("internal_steps", 1)
                    counters["left_commands"] += int(action == 0)
                    counters["right_commands"] += int(action == 2)
                    for counter, key in (("lane_changes_started", "lane_change_started"),
                                         ("lane_changes_completed", "lane_change_completed"),
                                         ("lane_changes_interrupted", "lane_change_interrupted")):
                        counters[counter] += int(info.get(key, False))
                row = {"policy": label, "seed": seed, "slot": initial["slot"],
                       "initial_lane": initial_lane, "goal_lane": goal_lane,
                       "initial_relation": "LEFT" if goal_lane < initial_lane else "RIGHT" if goal_lane > initial_lane else "SAME",
                       "success": int(info["is_success"]), "collision": int(info["is_collision"]),
                       "timeout": int(info["is_timeout"]), "reward": total_reward,
                       "elapsed_time": info["elapsed_time"], "final_distance": info["distance_to_goal"], **counters}
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                if (seed + 1) % 20 == 0:
                    print(f"{label}: {len(rows)}/100 stochastic episodes", flush=True)
    finally:
        env.close()
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == fingerprint
    return {"model": str(model_path.resolve()), "sha256": fingerprint,
            "environment": metadata["environment"], "summary": summary(rows)}


def main():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "stochastic_comparison.json"
    if destination.exists():
        raise FileExistsError(f"Preserve the existing result: {destination}")
    started = time.perf_counter()
    result = {"method": "direct PPO model.predict(observation, deterministic=False)",
              "seeds": [10000, 10099], "torch_seed_at_each_reset": "episode_seed + 1000000",
              "additional_training_steps": 0, "reflection_or_action_transforms": False,
              "caution": "One seeded stochastic trajectory per scenario; finite evaluation, not a reliable-fix claim. Rewards differ between models; compare outcomes rather than raw returns.",
              "models": {}}
    for label, path in MODELS.items():
        result["models"][label] = evaluate(label, path)
        print(f"{label} result: {result['models'][label]['summary']['success_rate']:.1%} success", flush=True)
    result["seconds"] = time.perf_counter() - started
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: data["summary"] for label, data in result["models"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
