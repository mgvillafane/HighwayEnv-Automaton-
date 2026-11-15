"""Evaluate opt-in RIGHT canonicalization without changing rewards or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment, evaluate_policy, summarize
from baseline_dynamic_waypoint.reflection_policy import RightCanonicalPolicy
from baseline_waypoint.evaluate import save_rows


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models" / "ppo_singlecar_committed57_seed7_200k.zip"
SOURCE_RUN = ROOT / "results" / SOURCE.stem


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key not in ("policy", "slot"):
                row[key] = float(row[key])
    return rows


def verify_other_goals(before, after):
    previous = {int(row["seed"]): row for row in before}
    checked = 0
    for row in after:
        if row["slot"] == "right":
            continue
        original = previous[int(row["seed"])]
        for key in ("success", "collision", "timeout", "length", "physics_steps"):
            assert row[key] == original[key], (row["seed"], key)
        assert abs(row["reward"] - original["reward"]) < 1e-8
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="ppo_singlecar_right_canonical_seed7")
    parser.add_argument("--fresh-episodes", type=int, default=200)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.fresh_episodes < 1:
        parser.error("Use a directory name and a positive fresh episode count.")
    output = ROOT / "results" / args.run_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    fingerprint = sha256(SOURCE)
    model, env, metadata = load_experiment(SOURCE)
    controller = RightCanonicalPolicy(model)
    run = {"experiment": "right_objective_policy_canonicalization", "status": "running",
           "controller": "RIGHT-only observation reflection and inverse action mapping",
           "source_model": str(SOURCE.resolve()), "source_sha256": fingerprint,
           "additional_training_steps": 0, "reward_changed": False,
           "environment": metadata["environment"],
           "comparison_seeds": [10000, 10099], "fresh_seeds": [30000, 30000 + args.fresh_episodes - 1]}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    try:
        before = numeric_rows(SOURCE_RUN / "evaluation_episodes.csv")
        after = evaluate_policy(env, controller, 100, 10000, "ppo_right_canonical")
        checked = verify_other_goals(before, after)
        save_rows(output / "evaluation_episodes.csv", after)
        print("Original held-out evaluation complete: " + str(summarize(after)), flush=True)
        fresh_before, fresh_after = [], []
        for offset in range(0, args.fresh_episodes, 25):
            count = min(25, args.fresh_episodes - offset)
            fresh_before += evaluate_policy(env, model, count, 30000 + offset, "ppo_original")
            fresh_after += evaluate_policy(env, controller, count, 30000 + offset, "ppo_right_canonical")
            print(f"Fresh paired evaluation {offset + count}/{args.fresh_episodes}", flush=True)
        checked += verify_other_goals(fresh_before, fresh_after)
        save_rows(output / "fresh_before_episodes.csv", fresh_before)
        save_rows(output / "fresh_after_episodes.csv", fresh_after)
        assert sha256(SOURCE) == fingerprint
        result = {"original_seeds": {"before": summarize(before), "after": summarize(after)},
                  "fresh_seeds": {"before": summarize(fresh_before), "after": summarize(fresh_after)},
                  "verification": {"source_checkpoint_unchanged": True,
                                   "non_right_episodes_identical": checked,
                                   "reward_weights_unchanged": env.reward_weights == metadata["environment"]["reward_weights"]}}
    finally:
        env.close()
    run.update(status="complete", seconds=time.perf_counter() - started)
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    slots = ("back", "front", "left", "right")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (group, title) in zip(axes, (("original_seeds", "Original 100 evaluation scenarios"),
                                          ("fresh_seeds", f"{args.fresh_episodes} fresh scenarios"))):
        for label, offset, color in (("before", -0.18, "#94a3b8"), ("after", 0.18, "#2563eb")):
            values = [100 * result[group][label]["success_rate_by_slot"][slot] for slot in slots]
            bars = axis.bar([i + offset for i in range(4)], values, width=0.36, label=label, color=color)
            axis.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
        axis.set_xticks(range(4), [slot.upper() for slot in slots])
        axis.set_ylim(0, 112)
        axis.set_title(title)
        axis.set_ylabel("Success (%)")
        axis.legend()
    figure.suptitle("RIGHT-only policy reflection — same reward, no additional training")
    figure.tight_layout()
    figure.savefig(output / "results.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
