"""Compare direct PPO reward-guidance runs on matched old and fresh scenarios.

This program never trains models or applies an observation/action adapter. Each
checkpoint is evaluated in the exact environment and reward saved in its sidecar.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment, evaluate_policy, summarize
from baseline_dynamic_waypoint.config import experiment_name
from baseline_waypoint.evaluate import save_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "models" / "ppo_singlecar_committed57_seed7_200k.zip"
DEFAULT_CONTROL = ROOT / "models" / "ppo_singlecar_reward_baseline_seed7_200k.zip"
DEFAULT_GUIDED = ROOT / "models" / "ppo_singlecar_reward_guided_seed7_200k.zip"
DEFAULT_FRESH = ROOT / "models" / "ppo_singlecar_reward_guided_fresh_seed7_200k.zip"
DEFAULT_APPROACH = ROOT / "models" / "ppo_singlecar_reward_approach_seed7_200k.zip"
SLOTS = ("front", "back", "left", "right")
LABELS = {"source": "Original checkpoint", "control": "Baseline +200k", "guided": "Guided reward +200k",
          "fresh_init": "Guided +200k from scratch", "approach_followup": "Stronger reward +200k follow-up"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key not in ("policy", "slot"):
                row[key] = float(value)
    return rows


def verify_matched_scenarios(reference, candidate):
    """Seed equality alone is insufficient if an environment setting changed."""
    previous = {int(row["seed"]): row for row in reference}
    assert len(previous) == len(reference) == len(candidate)
    assert set(previous) == {int(row["seed"]) for row in candidate}
    for row in candidate:
        original = previous[int(row["seed"])]
        assert original["slot"] == row["slot"]
        for key in ("initial_lane", "goal_lane", "lane_changes_required"):
            assert original[key] == row[key], (row["seed"], key)
        for key in ("anchor_initial_gap", "initial_distance"):
            assert abs(original[key] - row[key]) < 1e-8, (row["seed"], key)
    return len(candidate)


def goal_counts(rows):
    result = {}
    for slot in SLOTS:
        selected = [row for row in rows if row["slot"] == slot]
        if selected:
            result[slot] = {"episodes": len(selected),
                            **{key: int(sum(row[key] for row in selected))
                               for key in ("success", "collision", "timeout")}}
    return result


def source_cache(path, metadata, episodes, seed):
    """Reuse only an already verified *direct original PPO* evaluation."""
    cache_dir = ROOT / "results" / "ppo_singlecar_right_canonical_seed7"
    table, manifest = cache_dir / "fresh_before_episodes.csv", cache_dir / "run.json"
    if not (table.exists() and manifest.exists()):
        return None
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("source_sha256") != digest(path) or saved.get("environment") != metadata["environment"]:
        return None
    rows = read_rows(table)
    if (len(rows) != episodes or {int(row["seed"]) for row in rows} != set(range(seed, seed + episodes))
            or any(row["policy"] != "ppo_original" for row in rows)):
        return None
    return rows, {"method": "cached direct original PPO; no adapter applied",
                  "path": str(table.resolve()), "sha256": digest(table)}


def validate_protocol(paths, metadata):
    expected_source = digest(paths["source"])
    for label in ("control", "guided"):
        saved = metadata[label]
        if saved.get("status") != "trained" or saved.get("actual_timesteps") != 200000:
            raise ValueError(f"{label} has not completed its required 200,000 training decisions.")
        if saved["initialization"]["sha256"] != expected_source:
            raise ValueError(f"{label} did not initialize from the requested source checkpoint.")
        if saved["environment"]["env_config"] != metadata["source"]["environment"]["env_config"]:
            raise ValueError(f"{label} simulator configuration differs from the source.")
    if metadata["control"]["ppo_config"] != metadata["guided"]["ppo_config"]:
        raise ValueError("Control and guided PPO settings differ.")
    if metadata["control"]["seed"] != metadata["guided"]["seed"]:
        raise ValueError("Control and guided training seeds differ.")
    for label in ("source", "control", "guided"):
        goal = deepcopy(metadata[label]["environment"]["goal_config"])
        goal.pop("front_lane_blend", None)  # This flag affects reward potential only.
        reference = deepcopy(metadata["source"]["environment"]["goal_config"])
        reference.pop("front_lane_blend", None)
        if goal != reference:
            raise ValueError(f"{label} success geometry or sampling differs from the source.")
    if metadata["control"]["environment"]["reward_weights"] != metadata["source"]["environment"]["reward_weights"]:
        raise ValueError("Baseline control rewards differ from the source.")


def write_plot(output, result):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    labels = [label for label in LABELS if label in result["models"]]
    width = 0.72 / len(labels)
    colors = {"source": "#94a3b8", "control": "#f59e0b", "guided": "#2563eb", "fresh_init": "#059669",
              "approach_followup": "#dc2626"}
    for axis, cohort, title in zip(axes, ("fixed_100", "fresh"),
                                   ("Original 100 held-out scenarios", f"{result['protocol']['fresh_episodes']} fresh scenarios")):
        for index, label in enumerate(labels):
            offset = (index - (len(labels) - 1) / 2) * width
            summary = result["evaluations"][cohort][label]["summary"]
            values = [100 * summary["success_rate_by_slot"].get(slot, 0) for slot in SLOTS]
            bars = axis.bar(np.arange(len(SLOTS)) + offset, values, width=width * 0.96,
                            label=LABELS[label], color=colors[label])
            axis.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        axis.set_xticks(range(len(SLOTS)), [slot.upper() for slot in SLOTS])
        axis.set_ylim(0, 112)
        axis.set_title(title)
        axis.set_ylabel("Success (%)")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2 if len(labels) == 4 else 3, frameon=False)
    figure.suptitle("Reward guidance: matched PPO training, direct actions, no reflection")
    figure.tight_layout(rect=(0, 0.12 if len(labels) >= 4 else 0.075, 1, 0.96))
    figure.savefig(output / "success_comparison.png", dpi=170)
    plt.close(figure)


def write_report(output, result):
    labels = [label for label in LABELS if label in result["models"]]
    target_findings = []
    for slot in ("right", "front"):
        before = result["evaluations"]["fresh"]["control"]["by_goal"][slot]
        after = result["evaluations"]["fresh"]["guided"]["by_goal"][slot]
        target_findings.append(f"{slot.upper()} {before['success']}/{before['episodes']} → {after['success']}/{after['episodes']}")
    lines = ["# Reward guidance comparison", "",
             "Two PPO runs each trained for **200,000 additional policy decisions**, initialized from the same saved actor and optimizer. "
             "The training seed, PPO settings, randomized two-car environment, 30-second limit, 57 observations, and completed lane-change execution were kept the same. "
             "The guided run changes reward guidance only. **Every evaluation uses the trained PPO directly: no reflection, adapter, or scripted actions.**", "",
             "In the fresh-scenario matched warm-start comparison, success changes are **" + "; ".join(target_findings) + "**. "
             "These measured outcomes, rather than larger shaped returns, determine whether the requested goals improved.", "",
             "## Results", "",
             "Success and collision rates are the main comparison. Recorded episode returns use each model's own reward weights, "
             "so larger guided returns alone do not establish improvement.", ""]
    for cohort, title in (("fixed_100", "Original held-out 100 scenarios"), ("fresh", "Fresh scenarios")):
        lines += [f"### {title}", "",
                  "| Policy | Success | Collisions | Timeouts | Mean return (own reward) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label in labels:
            row = result["evaluations"][cohort][label]["summary"]
            count = row["episodes"]
            counts = [round(count * row[key]) for key in ("success_rate", "collision_rate", "timeout_rate")]
            lines.append(f"| {LABELS[label]} | {counts[0]}/{count} ({row['success_rate']:.1%}) | "
                         f"{counts[1]}/{count} ({row['collision_rate']:.1%}) | {counts[2]}/{count} ({row['timeout_rate']:.1%}) | "
                         f"{row['average_episode_reward']:.2f} |")
        lines += ["", "| Objective | " + " | ".join(LABELS[label] for label in labels) + " |",
                  "| --- | " + " | ".join("---:" for _ in labels) + " |"]
        for slot in SLOTS:
            values = []
            for label in labels:
                row = result["evaluations"][cohort][label]["by_goal"][slot]
                values.append(f"{row['success']}/{row['episodes']} ({row['success']/row['episodes']:.1%})")
            lines.append(f"| {slot.upper()} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["![Success comparison](success_comparison.png)", "",
              "## Reward change", "",
              "Per 0.1-second physics tick, the original reward remains:", "",
              "`r_base = 50*success - 50*collision + 0.1*(d_previous-d_current) - 0.05 + 0.2*inside_valid_slot`", "",
              "The guided profile adds:", "",
              "`r_guided = r_base + 1.0*(L_previous-L_current) + 0.1*(S_previous-S_current) + 0.5*(C_current-C_previous)`", "",
              "- `L`: lateral error in meters. Side/BACK objectives target the goal lane. For FRONT, lateral guidance initially targets a passing lane, "
              "then blends continuously toward the goal lane as ego's lead grows from 5 to 15 m. This avoids an abrupt penalty at a single passing threshold.",
              "- `S`: speed mismatch in m/s, multiplied by `max(0, 1-|longitudinal_error|/30)` so distant ego can catch up before matching speed.",
              "- `C`: ego's longitudinal lead over traffic, clipped to [-150, 20] m, for FRONT only; zero for other objectives.", "",
              "The reward encourages reaching the appropriate lane rather than paying for RIGHT commands. "
              "It also provides FRONT credit for gaining position through a pass. Success/collision bonuses and success tolerances are unchanged. "
              "Each committed maneuver still returns the sum of all internal physics-tick rewards. "
              "This difference-based guidance changes the learning objective; no claim of discounted policy invariance is made.", "",
              "## Protocol and limits", "",
              f"Fixed evaluation seeds: {result['protocol']['fixed_seeds'][0]}–{result['protocol']['fixed_seeds'][1]}. "
              f"Fresh evaluation seeds: {result['protocol']['fresh_seeds'][0]}–{result['protocol']['fresh_seeds'][1]}. "
              "Before/after comparisons use identical sampled scenarios, verified by goal, initial lanes, anchor gap, and initial distance.", "",
              "The original checkpoint is included as a reference. The controlled training comparison is **guided +200k versus baseline +200k**; "
              "comparing only against the original checkpoint would confound reward changes with extra training. "
              "This is one training seed, with deterministic evaluation on finite held-out sets; it is not a multi-seed robustness claim.", "",
              "Training budgets count PPO policy decisions. Completed lane changes contain multiple 0.1-second physics ticks, "
              "so equal decision budgets need not contain equal simulated time; actual physics totals are recorded below.", "",
              "The pre-existing checkpoint and experiments remain unchanged. Models load their own saved reward profiles and goal settings. "
              "Fresh evaluations use saved environments; no canonicalization code is imported.", "",
              "## Artifacts", "",
              "[Machine-readable comparison](comparison.json) · [Summary CSV](summary.csv)", ""]
    if "fresh_init" in labels:
        lines += ["The additional **guided +200k from scratch** run uses the same guided reward, environment, "
                  "PPO settings, and seed, but a fresh actor/critic/optimizer. It tests whether escaping the inherited policy bias helps. "
                  "Because initialization differs, its comparison is separate from the matched warm-start reward experiment.", ""]
    if "approach_followup" in labels:
        lines += ["The **stronger reward +200k follow-up** starts from the completed from-scratch guided model, "
                  "then trains another 200,000 decisions (400,000 lifetime decisions). It increases collision cost to 100, "
                  "distance-progress weight to 0.5, lateral guidance to 2.0, speed-match guidance to 0.5, and FRONT lead guidance to 1.0. "
                  "Success/holding/time terms and success geometry stay unchanged. This is an exploratory follow-up: "
                  "both reward and additional training differ from its parent, so it is not the matched causal reward comparison.", "",
                  "Its per-physics-tick reward is:", "",
                  "`r_approach = 50*success - 100*collision + 0.5*(d_previous-d_current) - 0.05 + 0.2*inside_valid_slot "
                  "+ 2.0*(L_previous-L_current) + 0.5*(S_previous-S_current) + 1.0*(C_current-C_previous)`", ""]
    lines += ["Saved reward weights (collision/time weights are subtracted):", "",
              "| Profile | Goal | Collision | Progress | Time | Hold | Lane | Speed match | FRONT lead |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in ("control", "guided", "approach_followup"):
        if label in result["models"]:
            weights = result["models"][label]["metadata"]["environment"]["reward_weights"]
            profile = result["models"][label]["metadata"].get("reward_profile", label)
            lines.append(f"| {profile} | " + " | ".join(f"{weights.get(key, 0):g}" for key in
                         ("goal", "collision", "progress", "time", "hold", "lane", "speed_match", "overtake")) + " |")
    lines.append("")
    for label in labels:
        model = result["models"][label]
        stem = Path(model["path"]).stem
        lines += [f"- **{LABELS[label]}:** model `{stem}`; SHA256 `{model['sha256']}`; "
                  f"additional training in that run: {model['metadata'].get('actual_timesteps', 0):,} decisions; "
                  f"lifetime counter: {model['metadata'].get('total_lifetime_timesteps', model['metadata'].get('actual_timesteps', 0)):,} decisions; "
                  f"physics ticks in that run: {model['metadata'].get('actual_physics_steps', 0):,}.",
                  f"  [Model](../../models/{stem}.zip) · [Saved environment](../../models/{stem}.json) · "
                  f"[Videos](../{stem}/videos/all_goals.mp4)"]
    lines += ["", "[Comparison runner](../../compare_reward_experiments.py) · [Reward configuration](../../config.py)", "",
              "Fresh source evaluation provenance:", "", "```json",
              json.dumps(result["evaluations"]["fresh"]["source"]["provenance"], indent=2), "```", "",
              "Run after both training jobs complete:", "", "```powershell",
              ".\\baseline_waypoint\\.venv\\Scripts\\python.exe -B -m baseline_dynamic_waypoint.compare_reward_experiments", "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(output, result):
    rows = []
    for cohort in ("fixed_100", "fresh"):
        for label in LABELS:
            if label not in result["models"]:
                continue
            data = result["evaluations"][cohort][label]
            row = {"cohort": cohort, "policy": label,
                   **{key: data["summary"][key] for key in ("episodes", "success_rate", "collision_rate", "timeout_rate", "average_episode_reward")}}
            for slot in SLOTS:
                row[f"{slot}_successes"] = data["by_goal"][slot]["success"]
                row[f"{slot}_episodes"] = data["by_goal"][slot]["episodes"]
            rows.append(row)
    save_rows(output / "summary.csv", rows)
    return rows


def append_fresh_initialization(output, path, *, followup=False):
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    label = "approach_followup" if followup else "fresh_init"
    if label in result["models"]:
        raise ValueError(f"The {label} comparison already exists; use --report-only to regenerate its report.")
    for existing in result["models"].values():
        if digest(Path(existing["path"])) != existing["sha256"]:
            raise ValueError("A checkpoint from the saved warm-start comparison has changed.")
    path = path.resolve()
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if followup and "fresh_init" not in result["models"]:
        raise ValueError("Append the fresh guided parent before appending its stronger-reward follow-up.")
    parent = result["models"]["fresh_init" if followup else "guided"]
    reference = parent["metadata"]
    if metadata.get("status") != "trained" or metadata.get("actual_timesteps") != 200000:
        raise ValueError(f"The {label} experiment has not completed 200,000 additional decisions.")
    if followup:
        if ((metadata.get("initialization") or {}).get("sha256") != parent["sha256"]
                or metadata.get("total_lifetime_timesteps") != 400000
                or metadata.get("reward_profile") != "guided_approach"):
            raise ValueError("The stronger reward run must initialize from the fresh guided parent and total 400,000 lifetime decisions.")
        for key in ("env_config", "goal_config"):
            if metadata["environment"][key] != reference["environment"][key]:
                raise ValueError(f"Follow-up changes non-reward {key}.")
        expected = {**reference["environment"]["reward_weights"], "collision": 100.0, "progress": 0.5,
                    "lane": 2.0, "speed_match": 0.5, "overtake": 1.0}
        if metadata["environment"]["reward_weights"] != expected:
            raise ValueError("Follow-up saved weights differ from the declared stronger-approach profile.")
    elif metadata.get("initialization") is not None or metadata["environment"] != reference["environment"]:
        raise ValueError("The fresh guided run must use fresh initialization and the matched guided environment/reward.")
    for key in ("ppo_config", "seed"):
        if metadata[key] != reference[key]:
            raise ValueError(f"The {label} experiment differs in {key}.")
    fingerprint = digest(path)
    fixed = ROOT / "results" / path.stem / "evaluation_episodes.csv"
    fixed_rows = read_rows(fixed)
    reference_fixed = read_rows(Path(result["evaluations"]["fixed_100"]["source"]["provenance"]["path"]))
    matched_fixed = verify_matched_scenarios(reference_fixed, fixed_rows)
    model, env, _ = load_experiment(path)
    rows = []
    seed = result["protocol"]["fresh_seeds"][0]
    episodes = result["protocol"]["fresh_episodes"]
    try:
        for offset in range(0, episodes, 25):
            count = min(25, episodes - offset)
            rows += evaluate_policy(env, model, count, seed + offset, label)
            print(f"{label}: fresh evaluation {offset + count}/{episodes}", flush=True)
    finally:
        env.close()
    save_rows(output / f"{label}_fresh_episodes.csv", rows)
    reference_fresh = read_rows(output / "source_fresh_episodes.csv")
    matched_fresh = verify_matched_scenarios(reference_fresh, rows)
    assert digest(path) == fingerprint
    result["models"][label] = {"path": str(path), "sha256": fingerprint, "metadata": metadata,
                              "initialization": "continued from fresh guided checkpoint" if followup else "fresh; no source checkpoint"}
    result["evaluations"]["fixed_100"][label] = {
        "summary": summarize(fixed_rows), "by_goal": goal_counts(fixed_rows),
        "provenance": {"method": "saved direct PPO evaluation", "path": str(fixed.resolve()), "sha256": digest(fixed)}}
    result["evaluations"]["fresh"][label] = {
        "summary": summarize(rows), "by_goal": goal_counts(rows),
        "provenance": {"method": "new deterministic direct PPO evaluation", "checkpoint": str(path)}}
    result["verification"]["matched_fixed_scenarios"][label] = matched_fixed
    result["verification"]["matched_fresh_scenarios"][label] = matched_fresh
    if followup:
        result["verification"]["followup_parent_sha_and_400k_lifetime_verified"] = True
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = write_summary(output, result)
    write_plot(output, result)
    write_report(output, result)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--guided", type=Path, default=DEFAULT_GUIDED)
    parser.add_argument("--output-name", default="reward_guidance_comparison_seed7")
    parser.add_argument("--fresh-episodes", type=int, default=200)
    parser.add_argument("--fresh-seed", type=int, default=30000)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--append-fresh-model", type=Path, nargs="?", const=DEFAULT_FRESH,
                        help="Append a completed from-scratch guided run to an existing comparison without repeating warm evaluations.")
    parser.add_argument("--append-followup-model", type=Path, nargs="?", const=DEFAULT_APPROACH,
                        help="Append the completed stronger-reward continuation of the fresh guided model.")
    args = parser.parse_args()
    output = ROOT / "results" / experiment_name(args.output_name)
    if args.append_followup_model:
        torch.set_num_threads(1)
        append_fresh_initialization(output, args.append_followup_model, followup=True)
        return
    if args.append_fresh_model:
        torch.set_num_threads(1)
        append_fresh_initialization(output, args.append_fresh_model)
        return
    if args.report_only:
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        write_plot(output, result)
        write_report(output, result)
        return
    if args.fresh_episodes < 1:
        parser.error("fresh-episodes must be positive.")
    paths = {label: getattr(args, label).resolve() for label in ("source", "control", "guided")}
    metadata = {label: json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                for label, path in paths.items()}
    validate_protocol(paths, metadata)
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"fixed_seeds": [10000, 10099],
                           "fresh_seeds": [args.fresh_seed, args.fresh_seed + args.fresh_episodes - 1],
                           "fresh_episodes": args.fresh_episodes, "controller": "direct saved PPO; no reflection",
                           "training_decisions_per_comparison_run": 200000,
                           "returns_comparable_across_reward_profiles": False},
              "models": {label: {"path": str(path), "sha256": digest(path), "metadata": metadata[label]}
                         for label, path in paths.items()},
              "evaluations": {"fixed_100": {}, "fresh": {}}, "verification": {}}
    fixed_rows, fresh_rows = {}, {}
    for label, path in paths.items():
        fixed = ROOT / "results" / path.stem / "evaluation_episodes.csv"
        fixed_rows[label] = read_rows(fixed)
        if len(fixed_rows[label]) != 100 or {int(row["seed"]) for row in fixed_rows[label]} != set(range(10000, 10100)):
            raise ValueError(f"{label} saved fixed evaluation does not cover seeds 10000–10099.")
        result["evaluations"]["fixed_100"][label] = {
            "summary": summarize(fixed_rows[label]), "by_goal": goal_counts(fixed_rows[label]),
            "provenance": {"method": "saved direct PPO evaluation", "path": str(fixed.resolve()), "sha256": digest(fixed)}}
        cached = source_cache(path, metadata[label], args.fresh_episodes, args.fresh_seed) if label == "source" else None
        if cached:
            fresh_rows[label], provenance = cached
            print(f"{label}: reused verified direct-PPO source evaluation on {len(fresh_rows[label])} fresh seeds.", flush=True)
        else:
            model, env, _ = load_experiment(path)
            fresh_rows[label] = []
            try:
                for offset in range(0, args.fresh_episodes, 25):
                    count = min(25, args.fresh_episodes - offset)
                    fresh_rows[label] += evaluate_policy(env, model, count, args.fresh_seed + offset, label)
                    print(f"{label}: fresh evaluation {offset + count}/{args.fresh_episodes}", flush=True)
            finally:
                env.close()
            provenance = {"method": "new deterministic direct PPO evaluation", "checkpoint": str(path)}
            del model
        save_rows(output / f"{label}_fresh_episodes.csv", fresh_rows[label])
        result["evaluations"]["fresh"][label] = {
            "summary": summarize(fresh_rows[label]), "by_goal": goal_counts(fresh_rows[label]), "provenance": provenance}
        assert digest(path) == result["models"][label]["sha256"]
    result["verification"] = {"matched_fixed_scenarios": {
        label: verify_matched_scenarios(fixed_rows["source"], fixed_rows[label]) for label in ("control", "guided")},
        "matched_fresh_scenarios": {
        label: verify_matched_scenarios(fresh_rows["source"], fresh_rows[label]) for label in ("control", "guided")},
        "all_source_checkpoints_unchanged": True,
        "training_metadata_protocol_checks_passed": True,
        "direct_ppo_only": True}
    result["seconds"] = time.perf_counter() - started
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_rows = write_summary(output, result)
    write_plot(output, result)
    write_report(output, result)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
