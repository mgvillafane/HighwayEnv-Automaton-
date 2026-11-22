"""Generate the experiment report from completed, verified evaluation files."""
import csv
import json
from pathlib import Path
from statistics import mean


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "ppo_singlecar_committed57_seed7_200k"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def statistics(data):
    return {"episodes": len(data),
            "success": sum(int(row["success"]) for row in data),
            "collision": sum(int(row["collision"]) for row in data),
            "timeout": sum(int(row["timeout"]) for row in data),
            "reward": mean(float(row["reward"]) for row in data)}


def count_rate(value, total):
    return f"{value}/{total} ({100 * value / total:.1f}%)"


def main():
    evaluation = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))
    run = json.loads((RUN / "run.json").read_text(encoding="utf-8"))
    cohorts = [
        ("Original 100 scenarios", "original_seeds", (10000, 10099),
         rows(SOURCE / "evaluation_episodes.csv"), rows(RUN / "evaluation_episodes.csv")),
        ("Fresh 200 scenarios", "fresh_seeds", (30000, 30199),
         rows(RUN / "fresh_before_episodes.csv"), rows(RUN / "fresh_after_episodes.csv")),
    ]
    assert run["status"] == "complete"
    assert run["additional_training_steps"] == 0 and not run["reward_changed"]
    for title, key, seeds, before, after in cohorts:
        assert [row["seed"] for row in before] == [row["seed"] for row in after]
        assert len(before) == seeds[1] - seeds[0] + 1
        for label, data in (("before", before), ("after", after)):
            calculated = statistics(data)
            expected = evaluation[key][label]
            assert calculated["episodes"] == expected["episodes"]
            for field in ("success", "collision", "timeout"):
                assert abs(calculated[field] / len(data) - expected[field + "_rate"]) < 1e-10
            assert abs(calculated["reward"] - expected["average_episode_reward"]) < 1e-8
    verification = evaluation["verification"]
    assert verification["source_checkpoint_unchanged"] and verification["reward_weights_unchanged"]
    original_stats = [statistics(group) for group in cohorts[0][3:]]
    fresh_stats = [statistics(group) for group in cohorts[1][3:]]
    lines = [
        "# RIGHT-only reflection experiment", "",
        "The experiment reuses the existing PPO's successful LEFT strategy for RIGHT objectives. "
        "Only RIGHT observations are reflected across the middle lane, and LEFT/RIGHT output actions "
        "are swapped back before execution. FRONT, BACK, and LEFT use the original policy directly.", "",
        "**No additional training was run. Model weights, reward, simulator dynamics, scenario sampling, "
        "30-second episode duration, and committed lane-change execution are unchanged.** "
        "This measures an observation/action adapter, not a newly trained policy.", "",
        f"Overall success increased from **{count_rate(original_stats[0]['success'], 100)} to "
        f"{count_rate(original_stats[1]['success'], 100)}** on the original evaluation seeds, and from "
        f"**{count_rate(fresh_stats[0]['success'], 200)} to {count_rate(fresh_stats[1]['success'], 200)}** "
        "on fresh seeds. The tables below give the full outcomes.", "",
    ]
    for title, key, seeds, before, after in cohorts:
        lines += [f"## {title}", "", f"Paired deterministic evaluation on seeds {seeds[0]}–{seeds[1]}; "
                  "each before/after pair uses exactly the same sampled scenario.", "",
                  "| Goal | Episodes | Success before → after | Collisions before → after | Mean return before → after |",
                  "| --- | ---: | --- | --- | ---: |"]
        for slot in ("front", "back", "left", "right", "all"):
            a = statistics([row for row in before if slot == "all" or row["slot"] == slot])
            b = statistics([row for row in after if slot == "all" or row["slot"] == slot])
            n = a["episodes"]
            label = "**Overall**" if slot == "all" else slot.upper()
            lines.append(f"| {label} | {n} | {count_rate(a['success'], n)} → {count_rate(b['success'], n)} "
                         f"| {count_rate(a['collision'], n)} → {count_rate(b['collision'], n)} "
                         f"| {a['reward']:.2f} → {b['reward']:.2f} |")
        a, b = statistics(before), statistics(after)
        lines += ["", f"Timeouts: {count_rate(a['timeout'], a['episodes'])} → "
                  f"{count_rate(b['timeout'], b['episodes'])}.", ""]
    lines += [
        "## What the result establishes", "",
        "RIGHT benefits from reusing the already learned LEFT behavior under the existing reward. "
        "The original network had not learned that reflection on its own. This supports the earlier "
        "diagnosis of a learned goal-dependent asymmetry rather than a RIGHT steering or y-sign error.", "",
        f"All **{verification['non_right_episodes_identical']} non-RIGHT episodes** across the two cohorts "
        "were checked against the original policy: success, collision, timeout, decision count, physics "
        "ticks, and reward match. The source checkpoint hash was also verified unchanged.", "",
        "FRONT remains unresolved. Reflection of RIGHT objectives does not supply the overtaking/approach "
        "strategy that FRONT requires. These finite deterministic evaluations also do not establish "
        "performance on curved roads, more traffic, different lane counts, or different observation scales.", "",
        "## Videos and plot", "",
        "[RIGHT before and after, same scenario](videos/right_before_after.mp4) · "
        "[All four objectives](videos/all_goals.mp4) · [Outcome comparison plot](results.png)", "",
        "![Success before and after](results.png)", "",
        "The RIGHT comparison uses seed 10000: the original policy times out after 30 seconds; "
        "the adapter reaches the goal in 7.0 seconds. Videos show every 0.1-second physics tick at "
        "simulator speed. In the side-by-side comparison, the shorter video's final frame is held. "
        "The four-goal reel uses the first evaluation seed for each objective, without selecting outcomes.", "",
        "## Scope and reproduction", "",
        "This is an opt-in experiment in `ppo_singlecar_right_canonical_seed7`. "
        "The previous checkpoint and experiment remain available and unchanged. "
        "Loading the previous model normally keeps its original behavior; applying "
        "`RightCanonicalPolicy(model)` explicitly enables this experiment. No global environment "
        "default was changed.", "",
        "The adapter currently requires the tested three straight 4 m lanes, 12 m y normalization "
        "scales, five discrete actions, and 57-input committed-lane observation contract. "
        "Unsupported configurations are rejected.", "",
        "Implementation: [policy adapter](../../reflection_policy.py), "
        "[seven passing contract tests](../../test_reflection_policy.py), "
        "[experiment runner](../../experiment_right_canonical.py), "
        "[video recorder](../../record_reflection_experiment.py).", "",
        "Raw results: [evaluation summary](evaluation.json), "
        "[original-seed adapter episodes](evaluation_episodes.csv), "
        "[fresh original episodes](fresh_before_episodes.csv), "
        "[fresh adapter episodes](fresh_after_episodes.csv), [run configuration](run.json).", "",
        "Reproduce in a fresh output directory:", "", "```powershell",
        ".\\baseline_waypoint\\.venv\\Scripts\\python.exe -B -m baseline_dynamic_waypoint.experiment_right_canonical "
        "--run-name ppo_singlecar_right_canonical_seed7_repeat --fresh-episodes 200", "```", "",
    ]
    (RUN / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote EXPERIMENT_REPORT.md from verified CSV/JSON results.")


if __name__ == "__main__":
    main()
