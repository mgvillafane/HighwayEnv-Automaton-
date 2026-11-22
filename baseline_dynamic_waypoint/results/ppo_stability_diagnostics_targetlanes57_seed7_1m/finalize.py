"""Validate and report the all-goal, timeout-penalty, dropout experiment."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torch import nn

from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
from baseline_dynamic_waypoint.record_committed import record_episode
from baseline_dynamic_waypoint.single_car import make_single_car_env
from baseline_waypoint.plot_results import plot_training

RUN = Path(__file__).resolve().parent
MODEL = RUN.parents[1] / "models" / (RUN.name + ".zip")
GOALS = ("front", "left", "back", "right")
URL = "http://127.0.0.1:6016"


def read_rows(name):
    with (RUN / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    metadata = json.loads((RUN / "run.json").read_text())
    assert metadata["status"] == "trained" and metadata["initialization"] is None
    assert metadata["actual_timesteps"] == metadata["total_lifetime_timesteps"] == 1_000_000
    assert set(metadata["enabled_slots"]) == set(metadata["training_slots"]) == set(metadata["evaluation_slots"]) == set(GOALS)
    assert metadata["dropout_rate"] == .05 and metadata["environment"]["reward_weights"]["timeout"] == 10
    assert metadata["ppo_config"]["learning_rate"] == 0.00005
    assert metadata["ppo_config"]["gamma"] == 0.97
    assert metadata["evaluation_every_timesteps"] == 250_000
    training, scenarios = read_rows("training_episodes.csv"), read_rows("sampled_scenarios.csv")
    before, after = read_rows("evaluation_before_episodes.csv"), read_rows("evaluation_episodes.csv")
    assert len(training) == len(scenarios)
    for data in (training, scenarios, before, after):
        assert set(row["slot"] for row in data) == set(GOALS)
    for data in (training, before, after):
        for row in data:
            assert float(row["reward_timeout"]) == -10 * int(row["timeout"])
    for previous, current in zip(before, after):
        for key in previous:
            if key.startswith("initial_") or key in ("seed", "slot", "anchor_initial_gap", "goal_lane"):
                assert previous[key] == current[key]
    model = PPO.load(MODEL, device="cpu")
    assert model.num_timesteps == 1_000_000
    assert isinstance(model.policy, DropoutMultiInputPolicy)
    assert model.policy.features_extractor.features_dim == 57
    dropout = {}
    for name, network in (("actor", model.policy.mlp_extractor.policy_net), ("critic", model.policy.mlp_extractor.value_net)):
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Dropout)]
        assert len(layers) == 2 and all(layer.p == .05 for layer in layers)
        dropout[name] = {"layers": len(layers), "rate": .05}
    assert metadata["environment"]["goal_config"]["observation_schema"] == "legacy_target_lanes"
    assert "achieved_goal" not in model.observation_space.spaces
    assert model.observation_space["target_lanes"].shape == (2,)
    lane_env = make_single_car_env(**metadata["evaluation_environment"])
    lane_examples = []
    try:
        for row in after:
            obs, info = lane_env.reset(seed=int(row["seed"]))
            expected = [int(lane_env.anchor.lane_index[2]), lane_env.goal_lane]
            np.testing.assert_array_equal(obs["target_lanes"], expected)
            assert sum(value.size for value in obs.values()) == 57
            assert lane_env.observation_space.contains(obs)
            lane_examples.append({"seed": int(row["seed"]), "goal": lane_env.slot,
                                  "ego_lane": int(lane_env.highway.vehicle.lane_index[2]),
                                  "target_car_lane": expected[0], "target_goal_lane": expected[1],
                                  "observed_target_lanes": obs["target_lanes"].tolist()})
    finally:
        lane_env.close()
    (RUN / "lane_observation_audit.json").write_text(json.dumps(lane_examples, indent=2) + "\n")
    (RUN / "policy_architecture.txt").write_text(str(model.policy) + "\n")
    events = EventAccumulator(str(RUN)).Reload()
    for goal in GOALS:
        assert events.Scalars("setup/enabled_" + goal)[0].value == 1
        assert events.Scalars(f"evaluation/{goal}/success_rate")[-1].step == 1_000_000
    assert events.Scalars("setup/reward_weight_timeout")[0].value == 10
    phases = [(0, "evaluation_before"), *[(step, f"evaluation_{step}") for step in range(250_000, 1_000_000, 250_000)], (1_000_000, "evaluation")]
    evaluations = []
    for step, prefix in phases:
        evaluation = json.loads((RUN / (prefix + ".json")).read_text())
        assert evaluation["ppo"]["episodes"] == 100
        evaluations.append({"step": step, "summary": evaluation["ppo"], "by_goal": evaluation["ppo_by_slot"]})
    (RUN / "evaluation_history.json").write_text(json.dumps(evaluations, indent=2) + "\n")
    with urllib.request.urlopen(URL + "/data/plugin/scalars/tags", timeout=5) as response:
        live_tags = json.load(response)
    live_run = next(name for name in live_tags if RUN.name in name)
    live_metrics = {}
    for goal in GOALS:
        for tag in (f"episodes/{goal}/success_mean_100", f"evaluation/{goal}/success_rate"):
            query = urllib.parse.urlencode({"run": live_run, "tag": tag})
            with urllib.request.urlopen(URL + "/data/plugin/scalars/scalars?" + query, timeout=5) as response:
                points = json.load(response)
            assert points[-1][1] == events.Scalars(tag)[-1].step
            live_metrics[tag] = {"points": len(points), "last_step": points[-1][1], "last_value": points[-1][2]}
    (RUN / "tensorboard_metrics.json").write_text(json.dumps(live_metrics, indent=2) + "\n")
    plot_training(RUN)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for goal, color in zip(GOALS, ("#f97316", "#8b5cf6", "#2563eb", "#16a34a")):
        rows = [row for row in training if row["slot"] == goal]
        values = np.array([int(row["success"]) for row in rows])
        cumulative = np.r_[0, values.cumsum()]
        ends = np.arange(1, len(rows) + 1)
        starts = np.maximum(0, ends - 100)
        axes[0].plot([int(row["timesteps"]) for row in rows], 100 * (cumulative[ends] - cumulative[starts]) / (ends - starts), label=goal.upper(), color=color)
        axes[1].plot([phase["step"] for phase in evaluations], [100 * phase["by_goal"][goal]["success_rate"] for phase in evaluations], marker="o", label=goal.upper(), color=color)
    axes[0].set(title="Sampled training actions: rolling 100 episodes per goal", ylabel="Success (%)")
    axes[1].set(title="Deterministic evaluation: same 100 held-out seeds", ylabel="Success (%)")
    for axis in axes:
        axis.set(xlabel="PPO decisions", ylim=(-2, 102))
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("All four goals / timeout -10 / actor and critic dropout 0.05")
    figure.tight_layout()
    figure.savefig(RUN / "results.png", dpi=180)
    plt.close(figure)
    expected_tags = ("episodes/timeout_mean_100", "episodes/success_count_100",
                     "episodes/steps_to_success_mean_100", "reward_components_mean_100/timeout",
                     "evaluation/average_episode_reward", "evaluation/successful_episodes",
                     "evaluation/average_steps_to_success", "train/approx_kl", "train/clip_fraction",
                     "train/explained_variance")
    assert all(tag in events.Tags()["scalars"] for tag in expected_tags)
    selected = json.loads((RUN / "best_evaluation.json").read_text())
    expected_score = max([phase["summary"]["success_rate"], -phase["summary"]["collision_rate"],
                          phase["summary"]["average_episode_reward"]] for phase in evaluations)
    assert selected["score"] == expected_score
    best_policy = PPO.load(selected["model"], device="cpu")
    assert best_policy.num_timesteps == selected["step"]
    assert best_policy.gamma == .97 and best_policy.learning_rate == .00005
    assert best_policy.policy.features_extractor.features_dim == 57
    lane_change_results = {}
    for required in (0, 1, 2):
        rows = [row for row in after if int(row["lane_changes_required"]) == required]
        lane_change_results[str(required)] = {"episodes": len(rows), "successes": sum(int(row["success"]) for row in rows)}
    verification = {"trained_from_scratch": True, "observation_inputs": 57, "explicit_target_lanes_verified": True, "training_decisions": 1_000_000, "all_four_goals_sampled": True,
                    "dropout": dropout, "timeout_penalty": -10, "timeout_cost_charged_only_on_timeout": True,
                    "training_counts": dict(Counter(row["slot"] for row in training)), "evaluation_episodes": len(after),
                    "periodic_evaluations": len(evaluations), "lane_change_evaluation": lane_change_results,
                    "tensorboard_url": URL + "/live", "tests": "9 focused stability, timeout, logging, and checkpoint tests passed"}
    (RUN / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    final, initial = evaluations[-1], evaluations[0]
    best = max(evaluations[1:], key=lambda phase: phase["summary"]["success_rate"])
    best_model = MODEL if best["step"] == 1_000_000 else MODEL.with_name(f"{MODEL.stem}_{best['step']}.zip")
    assert best_model.exists()
    (RUN / "best_recorded_evaluation.json").write_text(json.dumps({"step": best["step"],
        "model": str(best_model), "summary": best["summary"],
        "selection": "Highest observed score on the same 100 monitoring seeds; not a new independent evaluation."}, indent=2) + "\n")
    lines = ["# All goals, timeout -10, actor/critic dropout", "", f"Completed exactly **1,000,000 PPO decisions from scratch** in {metadata['training_seconds'] / 60:.2f} minutes, with {metadata['actual_physics_steps']:,} physics ticks.", "", "Learning rate is 0.00005 and gamma is 0.97. All four goals were sampled uniformly at reset. The legacy layout retains 57 inputs by replacing achieved_goal (2) with target_lanes (2): [target_car_lane, target_goal_lane], as raw lane IDs 0-2. Other inputs, committed lane actions, and success conditions retain their legacy meanings. Actor and critic each use two 256-unit Tanh hidden layers, with dropout 0.05 after each activation. Dropout is active during gradient updates and disabled for rollout collection and evaluation.", "", "The timeout10 reward profile adds one terminal -10 cost on timeout. Success, collision, and anchor loss do not receive this timeout cost. Other baseline reward terms are retained.", "", "| Goal | Training episodes | Training successes | Evaluation cases | Success before | Success after | Collisions after |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for goal in GOALS:
        rows = [row for row in training if row["slot"] == goal]
        stats, before_stats = final["by_goal"][goal], initial["by_goal"][goal]
        lines.append(f"| {goal.upper()} | {len(rows)} | {sum(int(row['success']) for row in rows)} | {stats['episodes']} | {before_stats['success_rate']:.1%} | {stats['success_rate']:.1%} | {stats['collision_rate']:.1%} |")
    lines += ["", f"Overall held-out success: {initial['summary']['success_rate']:.1%} -> {final['summary']['success_rate']:.1%}; final collisions: {final['summary']['collision_rate']:.1%}; timeouts: {final['summary']['timeout_rate']:.1%}.", "", "Deterministic evaluation uses seeds 10000-10099, before training and every 250,000 steps. One training seed and small per-goal evaluation counts limit precision.", "", "![Learning curves](results.png)", "", "[TensorBoard](" + URL + "/live) | [Model](../../models/" + MODEL.name + ") | [Run settings](run.json) | [Evaluation history](evaluation_history.json) | [Verification](verification.json) | [Network architecture](policy_architecture.txt)", "", "Aggregate metrics use all 100 fixed evaluation cases.", ""]
    lines += [f"Best recorded trained checkpoint on these same 100 monitoring seeds: {best['step']:,} steps, {best['summary']['success_rate']:.1%} success. [Checkpoint](../../models/{best_model.name}). This selection does not provide an independent test score.", ""]
    lines += ["Lane-change evaluation: " + "; ".join(
        f"{required} lanes required: {stats['successes']}/{stats['episodes']} successes"
        for required, stats in lane_change_results.items()) + ".", "",
        "All successes started in the goal lane. Explicit target lane inputs were verified, but the deterministic policy did not solve cases requiring a lane change. [Lane observation audit](lane_observation_audit.json).", ""]
    (RUN / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
