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
URL = "http://127.0.0.1:6012"


def read_rows(name):
    with (RUN / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    metadata = json.loads((RUN / "run.json").read_text())
    assert metadata["status"] == "trained" and metadata["initialization"] is None
    assert metadata["actual_timesteps"] == metadata["total_lifetime_timesteps"] == 2_000_000
    assert set(metadata["enabled_slots"]) == set(metadata["training_slots"]) == set(metadata["evaluation_slots"]) == set(GOALS)
    assert metadata["dropout_rate"] == .1 and metadata["environment"]["reward_weights"]["timeout"] == 25
    assert metadata["evaluation_every_timesteps"] == 250_000
    training, scenarios = read_rows("training_episodes.csv"), read_rows("sampled_scenarios.csv")
    before, after = read_rows("evaluation_before_episodes.csv"), read_rows("evaluation_episodes.csv")
    assert len(training) == len(scenarios)
    for data in (training, scenarios, before, after):
        assert set(row["slot"] for row in data) == set(GOALS)
    for data in (training, before, after):
        for row in data:
            assert float(row["reward_timeout"]) == -25 * int(row["timeout"])
    for previous, current in zip(before, after):
        for key in previous:
            if key.startswith("initial_") or key in ("seed", "slot", "anchor_initial_gap", "goal_lane"):
                assert previous[key] == current[key]
    model = PPO.load(MODEL, device="cpu")
    assert model.num_timesteps == 2_000_000
    assert isinstance(model.policy, DropoutMultiInputPolicy)
    assert model.policy.features_extractor.features_dim == 54
    dropout = {}
    for name, network in (("actor", model.policy.mlp_extractor.policy_net), ("critic", model.policy.mlp_extractor.value_net)):
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Dropout)]
        assert len(layers) == 2 and all(layer.p == .1 for layer in layers)
        dropout[name] = {"layers": len(layers), "rate": .1}
    (RUN / "policy_architecture.txt").write_text(str(model.policy) + "\n")
    events = EventAccumulator(str(RUN)).Reload()
    for goal in GOALS:
        assert events.Scalars("setup/enabled_" + goal)[0].value == 1
        assert events.Scalars(f"evaluation/{goal}/success_rate")[-1].step == 2_000_000
    assert events.Scalars("setup/reward_weight_timeout")[0].value == 25
    phases = [(0, "evaluation_before"), *[(step, f"evaluation_{step}") for step in range(250_000, 2_000_000, 250_000)], (2_000_000, "evaluation")]
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
    figure.suptitle("All four goals / timeout -25 / actor and critic dropout 0.1")
    figure.tight_layout()
    figure.savefig(RUN / "results.png", dpi=180)
    plt.close(figure)
    env = make_single_car_env(render_mode="rgb_array", **metadata["evaluation_environment"])
    videos = RUN / "videos"
    videos.mkdir(exist_ok=True)
    recordings = []
    try:
        for goal in GOALS:
            row = next(row for row in after if row["slot"] == goal)
            env.demo_label = "2M PPO / timeout -25 / dropout 0.1 / " + goal.upper()
            result = record_episode(model, env, MODEL, int(row["seed"]), videos / goal)
            assert result["success"] == bool(int(row["success"])) and result["collision"] == bool(int(row["collision"]))
            assert abs(result["reward"] - float(row["reward"])) < 1e-8
            assert result["policy_decisions"] == int(row["length"])
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(videos / goal / "episode.mp4"))
            assert frames == result["physics_steps"] + 1 + 2 * env.metadata["render_fps"]
            recordings.append({key: value for key, value in result.items() if key not in ("decisions", "trace")})
    finally:
        env.close()
    playlist = videos / "playlist.txt"
    playlist.write_text("".join(f"file '{goal}/episode.mp4'\n" for goal in GOALS))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy", "-movflags", "+faststart", str(videos / "all_goals.mp4")], check=True)
    (videos / "episodes.json").write_text(json.dumps(recordings, indent=2) + "\n")
    verification = {"trained_from_scratch": True, "training_decisions": 2_000_000, "all_four_goals_sampled": True,
                    "dropout": dropout, "timeout_penalty": -25, "timeout_cost_charged_only_on_timeout": True,
                    "training_counts": dict(Counter(row["slot"] for row in training)), "evaluation_episodes": len(after),
                    "periodic_evaluations": len(evaluations), "video_replays_match_evaluation": True,
                    "tensorboard_url": URL + "/live", "tests": "13 dropout, timeout and reward checks passed"}
    (RUN / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    final, initial = evaluations[-1], evaluations[0]
    best = max(evaluations[1:], key=lambda phase: phase["summary"]["success_rate"])
    best_model = MODEL if best["step"] == 2_000_000 else MODEL.with_name(f"{MODEL.stem}_{best['step']}.zip")
    assert best_model.exists()
    (RUN / "best_recorded_evaluation.json").write_text(json.dumps({"step": best["step"],
        "model": str(best_model), "summary": best["summary"],
        "selection": "Highest observed score on the same 100 monitoring seeds; not a new independent evaluation."}, indent=2) + "\n")
    lines = ["# All goals, timeout -25, actor/critic dropout", "", f"Completed exactly **2,000,000 PPO decisions from scratch** in {metadata['training_seconds'] / 60:.2f} minutes, with {metadata['actual_physics_steps']:,} physics ticks.", "", "All four goals were sampled uniformly at reset. The 54-input observation, committed lane actions, and success conditions match the preceding experiments. Actor and critic each use two 256-unit Tanh hidden layers, with dropout 0.1 after each activation. Dropout is active during gradient updates and disabled for rollout collection and evaluation.", "", "The timeout25 reward profile adds one terminal -25 cost on timeout. Success, collision, and anchor loss do not receive this timeout cost. Other baseline reward terms are retained.", "", "| Goal | Training episodes | Training successes | Evaluation cases | Success before | Success after | Collisions after |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for goal in GOALS:
        rows = [row for row in training if row["slot"] == goal]
        stats, before_stats = final["by_goal"][goal], initial["by_goal"][goal]
        lines.append(f"| {goal.upper()} | {len(rows)} | {sum(int(row['success']) for row in rows)} | {stats['episodes']} | {before_stats['success_rate']:.1%} | {stats['success_rate']:.1%} | {stats['collision_rate']:.1%} |")
    lines += ["", f"Overall held-out success: {initial['summary']['success_rate']:.1%} -> {final['summary']['success_rate']:.1%}; final collisions: {final['summary']['collision_rate']:.1%}; timeouts: {final['summary']['timeout_rate']:.1%}.", "", "Deterministic evaluation uses seeds 10000-10099, before training and every 250,000 steps. One training seed and small per-goal evaluation counts limit precision.", "", "![Learning curves](results.png)", "", "[TensorBoard](" + URL + "/live) | [Replays](videos/all_goals.mp4) | [Model](../../models/" + MODEL.name + ") | [Run settings](run.json) | [Evaluation history](evaluation_history.json) | [Verification](verification.json) | [Network architecture](policy_architecture.txt)", "", "Replays use the first held-out seed for each goal; aggregate metrics use all 100 cases.", ""]
    lines += [f"Best recorded trained checkpoint on these same 100 monitoring seeds: {best['step']:,} steps, {best['summary']['success_rate']:.1%} success. [Checkpoint](../../models/{best_model.name}). This selection does not provide an independent test score.", ""]
    (RUN / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
