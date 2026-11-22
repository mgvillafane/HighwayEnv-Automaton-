"""Verify, compare and record the 54-input front/right continuation."""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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

from baseline_dynamic_waypoint.record_committed import record_episode
from baseline_dynamic_waypoint.single_car import make_single_car_env
from baseline_waypoint.plot_results import plot_training

RUN = Path(__file__).resolve().parent
MODEL = RUN.parents[1] / "models" / f"{RUN.name}.zip"
SLOTS = ("front", "right", "back", "left")


def read_rows(name):
    with (RUN / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rate(rows, key="success"):
    return sum(int(row[key]) for row in rows) / len(rows)


def goal_rows(rows, slot):
    return [row for row in rows if row["slot"] == slot]


def main():
    torch.set_num_threads(1)
    metadata = json.loads((RUN / "run.json").read_text())
    assert metadata["status"] == "trained"
    assert metadata["actual_timesteps"] == metadata["requested_timesteps"] == 200_000
    assert metadata == json.loads(MODEL.with_suffix(".json").read_text())
    source = Path(metadata["initialization"]["model"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["initialization"]["sha256"]
    source_metadata = json.loads(source.with_suffix(".json").read_text())
    original_environment = deepcopy(metadata["environment"])
    assert original_environment["goal_config"].pop("sampled_slots") == ["right", "front"]
    assert original_environment == metadata["evaluation_environment"]
    assert original_environment["goal_config"].pop("observation_schema") == "lane_goal_v2"
    assert original_environment == source_metadata["environment"]
    assert metadata["initialization"]["kind"] == "observation_schema_weight_transfer"
    assert metadata["initialization"]["optimizer_reset"] is True
    assert metadata["initialization"]["policy_outputs_preserved"] is False

    before = read_rows("evaluation_before_episodes.csv")
    after = read_rows("evaluation_episodes.csv")
    assert len(before) == len(after) == 100
    for previous, current in zip(before, after):
        for key in previous:
            if key.startswith("initial_") or key in ("seed", "slot", "anchor_initial_gap", "goal_lane", "lane_changes_required"):
                assert previous[key] == current[key], (key, previous, current)
    with (RUN.parent / source.stem / "evaluation_episodes.csv").open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    assert len(source_rows) == len(before)
    for previous, cached in zip(before, source_rows):
        for key in ("seed", "slot", "anchor_initial_gap", "initial_lane", "goal_lane", "initial_distance"):
            assert previous[key] == cached[key], (key, previous, cached)

    training = read_rows("training_episodes.csv")
    scenarios = read_rows("sampled_scenarios.csv")
    counts = Counter(row["slot"] for row in scenarios)
    assert set(counts) == {"front", "right"}
    assert len(scenarios) == len(training)
    assert all(a["slot"] == b["slot"] and a["timesteps"] == b["timesteps"] for a, b in zip(training, scenarios))
    assert all(12 <= float(row["traffic_speed"]) <= 18 for row in scenarios)
    assert all(int(row["traffic_lane"]) < 2 for row in scenarios if row["slot"] == "right")
    model = PPO.load(MODEL, device="cpu")
    assert model.num_timesteps == 200_000
    assert model.policy.features_extractor.features_dim == 54
    assert set(model.observation_space.spaces) == {"vehicles", "goal_delta", "anchor", "lane_difference", "desired_goal", "hold_progress"}
    assert model.observation_space["vehicles"].shape == (7, 6)
    assert model.observation_space["desired_goal"].shape == (3,)
    events = EventAccumulator(str(RUN)).Reload()
    assert events.Scalars("evaluation/success_rate")[-1].step == 200_000
    plot_training(RUN)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for rows, offset, label, color in ((source_rows, -.25, "Original 57-input policy", "#94a3b8"),
                                       (before, 0, "54 inputs, before training", "#f59e0b"),
                                       (after, .25, "54 inputs, after 200,000 steps", "#2563eb")):
        values = [100 * rate(goal_rows(rows, slot)) for slot in SLOTS]
        bars = axes[0].bar(np.arange(4) + offset, values, width=.24, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=8)
    axes[0].set(xticks=np.arange(4), xticklabels=[slot.upper() for slot in SLOTS], ylim=(0, 115), ylabel="Success (%)", title="Same 100 held-out scenarios")
    axes[0].legend(loc="upper left", fontsize=8)
    for slot, color in (("front", "#f97316"), ("right", "#16a34a")):
        rows = goal_rows(training, slot)
        successes = np.array([int(row["success"]) for row in rows])
        cumulative = np.r_[0, successes.cumsum()]
        ends = np.arange(1, len(rows) + 1)
        starts = np.maximum(0, ends - 100)
        axes[1].plot([int(row["timesteps"]) for row in rows], 100 * (cumulative[ends] - cumulative[starts]) / (ends - starts), label=slot.upper(), color=color)
    axes[1].set(xlabel="Additional PPO decisions", ylabel="Training success (%)", ylim=(-2, 102), title="Rolling success: up to 100 episodes per goal")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("54-input observation: target-car-relative goal and lane information")
    figure.tight_layout()
    figure.savefig(RUN / "results.png", dpi=180)
    plt.close(figure)

    videos = RUN / "videos"
    videos.mkdir(exist_ok=True)
    env = make_single_car_env(render_mode="rgb_array", **metadata["evaluation_environment"])
    recordings = []
    try:
        for slot in SLOTS:
            row = goal_rows(after, slot)[0]
            env.demo_label = f"54-input PPO / front-right training / {slot.upper()} / first evaluation seed"
            result = record_episode(model, env, MODEL, int(row["seed"]), videos / slot)
            assert result["slot"] == slot
            assert result["success"] == bool(int(row["success"]))
            assert result["collision"] == bool(int(row["collision"]))
            assert abs(result["reward"] - float(row["reward"])) < 1e-8
            assert result["policy_decisions"] == int(row["length"])
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(videos / slot / "episode.mp4"))
            assert frames == result["physics_steps"] + 1 + 2 * env.metadata["render_fps"]
            recordings.append({key: value for key, value in result.items() if key not in ("decisions", "trace")})
    finally:
        env.close()
    playlist = videos / "playlist.txt"
    playlist.write_text("".join(f"file '{slot}/episode.mp4'\n" for slot in SLOTS))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy", "-movflags", "+faststart", str(videos / "all_goals.mp4")], check=True)
    (videos / "episodes.json").write_text(json.dumps(recordings, indent=2) + "\n")
    verification = {"exact_additional_decisions": model.num_timesteps, "source_checkpoint_unchanged": True,
                    "observation_schema": "lane_goal_v2", "input_values": 54,
                    "initialization": "Partial transfer of retained weights; optimizer reset; removed/redefined inputs omitted",
                    "physics_rewards_and_success_conditions_match_source": True,
                    "source_and_transferred_initial_scenarios_identical": True,
                    "training_episode_counts": dict(counts),
                    "training_successes": {slot: sum(int(row['success']) for row in goal_rows(training, slot)) for slot in ("front", "right")},
                    "before_after_initial_scenarios_identical": True, "evaluation_episodes": len(after),
                    "tensorboard_final_step": 200_000, "video_replays_match_evaluation": True}
    (RUN / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    lines = ["# 54-input front/right experiment", "", f"Initialized from `{source.name}` by transferring compatible input weights and the remaining network layers. The optimizer was reset. Removed/redefined input columns were omitted and new input connections started at zero, so this changes the policy immediately before training.", "", f"Completed exactly **200,000 additional PPO decisions**, covering {metadata['actual_physics_steps']:,} physics ticks in {metadata['training_seconds'] / 60:.2f} minutes.", "", "The observation has six fields: vehicles (7x6, including lane ID), goal_delta (2), anchor (4), lane_difference (2), desired_goal (3: target car ID plus target-car-relative x/y), and hold_progress (1). Relative goal x/y are divided by [200, 12].", "", "Training sampled RIGHT and FRONT uniformly at reset. Physics, rewards, traffic speeds, committed actions, and success rules match the source checkpoint. Evaluation samples all four goals.", "", f"Completed training episodes: FRONT {counts['front']}, RIGHT {counts['right']}; BACK 0, LEFT 0. Episode counts need not be exactly equal.", "", "Deterministic evaluation uses the same 100 held-out seeds (10000-10099), with identical initial scenarios before and after. The before column measures the transferred 54-input policy, rather than the original 57-input policy.", "", "| Goal | Episodes | Original 57-input policy | Transferred 54-input before | 54-input after 200k | Collisions after | Timeouts after |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for slot in SLOTS:
        old, new = goal_rows(before, slot), goal_rows(after, slot)
        lines.append(f"| {slot.upper()} | {len(new)} | {rate(goal_rows(source_rows, slot)):.1%} | {rate(old):.1%} | {rate(new):.1%} | {rate(new, 'collision'):.1%} | {rate(new, 'timeout'):.1%} |")
    lines += ["", f"Overall success: {rate(before):.1%} -> {rate(after):.1%}. Collisions: {rate(before, 'collision'):.1%} -> {rate(after, 'collision'):.1%}.", "", "A single training seed and small per-goal evaluation counts limit precision.", "", "![Results](results.png)", "", "[Watch all four goals](videos/all_goals.mp4). Videos use the first evaluation seed for each goal, with every physics tick retained; outcomes, rewards, lengths, and frame counts were verified.", "", "| Goal | Seed | Outcome | Video |", "| --- | ---: | --- | --- |"]
    lines += [f"| {row['slot'].upper()} | {row['seed']} | {row['outcome']} | [Watch](videos/{row['slot']}/episode.mp4) |" for row in recordings]
    previous_run = RUN.parent / "ppo_singlecar_front_right_seed7_200k" / "evaluation.json"
    previous_stats = json.loads(previous_run.read_text())["ppo"]
    comparison = {"original_57_input_policy": json.loads((RUN.parent / source.stem / "evaluation.json").read_text())["ppo"],
                  "transferred_54_input_before": json.loads((RUN / "evaluation_before.json").read_text())["ppo"],
                  "trained_54_input_after": json.loads((RUN / "evaluation.json").read_text())["ppo"],
                  "previous_57_input_front_right_200k": previous_stats,
                  "caution": "Schema change also removes/redefines inputs and resets the optimizer; this is not an isolated test of adding lanes."}
    (RUN / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines += ["", f"The previous 57-input front/right-only 200k continuation achieved {previous_stats['success_rate']:.1%} overall success. Its optimizer and inputs were retained; this experiment changed inputs and reset the optimizer, so that comparison does not isolate the effect of lane features.", "", "[Model](../../models/" + MODEL.name + ") | [Run settings](run.json) | [Evaluation](evaluation.json) | [Training curves](training_curves.png) | [Verification](verification.json) | [Comparison](comparison.json)", "", "Repeat with a fresh run name:", "", "```powershell", ".\\baseline_waypoint\\.venv\\Scripts\\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema lane_goal_v2 --training-slots right front --timesteps 200000 --num-envs 4 --rollout-steps 250 --seed 7 --init-model baseline_dynamic_waypoint/models/ppo_singlecar_committed57_seed7_200k.zip --reward-profile baseline --run-name YOUR_NEW_RUN --eval-episodes 100 --eval-seed 10000 --quiet", "```", ""]
    (RUN / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)
    print("Report, plots and videos saved: " + str(RUN), flush=True)


if __name__ == "__main__":
    main()
