"""Record fixed LEFT/RIGHT examples and all side-goal failures from the final model."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
VIDEOS = Path(__file__).resolve().parent
RUN = VIDEOS.parent
ROOT = Path(__file__).resolve().parents[4]
OUTPUT = VIDEOS / "left_right"
sys.path.insert(0, str(ROOT))

import imageio_ffmpeg
import torch
from baseline_dynamic_waypoint.evaluate import load_experiment
from record_experiment_episodes import record_episode


def concatenate(rows, name):
    playlist = OUTPUT / (name + "_playlist.txt")
    playlist.write_text("".join(f"file '{row['video']}'\n" for row in rows), encoding="utf-8")
    target = OUTPUT / (name + ".mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(target)], check=True)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(target))
    assert frames == sum(row["video_frames"] for row in rows)


def main():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "baseline_dynamic_waypoint/models" / (RUN.name + ".zip")
    with (RUN / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["slot"] in ("left", "right")]
    selected = []
    for goal in ("left", "right"):
        cases = [row for row in rows if row["slot"] == goal]
        selected += [(row, "first_two_evaluation_seeds") for row in cases[:2]]
        selected += [(row, "evaluation_failure") for row in cases[2:]
                     if int(row["timeout"]) or int(row["collision"])]
    model, env, metadata = load_experiment(model_path, render_mode="rgb_array")
    assert model.num_timesteps == metadata["actual_timesteps"] == 2_000_000
    env.demo_label = "Final 2M PPO / LEFT and RIGHT evaluation replay"
    title = "PPO | 2,000,000 steps | route progress + tracking | LEFT / RIGHT"
    results = []
    try:
        for number, (reference, selection) in enumerate(selected, 1):
            episode = record_episode(model, env, reference, OUTPUT, title, number, "ppo")
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(OUTPUT / episode["video"]))
            assert frames == episode["video_frames"]
            episode.update(selection=selection, verified_encoded_frames=frames,
                           encoded_seconds=seconds, initial_lane=int(reference["initial_lane"]),
                           goal_lane=int(reference["goal_lane"]))
            results.append(episode)
    finally:
        env.close()
    for goal in ("left", "right"):
        concatenate([row for row in results if row["slot"] == goal], goal + "_examples")
    concatenate(results, "left_and_right")
    report = {"model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "training_timesteps": 2_000_000, "deterministic": True,
              "selection": "First two evaluation seeds for each side goal, plus remaining failed side episodes.",
              "episodes": results}
    (OUTPUT / "episodes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# LEFT and RIGHT: final 2M route-progress model", "",
             "Final evaluation: LEFT 33/33 success; RIGHT 21/22 success (one timeout).", "",
             "The first two evaluation seeds for each goal are included, plus the single RIGHT timeout.",
             "Every replay is checked against its evaluation outcome, return, step count, and duration.",
             "All simulation ticks are recorded at 10 fps, with a two-second final-frame hold.", "",
             "[LEFT examples](left_examples.mp4) | [RIGHT examples](right_examples.mp4) | "
             "[Combined video](left_and_right.mp4)", "",
             "| Goal | Seed | Initial / goal lane | Outcome | Seconds | Video |",
             "| --- | ---: | --- | --- | ---: | --- |"]
    lines += [f"| {row['slot'].upper()} | {row['seed']} | {row['initial_lane']} / {row['goal_lane']} | "
              f"{row['outcome']} | {row['simulated_seconds']:.1f} | [Watch]({row['video']}) |"
              for row in results]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified LEFT/RIGHT videos: " + str(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
