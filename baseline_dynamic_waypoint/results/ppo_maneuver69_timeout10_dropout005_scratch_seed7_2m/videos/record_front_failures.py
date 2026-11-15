"""Replay every failed FRONT episode from the final 2M-step evaluation."""
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
OUTPUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import imageio_ffmpeg
import torch
from baseline_dynamic_waypoint.evaluate import load_experiment
from record_experiment_episodes import record_episode


def main():
    torch.set_num_threads(1)
    run = OUTPUT.parent
    model_path = ROOT / "baseline_dynamic_waypoint/models" / (run.name + ".zip")
    with (run / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        front = [row for row in csv.DictReader(handle) if row["slot"] == "front"]
    selected = [row for row in front if int(row["timeout"]) or int(row["collision"])]
    if not selected:
        raise ValueError("No failed FRONT evaluation episodes exist.")
    model, env, metadata = load_experiment(model_path, render_mode="rgb_array")
    assert metadata["actual_timesteps"] == model.num_timesteps == 2_000_000
    env.demo_label = "Final 2M PPO / FRONT failure replay"
    title = "PPO | 2,000,000 training steps | maneuver guidance | FRONT failures"
    episodes = []
    try:
        for number, reference in enumerate(selected, 1):
            episode = record_episode(model, env, reference, OUTPUT, title, number, "ppo")
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(OUTPUT / episode["video"]))
            assert frames == episode["video_frames"]
            episode.update(verified_encoded_frames=frames, video_seconds=seconds)
            episodes.append(episode)
    finally:
        env.close()
    playlist = OUTPUT / "front_failures_playlist.txt"
    playlist.write_text("".join(f"file '{row['video']}'\n" for row in episodes), encoding="utf-8")
    combined = OUTPUT / "front_failures.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(combined)], check=True)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(combined))
    assert frames == sum(row["video_frames"] for row in episodes)
    report = {"model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "training_timesteps": 2_000_000, "front_evaluation_episodes": len(front),
              "front_timeouts": sum(int(row["timeout"]) for row in front),
              "front_collisions": sum(int(row["collision"]) for row in front),
              "selection": "All FRONT timeouts and collisions from the final evaluation.",
              "combined_video": combined.name, "episodes": episodes}
    (OUTPUT / "front_failures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final 2M-step FRONT failures", "",
             f"Final evaluation: {len(front)} FRONT episodes; {report['front_timeouts']} timeouts; "
             f"{report['front_collisions']} collisions.", "",
             "Every failed FRONT evaluation episode is included. Replays use the final PPO",
             "directly and verify outcome, return, duration, steps, and encoded frame count.",
             "Videos play at simulation speed with a two-second final-frame hold.", "",
             "[Watch all FRONT failures](front_failures.mp4)", "",
             "| Seed | Outcome | Seconds | Video |", "| ---: | --- | ---: | --- |"]
    lines += [f"| {row['seed']} | {row['outcome']} | {row['simulated_seconds']:.1f} | "
              f"[Watch]({row['video']}) |" for row in episodes]
    (OUTPUT / "FRONT_FAILURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified all FRONT failure recordings: " + str(combined), flush=True)


if __name__ == "__main__":
    main()
