"""Record fixed RIGHT/FRONT scenarios from this run's stopped checkpoint."""
import csv
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

from baseline_dynamic_waypoint.evaluate import evaluate_policy, load_experiment
from record_experiment_episodes import record_episode


def concatenate(paths, target):
    playlist = target.with_suffix(".txt")
    playlist.write_text(
        "".join(f"file '{path.name}'\n" for path in paths), encoding="utf-8"
    )
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-y", "-f", "concat", "-safe", "1", "-i", str(playlist),
         "-c", "copy", "-movflags", "+faststart", str(target)],
        check=True,
    )


def main():
    torch.set_num_threads(1)
    model_path = ROOT / "baseline_dynamic_waypoint/models" / (OUTPUT.parent.name + ".zip")
    model, env, metadata = load_experiment(model_path, render_mode="rgb_array")
    steps = metadata["actual_timesteps"]
    assert steps == model.num_timesteps
    selected = [("right", 10000), ("right", 10001), ("front", 10002), ("front", 10003)]
    references, episodes = [], []
    title = f"PPO | {steps:,} training steps | timeout=10 | dropout=0.05"
    env.demo_label = f"Stopped checkpoint: {steps:,} steps"
    try:
        for number, (slot, seed) in enumerate(selected, 1):
            reference = evaluate_policy(env, model, 1, seed, "ppo")[0]
            assert reference["slot"] == slot
            references.append(reference)
            # Save the measured reference before replaying it with rendering.
            with (OUTPUT / "selected_evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(reference))
                writer.writeheader()
                writer.writerows(references)
            episode = record_episode(model, env, reference, OUTPUT, title, number, "ppo")
            count, seconds = imageio_ffmpeg.count_frames_and_secs(str(OUTPUT / episode["video"]))
            assert count == episode["video_frames"]
            episode.update(
                verified_encoded_frames=count,
                encoded_seconds=seconds,
                source_checkpoint_steps=steps,
                selection="First two fixed evaluation seeds for this goal; outcomes not selected.",
            )
            episodes.append(episode)
    finally:
        env.close()
    for slot in ("right", "front"):
        concatenate([OUTPUT / row["video"] for row in episodes if row["slot"] == slot],
                    OUTPUT / f"{slot}_examples.mp4")
    concatenate([OUTPUT / row["video"] for row in episodes], OUTPUT / "right_and_front.mp4")
    (OUTPUT / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# RIGHT and FRONT videos at {steps:,} saved training steps", "",
        "Training was stopped at the user's request. These recordings use the latest saved",
        "checkpoint with direct deterministic PPO actions. Each video includes every",
        "0.1-second simulation tick and a two-second hold of the final frame.", "",
        "The first two evaluation seeds for each goal were used regardless of outcome.", "",
        "[RIGHT examples](right_examples.mp4) | [FRONT examples](front_examples.mp4) | "
        "[Combined video](right_and_front.mp4)", "",
        "| Goal | Seed | Outcome | Simulated seconds | Video |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    lines += [
        f"| {row['slot'].upper()} | {row['seed']} | {row['outcome']} | "
        f"{row['simulated_seconds']:.1f} | [Watch]({row['video']}) |" for row in episodes
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified videos saved to " + str(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
