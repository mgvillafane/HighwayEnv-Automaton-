"""Record the RIGHT-only reflection experiment without changing PPO weights."""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.record_committed import record_episode
from baseline_dynamic_waypoint.reflection_policy import RightCanonicalPolicy


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "ppo_singlecar_committed57_seed7_200k.zip"
DEFAULT_RUN = ROOT / "results" / "ppo_singlecar_right_canonical_seed7"


def paired_video(before, after, output, before_outcome, after_outcome):
    """Play both at physical speed, retaining the shorter final state."""
    with imageio.get_reader(str(before)) as left, imageio.get_reader(str(after)) as right:
        frames_left = [frame for frame in left]
        frames_right = [frame for frame in right]
        fps = float(left.get_meta_data()["fps"])
        assert fps == float(right.get_meta_data()["fps"])
    width = frames_left[0].shape[1]
    assert frames_right[0].shape == frames_left[0].shape
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    count = max(len(frames_left), len(frames_right))
    with imageio.get_writer(str(output), fps=fps, codec="libx264", macro_block_size=1,
                            output_params=["-movflags", "+faststart"]) as writer:
        for index in range(count):
            a = frames_left[min(index, len(frames_left) - 1)]
            b = frames_right[min(index, len(frames_right) - 1)]
            combined = np.concatenate([a, b], axis=1)
            frame = Image.new("RGB", (combined.shape[1], combined.shape[0] + 42), (17, 24, 38))
            frame.paste(Image.fromarray(combined), (0, 42))
            draw = ImageDraw.Draw(frame)
            draw.text((14, 8), f"BEFORE: original PPO | {before_outcome}", font=font, fill=(240, 210, 140))
            draw.text((width + 14, 8), f"AFTER: RIGHT reflection | {after_outcome}", font=font,
                      fill=(120, 245, 165))
            writer.append_data(np.asarray(frame))
    actual, _ = imageio_ffmpeg.count_frames_and_secs(str(output))
    assert actual == count
    return {"frames": actual, "fps": fps, "shorter_video_final_state_held": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--evaluation-csv", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    model, env, metadata = load_experiment(args.model, "rgb_array")
    policy = RightCanonicalPolicy(model)
    source_run = ROOT / "results" / args.model.stem
    output = args.run / "videos"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.evaluation_csv or args.run / "evaluation_episodes.csv"
    expected = {}
    if evaluation_path.exists():
        with evaluation_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {int(row["seed"]): row for row in rows}
    else:
        rows = []
        for seed in range(10000, 10100):
            _, initial = env.reset(seed=seed)
            rows.append({"seed": seed, "slot": initial["slot"]})
    results = []
    try:
        for slot in ("back", "front", "left", "right"):
            seed = int(next(row for row in rows if row["slot"] == slot)["seed"])
            env.demo_label = "PPO + RIGHT reflection (no retraining)" if slot == "right" else "Original PPO (unchanged)"
            # The shared recorder traces every internal physics tick and checks
            # completion before accepting another policy action.
            with contextlib.redirect_stdout(io.StringIO()):
                result = record_episode(policy, env, args.model, seed, output / slot)
            result.update(controller="trained_ppo_with_right_only_canonicalization",
                          source_model=str(args.model.resolve()),
                          policy_adapter="RightCanonicalPolicy",
                          reflected=slot == "right", retrained=False, reward_changed=False)
            if seed in expected:
                row = expected[seed]
                assert result["success"] == bool(int(row["success"]))
                assert result["collision"] == bool(int(row["collision"]))
                assert abs(result["reward"] - float(row["reward"])) < 1e-8
                assert result["policy_decisions"] == int(row["length"])
                result["matches_evaluation"] = True
            frame_count, _ = imageio_ffmpeg.count_frames_and_secs(str(output / slot / "episode.mp4"))
            assert frame_count == result["physics_steps"] + 1 + 2 * env.metadata["render_fps"]
            result["verified_video_frames"] = frame_count
            (output / slot / "episode.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("trace", "decisions")}), flush=True)
    finally:
        env.close()
    playlist = output / "playlist.txt"
    playlist.write_text("".join(f"file '{row['slot']}/episode.mp4'\n" for row in results), encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(output / "all_goals.mp4")], check=True)
    right_result = next(row for row in results if row["slot"] == "right")
    baseline_path = source_run / "videos" / "right" / "episode.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["seed"] == right_result["seed"]
    comparison = paired_video(source_run / "videos" / "right" / "episode.mp4",
                              output / "right" / "episode.mp4", output / "right_before_after.mp4",
                              baseline["outcome"], right_result["outcome"])
    (output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({"episodes": len(results),
        "evaluation_csv": str(evaluation_path), "all_physics_ticks_recorded": True,
        "reward_changed": False, "retrained": False, "comparison": comparison}, indent=2) + "\n", encoding="utf-8")
    lines = ["# RIGHT reflection experiment videos", "", "Existing PPO weights; only RIGHT objectives use reflected observations/actions. Reward unchanged.", "",
             "Videos run at simulator speed and include every 0.1-second physics tick.", "",
             "[RIGHT before and after, same seed](right_before_after.mp4) · [All four goals](all_goals.mp4)", "",
             "| Goal | Seed | Outcome | Seconds |", "| --- | ---: | --- | ---: |"]
    lines += [f"| [{row['slot'].upper()}]({row['slot']}/episode.mp4) | {row['seed']} | {row['outcome']} | {row['seconds']:.1f} |" for row in results]
    (output / "SCENARIOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Videos saved to {output}", flush=True)


if __name__ == "__main__":
    main()
