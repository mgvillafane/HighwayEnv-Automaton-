"""Replay saved evaluation episodes and make full-episode and collision videos.

Run with baseline_waypoint/.venv/Scripts/python.exe -B record_experiment_episodes.py.
Videos live in each run's results/videos directory. Selected examples deliberately
include failures; the collision rates always come from the complete evaluation.
"""
import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    ("baseline_DQN", "dqn_quickstart_gamma09_seed2_200k", "dqn"),
    ("baseline_DQN", "dqn_quickstart_gamma095_seed2_200k_fast", "dqn"),
    ("baseline_dynamic_waypoint", "ppo_front20m_current_seed7_1m", "ppo"),
    ("baseline_dynamic_waypoint", "ppo_front20m_gamma099_seed7_1m", "ppo"),
)
FONT_PATH = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
FONT = ImageFont.truetype(str(FONT_PATH), 21) if FONT_PATH.exists() else ImageFont.load_default()


def outcome(row):
    if int(row["collision"]):
        return "COLLISION"
    if int(row.get("success", 0)):
        return "GOAL REACHED"
    if int(row.get("completed", 0)):
        return "COMPLETED"
    return "TIMED OUT" if int(row.get("timeout", 0)) else "ANCHOR LOST"


def select_examples(rows):
    crashes = [row for row in rows if int(row["collision"])]
    safe = [row for row in rows if not int(row["collision"])]
    selected_crashes, selected_safe = [], []
    # Show a collision for every moving-slot type when available.
    for slot in ("front", "back", "left", "right"):
        match = next((row for row in crashes if row.get("slot") == slot), None)
        if match is not None:
            selected_crashes.append(match)
    # Native DQN: include both early and late crashes.
    for index in (0, -1, len(crashes) // 2, 1, 2, 3):
        if crashes and len(selected_crashes) < 4 and crashes[index % len(crashes)] not in selected_crashes:
            selected_crashes.append(crashes[index % len(crashes)])
    for slot in ("front", "back", "left", "right"):
        match = next((row for row in safe if row.get("slot") == slot and int(row.get("success", 0))), None)
        if match is not None:
            selected_safe.append(match)
    # Include a timeout so avoiding a crash isn't confused with reaching the goal.
    timeout = next((row for row in safe if int(row.get("timeout", 0))), None)
    if timeout is not None and len(selected_safe) < 4:
        selected_safe.append(timeout)
    for row in safe:
        if len(selected_safe) < 4 and row not in selected_safe:
            selected_safe.append(row)
    return sorted(selected_crashes + selected_safe, key=lambda row: int(row["seed"]))


def annotate(frame, title, caption, status):
    frame = Image.fromarray(frame)
    canvas = Image.new("RGB", (frame.width, frame.height + 64), (15, 23, 38))
    canvas.paste(frame, (0, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 3), title, font=FONT, fill=(235, 240, 250))
    draw.text((14, 32), f"{caption}  |  {status}", font=FONT,
              fill=(255, 105, 105) if status == "COLLISION" else (145, 220, 250))
    return np.asarray(canvas).copy()


def writer(path, fps):
    return imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                              quality=8, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])


def concatenate(directory, names, target):
    playlist = directory / f"{Path(target).stem}_playlist.txt"
    playlist.write_text("".join(f"file '{name}'\n" for name in names), encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(directory / target)], check=True)


def record_episode(model, env, reference, output, title, number, policy):
    seed = int(reference["seed"])
    observation, initial = env.reset(seed=seed)
    fps = int(env.highway.config["simulation_frequency"])
    if policy == "ppo" and fps != int(env.env_config["policy_frequency"]):
        raise ValueError("PPO recording expects matching simulation and policy frequencies.")
    caption = f"Episode {number:02d}  |  seed {seed}"
    if "slot" in initial:
        caption += f"  |  target {initial['slot'].upper()}"
    stem = f"episode_{number:02d}_seed{seed}_{outcome(reference).lower().replace(' ', '_')}"
    recent = deque(maxlen=6 * fps)
    total_reward, steps, frame_count = 0.0, 0, 0
    env.render()
    viewer = env.highway.viewer
    # GoalViewer disables drawing for SDL dummy by default; it can draw offscreen.
    viewer.enabled = True
    env.render()

    with writer(output / f"{stem}.mp4", fps) as video:
        def capture(frame):
            nonlocal frame_count
            annotated = annotate(frame, title, caption, env.status)
            video.append_data(annotated)
            recent.append(annotated)
            frame_count += 1
            return annotated

        start = capture(viewer.get_image())
        imageio.imwrite(output / f"{stem}_start.png", start)
        if policy == "dqn":
            viewer.frame_callback = capture
        try:
            terminated = truncated = False
            while not (terminated or truncated):
                action = int(model.predict(observation, deterministic=True)[0])
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                env.render()
                if policy == "ppo":
                    capture(viewer.get_image())
        finally:
            if policy == "dqn":
                viewer.frame_callback = None
        final = annotate(viewer.get_image(), title, caption, env.status)
        imageio.imwrite(output / f"{stem}_final.png", final)
        for _ in range(2 * fps):
            video.append_data(final)
            frame_count += 1

    # Recording must reproduce the saved evaluation, including reward and length.
    matches = (bool(info["is_collision"]) == bool(int(reference["collision"]))
               and steps == int(reference["length"])
               and math.isclose(total_reward, float(reference["reward"]), rel_tol=1e-6, abs_tol=1e-5)
               and math.isclose(float(info["elapsed_time"]), float(reference["elapsed_time"]), abs_tol=1e-6)
               and all(bool(info[key]) == bool(int(reference[column]))
                       for key, column in (("is_success", "success"), ("is_timeout", "timeout"),
                                           ("is_completed", "completed"), ("is_anchor_lost", "anchor_lost"))
                       if column in reference))
    if not matches:
        raise RuntimeError(f"Recorded seed {seed} differs from the saved evaluation: {info}")
    expected_frames = 1 + round(float(info["elapsed_time"]) * fps) + 2 * fps
    if frame_count != expected_frames:
        raise RuntimeError(f"Seed {seed}: expected {expected_frames} frames, recorded {frame_count}.")
    result = {"episode": number, "seed": seed, "slot": reference.get("slot"),
              "outcome": outcome(reference), "collision": bool(info["is_collision"]),
              "reward": total_reward, "simulated_seconds": float(info["elapsed_time"]),
              "policy_steps": steps, "fps": fps, "video_frames": frame_count,
              "video": f"{stem}.mp4", "final_image": f"{stem}_final.png",
              "matches_saved_evaluation": matches}
    if result["collision"]:
        highlight = f"collision_{number:02d}_seed{seed}_slow.mp4"
        with writer(output / highlight, fps) as video:
            for frame in recent:
                video.append_data(frame)
                video.append_data(frame)  # Half-speed replay of the actual last six seconds.
            for _ in range(2 * fps):
                video.append_data(final)
        result["collision_video"] = highlight
    print(json.dumps(result), flush=True)
    return result


def make_contact_sheet(output, episodes):
    crashes = [row for row in episodes if row["collision"]]
    sheet = Image.new("RGB", (1200, math.ceil(len(crashes) / 2) * 246), (15, 23, 38))
    for index, row in enumerate(crashes):
        with Image.open(output / row["final_image"]) as frame:
            sheet.paste(frame.resize((600, 212)), ((index % 2) * 600, (index // 2) * 246))
        draw = ImageDraw.Draw(sheet)
        draw.text(((index % 2) * 600 + 10, (index // 2) * 246 + 215),
                  f"seed {row['seed']} | {row.get('slot') or 'native traffic'} | crash at {row['simulated_seconds']:.1f}s",
                  font=FONT, fill=(255, 140, 140))
    sheet.save(output / "collision_contact_sheet.jpg", quality=94)


def record_experiment(baseline, name, policy):
    run_dir = ROOT / baseline / "results" / name
    output = run_dir / "videos"
    output.mkdir(exist_ok=True)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    with (run_dir / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == policy]
    stats = evaluation[policy]
    assert len(rows) == stats["episodes"]
    crash_count = sum(int(row["collision"]) for row in rows)
    assert math.isclose(crash_count / len(rows), stats["collision_rate"])
    model_path = ROOT / baseline / "models" / f"{name}.zip"
    gamma = run[f"{policy}_config"]["gamma"]
    title = f"{policy.upper()}  |  gamma={gamma:g}  |  {run['actual_timesteps']:,} training steps  |  {name}"
    if policy == "dqn":
        from baseline_DQN.evaluate import load_experiment
    else:
        from baseline_dynamic_waypoint.evaluate import load_experiment
    model, env, metadata = load_experiment(model_path, render_mode="rgb_array")
    assert metadata["actual_timesteps"] == run["actual_timesteps"]
    episodes = []
    try:
        for number, reference in enumerate(select_examples(rows), 1):
            episodes.append(record_episode(model, env, reference, output, title, number, policy))
    finally:
        env.close()
    concatenate(output, [row["video"] for row in episodes], "all_episodes.mp4")
    concatenate(output, [row["collision_video"] for row in episodes if row["collision"]], "collisions_only.mp4")
    make_contact_sheet(output, episodes)
    with (output / "collisions.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(row for row in rows if int(row["collision"]))
    manifest = {"experiment": name, "model": str(model_path.relative_to(ROOT)), "gamma": gamma,
                "training_steps": run["actual_timesteps"], "evaluation_episodes": len(rows),
                "evaluation_collisions": crash_count, "evaluation_collision_rate": crash_count / len(rows),
                "selection": "Four collision examples and four non-collision examples, selected from the full evaluation; not a rate estimate.",
                "collision_replay": "Last six simulated seconds at half speed, then a two-second final-frame hold.",
                "episodes": episodes}
    (output / "episodes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {name}", "", f"Final {policy.upper()} model, gamma {gamma:g}, {run['actual_timesteps']:,} training steps.", "",
             f"Full saved evaluation: **{crash_count}/{len(rows)} collisions ({crash_count / len(rows):.0%})**.", "",
             "[Watch all eight episodes](all_episodes.mp4) | [Watch collision replays](collisions_only.mp4)", "",
             "Collision replays show the last six seconds at half speed and hold the terminal frame for two seconds.", "",
             "These examples deliberately include four crashes. Use the full 50-episode evaluation above for the collision rate.", "",
             "Every recording reproduced its saved evaluation outcome, reward, duration and policy-step count.", "",
             "| Episode | Seed | Slot | Outcome | Simulated seconds | Video | Collision replay |",
             "| --- | --- | --- | --- | ---: | --- | --- |"]
    for row in episodes:
        collision_link = f"[Half speed]({row['collision_video']})" if row["collision"] else ""
        lines.append(f"| {row['episode']} | {row['seed']} | {row.get('slot') or '-'} | {row['outcome']} | "
                     f"{row['simulated_seconds']:.1f} | [Watch]({row['video']}) | {collision_link} |")
    lines += ["", "![Collision final frames](collision_contact_sheet.jpg)", "",
              "[All evaluation collisions and seeds](collisions.csv) | [Recording metadata](episodes.json)", ""]
    (output / "EPISODES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"FINISHED {name}: {len(episodes)} videos, evaluation collisions {crash_count}/{len(rows)}", flush=True)
    return baseline, name, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", choices=[item[1] for item in EXPERIMENTS])
    args = parser.parse_args()
    torch.set_num_threads(1)
    completed = [record_experiment(*item) for item in EXPERIMENTS
                 if args.experiment is None or item[1] in args.experiment]
    lines = ["# New experiment episode videos", "",
             "Videos use the final trained checkpoints and the original saved evaluation settings.", "",
             "Each run has eight full episodes, including four selected collisions. Collision reels replay the last six seconds at half speed. "
             "Collision rates below use all 50 saved evaluation episodes, with paired seeds 10000-10049 within each task.", "",
             "DQN completion means surviving the 30-second episode. PPO success means reaching and holding the moving slot. "
             "The two algorithms have different tasks, so their rates are not a controlled algorithm comparison.", "",
             "| Experiment | Training steps | Evaluation collisions | All episodes | Collision replays | Episode list |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for baseline, name, manifest in completed:
        prefix = f"{baseline}/results/{name}/videos"
        lines.append(f"| {name} | {manifest['training_steps']:,} | "
                     f"{manifest['evaluation_collisions']}/{manifest['evaluation_episodes']} "
                     f"({manifest['evaluation_collision_rate']:.0%}) | [Watch]({prefix}/all_episodes.mp4) | "
                     f"[Watch]({prefix}/collisions_only.mp4) | [Details]({prefix}/EPISODES.md) |")
    (ROOT / "NEW_EXPERIMENT_EPISODES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
