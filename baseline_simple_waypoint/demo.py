"""Record one slot or all four slots, with a scripted controller or a saved PPO."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import torch

from baseline_dynamic_waypoint.config import SLOTS, experiment_name
from baseline_simple_waypoint.config import RESULTS_DIR, SEED
from baseline_simple_waypoint.environment import SimpleFollowingEnv
from baseline_simple_waypoint.scripted import scripted_action


def load_experiment(model_path, render_mode=None):
    from stable_baselines3 import PPO
    model_path = Path(model_path)
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("task") != "single_traffic_following":
        raise ValueError("Use a checkpoint from baseline_simple_waypoint/train.py.")
    env = SimpleFollowingEnv(render_mode=render_mode, **metadata["environment"])
    try:
        model = PPO.load(model_path, env=env, device="cpu")
    except Exception:
        env.close()
        raise
    return model, env, metadata


def record_scenario(env, model, model_path, seed, slot, output):
    output.mkdir(parents=True, exist_ok=True)
    trace = []
    obs, initial = env.reset(seed=seed, options={} if slot is None else {"slot": slot})
    initial_ego_speed = float(env.highway.vehicle.speed)
    initial_gap = float(env.anchor.position[0] - env.highway.vehicle.position[0])
    fps = env.metadata["render_fps"]
    env.render()
    frame = env.highway.viewer.get_image()
    imageio.imwrite(output / "scenario_start.png", frame)
    total_reward = 0.0
    with imageio.get_writer(str(output / "scenario.mp4"), fps=fps, codec="libx264",
                            macro_block_size=1, output_params=["-movflags", "+faststart"]) as video:
        video.append_data(frame)
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(model.predict(obs, deterministic=True)[0]) if model else scripted_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if env.highway.done:
                raise RuntimeError("The demonstration window was closed before the episode finished.")
            if env.render_mode == "rgb_array":
                env.render()
            frame = env.highway.viewer.get_image()
            video.append_data(frame)
            along, across = env.slot_error()
            trace.append({"time": info["elapsed_time"], "action": action,
                          "ego_speed": float(env.highway.vehicle.speed),
                          "traffic_speed": float(env.anchor.speed),
                          "gap": float(env.anchor.position[0] - env.highway.vehicle.position[0]),
                          "ego_lane": int(env.highway.vehicle.lane_index[2]),
                          "traffic_lane": int(env.anchor.lane_index[2]),
                          "goal_lane": env.goal_lane, "slot_error": [float(along), float(across)],
                          "hold_steps": env.hold_steps})
        imageio.imwrite(output / "scenario_final.png", frame)
        for _ in range(2 * fps):
            video.append_data(frame)
    summary = {"controller": "trained_ppo" if model else "scripted_untrained_demo",
               "model": str(model_path) if model_path else None,
               "seed": seed, "slot": env.slot, "total_vehicles": len(env.highway.road.vehicles),
               "initial_scenario": initial["initial_scenario"],
               "traffic_cars": 1, "initial_ego_speed": initial_ego_speed,
               "traffic_target_speed": float(env.anchor.target_speed),
               "initial_gap": initial_gap,
               "target_offset_from_traffic": (env.goal_position - env.anchor.position).tolist(),
               "required_hold_seconds": env.goal_config["hold_seconds"],
               "speed_tolerance": env.goal_config["speed_tolerance"],
               "outcome": info["termination_reason"], "success": bool(info["is_success"]),
               "collision": bool(info["is_collision"]), "simulated_seconds": info["elapsed_time"],
               "reward": total_reward, "policy_steps": len(trace), "trace": trace}
    (output / "scenario.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "trace"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name")
    parser.add_argument("--model", type=Path)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--slot", choices=SLOTS)
    choices.add_argument("--cycle-slots", action="store_true", help="Record BACK, FRONT, LEFT and RIGHT.")
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    label = "all_slots" if args.cycle_slots else args.slot or "sampled"
    output = RESULTS_DIR / experiment_name(args.output_name or f"demo_{label}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    mode = "rgb_array" if args.headless else "human"
    model = None
    if args.model:
        model, env, _ = load_experiment(args.model, mode)
    else:
        env = SimpleFollowingEnv(render_mode=mode)
    env.demo_label = "Trained PPO" if model else "Scripted demonstration (no training)"
    summaries = []
    try:
        for slot in ("back", "front", "left", "right") if args.cycle_slots else (args.slot,):
            directory = output / slot if args.cycle_slots else output
            summaries.append(record_scenario(env, model, args.model, args.seed, slot, directory))
    finally:
        env.close()
    if args.cycle_slots:
        playlist = output / "playlist.txt"
        playlist.write_text("".join(f"file '{row['slot']}/scenario.mp4'\n" for row in summaries), encoding="utf-8")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                        "-movflags", "+faststart", str(output / "all_scenarios.mp4")], check=True)
        (output / "scenarios.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        lines = ["# Two cars, four moving goals", "",
                 "[Watch all four scenarios](all_scenarios.mp4)", "",
                 "Scripted demonstrations. PPO training has not started.", "",
                 "| Goal | Outcome | Simulated seconds | Video | Final view |",
                 "| --- | --- | ---: | --- | --- |"]
        for row in summaries:
            slot = row["slot"]
            lines.append(f"| {slot.upper()} | {row['outcome']} | {row['simulated_seconds']:.1f} | "
                         f"[Watch]({slot}/scenario.mp4) | [Image]({slot}/scenario_final.png) |")
        (output / "SCENARIOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Videos saved in {output}", flush=True)


if __name__ == "__main__":
    main()
