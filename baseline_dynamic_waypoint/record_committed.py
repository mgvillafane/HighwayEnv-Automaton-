"""Record every physics frame of a policy using committed lane-change actions."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment


def record_episode(model, env, model_path, seed, output, *, deterministic=True, sampling_seed=None):
    output.mkdir(parents=True, exist_ok=True)
    obs, initial = env.reset(seed=seed)
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)
    fps = env.metadata["render_fps"]
    trace, decisions = [], []
    env.render()
    first = env.highway.viewer.get_image()
    imageio.imwrite(output / "start.png", first)
    total_reward = 0.0
    with imageio.get_writer(str(output / "episode.mp4"), fps=fps, codec="libx264",
                            macro_block_size=1, output_params=["-movflags", "+faststart"]) as video:
        video.append_data(first)

        def on_tick(observation, reward, terminated, truncated, info):
            env.render()
            video.append_data(env.highway.viewer.get_image())
            trace.append({"time": info["elapsed_time"], "decision": info["episode_policy_decisions"],
                          "policy_action": info["policy_action"], "applied_action": info["applied_action"],
                          "ego_lane": int(env.highway.vehicle.lane_index[2]),
                          "requested_lane": info["requested_lane"], "goal_lane": env.goal_lane,
                          "lane_change_in_progress": info["lane_change_in_progress"],
                          "ego_y": float(env.highway.vehicle.position[1]),
                          "lateral_speed": float(env.highway.vehicle.velocity[1]),
                          "hold_steps": info["hold_steps"], "reward": reward})

        env.physics_step_callback = on_tick
        try:
            terminated = truncated = False
            while not (terminated or truncated):
                action = int(model.predict(obs, deterministic=deterministic)[0])
                start = len(trace)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                decisions.append({"action": action, "first_tick": start, "ticks": info["internal_steps"],
                                  "started": info["lane_change_started"], "completed": info["lane_change_completed"],
                                  "interrupted": info["lane_change_interrupted"]})
                if not (terminated or truncated):
                    assert not info["lane_change_in_progress"]
                if info["lane_change_started"]:
                    ticks = trace[start:]
                    assert len({tick["requested_lane"] for tick in ticks}) == 1
                    assert all(tick["applied_action"] == 1 for tick in ticks[1:])
        finally:
            env.physics_step_callback = None
        final = env.highway.viewer.get_image()
        imageio.imwrite(output / "final.png", final)
        for _ in range(2 * fps):
            video.append_data(final)
    result = {"controller": "trained_ppo", "model": str(model_path), "seed": seed, "slot": env.slot,
              "deterministic": deterministic, "sampling_seed": sampling_seed,
              "initial_scenario": initial["initial_scenario"], "outcome": info["termination_reason"],
              "success": info["is_success"], "collision": info["is_collision"], "reward": total_reward,
              "seconds": info["elapsed_time"], "policy_decisions": len(decisions), "physics_steps": len(trace),
              "decisions": decisions, "trace": trace}
    (output / "episode.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("decisions", "trace")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    run = args.model.resolve().parent.parent / "results" / args.model.stem
    model, env, _ = load_experiment(args.model, "rgb_array")
    if not hasattr(env, "physics_step_callback"):
        env.close()
        raise ValueError("This recorder requires a committed-lane checkpoint.")
    env.demo_label = "Trained PPO / completed lane actions"
    rows = list(csv.DictReader((run / "evaluation_before_episodes.csv").open()))
    output = run / "videos"
    results = []
    try:
        for slot in ("back", "front", "left", "right"):
            seed = int(next(row for row in rows if row["slot"] == slot)["seed"])
            results.append(record_episode(model, env, args.model, seed, output / slot))
    finally:
        env.close()
    playlist = output / "playlist.txt"
    playlist.write_text("".join(f"file '{row['slot']}/episode.mp4'\n" for row in results))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(output / "all_goals.mp4")], check=True)
    for row in results:
        n, _ = imageio_ffmpeg.count_frames_and_secs(str(output / row["slot"] / "episode.mp4"))
        assert n == row["physics_steps"] + 1 + 20
    (output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Full-physics videos saved in {output}", flush=True)


if __name__ == "__main__":
    main()
