"""Record direct-PPO reward experiments and verify each replay against evaluation.

No reflection or scripted actions are used. Videos include every internal
physics tick of each committed lane-change action.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.record_committed import record_episode


ROOT = Path(__file__).resolve().parent
RUNS = (
    "ppo_singlecar_reward_baseline_seed7_200k",
    "ppo_singlecar_reward_guided_seed7_200k",
    "ppo_singlecar_reward_guided_fresh_seed7_200k",
    "ppo_singlecar_reward_approach_seed7_200k",
)
LABELS = {
    RUNS[0]: "Original reward / continued PPO",
    RUNS[1]: "Guided reward / continued PPO",
    RUNS[2]: "Guided reward / fresh PPO",
    RUNS[3]: "Approach reward / continued PPO",
}
FIXED_SEEDS = {"back": 10009, "front": 10002, "left": 10004, "right": 10000}


def concatenate(videos, playlist, output):
    playlist.write_text("".join(f"file '{video.resolve().as_posix()}'\n" for video in videos),
                        encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(playlist), "-c", "copy",
                    "-movflags", "+faststart", str(output)], check=True)


def ready(name):
    run = ROOT / "results" / name
    model = ROOT / "models" / f"{name}.zip"
    if not all(path.exists() for path in (model, model.with_suffix(".json"),
                                          run / "evaluation.json", run / "evaluation_episodes.csv")):
        return False
    metadata = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    return metadata.get("status") == "trained"


def cached_run(name):
    """Reuse existing verified recordings when model and evaluation are older."""
    run = ROOT / "results" / name
    output = run / "videos"
    verification = output / "verification.json"
    episodes = output / "episodes.json"
    if not verification.exists() or not episodes.exists():
        return None
    details = json.loads(verification.read_text(encoding="utf-8"))
    model = ROOT / "models" / f"{name}.zip"
    evaluation = run / "evaluation_episodes.csv"
    if not model.exists() or not evaluation.exists():
        return None
    if max(model.stat().st_mtime, evaluation.stat().st_mtime) > verification.stat().st_mtime:
        return None
    if not (details.get("direct_ppo") and details.get("outcome_reward_length_and_frame_counts_match")
            and not details.get("reflection_used")):
        return None
    results = json.loads(episodes.read_text(encoding="utf-8"))
    fixed = {row["slot"]: row for row in results if row["selection"] == "fixed_first_seed"}
    if set(fixed) != set(FIXED_SEEDS):
        return None
    for slot, seed in FIXED_SEEDS.items():
        if fixed[slot]["seed"] != seed or not (output / slot / "episode.mp4").exists():
            return None
    if not (output / "all_goals.mp4").exists():
        return None
    return [{key: value for key, value in row.items() if key not in ("trace", "decisions")} for row in results]


def selected_success(rows, slot):
    """Prefer successes involving an actual maneuver, then a required right move."""
    candidates = [row for row in rows if row["slot"] == slot and int(row["success"])]
    if not candidates:
        return None, None

    def characteristics(row):
        initial = int(float(row["initial_lane"]))
        goal = int(float(row["goal_lane"]))
        maneuver = int(row.get("physics_steps", row["length"])) > int(row["length"])
        return initial, goal, maneuver

    def rank(row):
        initial, goal, maneuver = characteristics(row)
        return (not maneuver, not (goal > initial), initial == goal, int(row["seed"]))

    chosen = min(candidates, key=rank)
    initial, goal, maneuver = characteristics(chosen)
    if maneuver:
        selection = "success_selected_with_lane_change"
    elif initial == goal:
        selection = "success_selected_initially_goal_aligned"
    else:
        selection = "success_selected"
    return chosen, selection


def record_run(name):
    run = ROOT / "results" / name
    model_path = ROOT / "models" / f"{name}.zip"
    with (run / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {int(row["seed"]): row for row in rows}
    output = run / "videos"
    output.mkdir(parents=True, exist_ok=True)
    model, env, metadata = load_experiment(model_path, "rgb_array")
    if not hasattr(env, "physics_step_callback"):
        env.close()
        raise ValueError("Reward experiment recorder requires committed lane actions.")
    results = []

    def record(slot, seed, directory, selection):
        env.demo_label = LABELS.get(name, name)
        if selection != "fixed_first_seed":
            env.demo_label += " / selected success"
        result = record_episode(model, env, model_path, seed, output / directory)
        row = expected[seed]
        assert row["slot"] == slot
        assert result["success"] == bool(int(row["success"]))
        assert result["collision"] == bool(int(row["collision"]))
        assert abs(result["reward"] - float(row["reward"])) < 1e-8
        assert result["policy_decisions"] == int(row["length"])
        if "physics_steps" in row:
            assert result["physics_steps"] == int(row["physics_steps"])
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(output / directory / "episode.mp4"))
        assert frames == result["physics_steps"] + 1 + 2 * env.metadata["render_fps"]
        result.update(directory=directory, selection=selection,
                      reward_profile=metadata.get("reward_profile", "baseline"),
                      reflection_used=False, direct_ppo=True,
                      evaluation_replay_matches=True, verified_video_frames=frames)
        result.update(left_lane_changes=sum(d["started"] and d["action"] == 0 for d in result["decisions"]),
                      right_lane_changes=sum(d["started"] and d["action"] == 2 for d in result["decisions"]))
        (output / directory / "episode.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        return result

    try:
        for slot, seed in FIXED_SEEDS.items():
            result = record(slot, seed, slot, "fixed_first_seed")
            if slot in ("front", "right") and not result["success"]:
                success, selection = selected_success(rows, slot)
                if success is not None:
                    record(slot, int(success["seed"]), f"{slot}_selected_success", selection)
    finally:
        env.close()
    fixed = [row for row in results if row["selection"] == "fixed_first_seed"]
    concatenate([output / row["directory"] / "episode.mp4" for row in fixed],
                output / "playlist.txt", output / "all_goals.mp4")
    (output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({
        "source_model": str(model_path.resolve()), "direct_ppo": True, "reflection_used": False,
        "all_physics_ticks_recorded": True, "episodes_checked": len(results),
        "outcome_reward_length_and_frame_counts_match": True,
    }, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {LABELS.get(name, name)}", "", "Direct trained PPO; no reflection or scripted actions.", "",
             "The four fixed scenarios use the first evaluation seed for each objective. Any extra success is explicitly selected.", "",
             "[All four fixed scenarios](all_goals.mp4)", "",
             "| Goal | Seed | Selection | Outcome | Seconds |", "| --- | ---: | --- | --- | ---: |"]
    lines += [f"| [{row['slot'].upper()}]({row['directory']}/episode.mp4) | {row['seed']} | {row['selection']} | {row['outcome']} | {row['seconds']:.1f} |"
              for row in results]
    (output / "SCENARIOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified direct-PPO recordings: {name}", flush=True)
    return [{key: value for key, value in row.items() if key not in ("trace", "decisions")} for row in results]


def comparisons(names, summaries):
    output = ROOT / "results" / "reward_guidance_comparison_seed7" / "videos"
    output.mkdir(parents=True, exist_ok=True)
    for slot in ("front", "right"):
        videos = [ROOT / "results" / name / "videos" / slot / "episode.mp4" for name in names]
        concatenate(videos, output / f"{slot}_playlist.txt", output / f"{slot}_comparison.mp4")
    (output / "episodes.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reward experiment video comparison", "", "No reflection: direct PPO predictions in every experiment.", "",
             "The comparison reels show the same fixed scenario sequentially in this order:", ""]
    lines += [f"{index + 1}. {LABELS.get(name, name)}" for index, name in enumerate(names)]
    lines += ["", "[FRONT comparison, seed 10002](front_comparison.mp4)", "",
              "[RIGHT comparison, seed 10000](right_comparison.mp4)", "",
              "Each episode includes all simulation ticks and a two-second final-frame hold.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", "--models", dest="runs", nargs="+", default=list(RUNS),
                        help="Record these model stems; verified existing recordings are reused.")
    parser.add_argument("--wait", action="store_true", help="Wait for final models and evaluation CSV files.")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    pending = list(args.runs)
    summaries = {}
    last_notice = 0.0
    while pending:
        available = [name for name in pending if ready(name)]
        if not available:
            if not args.wait:
                raise FileNotFoundError(f"Final models/evaluations are not ready: {pending}")
            if time.monotonic() - last_notice > 30:
                print("Waiting for final evaluation: " + ", ".join(pending), flush=True)
                last_notice = time.monotonic()
            time.sleep(2)
            continue
        for name in available:
            summaries[name] = cached_run(name)
            if summaries[name] is None:
                summaries[name] = record_run(name)
            else:
                print(f"Reusing verified recordings: {name}", flush=True)
            pending.remove(name)
    # Include earlier experiments in the comparison even when only the newly
    # trained model was requested. Loading their verified metadata does not
    # rerender or rerun any previous episode.
    comparison_names = list(dict.fromkeys([*RUNS, *args.runs]))
    for name in comparison_names:
        if name not in summaries:
            cached = cached_run(name)
            if cached is not None:
                summaries[name] = cached
    comparison_names = [name for name in comparison_names if name in summaries]
    comparisons(comparison_names, summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
