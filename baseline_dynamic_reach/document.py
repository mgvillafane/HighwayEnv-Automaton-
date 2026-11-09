"""Figures and tables for EXPERIMENT_NOTES.md, generated from the real environment and runs.

    python -B baseline_dynamic_reach/document.py task              # geometry, goals, frames, observation
    python -B baseline_dynamic_reach/document.py arm B0            # curves, results table, IV diff
    python -B baseline_dynamic_reach/document.py rollouts B0_seed7 # one rollout per slot: MP4 + filmstrip + traces

Rollout episodes follow a rule fixed before training: for each slot, the first evaluation seed
(from 10,000 upward) whose naturally sampled slot is that slot. No outcome is looked at first.
"""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from baseline_dynamic_reach.config import (ARMS, EVAL_EPISODES, EVAL_SEED, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
                                           SEEDS, SLOTS, arm_config, run_name)
from baseline_dynamic_reach.ivcheck import describe, differences
from baseline_dynamic_reach.environment import ReachEnv

SLOT_COLORS = {"front": "#2a9d8f", "back": "#e76f51", "left": "#457b9d", "right": "#e9c46a"}
LANE_WIDTH = 4.0


def make_env(arm="B0", render_mode=None):
    return ReachEnv(render_mode=render_mode, **deepcopy(arm_config(arm)["environment"]))


def rollout_seeds(env, first=EVAL_SEED, count=EVAL_EPISODES):
    """For each slot, the first evaluation seed that samples it."""
    chosen = {}
    for seed in range(first, first + count):
        _, info = env.reset(seed=seed)
        chosen.setdefault(info["slot"], seed)
        if len(chosen) == len(SLOTS):
            break
    return {slot: chosen[slot] for slot in SLOTS if slot in chosen}


# --------------------------------------------------------------------------- task figures

def slot_geometry(goal_config, path):
    """The four slots around one anchor, to scale, in the anchor's lane frame."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for lane in range(3):
        ax.axhspan(lane * LANE_WIDTH - 2, lane * LANE_WIDTH + 2, color="#f1f3f5" if lane % 2 else "#e9ecef", zorder=0)
    for edge in (-2, 2, 6, 10):
        ax.axhline(edge, color="#868e96", lw=1, ls="-" if edge in (-2, 10) else "--", zorder=1)
    offsets, half_s, half_d = goal_config["slot_offsets"], goal_config["radius"], goal_config["lateral_tolerance"]
    centers = {"front": (offsets["front"], 4.0), "back": (-offsets["back"], 4.0), "left": (0.0, 0.0), "right": (0.0, 8.0)}
    ax.add_patch(Rectangle((-2.5, 3.0), 5.0, 2.0, color="#c2255c", zorder=3))
    ax.text(0, 4, "anchor", color="white", ha="center", va="center", fontsize=8, zorder=4)
    for slot, (s, d) in centers.items():
        ax.add_patch(Rectangle((s - half_s, d - half_d), 2 * half_s, 2 * half_d, fill=False,
                               ec=SLOT_COLORS[slot], lw=2.2, zorder=3))
        ax.plot(s, d, "o", color=SLOT_COLORS[slot], ms=4, zorder=4)
        ax.text(s + half_s + 0.6, d, slot.upper(), color=SLOT_COLORS[slot], ha="left", va="center",
                fontsize=9, fontweight="bold", zorder=4)
    # Offsets are drawn below the road so they cross no box or lane line.
    ax.plot([0, 0], [10.2, 11.6], color="#495057", lw=0.8)
    for s, label in ((offsets["front"], f"FRONT center +{offsets['front']:g} m"),
                     (-offsets["back"], f"BACK center -{offsets['back']:g} m")):
        ax.plot([s, s], [10.2, 11.6], color="#495057", lw=0.8)
        ax.annotate("", (s, 11.2), (0, 11.2), arrowprops=dict(arrowstyle="->", color="#495057"))
        ax.text(s / 2, 11.0, label, ha="center", va="bottom", fontsize=8, color="#495057")
    ax.annotate("direction of travel", (44, -2.9), (30, -2.9), va="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#495057"))
    ax.set(xlim=(-24, 46), ylim=(12.2, -3.6), xlabel="along the anchor's lane, relative to its center (m)",
           ylabel="lateral (m), lane 0 on top", yticks=[0, 4, 8])
    ax.set_aspect("equal")
    ax.set_title(f"Slot boxes: ego's center must be within ±{half_s:g} m along and ±{half_d:g} m across the slot lane",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def goal_samples(env, path, seed=EVAL_SEED, count=EVAL_EPISODES):
    """Every evaluation episode's starting goal, in ego's starting frame, plus distances and slot counts."""
    samples = []
    for episode_seed in range(seed, seed + count):
        _, info = env.reset(seed=episode_seed)
        ego = env.highway.vehicle
        samples.append({"seed": episode_seed, "slot": info["slot"],
                        "dx": float(info["goal_position"][0] - ego.position[0]),
                        "dy": float(info["goal_position"][1] - ego.position[1]),
                        "anchor_dx": float(info["anchor_position"][0] - ego.position[0]),
                        "distance": float(info["distance_to_goal"]),
                        "lane_changes": abs(int(info["goal_lane"]) - int(ego.lane_index[2])),
                        "anchor_speed": float(env.anchor.speed)})
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1))
    ax = fig.add_subplot(grid[0, :])
    for slot in SLOTS:
        points = [s for s in samples if s["slot"] == slot]
        ax.scatter([p["dx"] for p in points], [p["dy"] / LANE_WIDTH for p in points], s=22, alpha=0.75,
                   color=SLOT_COLORS[slot], label=f"{slot.upper()} ({len(points)})", edgecolor="none")
    ax.plot(0, 0, marker=">", color="black", ms=11, ls="none", label="ego at start")
    ax.set(xlabel="slot center minus ego's starting position, along the road (m)",
           ylabel="lanes to the right of ego", yticks=[-2, -1, 0, 1, 2])
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.set_title(f"Where the goal starts in the {count} evaluation episodes (seeds {seed:,}-{seed + count - 1:,})",
                 fontsize=10)
    ax = fig.add_subplot(grid[1, 0])
    bins = np.linspace(0, 160, 17)
    for slot in SLOTS:
        ax.hist([s["distance"] for s in samples if s["slot"] == slot], bins=bins, histtype="step", lw=2,
                color=SLOT_COLORS[slot], label=slot.upper())
    ax.set(xlabel="initial distance to slot center (m)", ylabel="episodes")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[1, 1])
    for slot in SLOTS:
        counts = np.bincount([s["lane_changes"] for s in samples if s["slot"] == slot], minlength=3)
        ax.plot(range(3), counts, "o-", color=SLOT_COLORS[slot], label=slot.upper())
    ax.set(xlabel="lane changes needed to reach the slot lane", ylabel="episodes", xticks=[0, 1, 2])
    ax = fig.add_subplot(grid[1, 2])
    ax.bar([slot.upper() for slot in SLOTS], [sum(s["slot"] == slot for s in samples) for slot in SLOTS],
           color=[SLOT_COLORS[slot] for slot in SLOTS])
    ax.set(ylabel="episodes", title="slot counts (anchor drawn first)")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return samples


def slot_frames(env, seeds, folder):
    """A real simulator frame at reset for each slot, as rendered during evaluation videos."""
    images = []
    for slot, seed in seeds.items():
        env.reset(seed=seed)
        image = env.render()
        plt.imsave(folder / f"frame_{slot}.png", image)
        images.append((slot, seed, image))
    fig, axes = plt.subplots(len(images), 1, figsize=(12, 3.6 * len(images)))
    for ax, (slot, seed, image) in zip(np.atleast_1d(axes), images):
        ax.imshow(image)
        ax.set_title(f"{slot.upper()} slot, evaluation seed {seed:,}: anchor outlined in magenta, slot box in green",
                     fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "slot_frames.png", dpi=110)
    plt.close(fig)


def observation_table(env, seed, path):
    """Every one of the 53 inputs at reset, with what it means and its physical value."""
    obs, info = env.reset(seed=seed)
    cfg = env.env_config["observation"]["features_range"]
    span = {key: (high - low) / 2 for key, (low, high) in cfg.items()}  # Kinematics maps [low, high] to [-1, 1].
    scale = env.goal_config["position_scale"]
    err_scale = env.goal_config["slot_error_scale"]
    rows = []
    features = ("presence", "x", "y", "vx", "vy")
    for index, row in enumerate(obs["vehicles"]):
        who = "ego" if index == 0 else f"car {index}"
        for feature, value in zip(features, row):
            if feature == "presence":
                meaning, physical = "1 if this row holds a car", f"{value:.0f}"
            elif index == 0 and feature == "x":
                meaning, physical = "ego progress since reset, / 200 m", f"{value * span['x']:+.1f} m"
            elif index == 0:
                meaning = {"y": "ego lateral position, / 12 m", "vx": "ego forward speed, / 30 m/s",
                           "vy": "ego lateral speed, / 10 m/s"}[feature]
                physical = f"{value * span[feature]:+.2f} {'m' if feature == 'y' else 'm/s'}"
            else:
                unit = "m" if feature in ("x", "y") else "m/s"
                meaning = f"{feature} relative to ego, / {span[feature]:g} {unit}"
                physical = f"{value * span[feature]:+.2f} {unit}"
            rows.append((f"vehicles[{index}].{feature}", who, f"{value:+.3f}", physical, meaning))
    for key, label in (("achieved_goal", "ego"), ("desired_goal", "slot"), ("goal_delta", "slot minus ego")):
        for axis, value, s in zip(("x", "y"), obs[key], scale):
            frame = ("difference, / " if key == "goal_delta"
                     else "position in the episode frame (origin: ego's start x, road y = 0), / ")
            rows.append((f"{key}.{axis}", label, f"{value:+.3f}", f"{value * s:+.2f} m", f"{axis} {frame}{s:g} m"))
    for axis, value in zip(("x", "y", "vx", "vy"), obs["anchor"]):
        unit = "m" if axis in ("x", "y") else "m/s"
        rows.append((f"anchor.{axis}", "anchor minus ego", f"{value:+.3f}", f"{value * span[axis]:+.2f} {unit}",
                     f"normalized like the Kinematics rows, clipped to [-1, 1]"))
    for slot, value in zip(SLOTS, obs["slot"]):
        rows.append((f"slot.{slot}", "target slot", f"{value:.0f}", "", "one-hot: which slot is the goal"))
    rows.append(("hold_progress", "ego", f"{obs['hold_progress'][0]:.3f}", "",
                 "decisions held in the box / required; always 0 before success in reach"))
    for name, value, s, unit in zip(("along", "across", "speed"), obs["slot_error"], err_scale, ("m", "m", "m/s")):
        what = {"along": "slot minus ego along the slot lane", "across": "slot lane center minus ego, across",
                "speed": "anchor speed minus ego speed"}[name]
        rows.append((f"slot_error.{name}", "local error", f"{value:+.3f}", f"{value * s:+.2f} {unit}",
                     f"{what}, / {s:g} {unit}, clipped to [-1, 1] (saturates beyond {s:g} {unit})"))
    lines = [f"Real observation at reset, evaluation seed {seed:,}: {info['slot'].upper()} slot, "
             f"initial distance {info['distance_to_goal']:.1f} m. {len(rows)} inputs.", "",
             "| Input | Refers to | Value | Physical | Meaning and scale |", "|---|---|---:|---:|---|"]
    lines += [f"| `{a}` | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def task_documents():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env("B0", render_mode="rgb_array")
    try:
        slot_geometry(env.goal_config, FIGURES_DIR / "slot_geometry.png")
        samples = goal_samples(env, FIGURES_DIR / "goal_samples.png")
        seeds = rollout_seeds(env)
        slot_frames(env, seeds, FIGURES_DIR)
        inputs = observation_table(env, EVAL_SEED, FIGURES_DIR / "observation_sample.md")
    finally:
        env.close()
    summary = {
        "rollout_seeds": seeds, "inputs": inputs,
        "slot_counts": {slot: sum(s["slot"] == slot for s in samples) for slot in SLOTS},
        "median_initial_distance_m": {slot: float(np.median([s["distance"] for s in samples if s["slot"] == slot]))
                                      for slot in SLOTS},
        "lane_changes_needed": {slot: np.bincount([s["lane_changes"] for s in samples if s["slot"] == slot],
                                                  minlength=3).tolist() for slot in SLOTS},
    }
    with (FIGURES_DIR / "goal_samples.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": samples}, handle, indent=2)
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- arm results

def read_training(run):
    with (RESULTS_DIR / run / "training_episodes.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def binned(rows, key, width, total, slot=None, minimum=30):
    """Mean of `key` over the episodes ending in each bin; bins with too few episodes are left empty."""
    edges = np.arange(0, total + width, width)
    values = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = [float(r[key]) for r in rows if low < int(r["timesteps"]) <= high and (slot is None or r["slot"] == slot)]
        values.append(np.mean(chosen) if len(chosen) >= minimum else np.nan)
    return (edges[:-1] + edges[1:]) / 2, np.array(values)


def learning_curves(arm, runs, path, width=25_000):
    data = {run: read_training(run) for run in runs}
    total = max(int(rows[-1]["timesteps"]) for rows in data.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for key, color in (("success", "#2b8a3e"), ("collision", "#c92a2a"), ("timeout", "#868e96")):
        curves = [binned(rows, key, width, total) for rows in data.values()]
        for x, y in curves:
            axes[0].plot(x, y, color=color, alpha=0.3, lw=1)
        axes[0].plot(curves[0][0], np.nanmean([y for _, y in curves], axis=0), color=color, lw=2.4, label=key)
    for slot in SLOTS:
        curves = [binned(rows, "success", width, total, slot) for rows in data.values()]
        for x, y in curves:
            axes[1].plot(x, y, color=SLOT_COLORS[slot], alpha=0.25, lw=1)
        axes[1].plot(curves[0][0], np.nanmean([y for _, y in curves], axis=0), color=SLOT_COLORS[slot], lw=2.4,
                     label=slot.upper())
    axes[0].set(title=f"{arm}: training outcomes (thin: each seed, thick: mean)", ylabel="fraction of episodes")
    axes[1].set(title=f"{arm}: training success by slot")
    for ax in axes:
        ax.set(xlabel=f"training transitions (episodes binned per {width // 1000}k)", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table(arm, runs):
    evaluations = {run: json.loads((RESULTS_DIR / run / "evaluation.json").read_text(encoding="utf-8")) for run in runs}
    def cells(result):
        low, high = result["success_ci95"]
        slots = " | ".join(f"{info['successes']}/{info['episodes']}" for info in result["by_slot"].values())
        return (f"{result['success_rate']:.1%} [{low:.0%}, {high:.0%}] | {result['collision_rate']:.1%} | "
                f"{result['timeout_rate']:.1%} | {result['anchor_lost_rate']:.1%} | {slots}")
    first = next(iter(evaluations.values()))
    header = " | ".join(slot.upper() for slot in first["ppo"]["by_slot"])
    lines = [f"| Policy | Success [95% CI] | Collision | Timeout | Anchor lost | {header} |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(first["ppo"]["by_slot"])]
    for run, evaluation in evaluations.items():
        lines.append(f"| PPO {arm}, seed {evaluation['run'].rsplit('seed', 1)[1]} | {cells(evaluation['ppo'])} |")
    rates = [evaluation["ppo"]["success_rate"] for evaluation in evaluations.values()]
    lines.append(f"| PPO {arm}, mean ± sd over seeds | {np.mean(rates):.1%} ± {np.std(rates, ddof=1) if len(rates) > 1 else 0:.1%} | | | | |"
                 + " |" * (len(first["ppo"]["by_slot"]) - 1))
    lines.append(f"| Uniform random | {cells(first['random'])} |")
    lines.append(f"| Scripted (reads true state) | {cells(first['scripted'])} |")
    return "\n".join(lines)


def arm_documents(arm):
    runs = [run_name(arm, seed) for seed in SEEDS if (RESULTS_DIR / run_name(arm, seed) / "evaluation.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No evaluated runs of {arm} yet.")
    learning_curves(arm, runs, RESULTS_DIR / f"{arm}_learning_curves.png")
    table = results_table(arm, runs)
    parent = ARMS[arm]["parent"]
    if parent is None:
        diff = "Root arm: no parent."
    else:
        mine = json.loads((MODELS_DIR / f"{runs[0]}.json").read_text(encoding="utf-8"))
        theirs = json.loads((MODELS_DIR / f"{run_name(parent, mine['seed'])}.json").read_text(encoding="utf-8"))
        diff = f"Difference from {parent} (ivcheck):\n{describe(differences(theirs, mine))}"
    text = f"{table}\n\n{diff}\n"
    (RESULTS_DIR / f"{arm}_summary.md").write_text(text, encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- rollouts

def rollouts(run):
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    env = ReachEnv(render_mode="rgb_array", **sidecar["environment"])
    folder = RESULTS_DIR / run / "rollouts"
    folder.mkdir(parents=True, exist_ok=True)
    traces, strips = {}, {}
    try:
        seeds = rollout_seeds(env)
        for slot, seed in seeds.items():
            obs, info = env.reset(seed=seed)
            frames, trace = [env.render()], []
            writer = imageio.get_writer(str(folder / f"{slot}_seed{seed}.mp4"), fps=env.metadata["render_fps"],
                                        codec="libx264", macro_block_size=1)
            writer.append_data(frames[0])
            done = False
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                ego, anchor = env.highway.vehicle, env.anchor
                along, across = env.slot_error()
                trace.append({"t": float(env.highway.time), "action": action, "along_error": float(along),
                              "across_error": float(across), "ego_speed": float(ego.speed),
                              "anchor_speed": float(anchor.speed), "ego_lane": int(ego.lane_index[2]),
                              "slot_lane": int(env.goal_lane)})
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frame = env.render()
                writer.append_data(frame)
                frames.append(frame)
            for _ in range(env.metadata["render_fps"]):
                writer.append_data(frames[-1])
            writer.close()
            picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 4)]
            strips[slot] = (seed, info["termination_reason"], info["elapsed_time"], picks)
            traces[slot] = trace
    finally:
        env.close()
    fig, axes = plt.subplots(len(strips), 4, figsize=(20, 1.9 * len(strips)))
    for row, (slot, (seed, outcome, elapsed, picks)) in enumerate(strips.items()):
        for col, image in enumerate(picks):
            ax = axes[row, col]
            ax.imshow(image)
            ax.axis("off")
            if col == 0:
                ax.set_title(f"{slot.upper()}, seed {seed:,}: {outcome} at {elapsed:.1f} s", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "filmstrip.png", dpi=90)
    plt.close(fig)
    fig, axes = plt.subplots(3, len(traces), figsize=(4.2 * len(traces), 7.5), sharex="col")
    for col, (slot, trace) in enumerate(traces.items()):
        t = [p["t"] for p in trace]
        axes[0, col].plot(t, [p["along_error"] for p in trace], color=SLOT_COLORS[slot])
        axes[0, col].axhspan(-3, 3, color=SLOT_COLORS[slot], alpha=0.15)
        axes[0, col].set_title(f"{slot.upper()}: {strips[slot][1]}", fontsize=10)
        axes[1, col].plot(t, [p["ego_lane"] for p in trace], label="ego lane", color="black")
        axes[1, col].plot(t, [p["slot_lane"] for p in trace], "--", label="slot lane", color=SLOT_COLORS[slot])
        axes[1, col].set_yticks([0, 1, 2])
        axes[1, col].invert_yaxis()
        axes[2, col].plot(t, [p["ego_speed"] for p in trace], label="ego", color="black")
        axes[2, col].plot(t, [p["anchor_speed"] for p in trace], "--", label="anchor", color="#c2255c")
        axes[2, col].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("slot minus ego along lane (m)\n(shaded: inside the box)")
    axes[1, 0].set_ylabel("lane")
    axes[2, 0].set_ylabel("speed (m/s)")
    axes[1, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "traces.png", dpi=130)
    plt.close(fig)
    with (folder / "traces.json").open("w", encoding="utf-8") as handle:
        json.dump({"seeds": seeds, "outcomes": {s: strips[s][1] for s in strips}, "traces": traces}, handle)
    print({slot: (seed, outcome, round(elapsed, 1)) for slot, (seed, outcome, elapsed, _) in strips.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("task")
    sub.add_parser("arm").add_argument("arm", choices=list(ARMS))
    sub.add_parser("rollouts").add_argument("run")
    args = parser.parse_args()
    if args.command == "task":
        task_documents()
    elif args.command == "arm":
        arm_documents(args.arm)
    else:
        rollouts(args.run)


if __name__ == "__main__":
    main()
