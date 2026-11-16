"""Saved example episodes for every stage of the study, always on the same watch list.

    python -B baseline_dynamic_reach/examples.py record B3_seed7                   # the watch list
    python -B baseline_dynamic_reach/examples.py record B3_seed7 --changed         # ... plus up to 3 episodes whose
                                                                                   #     outcome differs from the parent arm
    python -B baseline_dynamic_reach/examples.py record scripted                   # the scripted driver, same episodes
    python -B baseline_dynamic_reach/examples.py record B3_seed7 --episodes 10042 10077   # any episodes you choose
    python -B baseline_dynamic_reach/examples.py show B3_seed7 --episode 10042     # live window, real time
    python -B baseline_dynamic_reach/examples.py index                             # rewrite results/examples/README.md

An episode is named by its evaluation seed (10000-10199), the `seed` column of every evaluation CSV.

Written to results/examples/<run>/ for each recorded episode:
    <slot>_seed<episode>.mp4    the episode as the evaluation saw it (deterministic actions)
    <slot>_seed<episode>.png    four frames, then per step: slot error, lanes, speeds, gap to the car ahead,
                                the action taken, and (PPO only) the critic's value
    traces.json                 the numbers behind the figures
    summary.json                outcome of each recorded episode, and whether it matches the saved evaluation

The watch list is results/examples/watchlist.json. It was chosen once (Step 5) and is not changed afterwards,
so that every stage is seen in the same situations.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_dynamic_reach.blocked import ACTIONS, OUTPUT_DIR as TAG_DIR, TARGET_SLOTS
from baseline_dynamic_reach.config import ARMS, MODELS_DIR, RESULTS_DIR, arm_config, run_name
from baseline_dynamic_reach.document import SLOT_COLORS, record_episode
from baseline_dynamic_reach.environment import ReachEnv
from baseline_dynamic_waypoint.scripted import scripted_action

EXAMPLES_DIR = RESULTS_DIR / "examples"
WATCHLIST = EXAMPLES_DIR / "watchlist.json"
SCRIPTED = "scripted"
SCRIPTED_SAVED_IN = "B0_seed7"  # The scripted driver's saved rows are the same in every run (blocked.py checks this).
EXAMPLE_SEED = 7                # Stage examples are recorded for this training seed.
MAX_CHANGED = 3


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def watchlist():
    return read_json(WATCHLIST)["episodes"]


def tags():
    """results/step5_blocked/episodes.csv, by episode: the blocked tag and every arm's saved outcome."""
    path = TAG_DIR / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run blocked.py tag first.")
    with path.open(encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def saved_row(run, episode):
    """The same episode's row in the saved evaluation."""
    source, policy = (SCRIPTED_SAVED_IN, "scripted") if run == SCRIPTED else (run, "ppo")
    with (RESULTS_DIR / source / "evaluation_episodes.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == policy and int(row["seed"]) == episode:
                return row
    return None


def outcome_of(row):
    return next(name for name in ("success", "collision", "timeout", "anchor_lost") if int(row[name]))


def load(run, render_mode):
    """The environment a run was evaluated in, its policy, and the extra trace fields for it."""
    def road(env):
        ego = env.highway.vehicle
        front, _ = env.highway.road.neighbour_vehicles(ego, ego.lane_index)
        return {"front_gap": None if front is None else float(ego.lane_distance_to(front)),
                "front_speed": None if front is None else float(front.speed),
                "front_is_anchor": front is env.anchor,
                "target_lane": int(ego.target_lane_index[2]), "target_speed": float(ego.target_speed)}

    if run == SCRIPTED:
        env = ReachEnv(render_mode=render_mode, **deepcopy(arm_config("B0")["environment"]))
        return env, (lambda obs: scripted_action(env)), (lambda env, obs: road(env))
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    sidecar = read_json(MODELS_DIR / f"{run}.json")
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    env = ReachEnv(render_mode=render_mode, **deepcopy(sidecar["environment"]))
    spec = sidecar["environment"]["reward_weights"].get("potential_progress")
    shaping_weight = spec["weight"] if isinstance(spec, dict) else 0.0

    def extra(env, obs):
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            value = float(model.policy.predict_values(tensor)[0, 0])
        # Potential shaping adds weight x distance to every value (see blocked.py); this removes it.
        return {**road(env), "value": value, "value_without_shaping": value - shaping_weight * env.distance_to_goal()}

    return env, (lambda obs: model.predict(obs, deterministic=True)[0]), extra


def changed_against_parent(run, skip):
    """Up to MAX_CHANGED BACK/LEFT/RIGHT episodes whose saved outcome differs from the parent arm, same seed."""
    if run == SCRIPTED:  # The scripted driver is not an arm, so it has no parent.
        return []
    arm, seed = run.rsplit("_seed", 1)
    parent = ARMS[arm]["parent"]
    if parent is None:
        return []
    mine, theirs = run, run_name(parent, int(seed))
    broken, fixed = [], []
    for episode, row in tags().items():
        if mine not in row or theirs not in row:
            raise KeyError(f"{mine} or {theirs} is not in episodes.csv. Run blocked.py tag after evaluating the arm.")
        if row["slot"] in TARGET_SLOTS and episode not in skip and row[mine] != row[theirs]:
            (broken if row[theirs] == "S" else fixed).append(
                {"episode": episode, "group": "changed against parent",
                 "reason": f"{theirs} {row[theirs]} -> {mine} {row[mine]}"})
    picked = []  # Alternate broken and fixed, so both kinds are seen when both exist.
    while (broken or fixed) and len(picked) < MAX_CHANGED:
        for pool in (broken, fixed):
            if pool and len(picked) < MAX_CHANGED:
                picked.append(pool.pop(0))
    return picked


def figure(run, entry, record, tag_row, path):
    trace, slot = record["trace"], record["slot"]
    t = [p["t"] for p in trace]
    has_value = "value" in trace[0]
    rows = 6 if has_value else 5
    fig = plt.figure(figsize=(14, 5.0 + 1.55 * rows))
    grid = fig.add_gridspec(2 + rows, 2, height_ratios=[1.45, 1.45] + [1] * rows)
    times = np.linspace(0, record["length"], 4) / 10
    for index, image in enumerate(record["picks"]):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        ax.imshow(image)
        ax.set_title(f"t = {times[index]:.1f} s", fontsize=9, loc="left")
        ax.axis("off")
    axes = [fig.add_subplot(grid[2, :])]
    axes += [fig.add_subplot(grid[2 + i, :], sharex=axes[0]) for i in range(1, rows)]
    color = SLOT_COLORS[slot]
    axes[0].plot(t, [p["along_error"] for p in trace], color=color)
    axes[0].axhspan(-3, 3, color=color, alpha=0.15)
    axes[0].set_ylabel("slot minus ego\nalong lane (m)")
    axes[1].plot(t, [p["ego_lane"] for p in trace], color="black", label="ego lane")
    axes[1].plot(t, [p["target_lane"] for p in trace], ":", color="black", label="ego's commanded lane")
    axes[1].plot(t, [p["slot_lane"] for p in trace], "--", color=color, label="slot lane")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_ylim(2.3, -0.3)
    axes[1].set_ylabel("lane\n(0 = left)")
    axes[2].plot(t, [p["ego_speed"] for p in trace], color="black", label="ego")
    axes[2].plot(t, [p["anchor_speed"] for p in trace], "--", color="#c2255c", label="anchor")
    axes[2].plot(t, [np.nan if p["front_speed"] is None else p["front_speed"] for p in trace], color="#868e96",
                 label="car ahead in ego's lane")
    axes[2].set_ylabel("speed (m/s)")
    gaps = [np.nan if p["front_gap"] is None else p["front_gap"] for p in trace]
    axes[3].plot(t, gaps, color="#868e96")
    axes[3].axhline(5, color="#c92a2a", lw=0.8, ls="--")
    axes[3].set_ylim(0, 80)
    axes[3].set_ylabel("gap to the car ahead\nin ego's lane (m)\n(dashed: one car length)")
    axes[4].scatter(t, [p["action"] for p in trace], s=9, color="black")
    axes[4].set_yticks(range(len(ACTIONS)), ACTIONS, fontsize=7)
    axes[4].set_ylim(len(ACTIONS) - 0.5, -0.5)
    axes[4].set_ylabel("action")
    if has_value:
        axes[5].plot(t, [p["value"] for p in trace], color="#5f3dc4", label="critic value")
        axes[5].plot(t, [p["value_without_shaping"] for p in trace], "--", color="#5f3dc4",
                     label="value minus 0.1 x distance (shaping removed)")
        axes[5].axhline(0, color="#868e96", lw=0.8)
        weights = read_json(MODELS_DIR / f"{run}.json")["environment"]["reward_weights"]
        axes[5].set_ylabel(f"critic value\n(+{weights['goal']:g} reach, -{weights['collision']:g} crash)")
    for ax in axes:
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper right", ncol=3)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)")
    where = "" if tag_row is None else (
        f", {'blocked' if tag_row['blocked'] == 'True' else 'clear'}, "
        f"{tag_row['lane_changes_required']} lane change(s) needed")
    fig.suptitle(f"{run}, episode {record['seed']}: {slot.upper()} slot{where}. "
                 f"{record['outcome']} at {record['elapsed']:.1f} s.\n[{entry['group']}: {entry['reason']}]",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def record(run, episodes=None, changed=False):
    if episodes:
        entries = [{"episode": episode, "group": "chosen by hand", "reason": "requested with --episodes"}
                   for episode in episodes]
    else:
        entries = list(watchlist())
        if changed:
            entries += changed_against_parent(run, {entry["episode"] for entry in entries})
    folder = EXAMPLES_DIR / run
    folder.mkdir(parents=True, exist_ok=True)
    summary = {int(key): value for key, value in (read_json(folder / "summary.json", {})).items()}
    traces = {int(key): value for key, value in (read_json(folder / "traces.json", {})).items()}
    try:
        tag_rows = tags()
    except FileNotFoundError:
        tag_rows = {}
    env, act, extra = load(run, "rgb_array")
    try:
        for entry in entries:
            episode = entry["episode"]
            env.reset(seed=episode)  # Only to learn the slot for the file name; record_episode resets again.
            stem = f"{env.slot}_seed{episode}"
            result = record_episode(env, act, episode, folder / f"{stem}.mp4", extra)
            figure(run, entry, result, tag_rows.get(episode), folder / f"{stem}.png")
            saved = saved_row(run, episode)
            matches = None if saved is None else (outcome_of(saved) == outcome_of(result)
                                                  and int(saved["length"]) == result["length"])
            summary[episode] = {"episode": episode, "slot": result["slot"], "group": entry["group"],
                                "reason": entry["reason"], "outcome": outcome_of(result),
                                "elapsed": result["elapsed"], "length": result["length"], "reward": result["reward"],
                                "video": f"{stem}.mp4", "figure": f"{stem}.png", "matches_saved_evaluation": matches,
                                "saved_outcome": None if saved is None else outcome_of(saved),
                                "saved_length": None if saved is None else int(saved["length"])}
            traces[episode] = result["trace"]
            note = {True: "matches the saved evaluation", False: "DIFFERS FROM THE SAVED EVALUATION",
                    None: "no saved evaluation row"}[matches]
            print(f"{run} {episode} {result['slot']:5s} {outcome_of(result):9s} at {result['elapsed']:4.1f} s; {note}",
                  flush=True)
    finally:
        env.close()
    (folder / "summary.json").write_text(json.dumps(dict(sorted(summary.items())), indent=2), encoding="utf-8")
    (folder / "traces.json").write_text(json.dumps(dict(sorted(traces.items()))), encoding="utf-8")


def show(run, episode, speed=1.0):
    import pygame
    env, act, _ = load(run, "human")
    try:
        obs, info = env.reset(seed=episode)
        print(f"{run}, episode {episode}: {info['slot'].upper()} slot, slot lane {info['goal_lane']}, "
              f"{info['distance_to_goal']:.0f} m away. Close the window to stop.", flush=True)
        env.render()
        wait = int(1000 / (env.metadata["render_fps"] * speed))
        done = False
        while not done:
            action = int(act(obs))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if env.highway.done:  # The window was closed.
                return
            pygame.time.wait(wait)
        print(f"{info['termination_reason']} at {info['elapsed_time']:.1f} s, "
              f"{info['distance_to_goal']:.1f} m from the slot center", flush=True)
        for _ in range(20):
            pygame.time.wait(100)
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                return
    finally:
        env.close()


def index():
    """results/examples/README.md: the watch list under every stage, with links to what was recorded."""
    tag_rows = tags()
    arms = [arm for arm in ARMS if f"{run_name(arm, EXAMPLE_SEED)}" in next(iter(tag_rows.values()))]
    summaries = {path.parent.name: {int(k): v for k, v in read_json(path).items()}
                 for path in sorted(EXAMPLES_DIR.glob("*/summary.json"))}

    def links(run, episode):
        item = summaries.get(run, {}).get(episode)
        return "" if item is None else f" [video]({run}/{item['video']}) [figure]({run}/{item['figure']})"

    def cell(arm, episode):
        row = tag_rows[episode]
        seeds = sorted(int(key.rsplit("_seed", 1)[1]) for key in row if key.startswith(f"{arm}_seed"))
        return "".join(row[run_name(arm, seed)] for seed in seeds) + links(run_name(arm, EXAMPLE_SEED), episode)

    lines = [
        "# Example episodes, stage by stage",
        "",
        "Written by `examples.py index`; do not edit by hand. The watch list is `watchlist.json`: the same "
        "evaluation episodes are recorded for every stage, so a change between stages is seen in the same situation.",
        "",
        "- An episode is named by its evaluation seed (the `seed` column of every `evaluation_episodes.csv`).",
        "- Outcomes: S reached, C crash, T timeout, L anchor lost. Three letters are training seeds 7, 8, 9, "
        f"taken from the saved evaluations. Videos and figures are of training seed {EXAMPLE_SEED}.",
        "- Each figure shows four frames, then per step: slot error, lanes, speeds, gap to the car ahead, the "
        "action, and the critic's value (PPO only).",
        "- In the videos the anchor is outlined in magenta and the slot is the green box.",
        "",
        "```bash",
        "# Watch any episode live (run from the repository root):",
        ".venv/bin/python -B baseline_dynamic_reach/examples.py show B3_seed7 --episode 10002",
        "# Record any episodes you want (video, figure and trace into results/examples/<run>/):",
        ".venv/bin/python -B baseline_dynamic_reach/examples.py record B3_seed7 --episodes 10002 10045",
        "```",
        "",
        "## Watch list",
        "",
        "| Episode | Slot | Tag | Lane changes | Why it is on the list | " + " | ".join(arms) + " | Scripted |",
        "|---:|---|---|---:|---|" + "---|" * (len(arms) + 1),
    ]
    for entry in watchlist():
        episode = entry["episode"]
        row = tag_rows[episode]
        lines.append(f"| {episode} | {row['slot']} | {'blocked' if row['blocked'] == 'True' else 'clear'} | "
                     f"{row['lane_changes_required']} | {entry['group']}: {entry['reason']} | "
                     + " | ".join(cell(arm, episode) for arm in arms)
                     + f" | {row['scripted']}{links(SCRIPTED, episode)} |")
    listed = {entry["episode"] for entry in watchlist()}
    others = [(run, item) for run, items in summaries.items() for episode, item in items.items()
              if episode not in listed]
    if others:
        lines += ["", "## Other recorded episodes", "",
                  "| Run | Episode | Slot | Tag | Why | Outcome | |", "|---|---:|---|---|---|---|---|"]
        for run, item in others:
            row = tag_rows[item["episode"]]
            lines.append(f"| {run} | {item['episode']} | {item['slot']} | "
                         f"{'blocked' if row['blocked'] == 'True' else 'clear'} | {item['group']}: {item['reason']} | "
                         f"{item['outcome']} at {item['elapsed']:.1f} s |{links(run, item['episode'])} |")
    checked = [(run, item) for run, items in summaries.items() for item in items.values()
               if item["matches_saved_evaluation"] is not None]
    wrong = [(run, item) for run, item in checked if not item["matches_saved_evaluation"]]
    lines += ["", "## Check against the saved evaluations", "",
              f"{len(checked) - len(wrong)} of {len(checked)} recorded episodes have the same outcome and length "
              "as the same episode's row in the saved evaluation CSV."]
    lines += [f"- **{run}, episode {item['episode']}: recorded {item['outcome']} after {item['length']} steps, "
              f"saved {item['saved_outcome']} after {item['saved_length']} steps.**" for run, item in wrong]
    (EXAMPLES_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("run", help=f"a trained run such as B3_seed7, or '{SCRIPTED}'")
    rec.add_argument("--episodes", type=int, nargs="+", help="evaluation seeds; default: the watch list")
    rec.add_argument("--changed", action="store_true",
                     help=f"also record up to {MAX_CHANGED} episodes whose outcome differs from the parent arm")
    view = sub.add_parser("show")
    view.add_argument("run")
    view.add_argument("--episode", type=int, required=True)
    view.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    sub.add_parser("index")
    args = parser.parse_args()
    if args.command == "record":
        record(args.run, args.episodes, args.changed)
    elif args.command == "show":
        show(args.run, args.episode, args.speed)
    else:
        index()


if __name__ == "__main__":
    main()
