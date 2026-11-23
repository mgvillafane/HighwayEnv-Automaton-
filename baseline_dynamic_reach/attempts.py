"""The catalogue of every attempt in the reach study, generated from the saved configurations, evaluations and
training logs. EXPERIMENT_NOTES.md is the full log; this is the reference: one page per attempt with its reward
function, approach, every hyperparameter, results, training curves, demos and analysis.

An attempt is one training pipeline: a chain of runs, each starting from the previous run's trained policy,
judged by its last run. B0-B5 and P1-P6 are Steps 1-7; every training order x reward x network width that Step 8
trained to the end is one attempt.

    python -B baseline_dynamic_reach/attempts.py demos    # record a success and a crash demo per attempt (once)
    python -B baseline_dynamic_reach/attempts.py pages    # attempts/README.md, one page per attempt, figures

The demos follow a rule fixed before recording: training seed 7's final model, on its own evaluation episodes
(seeds 10,000-10,199), the first episode it reaches and the first it crashes in (a timeout if it never crashes or
never reaches). They are recorded with examples.py into results/examples/<run>/.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import (ARMS, MODELS_DIR, RESULTS_DIR, STEP8_ARMS, arm_config, run_name)

HERE = Path(__file__).resolve().parent
OUT = HERE / "attempts"
FIGURES = OUT / "figures"
DEMO_SEED = 7
GRID = RESULTS_DIR / "grid8"
B_P = [  # (attempt, step, date trained)
    ("B0", "Step 1", "2026-11-08"), ("B1", "Step 2", "2026-11-08"), ("B2", "Step 3", "2026-11-08"),
    ("B3", "Step 4", "2026-11-08"), ("B3m", "Step 5b", "2026-11-09"), ("B4", "Step 6", "2026-11-09"),
    ("B5", "Step 7a", "2026-11-15"), ("P1", "Step 7b", "2026-11-15"), ("P1s", "Step 7b", "2026-11-15"),
    ("P2", "Step 7c", "2026-11-15"), ("P3", "Step 7e", "2026-11-15"), ("P4", "Step 7f", "2026-11-15"),
    ("P5", "Step 7f", "2026-11-15"), ("P6", "Step 7g", "2026-11-15"),
]
ORDER_NAMES = {"N": "no practice", "H": "practice (natural start), then the full task",
               "E": "practice with easy starts, then the full task",
               "O": "the original order: practice with easy starts, practice with the reward, then the full task"}
STEP8_HEADINGS = {"Step 8a": "network width", "Step 8b": "training order", "Step 8c": "reward grid",
                  "Step 8c2": "time cost", "Step 8e": "256 wide on 8 seeds (8f adopted one)"}
COLORS = {"reached": "#2a78d6", "crash": "#eb6834", "timeout": "#898781", "ink": "#0b0b0b", "muted": "#898781",
          "grid": "#e1e0d9", "practice": "#f1f0ea", "surface": "#fcfcfb", "secondary": "#52514e", "seed": "#c3c2b7"}


# --- the attempts ---------------------------------------------------------------------------------------------
def seeds_of(arm):
    return sorted(int(p.stem.rsplit("_seed", 1)[1]) for p in MODELS_DIR.glob(f"{arm}_seed*.json")
                  if json.loads(p.read_text(encoding="utf-8")).get("status") == "trained"
                  and not json.loads(p.read_text(encoding="utf-8")).get("smoke"))


def lineage(arm):
    chain = [arm]
    while arm_config(chain[-1]).get("init_policy"):
        chain.append(arm_config(chain[-1])["init_policy"])
    return chain[::-1]


def step8_stage(order, reward, width):
    from baseline_dynamic_reach.config import TIME_REWARDS
    if width:
        return "Step 8a" if order == "E" else "Step 8e"
    if order == "E":
        return "Step 8a"
    if reward == "c1w01":
        return "Step 8b"
    return "Step 8c2" if reward in TIME_REWARDS else "Step 8c"


def attempts():
    found = []
    for arm, step, date in B_P:
        found.append({"id": arm, "kind": "BP", "step": step, "date": date, "final": arm, "long": None,
                      "lineage": lineage(arm), "seeds": seeds_of(arm)})
    for arm, info in STEP8_ARMS.items():
        if info["round"] != 3 or not seeds_of(arm):
            continue
        order, reward = info["order"], info["reward"]
        width = "" if info["width"] in (None, "", "n64") else info["width"]
        attempt_id = f"{order}_{reward}" + (f"_{width}" if width else "")
        longer = next((a for a, i in STEP8_ARMS.items() if i["round"] == 4 and i["order"] == order
                       and i["reward"] == reward and i["width"] == info["width"] and seeds_of(a)), None)
        found.append({"id": attempt_id, "kind": "S8", "step": step8_stage(order, reward, width),
                      "date": "2026-11-18 to 2026-11-20", "final": arm, "long": longer, "lineage": lineage(arm),
                      "seeds": seeds_of(arm), "order": order, "reward": reward, "width": width})
    order_key = {"Step 8a": 0, "Step 8b": 1, "Step 8c": 2, "Step 8c2": 3, "Step 8e": 4}
    b_p = [a for a in found if a["kind"] == "BP"]
    s8 = sorted((a for a in found if a["kind"] == "S8"),
                key=lambda a: (order_key[a["step"]], a["order"], a["reward"], a["width"]))
    return b_p + s8


# --- descriptions -----------------------------------------------------------------------------------------------
def task_of(arm):
    scenario = arm_config(arm)["environment"].get("scenario") or {}
    if not scenario:
        return "full task"
    return "practice with easy starts" if scenario.get("start_offsets") else "practice"


TASK_TEXT = {
    "full task": "the full task: all four slots, natural start (ego at 20 m/s, a 12 m/s car 35 m ahead), traffic "
                 "free to change lanes",
    "practice": "practice: only BACK/LEFT/RIGHT episodes in which a car sits in the slot's lane between ego and the "
                "slot, and that car is held in its lane (redrawn at reset until such an episode comes up), natural start",
    "practice with easy starts": "practice with easy starts: the practice episodes, but ego starts in the next lane "
                                 "10 m past, level with, or 15 m behind the blocking car, or at its natural start "
                                 "(one of the four at random)",
}


def fmt_number(x):
    return f"{x:g}"


def reward_lines(arm):
    """The reward of one run, term by term, in goal units (the scale times the raw weight)."""
    env = arm_config(arm)["environment"]
    w, scale = env["reward_weights"], env.get("reward_scale", 1.0)
    goal = w["goal"] * scale
    lines = [f"+{fmt_number(goal)} when ego reaches the slot (the episode ends)"]
    crash = w.get("collision", 0.0) * scale
    lines.append(f"−{fmt_number(crash)} on a crash (the episode ends)" if crash
                 else "0 on a crash: a crash ends the episode with nothing, like a timeout")
    if w.get("time"):
        lines.append(f"−{fmt_number(w['time'] * scale)} per decision (0.1 s); a whole 25 s episode costs "
                     f"{fmt_number(w['time'] * scale * 250)}")
    shaping = w.get("potential_progress")
    if isinstance(shaping, dict) and shaping.get("weight"):
        lines.append(f"+{fmt_number(shaping['weight'] * scale)} × (γ·Φ(next) − Φ(now)) per decision, Φ = −(distance "
                     f"to the slot in m), γ = {shaping['gamma']}: potential-based progress, which cannot change the "
                     "best policy")
    if w.get("progress"):
        lines.append(f"+{fmt_number(w['progress'] * scale)} per meter closed toward the slot")
    if w.get("timeout"):
        lines.append(f"−{fmt_number(w['timeout'] * scale)} on a timeout")
    lines.append("0 on a timeout (25 s) or a lost anchor")
    return lines


def reward_formula(arm):
    env = arm_config(arm)["environment"]
    w, scale = env["reward_weights"], env.get("reward_scale", 1.0)
    terms = [f"{fmt_number(w['goal'])}·[reached]"]
    if w.get("collision"):
        terms.append(f"− {fmt_number(w['collision'])}·[crash]")
    if w.get("time"):
        terms.append(f"− {fmt_number(w['time'])} per decision")
    shaping = w.get("potential_progress")
    if isinstance(shaping, dict) and shaping.get("weight"):
        terms.append(f"+ {fmt_number(shaping['weight'])}·(0.99·Φ′ − Φ)")
    inner = " ".join(terms)
    return f"r = {fmt_number(scale)} × ({inner})" if scale != 1 else f"r = {inner}"


def reward_short(arm):
    env = arm_config(arm)["environment"]
    w, scale = env["reward_weights"], env.get("reward_scale", 1.0)
    goal = w["goal"] * scale
    parts = [f"+{fmt_number(goal)} reach"]
    parts.append(f"−{fmt_number(w.get('collision', 0) * scale)} crash" if w.get("collision") else "crash 0")
    if w.get("time"):
        parts.append(f"−{fmt_number(w['time'] * scale)}/decision")
    shaping = w.get("potential_progress")
    if isinstance(shaping, dict) and shaping.get("weight"):
        parts.append(f"progress {fmt_number(shaping['weight'] * scale)}/m")
    return ", ".join(parts)


def width_of(arm):
    return arm_config(arm)["ppo_config"].get("policy_kwargs", {}).get("net_arch", {}).get("pi", [64, 64])


def sidecar(arm, seed):
    path = MODELS_DIR / f"{run_name(arm, seed)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- results ----------------------------------------------------------------------------------------------------
def full_task_file(run):
    arm = run.rsplit("_seed", 1)[0]
    if arm in STEP8_ARMS:
        return RESULTS_DIR / run / "eval8" / "s10000_final" / "evaluation_episodes.csv"
    if task_of(arm) != "full task":
        return RESULTS_DIR / run / "evaluation_on_B5" / "evaluation_episodes.csv"
    return RESULTS_DIR / run / "evaluation_episodes.csv"


def own_task_file(run):
    arm = run.rsplit("_seed", 1)[0]
    if arm in STEP8_ARMS:
        return RESULTS_DIR / run / ("eval8/pass10000_final" if task_of(arm) != "full task" else "eval8/s10000_final") \
            / "evaluation_episodes.csv"
    return RESULTS_DIR / run / "evaluation_episodes.csv"


@lru_cache(maxsize=None)
def episodes(path, policy="ppo"):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return tuple(row for row in csv.DictReader(handle) if row["policy"] == policy)


def outcome(row):
    return "S" if int(row["success"]) else "C" if int(row["collision"]) else "T"


def rates(rows):
    n = len(rows)
    letters = [outcome(r) for r in rows]
    reached, crash = letters.count("S") / n, letters.count("C") / n
    return {"n": n, "reached": reached, "crash": crash, "timeout": letters.count("T") / n, "score": reached - crash}


def summarize(files):
    """Per seed and mean: all, B/L/R and FRONT rates of the given evaluation files ({seed: path})."""
    per_seed = {}
    for seed, path in files.items():
        rows = episodes(str(path))
        if not rows:
            continue
        per_seed[seed] = {"all": rates(rows), "blr": rates([r for r in rows if r["slot"] != "front"]),
                          "front": rates([r for r in rows if r["slot"] == "front"]) if any(r["slot"] == "front" for r in rows) else None}
    if not per_seed:
        return None
    mean = {}
    for group in ("all", "blr", "front"):
        values = [s[group] for s in per_seed.values() if s[group]]
        mean[group] = {k: statistics.fmean(v[k] for v in values) for k in ("reached", "crash", "timeout", "score")} if values else None
    return {"per_seed": per_seed, "mean": mean}


@lru_cache(maxsize=None)
def references(set_path):
    found = {}
    for policy in ("scripted", "random"):
        rows = episodes(str(set_path), policy)
        if rows:
            found[policy] = {"all": rates(rows), "blr": rates([r for r in rows if r["slot"] != "front"]),
                             "front": rates([r for r in rows if r["slot"] == "front"])}
    return found


@lru_cache(maxsize=None)
def selection_tags():
    from baseline_dynamic_reach.grid import tags
    return tags()


def selection_stats(run):
    from baseline_dynamic_reach.grid import run_stats
    return run_stats(run, selection_tags())


@lru_cache(maxsize=None)
def fresh_stats(set_name):
    from baseline_dynamic_reach import grid
    first, count = {"s50000": (50_000, 500), "s60000": (60_000, 500)}[set_name]
    tags = grid.final_tags(set_name, first, count)
    runs = [run_name(a, s) for a in list(STEP8_ARMS) + ["P6"] for s in (7, 8, 9, 10, 13, 16, 19, 22, 25, 28)
            if (RESULTS_DIR / run_name(a, s) / "eval8" / f"{set_name}_final" / "evaluation_episodes.csv").exists()]
    return grid.final_test_stats(runs, tags, set_name, count), grid.final_references(tags, set_name)


# --- training curves --------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def training_log(run):
    import pandas as pd
    path = RESULTS_DIR / run / "training_episodes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["timesteps", "success", "collision", "timeout", "anchor_lost"])


@lru_cache(maxsize=None)
def checkpoint_curves():
    import pandas as pd
    frames = {}
    for folder in ("curves", "curves_8e"):  # 8e last: it covers the 9M pipelines.
        path = GRID / folder / "deterministic.csv"
        if path.exists():
            frame = pd.read_csv(path)
            for recipe, part in frame.groupby("recipe"):
                frames[recipe] = part
    return frames


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["seed"], "axes.labelcolor": COLORS["secondary"],
                         "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"], "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": COLORS["surface"],
                         "axes.facecolor": COLORS["surface"]})
    return plt


def training_figure(attempt, scripted_score):
    """Training outcomes (sampling policy, its own training task) per 1/20 of each round; with Step 8 checkpoint
    curves (deterministic, full task, 400 selection episodes) under it when they exist."""
    plt = style()
    chain = attempt["lineage"] + ([attempt["long"]] if attempt.get("long") else [])
    curve = checkpoint_curves().get(attempt["id"])
    rows = 2 if curve is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.1 * rows), sharex=True, squeeze=False)
    ax = axes[0][0]
    offset, first = 0, True
    for arm in chain:
        length = arm_config(arm)["requested_timesteps"]
        bins = 20
        size = length / bins
        per_seed = []
        for seed in attempt["seeds"]:
            log = training_log(run_name(arm, seed))
            if log is None or log.empty:
                continue
            index = np.minimum(log.timesteps // size, bins - 1)
            grouped = log.groupby(index)
            per_seed.append({"reached": grouped.success.mean().reindex(range(bins)).to_numpy(),
                             "crash": grouped.collision.mean().reindex(range(bins)).to_numpy(),
                             "timeout": (grouped.timeout.mean() + grouped.anchor_lost.mean()).reindex(range(bins)).to_numpy()})
        x = (offset + (np.arange(bins) + 0.5) * size) / 1e6
        if task_of(arm) != "full task":
            for a in axes[:, 0]:
                a.axvspan(offset / 1e6, (offset + length) / 1e6, color=COLORS["practice"], zorder=0, lw=0)
        for metric in ("reached", "crash", "timeout"):
            if not per_seed:
                continue
            values = np.array([s[metric] for s in per_seed], dtype=float)
            ax.fill_between(x, np.nanmin(values, 0), np.nanmax(values, 0), color=COLORS[metric], alpha=0.14, lw=0)
            ax.plot(x, np.nanmean(values, 0), color=COLORS[metric], lw=2, label=metric if first else None)
        first = False
        ax.text((offset + length / 2) / 1e6, 1.03, arm, ha="center", va="bottom", fontsize=8, color=COLORS["secondary"],
                transform=ax.get_xaxis_transform())
        offset += length
        for a in axes[:, 0]:
            a.axvline(offset / 1e6, color=COLORS["grid"], lw=1, zorder=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of training episodes")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="center right", frameon=False, fontsize=8)
    title = (f"{attempt['id']}: outcomes of training episodes (sampling policy, each run's own task)\n"
             f"line = mean of {len(attempt['seeds'])} seeds, band = lowest to highest seed")
    ax.set_title(title, loc="left", fontsize=9, color=COLORS["ink"], pad=18)
    if curve is not None:
        ax2 = axes[1][0]
        for seed, part in curve.groupby("seed"):
            part = part.sort_values("transitions")
            ax2.plot(part.transitions / 1e6, part.score, color=COLORS["reached"], lw=0.7, alpha=0.3)
        mean = curve.groupby("transitions").score.mean()
        ax2.plot(mean.index / 1e6, mean.values, color=COLORS["reached"], lw=2, label="score, mean of seeds")
        ax2.axhline(scripted_score, color=COLORS["secondary"], ls="--", lw=1, label=f"scripted driver ({scripted_score:+.3f})")
        ax2.set_ylabel("score (reached − crash)")
        ax2.set_title("deterministic policy at every 200k checkpoint, full task, 400 selection episodes (thin: each seed)",
                      loc="left", fontsize=9, color=COLORS["ink"])
        ax2.grid(axis="y", color=COLORS["grid"], lw=0.6)
        ax2.legend(loc="lower right", frameon=False, fontsize=8)
    axes[-1][0].set_xlabel("transitions (millions); shaded = practice rounds")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{attempt['id']}_training.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def overview_figure(rows, scripted):
    plt = style()
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(rows) + 1.2))
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        for score in row["seed_scores"]:
            ax.scatter(score, y, color=COLORS["seed"], s=10, zorder=2)
        ax.scatter(row["score"], y, color=COLORS["reached"], s=36, zorder=3)
    ax.axvline(scripted, color=COLORS["secondary"], ls="--", lw=1)
    ax.text(scripted, len(rows) - 0.4, " scripted driver", fontsize=8, color=COLORS["secondary"], va="bottom")
    ax.set_yticks(range(len(rows)), [r["id"] for r in rows][::-1], fontsize=7)
    ax.set_xlabel("score on the full task, 200 episodes (10,000–10,199): reached − crash; dots: seeds")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    fig.tight_layout()
    path = FIGURES / "overview.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --- demos --------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def blocked_tags():
    with (RESULTS_DIR / "step5_blocked" / "episodes.csv").open(encoding="utf-8") as handle:
        return {int(row["seed"]): row["blocked"] == "True" for row in csv.DictReader(handle)}


def demo_episodes(run):
    """The rule, fixed before recording, on the run's own evaluation episodes (10,000-10,199):
    - reached: the first FRONT episode it reaches; if none, the first blocked BACK/LEFT/RIGHT episode (a car in the
      slot's lane at reset); if none, the first episode it reaches. These show the hardest skill it has.
    - crash: the first episode it crashes in.
    - a timeout fills the place of either if it never happens."""
    rows = sorted(episodes(str(own_task_file(run))), key=lambda r: int(r["seed"]))
    practice = task_of(run.rsplit("_seed", 1)[0]) != "full task"
    reached = [r for r in rows if outcome(r) == "S"]
    hard = [] if practice else ([r for r in reached if r["slot"] == "front"]
                                or [r for r in reached if blocked_tags().get(int(r["seed"]))])
    picks = {letter: next((int(r["seed"]) for r in rows if outcome(r) == letter), None) for letter in "SCT"}
    if hard:
        picks["S"] = int(hard[0]["seed"])
    picks = {k: v for k, v in picks.items() if v is not None}
    chosen = [("reached", picks["S"])] if "S" in picks else []
    chosen += [("crash", picks["C"])] if "C" in picks else []
    if len(chosen) < 2 and "T" in picks:
        chosen.append(("timeout", picks["T"]))
    return chosen


def record_demos(run):
    from baseline_dynamic_reach import examples
    folder = examples.EXAMPLES_DIR / run
    done = json.loads((folder / "summary.json").read_text(encoding="utf-8")) if (folder / "summary.json").exists() else {}
    missing = [episode for _, episode in demo_episodes(run) if str(episode) not in done]
    if missing:
        examples.record(run, missing, group="demo", reason="attempts.py rule: first reached / first crash")
    return run, missing


def demos(args):
    runs = [f"{a['final']}_seed{DEMO_SEED}" for a in attempts()]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for run, missing in pool.map(record_demos, runs):
            print(f"{run}: recorded {missing or 'nothing new'}", flush=True)
    from baseline_dynamic_reach import examples
    examples.index()


# --- pages --------------------------------------------------------------------------------------------------------
def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def sc(x):
    return "–" if x is None else f"{x:+.3f}"


def table(header, rows):
    """Markdown table; a column is right-aligned when every filled cell is a number (percent, score, count)."""
    def numeric(cell):
        text = str(cell).strip("* ")
        return text[:1].isdigit() or (text[:1] in "+-−" and text[1:2].isdigit())

    align = []
    for i in range(len(header)):
        cells = [row[i] for row in rows if str(row[i]).strip("* –")]
        align.append("---:" if i and cells and all(numeric(c) for c in cells) else "---")
    return ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"] \
        + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def ppo_table(side):
    ppo = side["ppo_config"]
    arch = ppo.get("policy_kwargs", {}).get("net_arch", {})
    rows = [
        ["Algorithm", "PPO, Stable-Baselines3 " + side["versions"].get("stable-baselines3", "?")],
        ["Policy network", f"MLP {'-'.join(map(str, arch.get('pi', [64, 64])))} (tanh), separate from the value network"],
        ["Value network", f"MLP {'-'.join(map(str, arch.get('vf', [64, 64])))}"],
        ["Learning rate", fmt_number(ppo["learning_rate"]) + " (constant)"],
        ["Environments in parallel", side.get("num_envs", 3)],
        ["Steps per environment per update (n_steps)", ppo["n_steps"]],
        ["Transitions per update", ppo["n_steps"] * side.get("num_envs", 3)],
        ["Minibatch size", ppo["batch_size"]],
        ["Epochs per update", ppo["n_epochs"]],
        ["Discount γ", ppo["gamma"]],
        ["GAE λ", ppo["gae_lambda"]],
        ["Clip range", ppo["clip_range"]],
        ["Entropy coefficient", ppo["ent_coef"]],
        ["Value-loss coefficient", ppo.get("vf_coef", 0.5)],
        ["Gradient-norm clip (policy and value together)", ppo.get("max_grad_norm", 0.5)],
        ["Device", ppo.get("device", "cpu")],
    ]
    return table(["Setting", "Value"], rows)


def environment_table(side):
    env = side["environment"]
    c, g = env["env_config"], env["goal_config"]
    obs = c["observation"]
    rows = [
        ["Simulator", f"highway-env {side['versions'].get('highway-env', '?')}, highway-v0, {c['lanes_count']} lanes"],
        ["Traffic", f"{c['vehicles_count']} cars, density {c['vehicles_density']}, preferred speeds "
                    f"{c['traffic_speed_range'][0]:g}–{c['traffic_speed_range'][1]:g} m/s (IDM and MOBIL)"],
        ["Start", f"ego at {g['initial_speed']:g} m/s; a car {c['lead_vehicle_gap']:g} m ahead in ego's lane"],
        ["Decisions", f"{c['policy_frequency']} per second; episode limit {c['duration']:g} s "
                      f"({int(c['duration'] * c['policy_frequency'])} decisions)"],
        ["Actions", "LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER; target speeds "
                    + ", ".join(f"{v:g}" for v in c["action"]["target_speeds"]) + " m/s"],
        ["Observation", f"{obs['vehicles_count']} cars × {len(obs['features'])} ({', '.join(obs['features'])}), "
                        f"x ±{obs['features_range']['x'][1]} m, plus goal, anchor, slot type, slot error and hold "
                        "progress (53 numbers)"],
        ["Anchor and slot", f"anchor: a random car {g['distance_range'][0]:g}–{g['distance_range'][1]:g} m ahead, held in "
                            f"its lane; FRONT {g['slot_offsets']['front']:g} m ahead of it, BACK "
                            f"{g['slot_offsets']['back']:g} m behind, LEFT/RIGHT beside"],
        ["Success", f"ego's center within ±{g['radius']:g} m along and ±{g['lateral_tolerance']:g} m across the slot "
                    f"for {g['hold_seconds']:g} s (one decision)"],
        ["Reward scale", fmt_number(env.get("reward_scale", 1.0))],
        ["Task", TASK_TEXT[task_of(side["arm"])]],
    ]
    return table(["Setting", "Value"], rows)


def round_rows(attempt):
    rows = []
    for number, arm in enumerate(attempt["lineage"] + ([attempt["long"]] if attempt.get("long") else []), 1):
        side = sidecar(arm, attempt["seeds"][0])
        init = arm_config(arm).get("init_policy")
        rows.append([number, f"`{arm}`", task_of(arm), reward_short(arm),
                     f"{arm_config(arm)['requested_timesteps'] / 1e6:g}M",
                     f"`{init}`'s trained policy (same seed)" if init else "fresh weights",
                     side["platform"]["system"].replace("Darwin", "Mac").replace("Windows", "PC") if side else "–"])
    return table(["Run", "Arm", "Task", "Reward", "Transitions", "Starts from", "Trained on"], rows)


def full_task_section(attempt, refs):
    files = {s: full_task_file(run_name(attempt["final"], s)) for s in attempt["seeds"]}
    summary = summarize(files)
    lines = ["### Full task: 200 fixed episodes (10,000–10,199), final model of every seed", ""]
    if summary is None:
        return lines + ["No full-task evaluation saved."], None
    rows = []
    for seed, s in summary["per_seed"].items():
        rows.append([f"seed {seed}", pct(s["all"]["reached"]), pct(s["all"]["crash"]), pct(s["all"]["timeout"]),
                     sc(s["all"]["score"]), pct(s["blr"]["reached"]), pct(s["blr"]["crash"]),
                     pct(s["front"]["reached"]) if s["front"] else "–", pct(s["front"]["crash"]) if s["front"] else "–"])
    m = summary["mean"]
    rows.append([f"**mean of {len(summary['per_seed'])}**", f"**{pct(m['all']['reached'])}**", f"**{pct(m['all']['crash'])}**",
                 pct(m["all"]["timeout"]), f"**{sc(m['all']['score'])}**", pct(m["blr"]["reached"]), pct(m["blr"]["crash"]),
                 pct(m["front"]["reached"]) if m["front"] else "–", pct(m["front"]["crash"]) if m["front"] else "–"])
    for policy, r in refs.items():
        rows.append([f"{policy} driver", pct(r["all"]["reached"]), pct(r["all"]["crash"]), pct(r["all"]["timeout"]),
                     sc(r["all"]["score"]), pct(r["blr"]["reached"]), pct(r["blr"]["crash"]),
                     pct(r["front"]["reached"]), pct(r["front"]["crash"])])
    lines += table(["", "Reached", "Crash", "Timeout", "Score", "B/L/R reached", "B/L/R crash", "FRONT reached",
                    "FRONT crash"], rows)
    if attempt["kind"] == "BP" and task_of(attempt["final"]) != "full task":
        lines += ["", f"{attempt['id']} was trained only on practice episodes; this is the full task it never trained on "
                  "(evaluated with `--environment-of B5`)."]
    return lines, summary


def own_task_section(attempt):
    """Practice-task results of the runs that trained on practice episodes."""
    practice = [arm for arm in attempt["lineage"] if task_of(arm) != "full task"]
    if not practice:
        return []
    lines = ["", "### Practice task: 200 fixed practice episodes, final model", ""]
    if attempt["kind"] == "S8":
        lines += ["Step 8 measures every practice round on one practice task: the 200 natural-start practice episodes "
                  "(blocked BACK/LEFT/RIGHT, blocker held), whatever starts the round trained on.", ""]
    rows = []
    for arm in practice:
        summary = summarize({s: own_task_file(run_name(arm, s)) for s in attempt["seeds"]})
        if summary is None:
            continue
        per_seed = ", ".join(f"{100 * s['all']['reached']:.0f}%" for s in summary["per_seed"].values())
        m = summary["mean"]["all"]
        task = task_of(arm) if attempt["kind"] == "BP" else "practice, natural start (blocker held)"
        rows.append([f"`{arm}`", task, pct(m["reached"]), pct(m["crash"]), pct(m["timeout"]), per_seed])
    return lines + table(["Run", "Task", "Reached", "Crash", "Timeout", "Reached, by seed"], rows)


def selection_section(attempt):
    if attempt["kind"] != "S8":
        return []
    lines = ["", "### Step 8's own measure: 400 selection episodes (10,000–10,199 and 40,000–40,199), full task", "",
             "Every round's model on the full task. Rounds 3 and 4 are the mean of the final model and the checkpoints "
             "400k and 200k before the end, as the grid scored them; rounds 1–2 are the final model.", ""]
    rows = []
    chain = attempt["lineage"] + ([attempt["long"]] if attempt.get("long") else [])
    total = 0
    for arm in chain:
        total += arm_config(arm)["requested_timesteps"]
        stats = [selection_stats(run_name(arm, s)) for s in attempt["seeds"]]
        stats = [s for s in stats if s]
        if not stats:
            continue
        mean = lambda g, k: statistics.fmean(s[g][k] for s in stats)  # noqa: E731
        stuck = sum(s["front"]["T_rate"] >= 0.30 for s in stats)
        rows.append([f"`{arm}` (after {total / 1e6:g}M)", sc(mean("all", "score")), pct(mean("all", "S_rate")),
                     pct(mean("all", "C_rate")), pct(mean("all", "T_rate")), pct(mean("front", "S_rate")),
                     f"{stuck} of {len(stats)}", ", ".join(f"{s['all']['score']:+.2f}" for s in stats)])
    return lines + table(["Model", "Score", "Reached", "Crash", "Timeout", "FRONT reached",
                          "Seeds giving up on FRONT (≥ 30% timeouts)", "Score by seed"], rows)


def fresh_section(attempt):
    lines = []
    for set_name, label in (("s50000", "8d final test: 500 fresh episodes (50,000–50,499)"),
                            ("s60000", "8f test: 500 fresh episodes (60,000–60,499)")):
        stats, refs = fresh_stats(set_name)
        runs = [run_name(attempt["final"], s) for s in attempt["seeds"] if run_name(attempt["final"], s) in stats]
        if not runs:
            continue
        mean = lambda g, k: statistics.fmean(stats[r][g][k] for r in runs)  # noqa: E731
        rows = [[f"{attempt['id']} ({len(runs)} seeds)", sc(mean("all", "score")), pct(mean("all", "S_rate")),
                 pct(mean("all", "C_rate")), pct(mean("all", "T_rate")), pct(mean("front", "S_rate")),
                 ", ".join(f"{stats[r]['all']['score']:+.2f}" for r in runs)]]
        for policy, counts in refs.items():
            a = counts["all"]
            rows.append([f"{policy} driver", sc(a["score"]), pct(a["S"] / a["n"]), pct(a["C"] / a["n"]), pct(a["T"] / a["n"]),
                         pct(counts["front"]["S"] / counts["front"]["n"]), ""])
        lines += ["", f"### {label}", ""] + table(["Model", "Score", "Reached", "Crash", "Timeout", "FRONT reached",
                                                    "Score by seed"], rows)
    return lines


def demo_section(attempt):
    run = f"{attempt['final']}_seed{DEMO_SEED}"
    from baseline_dynamic_reach import examples
    folder = examples.EXAMPLES_DIR / run
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8")) if (folder / "summary.json").exists() else {}
    task = task_of(attempt["final"])
    lines = ["", "## Demos", "",
             f"Training seed {DEMO_SEED}'s final model on its own evaluation episodes ({TASK_TEXT[task].split(':')[0]}), "
             "chosen by a fixed rule: the first FRONT episode it reaches (else the first blocked BACK/LEFT/RIGHT episode, "
             "else the first it reaches; on practice episodes simply the first), and the first it crashes in; a timeout "
             "fills either place if it never happens. Each figure shows four "
             "frames, then per decision the slot error, lanes, speeds, gap to the car ahead, the action and the critic's "
             "value. In the videos the anchor is outlined in magenta and the slot is the green box.", ""]
    for label, episode in demo_episodes(run):
        item = summary.get(str(episode))
        if item is None:
            lines.append(f"- {label}: episode {episode} (not recorded yet; run `attempts.py demos`)")
            continue
        rel = f"../results/examples/{run}"
        check = "" if item.get("matches_saved_evaluation") in (True, None) else " **(differs from the saved evaluation)**"
        lines += [f"**{label.capitalize()}: episode {episode}, {item['slot'].upper()} slot, {item['outcome']} at "
                  f"{item['elapsed']:.1f} s.**{check} [Video]({rel}/{item['video']})", "",
                  f"![{attempt['id']} {label}, episode {episode}]({rel}/{item['figure']})", ""]
    return lines


def page(attempt, text, refs_full):
    final = attempt["final"]
    side = sidecar(final, attempt["seeds"][0])
    lines = [f"# {attempt['id']}: {text['title']}", "",
             f"*{attempt['step']}, trained {attempt['date']}. Seeds: {', '.join(map(str, attempt['seeds']))}. "
             f"[All attempts](README.md) · [Experiment notes](../EXPERIMENT_NOTES.md)*", "",
             f"**Status:** {text['status']}", "", "## Idea", "", text["idea"].strip(), "",
             "## Approach", "", text.get("approach", "").strip(), "", *round_rows(attempt), "",
             "## Reward function", "", f"Of the last run, `{final}` (per decision, in goal units after the reward scale):", ""]
    lines += [f"- {line}" for line in reward_lines(final)]
    lines += ["", f"Written out: `{reward_formula(final)}`, where [reached] and [crash] are 1 on the decision the "
              "episode ends that way and 0 otherwise.", ""]
    earlier = [arm for arm in attempt["lineage"][:-1] if reward_short(arm) != reward_short(final)]
    if earlier:
        lines += ["Earlier runs of the pipeline used:", ""]
        lines += [f"- `{arm}`: {reward_short(arm)} (`{reward_formula(arm)}`)" for arm in earlier]
        lines.append("")
    lines += ["## Hyperparameters", "", f"From `{final}`'s saved configuration (identical for every run of the pipeline "
              "except the task, reward and starting weights shown above).", "", *ppo_table(side), "",
              "<details><summary>Environment and task settings</summary>", "", *environment_table(side), "", "</details>", "",
              f"Software: highway-env {side['versions'].get('highway-env')}, gymnasium {side['versions'].get('gymnasium')}, "
              f"Stable-Baselines3 {side['versions'].get('stable-baselines3')}, PyTorch {side['versions'].get('torch')}, "
              f"NumPy {side['versions'].get('numpy')}, Python {side['platform'].get('python')}.", ""]
    section, summary = full_task_section(attempt, refs_full)
    lines += ["## Results", "", "Score = reached − crash per episode (+1, −1, 0 for a timeout), the study's measure "
              "since Step 8.", "", *section, *own_task_section(attempt), *selection_section(attempt), *fresh_section(attempt)]
    figure = training_figure(attempt, refs_selection_scripted())
    lines += ["", "## Training curves", "", f"![{attempt['id']} training curves](figures/{figure.name})", ""]
    lines += demo_section(attempt)
    lines += ["## Analysis", "", text["analysis"].strip(), ""]
    (OUT / f"{attempt['id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


@lru_cache(maxsize=None)
def refs_selection_scripted():
    from baseline_dynamic_reach.grid import reference_stats
    return reference_stats(selection_tags())["scripted"]["all"]["score"]


def index(rows, refs):
    scripted = refs["scripted"]["all"]["score"]
    overview_figure([r for r in rows if r["score"] is not None], scripted)
    lines = ["# Every attempt, with its reward, approach, settings, results, curves and demos", "",
             "*Generated by [`attempts.py`](../attempts.py) from the saved configurations, evaluations and training logs; "
             "do not edit by hand. The full log, with every prediction written before its run, is "
             "[`EXPERIMENT_NOTES.md`](../EXPERIMENT_NOTES.md); the summary is [`REPORT.md`](../REPORT.md).*", "",
             "**Task.** Ego must drive into a slot (FRONT, BACK, LEFT or RIGHT) of a moving traffic car within 25 s; one "
             "decision inside the slot box counts. **Score** = reached − crash per episode (+1, −1, 0 for a timeout).", "",
             "**An attempt** is one training pipeline: one or more runs, each starting from the previous run's trained "
             "policy. Its name says what it is:", "",
             "- **B0–B5** (Steps 1–7a): one reward change at a time on the full task, 1M transitions per run.",
             "- **P1–P6** (Step 7): practice on blocked episodes (a slow car in the slot's lane), then the full task.",
             "- **Step 8 pipelines**, named `<order>_<reward>[_n256]`, three rounds of 2M transitions:",
             "  - order: **N** no practice; **H** practice in round 1; **E** practice with easy starts in round 1; "
             "**O** the original P4 → P5 → P6 order (practice in rounds 1 and 2). Round 1 is always success only.",
             "  - reward: **c** crash cost (c0, c05 = 0.5, c1, c2 times the goal), **w** progress shaping "
             "(w0 none, w01 0.1 per m), **t** time cost (t1 = 0.001, t4 = 0.004 per decision);",
             "  - `_n256`: 256-256 networks instead of 64-64.", "",
             "Every page has the full reward function, the approach and every hyperparameter, the results (percentages "
             "and score), training curves, a success and a crash demo, and the idea and analysis behind it.", "",
             "## All attempts on the same 200 episodes", "",
             "Full task, episodes 10,000–10,199, final model, mean over seeds. The scripted driver reads the true "
             f"simulator state: {pct(refs['scripted']['all']['reached'])} reached, {pct(refs['scripted']['all']['crash'])} "
             f"crash, score {sc(scripted)}.", "", "![overview](figures/overview.png)", ""]
    groups = {"Steps 1–7: B0–B5 and P1–P6": [r for r in rows if not r["step"].startswith("Step 8")]}
    for step in dict.fromkeys(r["step"] for r in rows if r["step"].startswith("Step 8")):
        groups[f"{step}: {STEP8_HEADINGS[step]}"] = [r for r in rows if r["step"] == step]
    for heading, members in groups.items():
        lines += [f"### {heading}", ""]
        body = [[f"[{r['id']}]({r['id']}.md)", r["step"].replace("Step ", ""), r["title"], r["reward"], r["width"],
                 r["transitions"], r["seeds"], pct(r["reached"]), pct(r["crash"]), pct(r["timeout"]),
                 f"**{sc(r['score'])}**", r["status_short"]] for r in members]
        lines += table(["Attempt", "Step", "Approach", "Reward (last run)", "Network", "Training, whole pipeline", "Seeds",
                        "Reached", "Crash", "Timeout", "Score", "Status"], body) + [""]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pages(args):
    from baseline_dynamic_reach.attempts_text import text_for
    OUT.mkdir(exist_ok=True)
    refs_full = references(str(GRID / "references" / "s10000" / "evaluation_episodes.csv"))
    rows = []
    for attempt in attempts():
        if args.only and attempt["id"] not in args.only:
            continue
        text = text_for(attempt)
        summary = page(attempt, text, refs_full)
        m = summary["mean"]["all"] if summary else None
        total = sum(arm_config(a)["requested_timesteps"] for a in attempt["lineage"])
        rows.append({"id": attempt["id"], "step": attempt["step"], "title": text["title"], "reward": reward_short(attempt["final"]),
                     "width": "-".join(map(str, width_of(attempt["final"]))), "transitions": f"{total / 1e6:g}M",
                     "seeds": len(attempt["seeds"]), "reached": m and m["reached"], "crash": m and m["crash"],
                     "timeout": m and m["timeout"], "score": m and m["score"], "status_short": text["status_short"],
                     "seed_scores": [s["all"]["score"] for s in summary["per_seed"].values()] if summary else []})
        print(f"{attempt['id']}: page written", flush=True)
    if not args.only:
        index(rows, refs_full)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("demos")
    rec.add_argument("--jobs", type=int, default=16)
    pg = sub.add_parser("pages")
    pg.add_argument("--only", nargs="+", help="write only these attempts' pages (and no index)")
    sub.add_parser("list")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.command == "demos":
        demos(args)
    elif args.command == "pages":
        pages(args)
    else:
        for a in attempts():
            run = f"{a['final']}_seed{DEMO_SEED}"
            print(a["id"], a["step"], a["lineage"], a.get("long"), a["seeds"], demo_episodes(run))


if __name__ == "__main__":
    main()
