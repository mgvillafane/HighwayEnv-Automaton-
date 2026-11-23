"""Learning curves of Step 8 recipes over all their rounds: would more training help?

    python -B baseline_dynamic_reach/curves.py evaluate --jobs 28            # 8d recipes: rounds 1-3 (6M), 64 wide
    python -B baseline_dynamic_reach/curves.py plot
    python -B baseline_dynamic_reach/curves.py evaluate --set 8e --jobs 28   # 8e: rounds 1-4 (9M), 64 and 256 wide
    python -B baseline_dynamic_reach/curves.py plot --set 8e

Two kinds of curve, for the 8 seeds of each recipe (screening 7, 10, 13 and confirmation 16-28):
- deterministic: every 200k checkpoint of every round, scored as the grid scores (reached - crash on the 400
  selection episodes, full task), with grid_run's own evaluate command. Practice rounds are scored on the
  full task as well. Outputs go next to the grid's: results/<run>/eval8/<set>_<steps>/. These curve points
  are not part of the grid's score (grid.snapshot_labels keeps only the scored snapshots).
- training: outcomes of the training episodes themselves (stochastic policy, the round's own task), in 200k bins.

Nothing is trained and no model is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from baseline_dynamic_reach.config import (ARMS, RESULTS_DIR, ROUND_LENGTHS, STEP8_CHECKPOINT_EVERY,
                                           STEP8_CONFIRM_SEEDS, STEP8_SCREEN_SEEDS, arm_config, step8_name)
from baseline_dynamic_reach.grid_run import (FULL_TASK_ARM, SELECTION_SETS, evaluate_command, evaluation_output,
                                             run_process)

SEEDS = STEP8_SCREEN_SEEDS + STEP8_CONFIRM_SEEDS
ROUND = 2_000_000
# name: (order, reward, width, description). 8d: the five crash-cost finalists that tied, then success only in
# each order for contrast. 8e: three of those finalists, 64 and 256 wide, with round 4 (3M more).
RECIPE_SETS = {
    "8d": {"last_round": 3, "out": "curves", "recipes": {
        "N_c05w0t1": ("N", "c05w0t1", "", "No practice; crash 0.5 + time 0.001 (winner)"),
        "O_c1w0t4": ("O", "c1w0t4", "", "Original order; crash 1 + time 0.004"),
        "O_c1w01": ("O", "c1w01", "", "Original order; crash 1 + progress (P6's)"),
        "N_c1w0t4": ("N", "c1w0t4", "", "No practice; crash 1 + time 0.004"),
        "N_c1w01": ("N", "c1w01", "", "No practice; crash 1 + progress (P6's)"),
        "N_c0w0": ("N", "c0w0", "", "No practice; success only"),
        "O_c0w0": ("O", "c0w0", "", "Original order; success only"),
    }},
    "8e": {"last_round": 4, "out": "curves_8e", "recipes": {
        f"{cell}{suffix}": (cell[0], cell[2:], width, f"{text}, {label}")
        for cell, text in (("N_c05w0t1", "No practice; crash 0.5 + time 0.001 (winner)"),
                           ("N_c1w0t4", "No practice; crash 1 + time 0.004"),
                           ("O_c1w0t4", "Original order; crash 1 + time 0.004"))
        for suffix, width, label in (("", "", "64-64"), ("_n256", "n256", "256-256"))
    }},
}
RECIPES = RECIPE_SETS["8d"]["recipes"]
SCRIPTED_SELECTION = 0.725  # Scripted driver on the same 400 selection episodes (summary_8d.md).


def pipeline(order, reward, width="", last_round=3):
    """Arm names of rounds 1 .. last_round."""
    chain = [step8_name(order, last_round, reward, width)]
    while arm_config(chain[-1]).get("init_policy"):
        chain.append(arm_config(chain[-1])["init_policy"])
    assert len(chain) == last_round, chain
    return chain[::-1]


def round_start(round_number):
    """Transitions trained before this round starts."""
    return sum(ROUND_LENGTHS[r] for r in range(1, round_number))


def checkpoints(run):
    folder = RESULTS_DIR / run / "checkpoints"
    return sorted((int(path.stem.split("_")[-2]), path) for path in folder.glob("*.zip"))


def curve_labels(run, length):
    """Checkpoint labels of a round, without the last one (the final model stands for it)."""
    return [str(steps) for steps, _ in checkpoints(run) if steps <= length - STEP8_CHECKPOINT_EVERY / 2]


def chains(recipe_set):
    spec = RECIPE_SETS[recipe_set]
    for recipe, (order, reward, width, _) in spec["recipes"].items():
        yield recipe, pipeline(order, reward, width, spec["last_round"])


def evaluation_jobs(recipe_set):
    runs = sorted({f"{arm}_seed{seed}" for _, chain in chains(recipe_set) for arm in chain for seed in SEEDS})
    jobs = []
    for run in runs:
        arm = run.rsplit("_seed", 1)[0]
        full_task = FULL_TASK_ARM if arm_config(arm)["environment"].get("scenario") else None
        length = ROUND_LENGTHS[ARMS[arm]["round"]]
        for steps, path in checkpoints(run):
            if steps > length - STEP8_CHECKPOINT_EVERY / 2:
                continue  # The last checkpoint is the final model, which the grid has already scored.
            for name, (first, count) in SELECTION_SETS.items():
                job = (run, path, name, str(steps), first, count, full_task)
                if not (evaluation_output(run, name, str(steps)) / "evaluation_episodes.csv").exists():
                    jobs.append(job)
    return runs, jobs


def evaluate(args):
    runs, jobs = evaluation_jobs(args.set)
    print(f"{len(runs)} runs; {len(jobs)} checkpoint evaluations to do", flush=True)
    started = time.perf_counter()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for job in jobs:
            command, _ = evaluate_command(job)
            futures.append(pool.submit(run_process, f"{job[0]} {job[2]}_{job[3]}", command,
                                       RESULTS_DIR / "logs" / f"{job[0]}.log"))
        for done, future in enumerate(as_completed(futures), 1):
            label, code, _ = future.result()
            if code:
                failed += 1
                print(f"FAILED ({code}): {label}", flush=True)
            if done % 250 == 0 or done == len(futures):
                print(f"{time.strftime('%H:%M:%S')} {done}/{len(futures)} done, {failed} failed, "
                      f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    sys.exit(1 if failed else 0)


def deterministic_rows(recipe_set="8d"):
    """One row per (recipe, seed, snapshot): total transitions and outcome rates on the 400 selection episodes."""
    rows = []
    for recipe, chain in chains(recipe_set):
        for round_number, arm in enumerate(chain, 1):
            practice = bool(arm_config(arm)["environment"].get("scenario"))
            length = ROUND_LENGTHS[round_number]
            for seed in SEEDS:
                run = f"{arm}_seed{seed}"
                for label in curve_labels(run, length) + ["final"]:
                    frames = []
                    for name in SELECTION_SETS:
                        path = evaluation_output(run, name, label) / "evaluation_episodes.csv"
                        if not path.exists():
                            raise SystemExit(f"missing {path}; run `curves.py evaluate --set {recipe_set}` first")
                        frames.append(pd.read_csv(path))
                    episodes = pd.concat(frames)
                    assert len(episodes) == 400 and (episodes.policy == "ppo").all()
                    front = episodes[episodes.slot == "front"]
                    rows.append({
                        "recipe": recipe, "seed": seed, "round": round_number, "run": run, "practice_round": practice,
                        # A checkpoint is saved a few steps short of its nominal 200k mark (3 environments step together,
                        # e.g. 2,599,974 for 2.6M). Snap to the mark so the round windows below include it.
                        "transitions": round_start(round_number) + (length if label == "final" else
                                                                    round(int(label) / STEP8_CHECKPOINT_EVERY) * STEP8_CHECKPOINT_EVERY),
                        "checkpoint_steps": length if label == "final" else int(label),
                        "score": episodes.success.mean() - episodes.collision.mean(),
                        "reached": episodes.success.mean(), "crash": episodes.collision.mean(),
                        "timeout": episodes.timeout.mean(),
                        "front_reached": front.success.mean(), "front_crash": front.collision.mean(),
                        "front_timeout": front.timeout.mean(),
                    })
    return pd.DataFrame(rows)


def training_rows(recipe_set="8d"):
    """Training-episode outcomes in 200k bins (the round's own task; stochastic policy)."""
    rows = []
    for recipe, chain in chains(recipe_set):
        for round_number, arm in enumerate(chain, 1):
            practice = bool(arm_config(arm)["environment"].get("scenario"))
            bins = ROUND_LENGTHS[round_number] // STEP8_CHECKPOINT_EVERY
            for seed in SEEDS:
                episodes = pd.read_csv(RESULTS_DIR / f"{arm}_seed{seed}" / "training_episodes.csv")
                episodes["bin"] = np.minimum(episodes.timesteps // STEP8_CHECKPOINT_EVERY, bins - 1)
                for index, part in episodes.groupby("bin"):
                    front = part[part.slot == "front"]
                    rows.append({
                        "recipe": recipe, "seed": seed, "round": round_number, "practice_round": practice,
                        "transitions": round_start(round_number) + (index + 1) * STEP8_CHECKPOINT_EVERY,
                        "episodes": len(part), "score": part.success.mean() - part.collision.mean(),
                        "reached": part.success.mean(), "crash": part.collision.mean(), "timeout": part.timeout.mean(),
                        "front_timeout": front.timeout.mean() if len(front) else np.nan,
                    })
    return pd.DataFrame(rows)


def paired(differences):
    from baseline_dynamic_reach.grid import paired as grid_paired
    return grid_paired(list(differences))


def window(frame, low, high, column="score"):
    """Per-seed mean of a column over snapshots with low <= transitions <= high."""
    part = frame[(frame.transitions >= low) & (frame.transitions <= high)]
    return part.groupby("seed")[column].mean()


def slope(frame, low):
    """Per-seed least-squares slope (per 1M transitions) of the score from `low` transitions on, paired over seeds."""
    last = frame[frame.transitions >= low - 1]
    return paired(last.groupby("seed").apply(lambda g: np.polyfit(g.transitions / 1e6, g.score, 1)[0]))


def summary(det, out):
    lines = ["# Learning curves of the Step 8 recipes (6M transitions, 8 seeds)", "",
             "Deterministic score = reached − crash on the 400 selection episodes (full task), every 200k checkpoint. "
             "Window means are per seed, then paired over the 8 seeds (t-test). Practice rounds (order O, rounds 1–2) "
             "are trained on practice episodes but scored on the full task.", "",
             "| Recipe | End of round 1 (2M) | End of round 2 (4M) | Round 3, 4.2–5.0M | Round 3, 5.2–6.0M | "
             "Last 1M − previous 1M (p, seeds up) | Slope over 5.0–6.0M per 1M (p) | Peak snapshot (mean) |",
             "|---|---:|---:|---:|---:|---|---|---:|"]
    stats = {}
    for recipe in RECIPES:
        frame = det[det.recipe == recipe]
        end1 = frame[frame.transitions == ROUND].groupby("seed").score.mean()
        end2 = frame[frame.transitions == 2 * ROUND].groupby("seed").score.mean()
        early3 = window(frame, 4.2e6 - 1, 5.0e6 + 1)
        late3 = window(frame, 5.2e6 - 1, 6.0e6 + 1)
        gain = paired(late3 - early3)
        trend = slope(frame, 5.0e6)
        curve = frame.groupby("transitions").score.mean()
        stats[recipe] = {"gain": gain, "slope": trend}
        lines.append(f"| {recipe} | {end1.mean():+.3f} | {end2.mean():+.3f} | {early3.mean():+.3f} | {late3.mean():+.3f} | "
                     f"{gain['mean']:+.3f} (p {gain['p']:.2f}, {gain['ahead']} of 8) | "
                     f"{trend['mean']:+.3f} (p {trend['p']:.2f}) | {curve.max():+.3f} at {curve.idxmax() / 1e6:.1f}M |")
    lines += ["", "**Order N, round 3 against round 2.** Rounds 2 and 3 of order N have the same task and reward, so round 3 "
              "is 2M more transitions of the same training. Score = mean of the last 3 snapshots of each round.", "",
              "| Recipe | Round 2 (3.6–4.0M) | Round 3 (5.6–6.0M) | Difference (95% interval, p, seeds up) | Crash change | Timeout change |",
              "|---|---:|---:|---|---:|---:|"]
    for recipe, (order, _, _, _) in RECIPES.items():
        if order != "N":
            continue
        frame = det[det.recipe == recipe]
        two = frame[(frame.transitions >= 3.6e6 - 1) & (frame.transitions <= 4.0e6 + 1)].groupby("seed")
        three = frame[(frame.transitions >= 5.6e6 - 1) & (frame.transitions <= 6.0e6 + 1)].groupby("seed")
        diff = paired(three.score.mean() - two.score.mean())
        lines.append(f"| {recipe} | {two.score.mean().mean():+.3f} | {three.score.mean().mean():+.3f} | "
                     f"{diff['mean']:+.3f} ([{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}], p {diff['p']:.2f}, {diff['ahead']} of 8) | "
                     f"{(three.crash.mean() - two.crash.mean()).mean() * 100:+.1f} pts | "
                     f"{(three.timeout.mean() - two.timeout.mean()).mean() * 100:+.1f} pts |")
    lines += ["", "**FRONT give-up over round 3** (share of the 133 FRONT selection episodes that time out; mean of seeds):", "",
              "| Recipe | 4.2M | 4.6M | 5.0M | 5.4M | 5.8M | 6.0M | Seeds with ≥ 30% FRONT timeouts at 6.0M |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for recipe in RECIPES:
        frame = det[det.recipe == recipe]
        cells = []
        for point in (4.2e6, 4.6e6, 5.0e6, 5.4e6, 5.8e6, 6.0e6):
            near = frame[(frame.transitions - point).abs() < 50_000]
            cells.append(f"{near.front_timeout.mean() * 100:.0f}%")
        final = frame[frame.transitions == 3 * ROUND].set_index("seed").front_timeout
        stuck = ", ".join(str(seed) for seed, value in final.items() if value >= 0.3) or "none"
        lines.append(f"| {recipe} | " + " | ".join(cells) + f" | {stuck} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def summary_8e(det, out):
    """Round 4 (3M more of round 3's training) at each width: does the score keep rising, and does FRONT give-up fall?"""
    recipes = RECIPE_SETS["8e"]["recipes"]
    end3, end4 = round_start(4), round_start(5)
    lines = ["# Learning curves of Step 8e (9M transitions, 8 seeds, 64 and 256 wide)", "",
             "Deterministic score = reached − crash on the 400 selection episodes (full task), every 200k checkpoint. "
             "Round 4 continues round 3 for 3M more transitions (same task and reward). End of a round = mean of its last "
             "3 snapshots (the grid's scored snapshots). Per-seed values, paired over the 8 seeds (t-test).", "",
             "| Recipe | Width | End of round 2 (4M) | End of round 3 (6M) | End of round 4 (9M) | Round 4 − round 3 (95% interval, p, seeds up) | "
             "Slope over the last 1M, per 1M (p) | Peak snapshot (mean) |",
             "|---|---|---:|---:|---:|---|---|---:|"]
    for recipe, (_, _, width, _) in recipes.items():
        frame = det[det.recipe == recipe]
        two = window(frame, round_start(3) - 400_001, round_start(3) + 1)
        three = window(frame, end3 - 400_001, end3 + 1)
        four = window(frame, end4 - 400_001, end4 + 1)
        gain = paired(four - three)
        trend = slope(frame, end4 - 1_000_000)
        curve = frame.groupby("transitions").score.mean()
        lines.append(f"| {recipe.replace('_n256', '')} | {'256' if width else '64'} | {two.mean():+.3f} | {three.mean():+.3f} | "
                     f"{four.mean():+.3f} | {gain['mean']:+.3f} ([{gain['ci95'][0]:+.3f}, {gain['ci95'][1]:+.3f}], "
                     f"p {gain['p']:.2f}, {gain['ahead']} of 8) | {trend['mean']:+.3f} (p {trend['p']:.2f}) | "
                     f"{curve.max():+.3f} at {curve.idxmax() / 1e6:.1f}M |")
    lines += ["", "**FRONT give-up** (share of the 133 FRONT selection episodes that time out; mean of seeds; end of a round = "
              "mean of its last 3 snapshots):", "",
              "| Recipe | Width | End of round 2 | End of round 3 | End of round 4 | Seeds with ≥ 30% FRONT timeouts at the end of round 4 | "
              "FRONT reached, end of round 4 | FRONT crash, end of round 4 |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for recipe, (_, _, width, _) in recipes.items():
        frame = det[det.recipe == recipe]
        ends = [window(frame, end - 400_001, end + 1, "front_timeout") for end in (round_start(3), end3, end4)]
        stuck = ", ".join(str(seed) for seed, value in ends[2].items() if value >= 0.3) or "none"
        reached = window(frame, end4 - 400_001, end4 + 1, "front_reached").mean()
        crash = window(frame, end4 - 400_001, end4 + 1, "front_crash").mean()
        lines.append(f"| {recipe.replace('_n256', '')} | {'256' if width else '64'} | "
                     + " | ".join(f"{end.mean() * 100:.0f}%" for end in ends)
                     + f" | {stuck} | {reached * 100:.0f}% | {crash * 100:.0f}% |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Reference palette (dataviz skill): ink, muted labels, hairline grid, baseline; categorical slots 1-3.
    colors = {"ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
              "surface": "#fcfcfb", "practice": "#f1f0ea", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": colors["baseline"], "axes.labelcolor": colors["secondary"],
                         "xtick.color": colors["muted"], "ytick.color": colors["muted"], "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": colors["surface"], "axes.facecolor": colors["surface"]})
    return plt, colors


def plot(args):
    if args.set == "8e":
        return plot_8e()
    plt, c = style()
    out = RESULTS_DIR / "grid8" / RECIPE_SETS["8d"]["out"]
    out.mkdir(parents=True, exist_ok=True)
    det = deterministic_rows("8d")
    det.to_csv(out / "deterministic.csv", index=False)
    train = training_rows("8d")
    train.to_csv(out / "training.csv", index=False)
    summary(det, out)

    seed_color, mean_color, ref_color = c["baseline"], c["blue"], c["secondary"]
    overlay = {"N_c05w0t1": c["blue"], "O_c1w0t4": c["orange"], "N_c0w0": c["aqua"]}  # At most 3 series in one panel.

    def panel_grid(frame, column, ylabel, title, filename, ylim, reference=None, legend_loc="lower right"):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        recipes = list(RECIPES)
        for ax, recipe in zip(axes.flat, recipes):
            part = frame[frame.recipe == recipe]
            order = RECIPES[recipe][0]
            if order == "O":
                ax.axvspan(0, 4, color=c["practice"], zorder=0)
                ax.text(2, ylim[0] + 0.03 * (ylim[1] - ylim[0]), "practice episodes", ha="center", color=c["muted"], fontsize=8)
            for boundary in (2, 4):
                ax.axvline(boundary, color=c["grid"], lw=0.8, zorder=1)
            for seed, line in part.groupby("seed"):
                line = line.sort_values("transitions")
                ax.plot(line.transitions / 1e6, line[column], color=seed_color, lw=0.8, zorder=2)
            mean = part.groupby("transitions")[column].mean()
            ax.plot(mean.index / 1e6, mean.values, color=mean_color, lw=2.0, zorder=3)
            if reference is not None:
                ax.axhline(reference, color=ref_color, lw=1.0, ls="--", zorder=2)
            ax.set_title(RECIPES[recipe][3], fontsize=9, color=c["ink"], loc="left")
            ax.set_ylim(*ylim)
            ax.set_xlim(0, 6)
            ax.grid(axis="y", color=c["grid"], lw=0.5)
        ax = axes.flat[-1]
        for recipe, color in overlay.items():
            mean = frame[frame.recipe == recipe].groupby("transitions")[column].mean()
            ax.plot(mean.index / 1e6, mean.values, color=color, lw=2.0, label=RECIPES[recipe][3])
        if reference is not None:
            ax.axhline(reference, color=ref_color, lw=1.0, ls="--")
        for boundary in (2, 4):
            ax.axvline(boundary, color=c["grid"], lw=0.8)
        ax.set_title("Means: winner, best original order, success only", fontsize=9, color=c["ink"], loc="left")
        ax.legend(fontsize=7, frameon=False, loc=legend_loc, labelcolor=c["secondary"])
        ax.grid(axis="y", color=c["grid"], lw=0.5)
        for ax in axes[1]:
            ax.set_xlabel("transitions (millions); rounds end at 2M and 4M")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        note = "thin: each of 8 seeds; thick: mean"
        if reference is not None:
            note += f"; dashed: scripted driver ({reference:+.3f})"
        fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", fontsize=10, color=c["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)

    panel_grid(det, "score", "score (reached − crash)", "Deterministic score on the 400 selection episodes, every 200k checkpoint",
               "score.png", (-0.6, 1.0), SCRIPTED_SELECTION)
    panel_grid(det, "front_timeout", "FRONT episodes timed out", "FRONT give-up: share of the 133 FRONT selection episodes that time out",
               "front_timeout.png", (0, 1.0), legend_loc="upper left")
    panel_grid(train, "score", "training score (reached − crash)",
               "Training episodes (stochastic policy, the round's own task), 200k bins", "training.png", (-0.5, 1.0))
    sys.stdout.reconfigure(encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


def plot_8e():
    """One panel per recipe, 64 and 256 wide overlaid (two series: slots 1 and 2 of the reference palette)."""
    plt, c = style()
    spec = RECIPE_SETS["8e"]
    out = RESULTS_DIR / "grid8" / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    det = deterministic_rows("8e")
    det.to_csv(out / "deterministic.csv", index=False)
    summary_8e(det, out)
    widths = {"": ("64-64", c["blue"]), "n256": ("256-256", c["orange"])}
    cells = list(dict.fromkeys(recipe.replace("_n256", "") for recipe in spec["recipes"]))
    boundaries = [round_start(r) / 1e6 for r in (2, 3, 4)]
    end = round_start(5) / 1e6

    def panels(column, ylabel, title, filename, ylim, reference=None, legend_loc="lower right"):
        fig, axes = plt.subplots(1, len(cells), figsize=(15, 4.6), sharey=True)
        for ax, cell in zip(axes, cells):
            if cell.startswith("O"):
                ax.axvspan(0, 4, color=c["practice"], zorder=0)
                ax.text(2, ylim[0] + 0.03 * (ylim[1] - ylim[0]), "practice episodes", ha="center", color=c["muted"], fontsize=8)
            for boundary in boundaries:
                ax.axvline(boundary, color=c["grid"], lw=0.8, zorder=1)
            for width, (label, color) in widths.items():
                part = det[det.recipe == (f"{cell}_{width}" if width else cell)]
                for _, line in part.groupby("seed"):
                    line = line.sort_values("transitions")
                    ax.plot(line.transitions / 1e6, line[column], color=color, lw=0.6, alpha=0.22, zorder=2)
                mean = part.groupby("transitions")[column].mean()
                ax.plot(mean.index / 1e6, mean.values, color=color, lw=2.0, zorder=3, label=f"{label} (mean of 8 seeds)")
            if reference is not None:
                ax.axhline(reference, color=c["secondary"], lw=1.0, ls="--", zorder=2)
            ax.set_title(spec["recipes"][cell][3].rsplit(", ", 1)[0], fontsize=9, color=c["ink"], loc="left")
            ax.set_xlim(0, end)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color=c["grid"], lw=0.5)
            ax.set_xlabel("transitions (millions); rounds end at 2, 4, 6 and 9M")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8, frameon=False, loc=legend_loc, labelcolor=c["secondary"])
        note = "thin: each seed; thick: mean of 8 seeds; round 4 (6–9M) is 3M more of round 3's training"
        if reference is not None:
            note += f"; dashed: scripted driver ({reference:+.3f})"
        fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", fontsize=10, color=c["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)

    panels("score", "score (reached − crash)", "Step 8e: score on the 400 selection episodes, every 200k checkpoint, 64 vs 256 wide",
           "score.png", (-0.6, 1.0), SCRIPTED_SELECTION)
    panels("front_timeout", "FRONT episodes timed out", "Step 8e: FRONT give-up, share of the 133 FRONT selection episodes that time out",
           "front_timeout.png", (0, 1.0), legend_loc="upper right")
    sys.stdout.reconfigure(encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--jobs", type=int, default=28)
    run.add_argument("--set", choices=list(RECIPE_SETS), default="8d")
    drawing = sub.add_parser("plot")
    drawing.add_argument("--set", choices=list(RECIPE_SETS), default="8d")
    args = parser.parse_args()
    {"evaluate": evaluate, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
