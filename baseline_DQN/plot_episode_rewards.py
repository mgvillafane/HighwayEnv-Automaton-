"""Merge quickstart episode logs and plot rewards from independent training seeds."""
import argparse
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

from baseline_DQN.config import RESULTS_DIR, experiment_name
from baseline_DQN.summarize_quickstart import read_rows, rolling


def plot_rewards(seeds, window, output_name):
    if not seeds or len(set(seeds)) != len(seeds) or window < 1:
        raise ValueError("Provide distinct training seeds and a positive smoothing window.")
    runs, merged = [], []
    source_fields = None
    for seed in seeds:
        run_dir = RESULTS_DIR / f"dqn_quickstart_seed{seed}"
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (metadata["preset"] != "quickstart" or metadata["status"] != "trained"
                or metadata["seed"] != seed):
            raise ValueError(f"{run_dir.name} is not the expected completed quickstart run.")
        rows = read_rows(run_dir / "training_episodes.csv")
        if not rows:
            raise ValueError(f"No completed training episodes in {run_dir}.")
        fields = list(rows[0])
        if source_fields is not None and source_fields != fields:
            raise ValueError("Training episode CSV columns differ between runs.")
        source_fields = fields
        steps = np.array([int(row["timesteps"]) for row in rows])
        rewards = np.array([float(row["reward"]) for row in rows])
        if (np.any(np.diff(steps) <= 0) or not np.isfinite(rewards).all()
                or steps[-1] > metadata["actual_timesteps"]):
            raise ValueError(f"Invalid training steps or rewards in {run_dir}.")
        runs.append((seed, steps, rewards, rolling(rewards, window)))
        merged.extend({"training_seed": seed, "episode": episode, **row}
                      for episode, row in enumerate(rows, start=1))

    # Align independent runs by training transitions, without extrapolating beyond
    # any run's completed episodes. Each seed contributes equally to the mean.
    start = max(steps[0] for _, steps, _, _ in runs)
    end = min(steps[-1] for _, steps, _, _ in runs)
    if start >= end:
        raise ValueError("Runs have no overlapping range of training transitions.")
    common_steps = np.linspace(start, end, 400)
    mean_rewards = np.mean([np.interp(common_steps, steps, smoothed)
                            for _, steps, _, smoothed in runs], axis=0)

    output_dir = RESULTS_DIR / experiment_name(output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axis = plt.subplots(figsize=(12, 6.5), layout="constrained")
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
        for index, (seed, steps, rewards, smoothed) in enumerate(runs):
            color = colors[index % len(colors)]
            axis.scatter(steps, rewards, color=color, s=7, alpha=0.13,
                         linewidths=0, rasterized=True)
            axis.plot(steps, smoothed, color=color, linewidth=1.8,
                      label=f"Seed {seed} ({len(rewards):,} episodes)")
        axis.plot(common_steps, mean_rewards, color="#20252B", linewidth=3,
                  label="Mean across seeds", zorder=5)
        axis.set(title="DQN quickstart: training episode rewards",
                 xlabel="Training transitions per seed",
                 ylabel="Episode return (sum of native rewards)",
                 xlim=(0, max(steps[-1] for _, steps, _, _ in runs)))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.legend(loc="lower right", fontsize=9, framealpha=0.95)
        fig.suptitle(f"Seeds {', '.join(map(str, seeds))} | {len(merged):,} completed episodes\n"
                     f"Dots: every episode. Colored lines: trailing mean of up to {window} episodes.\n"
                     "Black line: equally weighted mean of aligned seed curves; exploration included.",
                     fontsize=10, color="#475569")
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"episode_rewards_merged.{extension}", dpi=200)
        plt.close(fig)

    merged.sort(key=lambda row: (int(row["timesteps"]), row["training_seed"]))
    csv_path = output_dir / "training_episodes_merged.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["training_seed", "episode", *source_fields])
        writer.writeheader()
        writer.writerows(merged)
    print(json.dumps({"episodes_by_seed": {str(seed): len(rewards)
                      for seed, _, rewards, _ in runs}, "total_episodes": len(merged),
                      "smoothing_window_episodes": window, "output_dir": str(output_dir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output-name", default="quickstart_comparison")
    args = parser.parse_args()
    plot_rewards(args.seeds, args.window, args.output_name)


if __name__ == "__main__":
    main()
