"""Plot pooled training-episode success for the four reward experiments."""
import csv
from plot_success_rates import CASES, GOALS, ROOT, OUT
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    values, records = [], []
    for label, name, *_ in CASES:
        with (ROOT/name/"training_episodes.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        goal_values = []
        for goal in GOALS:
            subset = [row for row in rows if row["slot"] == goal]
            successes = sum(int(row["success"]) for row in subset)
            total = len(subset)
            rate = 100*successes/total
            goal_values.append((successes, total, rate))
            records.append(dict(experiment=label, run=name, goal=goal, successes=successes,
                                training_episodes=total, success_percent=rate))
        values.append(goal_values)
    matrix = np.array([[value[2] for value in row] for row in values])
    fig, ax = plt.subplots(figsize=(12.5, 7.3), facecolor="white")
    fig.subplots_adjust(left=.31, right=.965, top=.79, bottom=.22)
    cmap = LinearSegmentedColormap.from_list("training_success", ["#F0F5F6", "#77B8BF", "#087E8B"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    labels = ["Simple reward\npretrained + 200k", "Timeout + dropout\n53 inputs; stopped run",
              "Route progress + tracking\n2M steps", "No tracking + FRONT side bonus\n1.5M steps"]
    ax.set_yticks(range(4), labels, fontsize=11, color="#233643")
    ax.set_xticks(range(4), [goal.upper() for goal in GOALS], fontsize=12, weight="bold", color="#233643")
    ax.tick_params(axis="both", length=0, pad=14)
    for spine in ax.spines.values(): spine.set_visible(False)
    for row in range(4):
        for col in range(4):
            successes, total, rate = values[row][col]
            label = "0%" if rate == 0 else f"{rate:.3f}%" if rate < .1 else f"{rate:.1f}%"
            color = "white" if rate >= 65 else "#233643"
            ax.text(col, row-.10, label, ha="center", va="center", fontsize=18, weight="bold", color=color)
            ax.text(col, row+.17, f"{successes:,} / {total:,}", ha="center", va="center", fontsize=11, color=color)
    ax.set_xticks(np.arange(-.5,4,1), minor=True)
    ax.set_yticks(np.arange(-.5,4,1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    for column in (1,3):
        ax.add_patch(Rectangle((column-.5,.5),1,1,fill=False,edgecolor="#D28D24",linewidth=2))
    fig.text(.055,.955,"Success rates across training history", fontsize=23, weight="bold", color="#233643", va="top")
    fig.text(.055,.895,"All completed training episodes; each cell shows success rate and successes / episodes.",
             fontsize=11.5, color="#536A78")
    fig.text(.055,.115,"Timeout/dropout recorded 1 FRONT and 1 RIGHT training success; both rare events are included.",
             fontsize=10.5, weight="bold", color="#233643")
    fig.text(.055,.072,"These pooled rates include exploration and changing policies throughout training. They are not final-policy evaluation rates.",
             fontsize=9.5, color="#536A78")
    fig.text(.055,.038,"Training budgets and initialization differ; the simple run's counts cover only its additional 200k steps.",
             fontsize=9.5, color="#536A78")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"training_history_success.{ext}", dpi=190, facecolor="white")
    plt.close(fig)
    with (OUT/"training_history_success.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(OUT/"training_history_success.png")


if __name__=="__main__": main()
