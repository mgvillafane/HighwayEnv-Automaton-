"""Compare final/latest deterministic monitoring results for four reward designs."""
import csv
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

CASES = [
    ("Simple reward", "ppo_singlecar_flexible30s_seed7_200k", "evaluation.json", 200_000,
     "Simple reward | pretrained + 200k", "#648096"),
    ("Timeout + dropout", "ppo_flexible53_timeout10_dropout005_scratch_seed7_2m", "evaluation_1250000.json", 1_250_000,
     "Timeout + dropout | latest evaluation, 1.25M", "#DF9A35"),
    ("Route progress + tracking", "ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m", "evaluation.json", 2_000_000,
     "Route progress + tracking | final, 2M", "#0B8F8F"),
    ("No tracking + FRONT side bonus", "ppo_maneuver69_notrack_side10_seed7_1p5m", "evaluation.json", 1_500_000,
     "No tracking + FRONT side bonus | final, 1.5M", "#8062B4"),
]
GOALS = ["left", "right", "back", "front"]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#233643", "text.color": "#233643",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    summary, rows = [], []
    for label, name, filename, steps, legend, color in CASES:
        source = ROOT / name / filename
        result = json.loads(source.read_text())
        stats = result["ppo"]
        episodes_file = source.with_name(source.stem + "_episodes.csv")
        with episodes_file.open(newline="", encoding="utf-8") as handle:
            episodes = [r for r in csv.DictReader(handle) if r["policy"] == "ppo"]
        assert len(episodes) == stats["episodes"] == 100
        assert [int(r["seed"]) for r in episodes] == list(range(10000, 10100))
        rates, counts = [], []
        for goal in GOALS:
            subset = [r for r in episodes if r["slot"] == goal]
            success = sum(int(r["success"]) for r in subset)
            assert len(subset) == stats["episodes_by_slot"][goal]
            assert abs(success / len(subset) - stats["success_rate_by_slot"][goal]) < 1e-12
            rates.append(100 * success / len(subset))
            counts.append([success, len(subset)])
            rows.append(dict(experiment=label, run=name, evaluation=filename,
                             evaluated_steps=steps, goal=goal, successes=success,
                             episodes=len(subset), success_percent=100*success/len(subset),
                             collisions=sum(int(r["collision"]) for r in subset),
                             timeouts=sum(int(r["timeout"]) for r in subset)))
        summary.append(dict(label=label, run=name, source=str(source), rates=rates,
                            counts=counts, overall_success=stats["success_rate"],
                            collision_rate=stats["collision_rate"], color=color, legend=legend))
    prior_matched = json.loads((ROOT / CASES[2][1] / "evaluation_1500000.json").read_text())["ppo"]
    latest = json.loads((ROOT / CASES[3][1] / "evaluation.json").read_text())["ppo"]
    assert prior_matched["success_rate_by_slot"] == latest["success_rate_by_slot"]
    assert prior_matched["collision_rate"] == 0 and latest["collision_rate"] == .04

    fig, ax = plt.subplots(figsize=(14, 8.4), facecolor="white")
    fig.subplots_adjust(left=.075, right=.985, top=.74, bottom=.25)
    x, width = np.arange(4), .195
    for index, case in enumerate(summary):
        positions = x + (index - 1.5) * width
        bars = ax.bar(positions, case["rates"], width=width*.92,
                      color=case["color"], label=case["legend"], zorder=3)
        for bar, value, count in zip(bars, case["rates"], case["counts"]):
            label = "100%" if value == 100 else "0%" if value == 0 else f"{value:.1f}%"
            ax.text(bar.get_x()+bar.get_width()/2, value+1.5, label,
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
            if value > 0:
                ax.text(bar.get_x()+bar.get_width()/2, max(value*.5, 9), f"{count[0]}/{count[1]}",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x, ["LEFT\n33 episodes", "RIGHT\n22 episodes", "BACK\n27 episodes", "FRONT\n18 episodes"])
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0,101,20))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel("Evaluation success rate", labelpad=15)
    ax.grid(axis="y", color="#DFE7EB", linewidth=.8, zorder=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.075,.955,"Goal success across four reward designs", fontsize=23, weight="bold", va="top")
    fig.text(.075,.906,"Deterministic evaluation: the same 100 monitoring seeds (10000–10099)",
             fontsize=12, color="#556B78")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.067,.873),
               frameon=False, ncols=2, fontsize=10.6, columnspacing=2.4, labelspacing=.95)
    fig.text(.075,.14,"Checkpoint selection", fontsize=10.5, weight="bold")
    fig.text(.075,.11,"Final evaluations are shown where available. The interrupted timeout/dropout run uses its latest evaluation at 1.25M.",
             fontsize=9.5, color="#556B78")
    fig.text(.075,.082,"The simple run had 1,000,448 pretrained steps plus 200k additional steps. Training budgets and initialization differ.",
             fontsize=9.5, color="#556B78")
    fig.text(.075,.048,"Equal-budget check: at 1.5M, both route variants scored LEFT/RIGHT/BACK 100% and FRONT 77.8%.",
             fontsize=10, weight="bold")
    fig.text(.075,.021,"Collisions at 1.5M: route + tracking 0/100; no tracking + FRONT side bonus 4/100. Reused monitoring seeds are not a fresh test set.",
             fontsize=9.2, color="#556B78")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"success_by_goal.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    with (OUT/"success_rates.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (OUT/"comparison.json").write_text(json.dumps({"selection":"Final/latest recorded evaluations; not best checkpoints",
        "goals":GOALS,"cases":summary,"equal_budget_1500000":{"route_tracking":prior_matched,"no_tracking_side_bonus":latest}},indent=2)+"\n")
    print(json.dumps([{ "experiment":r["label"],"rates":r["rates"]} for r in summary],indent=2))
    print(OUT/"success_by_goal.png")


if __name__ == "__main__":main()
