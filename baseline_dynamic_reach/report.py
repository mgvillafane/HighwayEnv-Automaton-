"""Tables and the outcome figure for REPORT.md, computed from the saved evaluations (nothing is re-run).

    python -B baseline_dynamic_reach/report.py           # tables and figure
    python -B baseline_dynamic_reach/report.py --pdf     # also REPORT.pdf from REPORT.md (needs Google Chrome)

Writes results/report/tables.md and results/report/outcomes_blr.png. The tables in REPORT.md are
pasted from tables.md; re-running this script and comparing the two is the check that none was
copied wrong.
"""
import argparse
import csv
import html
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_dynamic_reach.blocked import TARGET_SLOTS, count, evaluated_runs, saved_outcomes, scripted_outcomes
from baseline_dynamic_reach.config import ARMS, RESULTS_DIR, SEEDS, run_name

OUTPUT_DIR = RESULTS_DIR / "report"
CHANGE = {  # One short line per arm for the tables; the full reason is ARMS[arm]["reason"].
    "B0": "50 x success",
    "B1": "+ crash -50, from scratch",
    "B2": "B1, starting from B0's policy",
    "B3": "+ potential-based progress",
    "B3m": "B3 trained again on the Mac",
    "B4": "crash -50 -> -150",
}
# Outcome colors: the reference status palette (good / critical) and a neutral gray for timeouts.
COLORS = {"S": "#0ca30c", "C": "#d03b3b", "T": "#a8a79e"}
LABELS = {"S": "reached", "C": "crash", "T": "timeout"}
INK, MUTED, GRID = "#0b0b0b", "#5f5e58", "#e1e0d9"
# The episode compared in REPORT.md section 4, and the drivers shown (reference categorical order).
COMPARE_EPISODE = "10002"
COMPARE_RUNS = (("B3_seed7", "B3 (seed 7)", "#2a78d6"), ("B4_seed7", "B4 (seed 7)", "#eb6834"),
                ("scripted", "scripted driver", "#1baf7a"))


def episode_slots():
    with (RESULTS_DIR / "step5_blocked" / "episodes.csv").open(encoding="utf-8") as handle:
        return {int(row["seed"]): (row["slot"], row["blocked"] == "True") for row in csv.DictReader(handle)}


def split(outcomes, slots, keep):
    return [outcome for seed, outcome in sorted(outcomes.items()) if keep(*slots[seed])]


def pct(part, whole):
    return f"{100 * part / whole:.1f}%"


def tables(runs, scripted, random, slots):
    blr = lambda slot, blocked: slot in TARGET_SLOTS
    lines = ["# Tables for REPORT.md (written by report.py; do not edit by hand)", "",
             "## Outcomes per arm", "",
             "Overall: 200 evaluation episodes per seed, mean over seeds 7, 8, 9. "
             "BACK/LEFT/RIGHT: reached / crash / timeout of 138 episodes, per seed. FRONT: reached of 62, per seed.", "",
             "| Arm | Change from parent | Success (overall) | Crash | Timeout | BACK/LEFT/RIGHT, seeds 7 · 8 · 9 | FRONT |",
             "|---|---|---:|---:|---:|---|---|"]

    def row(name, change, per_seed):
        overall = {letter: statistics.mean(count(o.values(), letter) / len(o) for o in per_seed) for letter in "SCT"}
        sd = statistics.stdev(count(o.values(), "S") / len(o) for o in per_seed) if len(per_seed) > 1 else None
        success = f"{100 * overall['S']:.1f}%" + (f" ± {100 * sd:.1f}" if sd is not None else "")
        cells = " · ".join("/".join(str(count(split(o, slots, blr), letter)) for letter in "SCT") for o in per_seed)
        front = " · ".join(str(count(split(o, slots, lambda s, b: s == "front"), "S")) for o in per_seed)
        lines.append(f"| {name} | {change} | {success} | {100 * overall['C']:.1f}% | {100 * overall['T']:.1f}% | {cells} | {front} |")

    for arm, per_seed in runs.items():
        row(arm, CHANGE.get(arm, ARMS[arm]["iv"]), [per_seed[seed] for seed in SEEDS])
    row("Scripted", "rule-based, reads the true state", [scripted])
    row("Random", "uniform random actions", [random])

    lines += ["", "## BACK/LEFT/RIGHT by tag", "",
              "Blocked: at reset a car is in the slot's lane, between ego and the slot (45 episodes); clear: the other 93. "
              "Reached / crash / timeout, summed over the three seeds (so out of 3 × 45 = 135 and 3 × 93 = 279).", "",
              "| Arm | Clear: reached / crash / timeout (of 279) | Blocked: reached / crash / timeout (of 135) |", "|---|---|---|"]
    for arm, per_seed in runs.items():
        cells = []
        for tag in (False, True):
            outs = [x for seed in SEEDS for x in split(per_seed[seed], slots, lambda s, b: s in TARGET_SLOTS and b == tag)]
            cells.append("/".join(str(count(outs, letter)) for letter in "SCT"))
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} |")
    cells = []
    for tag in (False, True):
        outs = split(scripted, slots, lambda s, b: s in TARGET_SLOTS and b == tag)
        cells.append("/".join(str(count(outs, letter)) for letter in "SCT") + f" (one run, of {len(outs)})")
    lines.append(f"| Scripted | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def figure(runs, scripted, slots, path):
    blr = lambda slot, blocked: slot in TARGET_SLOTS
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width, x, ticks = 0.26, 0.0, []
    groups = [(arm, [runs[arm][seed] for seed in SEEDS]) for arm in runs] + [("Scripted", [scripted])]
    for name, per_seed in groups:
        offsets = [(i - (len(per_seed) - 1) / 2) * (width + 0.04) for i in range(len(per_seed))]
        for offset, outcomes in zip(offsets, per_seed):
            outs = split(outcomes, slots, blr)
            bottom = 0
            for letter in "SCT":
                height = count(outs, letter)
                if height:
                    ax.bar(x + offset, height, width, bottom=bottom, color=COLORS[letter],
                           edgecolor="#fcfcfb", linewidth=1.5)
                bottom += height
        ticks.append((x, name))
        x += 1.15
    reference = count(split(scripted, slots, blr), "S")
    ax.axhline(reference, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks], fontsize=10, color=INK)
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 46, 92, 138])
    ax.set_ylabel("BACK/LEFT/RIGHT episodes (of 138)", fontsize=9, color=MUTED)
    ax.set_title("Outcomes on BACK/LEFT/RIGHT, one bar per training seed (7, 8, 9)", fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", length=0, colors=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[letter]) for letter in "SCT"]
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
    ax.legend(handles, [LABELS[letter] for letter in "SCT"] + [f"scripted\nreaches {reference}"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


REPORT = Path(__file__).resolve().parent / "REPORT.md"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
REPO_URL = "https://github.com/mgvillafane/HighwayEnv-Automaton-/blob/{branch}/baseline_dynamic_reach/"
PRINT_CSS = """
@page { size: Letter; margin: 16mm 15mm 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.42;
       color: #1b1b19; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; padding-bottom: 2pt; border-bottom: 1px solid #d9d8d0; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 11pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 16pt; }
code { font-family: Menlo, monospace; font-size: 8.4pt; background: #f2f1ec; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.2pt; break-inside: avoid; }
th, td { border: 1px solid #d9d8d0; padding: 3pt 4pt; vertical-align: top; text-align: left; }
th { background: #f2f1ec; }
img { max-width: 100%; display: block; margin: 6pt auto; }
td img { margin: 0; }
p > img:only-child { break-inside: avoid; }
a { color: #1f4e9c; text-decoration: none; }
em { color: #45443f; }
"""


def pdf(branch):
    """REPORT.md -> REPORT.pdf: Markdown to HTML here, then Chrome's own print engine (headless)."""
    import markdown
    if not CHROME.exists():
        raise SystemExit(f"Google Chrome not found at {CHROME}; the PDF needs it.")
    text = REPORT.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    # Links to files in this folder point to the branch on GitHub, so they work outside this machine.
    # Images stay local and are embedded by Chrome.
    body = re.sub(r'href="(?!https?:|#)([^"]+)"', lambda m: f'href="{REPO_URL.format(branch=branch)}{m.group(1)}"', body)
    title = html.escape(text.splitlines()[0].lstrip("# "))
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{REPORT.parent.as_uri()}/"><style>{PRINT_CSS}</style></head><body>{body}</body></html>')
    target = REPORT.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "report.html"
        source.write_text(page, encoding="utf-8")
        subprocess.run([str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--user-data-dir={folder}/profile",
                        f"--print-to-pdf={target}", source.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print(f"Wrote {target}.")


def compare_figure(path):
    """Speed, gap to the car ahead and distance to the slot over time, one line per driver, from the saved
    example traces (examples.py record). A crash ends a line with an x."""
    import json
    traces = {}
    for run, label, color in COMPARE_RUNS:
        saved = json.loads((RESULTS_DIR / "examples" / run / "traces.json").read_text(encoding="utf-8"))
        traces[run] = saved[COMPARE_EPISODE]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    fig.patch.set_facecolor("#fcfcfb")
    panels = (("ego_speed", "ego speed (m/s)"), ("front_gap", "gap to the car ahead (m)"),
              ("along_error", "distance to the slot (m)"))
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for run, label, color in COMPARE_RUNS:
            points = [p for p in traces[run] if p.get(key) is not None]
            t = [p["t"] for p in points]
            y = [p[key] for p in points]
            ax.plot(t, y, color=color, lw=2, label=label)
            crashed = len(traces[run]) < 240 and points and traces[run][-1].get("front_gap", 99) < 8
            if crashed:
                ax.plot(t[-1], y[-1], marker="x", color=color, ms=9, mew=2.5)
        if key == "ego_speed":
            front = [p for p in traces["scripted"] if p.get("front_speed") is not None]
            ax.plot([p["t"] for p in front], [p["front_speed"] for p in front], color=MUTED, lw=1, ls=(0, (4, 3)),
                    label="car ahead")
        if key == "front_gap":
            ax.axhline(5, color=MUTED, lw=0.8, ls=":")
            ax.text(11.2, 6.5, "one car length", fontsize=7.5, color=MUTED, ha="right")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(length=0, colors=MUTED, labelsize=8)
        ax.set_xlabel("time (s)", fontsize=8.5, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Episode {COMPARE_EPISODE} (BACK slot behind the 12 m/s lead car): x = crash", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", action="store_true", help="also render REPORT.md to REPORT.pdf")
    parser.add_argument("--branch", default="dynamic-reach-b4", help="branch the PDF's file links point to on GitHub")
    args = parser.parse_args()
    runs = evaluated_runs()
    scripted = scripted_outcomes(runs)
    first = next(iter(runs))
    random = saved_outcomes(run_name(first, SEEDS[0]), "random")
    slots = episode_slots()
    if set(slots) != set(scripted):
        raise ValueError("episodes.csv does not cover the saved evaluation episodes; run blocked.py tag first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables.md").write_text(tables(runs, scripted, random, slots), encoding="utf-8")
    figure(runs, scripted, slots, OUTPUT_DIR / "outcomes_blr.png")
    compare_figure(OUTPUT_DIR / f"episode_{COMPARE_EPISODE}.png")
    print(f"Arms: {', '.join(runs)}. Wrote {OUTPUT_DIR / 'tables.md'} and {OUTPUT_DIR / 'outcomes_blr.png'}.")
    if args.pdf:
        pdf(args.branch)


if __name__ == "__main__":
    main()
