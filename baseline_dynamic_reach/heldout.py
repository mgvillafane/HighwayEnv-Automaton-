"""Step 7h: P6 and B5 on 200 episodes no earlier step looked at (seeds 40,000-40,199).

    python -B baseline_dynamic_reach/evaluate.py --run P6_seed7 --seed 40000 --output-name heldout_40000/P6_seed7
    ... (the same for P6_seed8/9 and B5_seed7/8/9)
    python -B baseline_dynamic_reach/heldout.py

Tags every episode as blocked or clear with blocked.py's own test, joins the saved outcomes,
puts them beside the same runs on the usual 200 episodes (seeds 10,000-10,199), and checks the
five predictions written in EXPERIMENT_NOTES.md (Step 7h) before the evaluation was run.
Writes results/heldout_40000/summary.md and episodes.csv.
"""
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dynamic_reach.blocked import saved_outcomes, tag_episodes
from baseline_dynamic_reach.config import EVAL_EPISODES, EVAL_SEED, RESULTS_DIR, SEEDS, run_name
from baseline_dynamic_reach.environment import TARGET_SLOTS

HELDOUT_SEED = 40_000
FOLDER = f"heldout_{HELDOUT_SEED}"
ARMS = ("P6", "B5")


def outcomes(sets):
    """{(set, policy label): {episode seed: S/C/T/L}} for both episode sets."""
    table = {}
    for name, prefix in sets.items():
        for arm in ARMS:
            for seed in SEEDS:
                table[(name, f"{arm} seed {seed}")] = saved_outcomes(f"{prefix}{run_name(arm, seed)}")
        reference = f"{prefix}{run_name('P6', SEEDS[0])}"
        for policy, label in (("scripted", "Scripted"), ("random", "Random")):
            table[(name, label)] = saved_outcomes(reference, policy)
    missing = [key for key, value in table.items() if value is None]
    if missing:
        raise FileNotFoundError(f"No saved evaluation for {missing}")
    return table


def tally(tags, result):
    groups = {
        "blr": [t for t in tags if t["slot"] in TARGET_SLOTS],
        "clear": [t for t in tags if t["slot"] in TARGET_SLOTS and not t["blocked"]],
        "blocked": [t for t in tags if t["slot"] in TARGET_SLOTS and t["blocked"]],
        "front": [t for t in tags if t["slot"] == "front"],
        "all": tags,
    }
    out = {}
    for group, rows in groups.items():
        letters = [result[t["seed"]] for t in rows]
        out[group] = {"n": len(rows), **{letter: letters.count(letter) for letter in "SCTL"}}
        out[group]["rate"] = out[group]["S"] / len(rows) if rows else float("nan")
    return out


def cell(group):
    return f"{group['S']}/{group['C']}/{group['T']} ({100 * group['rate']:.1f}%)"


def main():
    sets = {"usual": "", "held-out": f"{FOLDER}/"}
    tags = {"usual": tag_episodes(EVAL_EPISODES, EVAL_SEED), "held-out": tag_episodes(EVAL_EPISODES, HELDOUT_SEED)}
    table = outcomes(sets)
    stats = {key: tally(tags[key[0]], result) for key, result in table.items()}
    labels = [label for (name, label) in table if name == "usual"]

    lines = [f"# Step 7h: P6 and B5 on held-out episodes (seeds {HELDOUT_SEED:,}–{HELDOUT_SEED + EVAL_EPISODES - 1:,})", ""]
    for name in sets:
        held = tags[name]
        slots = {slot: sum(t["slot"] == slot for t in held) for slot in ("front", "back", "left", "right")}
        blocked = sum(t["blocked"] for t in held if t["slot"] in TARGET_SLOTS)
        blr = sum(t["slot"] in TARGET_SLOTS for t in held)
        lines += [f"## {name.capitalize()} episodes", "",
                  f"Slots: FRONT {slots['front']}, BACK {slots['back']}, LEFT {slots['left']}, RIGHT {slots['right']}. "
                  f"BACK/LEFT/RIGHT: {blr}, of which {blocked} blocked at reset.", "",
                  "Cells are reached / crash / timeout (success rate).", "",
                  "| Policy | BACK/LEFT/RIGHT | clear | blocked | FRONT | All 200 |",
                  "|---|---|---|---|---|---|"]
        for label in labels:
            s = stats[(name, label)]
            lines.append(f"| {label} | {cell(s['blr'])} | {cell(s['clear'])} | {cell(s['blocked'])} | "
                         f"{cell(s['front'])} | {cell(s['all'])} |")
        lines.append("")

    def rate(name, label, group="blr"):
        return 100 * stats[(name, label)][group]["rate"]

    def crashes(name, label):
        return stats[(name, label)]["all"]["C"]

    p6 = [f"P6 seed {seed}" for seed in SEEDS]
    b5 = [f"B5 seed {seed}" for seed in SEEDS]
    lead = {name: [rate(name, p) - rate(name, "Scripted") for p in p6] for name in sets}
    over_b5 = [rate("held-out", p) - rate("held-out", b) for p, b in zip(p6, b5)]
    shrink = [u - h for u, h in zip(lead["usual"], lead["held-out"])]
    checks = [
        ("1. P6 beats the scripted driver on BACK/LEFT/RIGHT by >= 5 points, every seed",
         all(x >= 5 for x in lead["held-out"]), "leads " + ", ".join(f"{x:+.1f}" for x in lead["held-out"])),
        ("2. P6 beats B5 on BACK/LEFT/RIGHT by >= 5 points, every seed",
         all(x >= 5 for x in over_b5), "leads " + ", ".join(f"{x:+.1f}" for x in over_b5)),
        ("3. The lead over the scripted driver shrinks by <= 7 points on at least 2 of 3 seeds",
         sum(x <= 7 for x in shrink) >= 2,
         "usual " + ", ".join(f"{x:+.1f}" for x in lead["usual"]) + "; held-out "
         + ", ".join(f"{x:+.1f}" for x in lead["held-out"]) + "; shrink " + ", ".join(f"{x:+.1f}" for x in shrink)),
        ("4. FRONT: P6 seed 7 at least matches the scripted driver",
         rate("held-out", p6[0], "front") >= rate("held-out", "Scripted", "front"),
         f"{rate('held-out', p6[0], 'front'):.1f}% vs {rate('held-out', 'Scripted', 'front'):.1f}%"),
        ("5. P6 crashes more often than the scripted driver, every seed",
         all(crashes("held-out", p) > crashes("held-out", "Scripted") for p in p6),
         "crashes " + ", ".join(str(crashes("held-out", p)) for p in p6) + f" vs {crashes('held-out', 'Scripted')}"),
    ]
    lines += ["## Predictions (written before the evaluation)", "", "| Prediction | Held | Numbers |", "|---|---|---|"]
    lines += [f"| {text} | {'yes' if held else '**no**'} | {numbers} |" for text, held, numbers in checks]
    out = RESULTS_DIR / FOLDER
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "slot", "blocked", "lane_changes_required", "lead_car_anchor"] + labels
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in tags["held-out"]:
            writer.writerow({**t, **{label: table[("held-out", label)][t["seed"]] for label in labels}})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
