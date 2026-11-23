"""Step 8 analysis: score every run, compare cells seed by seed, apply the winner rules, keep the registry.

    python -B baseline_dynamic_reach/grid.py 8a                 # network width
    python -B baseline_dynamic_reach/grid.py 8b --width n64     # training order
    python -B baseline_dynamic_reach/grid.py 8c --order E --width n64
    python -B baseline_dynamic_reach/grid.py 8d --cells O_c05w0 N_c1w01 --width n64   # confirmation + final test
    python -B baseline_dynamic_reach/grid.py 8f                 # 256-wide recipe vs the winner, fresh episodes
    python -B baseline_dynamic_reach/grid.py registry           # registry.csv and checkpoints_manifest.csv only

Score of an episode: +1 reached, -1 crash, 0 timeout or anchor lost (the user's rule, 2026-11-18). A run's
score is its mean over the 400 selection episodes (seeds 10,000-10,199 and 40,000-40,199), averaged over its
scored snapshots: for the last round of a pipeline, the checkpoints nearest 1.6M and 1.8M and the final model.

The replicate is the training run (seed), not the episode: two cells are compared through their per-seed
score differences (paired by seed; a cell's runs on one seed share their round-1 policy where the pipelines
do), with a paired t-test and 95% interval over seeds. Every compared pair is first checked with
ivcheck.check_pipeline, so it differs only in the factor being compared.
Outputs: results/grid8/ (summary_<stage>.md, cells_<stage>.csv, figures, registry.csv, checkpoints_manifest.csv).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dynamic_reach.config import (ALL_REWARDS, ARMS, MODELS_DIR, P6_REWARD, RESULTS_DIR, REWARDS, STEP8_ARMS,
                                           STEP8_CONFIRM_SEEDS, STEP8_SCREEN_SEEDS, TIME_LEVELS, reward_parts,
                                           run_name, step8_name)
from baseline_dynamic_reach.environment import TARGET_SLOTS

GRID_DIR = RESULTS_DIR / "grid8"
SELECTION = (("s10000", 10_000), ("s40000", 40_000))
EPISODES = 200
ORDER_SIMPLICITY = ("N", "H", "E", "O")  # Fewer special rounds first.
WIDTH_BAR = 0.10                          # 8a: the larger network must add at least this much score.
ALPHA = 0.05


# --- statistics (no SciPy in the lock file) -------------------------------------------------------------
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for numerator in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t)) if math.isfinite(t) else 0.0


def t_quantile(probability, df):
    """t such that P(T <= t) = probability (probability > 0.5), by bisection."""
    low, high = 0.0, 1e4
    for _ in range(200):
        middle = (low + high) / 2
        if 1 - t_two_sided_p(middle, df) / 2 < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def paired(differences):
    """Mean, 95% interval, t-test p and sign count of per-seed differences."""
    n = len(differences)
    mean = statistics.fmean(differences)
    if n < 2:
        return {"n": n, "mean": mean, "ci95": [float("nan")] * 2, "p": float("nan"), "ahead": sum(d > 0 for d in differences)}
    sd = statistics.stdev(differences)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else (math.inf if mean else 0.0)
    half = t_quantile(0.975, n - 1) * se
    return {"n": n, "mean": mean, "ci95": [mean - half, mean + half], "p": t_two_sided_p(abs(t), n - 1),
            "ahead": sum(d > 0 for d in differences)}


# --- outcomes and scores ---------------------------------------------------------------------------------
def episode_score(letter):
    return 1 if letter == "S" else -1 if letter == "C" else 0


def tags():
    from baseline_dynamic_reach.blocked import tag_episodes
    path = GRID_DIR / "selection_tags.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    found = {}
    for _, first in SELECTION:
        for row in tag_episodes(EPISODES, first):
            found[row["seed"]] = {"slot": row["slot"], "blocked": row["blocked"]}
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found), encoding="utf-8")
    return found


def snapshot_outcomes(run, label):
    """{episode seed: letter} over both selection sets for one snapshot, or None if not evaluated yet."""
    from baseline_dynamic_reach.blocked import saved_outcomes
    merged = {}
    for name, _ in SELECTION:
        found = saved_outcomes(run, "ppo", f"eval8/{name}_{label}/evaluation_episodes.csv")
        if found is None:
            return None
        merged.update(found)
    return merged


def snapshot_labels(run):
    """The scored snapshots that have been evaluated: the final model, plus for a round 3 or 4 the checkpoints
    400k and 200k before the end (grid_run.SCORED_BEFORE_END). curves.py evaluates every other checkpoint into
    the same eval8 folder; those are curve points, not part of the score (fixed 2026-11-20: before this,
    every evaluated label was averaged, which only mattered once curves.py had run)."""
    from baseline_dynamic_reach.grid_run import SCORED_BEFORE_END
    folder = RESULTS_DIR / run / "eval8"
    found = {path.name.split("_", 1)[1] for path in folder.glob("s10000_*")} if folder.exists() else set()
    scored = {"final"} & found
    arm = run.rsplit("_seed", 1)[0]
    numeric = sorted(int(label) for label in found if label.isdigit())
    if ARMS.get(arm, {}).get("round") in (3, 4) and numeric:
        sidecar = MODELS_DIR / f"{run}.json"
        total = json.loads(sidecar.read_text(encoding="utf-8"))["requested_timesteps"]
        for before in SCORED_BEFORE_END:
            nearest = min(numeric, key=lambda steps: abs(steps - (total - before)))
            if abs(nearest - (total - before)) <= 1_000:  # Checkpoint steps fall a few transitions short of round numbers.
                scored.add(str(nearest))
    return sorted(scored, key=lambda label: (label == "final", int(label) if label.isdigit() else 0))


def group_counts(outcomes, episode_tags):
    groups = {"all": lambda t: True, "blr": lambda t: t["slot"] in TARGET_SLOTS,
              "blocked": lambda t: t["slot"] in TARGET_SLOTS and t["blocked"], "front": lambda t: t["slot"] == "front"}
    counts = {}
    for name, keep in groups.items():
        letters = [letter for seed, letter in outcomes.items() if keep(episode_tags[seed])]
        n = len(letters)
        counts[name] = {"n": n, "S": letters.count("S"), "C": letters.count("C"),
                        "T": letters.count("T") + letters.count("L"),
                        "score": sum(map(episode_score, letters)) / n if n else float("nan")}
    return counts


def run_stats(run, episode_tags):
    """Mean over scored snapshots of each group's rates and score; None if the run has no evaluation."""
    labels = snapshot_labels(run)
    per_snapshot = {}
    for label in labels:
        outcomes = snapshot_outcomes(run, label)
        if outcomes is not None:
            per_snapshot[label] = group_counts(outcomes, episode_tags)
    if not per_snapshot:
        return None
    stats = {}
    for group in ("all", "blr", "blocked", "front"):
        values = [snap[group] for snap in per_snapshot.values()]
        n = values[0]["n"]
        stats[group] = {"n": n, "score": statistics.fmean(v["score"] for v in values),
                        **{f"{k}_rate": statistics.fmean(v[k] / n for v in values) if n else float("nan") for k in "SCT"}}
    stats["snapshots"] = sorted(per_snapshot)
    stats["final_score"] = per_snapshot["final"]["all"]["score"] if "final" in per_snapshot else float("nan")
    return stats


def reference_stats(episode_tags):
    from baseline_dynamic_reach.blocked import saved_outcomes
    found = {}
    for policy in ("scripted", "random"):
        merged = {}
        for name, _ in SELECTION:
            outcomes = saved_outcomes(f"grid8/references/{name}", policy)
            if outcomes is None:
                return {}
            merged.update(outcomes)
        found[policy] = group_counts(merged, episode_tags)
    return found


def p6_reference(episode_tags):
    """P6 (Mac, 1M per round, P4 -> P5 -> P6) on the same 400 episodes: its usual and Step 7h evaluations."""
    from baseline_dynamic_reach.blocked import saved_outcomes
    per_seed = []
    for seed in (7, 8, 9):
        usual, held = saved_outcomes(f"P6_seed{seed}"), saved_outcomes(f"heldout_40000/P6_seed{seed}")
        if usual is None or held is None:
            return {}
        per_seed.append(group_counts({**usual, **held}, episode_tags))
    mean = {group: {key: statistics.fmean(c[group][key] for c in per_seed) for key in ("n", "S", "C", "T", "score")}
            for group in per_seed[0]}
    return {"P6 (Mac; mean of seeds 7, 8, 9)": mean,
            **{f"P6 seed {seed}": counts for seed, counts in zip((7, 8, 9), per_seed)}}


def cell_runs(arm, seeds):
    return {seed: run_name(arm, seed) for seed in seeds}


def compare(arm_a, arm_b, factor, seeds, stats):
    """Per-seed score differences a - b, after checking each pair differs only in `factor`."""
    from baseline_dynamic_reach.ivcheck import check_pipeline
    differences, seeds_used = [], []
    for seed in seeds:
        run_a, run_b = run_name(arm_a, seed), run_name(arm_b, seed)
        if stats.get(run_a) is None or stats.get(run_b) is None:
            continue
        check_pipeline(run_a, run_b, factor)
        differences.append(stats[run_a]["all"]["score"] - stats[run_b]["all"]["score"])
        seeds_used.append(seed)
    result = paired(differences) if differences else None
    if result:
        result.update(a=arm_a, b=arm_b, factor=factor, seeds=seeds_used, differences=differences,
                      crash_change=statistics.fmean(stats[run_name(arm_a, s)]["all"]["C_rate"] - stats[run_name(arm_b, s)]["all"]["C_rate"]
                                                    for s in seeds_used))
    return result


def beats(result):
    """The more complex option replaces the simpler one: paired t-test p < 0.05 and ahead on >= 2 of 3 seeds
    (in general, on at least two thirds of the seeds)."""
    return bool(result and result["mean"] > 0 and result["p"] < ALPHA and result["ahead"] >= math.ceil(2 * result["n"] / 3))


# --- tables -----------------------------------------------------------------------------------------------
def fmt(x, digits=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}f}" if digits == 3 else f"{x:.{digits}f}"


def cell_rows(arms, seeds, stats):
    rows = []
    for arm in arms:
        runs = cell_runs(arm, seeds)
        scored = {seed: stats.get(run) for seed, run in runs.items() if stats.get(run)}
        if not scored:
            continue
        mean = lambda group, key: statistics.fmean(s[group][key] for s in scored.values())  # noqa: E731
        rows.append({"arm": arm, "seeds": len(scored),
                     "score": mean("all", "score"), "score_by_seed": {seed: s["all"]["score"] for seed, s in scored.items()},
                     "success": mean("all", "S_rate"), "crash": mean("all", "C_rate"), "timeout": mean("all", "T_rate"),
                     "blr_success": mean("blr", "S_rate"), "blocked_success": mean("blocked", "S_rate"),
                     "front_success": mean("front", "S_rate"), "front_crash": mean("front", "C_rate"),
                     "final_score": statistics.fmean(s["final_score"] for s in scored.values())})
    return rows


def cells_table(rows, references):
    lines = ["| Cell | Score (mean of seeds) | Per seed | Reached | Crash | Timeout | B/L/R reached | Blocked reached | "
             "FRONT reached | FRONT crash | Final snapshot score |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        per_seed = ", ".join(f"{score:+.3f}" for score in r["score_by_seed"].values())
        lines.append(f"| {r['arm']} | **{r['score']:+.3f}** | {per_seed} | {r['success']:.1%} | {r['crash']:.1%} | "
                     f"{r['timeout']:.1%} | {r['blr_success']:.1%} | {r['blocked_success']:.1%} | {r['front_success']:.1%} | "
                     f"{r['front_crash']:.1%} | {r['final_score']:+.3f} |")
    for policy, counts in references.items():
        a = counts["all"]
        label = policy if policy.startswith("P6") else f"{policy.capitalize()} driver"
        lines.append(f"| {label} (reference) | {a['score']:+.3f} | | {a['S'] / a['n']:.1%} | "
                     f"{a['C'] / a['n']:.1%} | {a['T'] / a['n']:.1%} | {counts['blr']['S'] / counts['blr']['n']:.1%} | "
                     f"{counts['blocked']['S'] / counts['blocked']['n']:.1%} | {counts['front']['S'] / counts['front']['n']:.1%} | "
                     f"{counts['front']['C'] / counts['front']['n']:.1%} | |")
    return lines


def comparisons_table(results):
    lines = ["| Comparison (a − b) | Factor | Per-seed differences | Mean | 95% interval | p (paired t) | a ahead | Crash rate change |",
             "|---|---|---|---:|---|---:|---:|---:|"]
    for r in results:
        if r is None:
            continue
        lines.append(f"| {r['a']} − {r['b']} | {r['factor']} | {', '.join(f'{d:+.3f}' for d in r['differences'])} | "
                     f"{r['mean']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.3f} | "
                     f"{r['ahead']} of {r['n']} | {r['crash_change']:+.1%} |")
    return lines


def figure(rows, references, filename, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(rows) + 2), 3.6))
    for i, r in enumerate(rows):
        values = list(r["score_by_seed"].values())
        ax.scatter([i] * len(values), values, color="#4c78a8", s=18, zorder=3)
        ax.hlines(r["score"], i - 0.3, i + 0.3, color="#1f3b5c", linewidth=2, zorder=4)
    for policy, style in (("scripted", "--"), ("random", ":")):
        if policy in references:
            ax.axhline(references[policy]["all"]["score"], color="#888888", linestyle=style, linewidth=1)
            ax.annotate(policy, (len(rows) - 0.5, references[policy]["all"]["score"]), fontsize=8, color="#555555",
                        va="bottom", ha="right")
    ax.set_xticks(range(len(rows)), [r["arm"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score (reached − crash rate)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRID_DIR / filename, dpi=150)
    plt.close(fig)


def heatmap(rows, order, width, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baseline_dynamic_reach.config import CRASH_LEVELS, PROGRESS_LEVELS
    lookup = {r["arm"]: r["score"] for r in rows}
    grid = [[lookup.get(step8_name(order, 3, f"{c}{w}", width), float("nan")) for c in CRASH_LEVELS] for w in PROGRESS_LEVELS]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    image = ax.imshow(grid, cmap="viridis", aspect="auto")
    low, high = image.get_clim()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if not math.isnan(value):  # Dark text on the light (high) end of viridis, white on the dark end.
                light = high > low and (value - low) / (high - low) > 0.6
                ax.text(x, y, f"{value:+.3f}", ha="center", va="center", color="black" if light else "white", fontsize=9)
    ax.set_xticks(range(len(CRASH_LEVELS)), [f"crash {v:g}" for v in CRASH_LEVELS.values()])
    ax.set_yticks(range(len(PROGRESS_LEVELS)), [f"progress {v:g}" for v in PROGRESS_LEVELS.values()])
    ax.set_title(f"Order {order}: mean score after round 3", loc="left", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(GRID_DIR / filename, dpi=150)
    plt.close(fig)


# --- registry ---------------------------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry(episode_tags, include_smoke=False):
    """One row per Step 8 run (registry.csv) and one per saved checkpoint (checkpoints_manifest.csv)."""
    rows, manifest = [], []
    for path in sorted(MODELS_DIR.glob("*.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        arm = sidecar.get("arm")
        if arm not in STEP8_ARMS or (sidecar.get("smoke") and not include_smoke):
            continue
        info, run = STEP8_ARMS[arm], sidecar["run_name"]
        weights = sidecar["environment"]["reward_weights"]
        model = path.with_suffix(".zip")
        s = run_stats(run, episode_tags)
        scored = set(s["snapshots"]) if s else set()
        rows.append({"run": run, "arm": arm, "order": info["order"], "round": info["round"], "reward": info["reward"] or "success only",
                     "crash_cost": weights["collision"] / weights["goal"], "progress": weights["potential_progress"]["weight"],
                     "width": info["width"], "seed": sidecar["seed"], "init_run": run_name(sidecar["init_policy"], sidecar["seed"]) if sidecar.get("init_policy") else "",
                     "status": sidecar["status"], "transitions": sidecar.get("actual_timesteps"),
                     "training_minutes": round(sidecar.get("training_seconds", 0) / 60, 1),
                     "model_file": model.relative_to(RESULTS_DIR.parent).as_posix(), "model_sha256": sha256(model) if model.exists() else "",
                     "git_head": sidecar.get("code", {}).get("git_head"), "source_sha256": sidecar.get("code", {}).get("source_sha256"),
                     "machine": f"{sidecar['platform']['system']} {sidecar['platform']['release']}, Python {sidecar['platform']['python']}",
                     "selection_score": round(s["all"]["score"], 4) if s else "", "selection_reached": round(s["all"]["S_rate"], 4) if s else "",
                     "selection_crash": round(s["all"]["C_rate"], 4) if s else "", "scored_snapshots": " ".join(sorted(scored)),
                     "evaluations": (RESULTS_DIR / run / "eval8").relative_to(RESULTS_DIR.parent).as_posix()})
        for checkpoint in sorted((RESULTS_DIR / run / "checkpoints").glob("*.zip")):
            steps = checkpoint.stem.split("_")[-2]
            manifest.append({"run": run, "transitions": int(steps), "file": checkpoint.relative_to(RESULTS_DIR.parent).as_posix(),
                             "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint), "scored": steps in scored})
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("registry.csv", rows), ("checkpoints_manifest.csv", manifest)):
        if table:
            with (GRID_DIR / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    return rows, manifest


# --- stages -------------------------------------------------------------------------------------------------
def load_stats(arms, seeds, episode_tags):
    return {run_name(arm, seed): run_stats(run_name(arm, seed), episode_tags) for arm in arms for seed in seeds}


def stage_8a(seeds, episode_tags, references):
    small, large = step8_name("E", 3, P6_REWARD), step8_name("E", 3, P6_REWARD, "n256")
    arms = [step8_name("E", r, P6_REWARD if r > 1 else None, w) for w in ("", "n256") for r in (1, 2, 3)]
    stats = load_stats(arms, seeds, episode_tags)
    rows = cell_rows(arms, seeds, stats)
    result = compare(large, small, "width", seeds, stats)
    decision = "not decided (missing runs)"
    if result:
        adopt = (result["mean"] >= WIDTH_BAR and result["ahead"] == result["n"] and result["crash_change"] <= 0)
        decision = (f"**Adopt 256-256**: +{result['mean']:.3f} ≥ {WIDTH_BAR}, ahead on all {result['n']} seeds, crashes not higher."
                    if adopt else
                    f"**Keep 64-64** (the larger network is not needed): change {result['mean']:+.3f} (bar +{WIDTH_BAR}), "
                    f"ahead on {result['ahead']} of {result['n']} seeds, crash rate {result['crash_change']:+.1%}.")
    lines = ["# Step 8a: network width (E order, P6's reward, seeds " + ", ".join(map(str, seeds)) + ")", "",
             "Bar, written before the runs: 256-256 replaces 64-64 only if the mean score rises by at least 0.10, "
             "it is ahead on all 3 seeds, and crashes do not rise.", "",
             *cells_table(rows, references), "", *comparisons_table([result]), "", decision]
    figure(rows, references, "stage_8a.png", "Step 8a: score by run (dots: seeds; bar: mean)")
    return lines + ["", "![scores](stage_8a.png)"], rows, stats


def stage_8b(seeds, width, episode_tags, references):
    w = "" if width == "n64" else width
    last = {order: step8_name(order, 3, P6_REWARD, w) for order in ORDER_SIMPLICITY}
    arms = [step8_name(o, r, P6_REWARD if r > 1 else None, w) for o in ORDER_SIMPLICITY for r in (1, 2, 3) if not (o == "O" and r == 1)]
    stats = load_stats(arms, seeds, episode_tags)
    rows = cell_rows(arms, seeds, stats)
    # Only one-factor pairs (check_pipeline refuses the rest): O differs from N and H in two rounds.
    pairs = [("H", "N", "practice"), ("E", "H", "easy starts"), ("E", "N", "practice"), ("O", "E", "round-2 task")]
    results = {(a, b): compare(last[a], last[b], f, seeds, stats) for a, b, f in pairs}
    # Walk up the simplicity order along one-factor comparisons: a more complex order replaces the current
    # choice only if it beats it by the rule. O can only be reached through E (O vs E is its one-factor pair).
    chosen = "N"
    for candidate in ORDER_SIMPLICITY[1:]:
        if (candidate, chosen) in results and beats(results[(candidate, chosen)]):
            chosen = candidate
    scores = {o: next((r["score"] for r in rows if r["arm"] == last[o]), float("nan")) for o in ORDER_SIMPLICITY}
    best = max(scores, key=lambda o: -math.inf if math.isnan(scores[o]) else scores[o])
    decision = (f"**Rule choice: order {chosen}.** Highest mean score: order {best} ({scores[best]:+.3f}). "
                + ("They agree." if best == chosen else
                   f"Order {best} scores higher but does not beat order {chosen} by the rule (within noise); "
                   "per the plan, both get the reward grid in 8c unless the review decides otherwise."))
    final_rows = [r for r in rows if r["arm"] in last.values()]
    lines = [f"# Step 8b: training order at P6's reward (width {width}, seeds " + ", ".join(map(str, seeds)) + ")", "",
             "Simplicity order N < H < E < O. A more complex order replaces a simpler one only if its per-seed score "
             "difference has paired t-test p < 0.05 and it is ahead on at least 2 of 3 seeds.", "",
             "## Last round of each order", "", *cells_table(final_rows, references), "",
             "## Every round", "", *cells_table(rows, {}), "", "## Comparisons", "",
             *comparisons_table(list(results.values())), "", decision, "", "![scores](stage_8b.png)"]
    figure(final_rows, references, "stage_8b.png", "Step 8b: score after round 3, by training order")
    return lines, rows, stats


def reward_terms(reward):
    crash, progress, time = reward_parts(reward)
    return int(crash != "c0") + int(progress != "w0") + int(bool(time))


def choose_reward(cells, scores, seeds, stats):
    """(best, chosen): the best-scoring reward, and the simplest reward it does not beat by the rule
    (fewer non-zero terms first; among equally simple ones, the higher score first)."""
    ready = [r for r in scores if not math.isnan(scores[r])]
    best = max(ready, key=lambda r: scores[r])
    for reward in sorted(ready, key=lambda r: (reward_terms(r), -scores[r])):
        if reward == best or not beats(compare(cells[best], cells[reward], "reward", seeds, stats)):
            return best, reward
    return best, best


def stage_8c2(seeds, orders, width, episode_tags, references):
    """Time cost at crash 0.5 and 1 (progress 0), then the reward rule again over all 12 rewards."""
    w = "" if width == "n64" else width
    lines, all_rows, all_stats, finalists_by_order = [], [], {}, {}
    front = [s for s, t in episode_tags.items() if t["slot"] == "front"]
    for order in orders:
        cells = {reward: step8_name(order, 3, reward, w) for reward in ALL_REWARDS}
        arms = [step8_name(order, r, reward, w) for reward in ALL_REWARDS for r in (2, 3)]
        stats = load_stats(arms, seeds, episode_tags)
        block = [reward for c in ("c05", "c1") for reward in (f"{c}w0", *(f"{c}w0{t}" for t in TIME_LEVELS))]
        rows = cell_rows([cells[r] for r in block], seeds, stats)
        comparisons = []
        for c in ("c05", "c1"):
            levels = [f"{c}w0", *(f"{c}w0{t}" for t in TIME_LEVELS)]
            for low, high in zip(levels, levels[1:]):
                comparisons.append(compare(cells[high], cells[low], "time cost", seeds, stats))
        give_up = ["| Cell | FRONT timeouts by seed (final model, of " + f"{len(front)})" + " |", "|---|---|"]
        for reward in block:
            counts = []
            for seed in seeds:
                outcomes = snapshot_outcomes(run_name(cells[reward], seed), "final")
                counts.append("–" if outcomes is None else str(sum(outcomes[s] in "TL" for s in front)))
            give_up.append(f"| {cells[reward]} | {' / '.join(counts)} |")
        all_rows_order = cell_rows(list(cells.values()), seeds, stats)
        scores = {reward: next((r["score"] for r in all_rows_order if r["arm"] == arm), float("nan"))
                  for reward, arm in cells.items()}
        best, chosen = choose_reward(cells, scores, seeds, stats)
        top = sorted((r for r in scores if not math.isnan(scores[r])), key=lambda r: -scores[r])[:3]
        finalists = list(dict.fromkeys([chosen, *top, P6_REWARD]))
        finalists_by_order[order] = finalists
        ranking = sorted(all_rows_order, key=lambda r: -r["score"])
        lines += [f"# Step 8c2: time cost, order {order} (width {width}, seeds " + ", ".join(map(str, seeds)) + ")", "",
                  "Time cost per decision (0.1 s) as a multiple of the goal: t1 = 0.001 (Marcos's weight), t4 = 0.004. "
                  "Progress 0. Compared with the same crash cost without a time cost (from 8c).", "",
                  *cells_table(rows, references), "", *give_up, "", "## Time cost (neighbouring levels)", "",
                  *comparisons_table(comparisons), "",
                  "## All 12 rewards of this order, best first", "", *cells_table(ranking, {}), "",
                  f"**Best score: {best} ({scores[best]:+.3f}). Rule choice: {chosen}. Finalists for 8d: "
                  f"{', '.join(finalists)}.**", ""]
        all_rows += all_rows_order
        all_stats.update(stats)
    return lines, all_rows, all_stats, finalists_by_order


def stage_8c(seeds, orders, width, episode_tags, references):
    w = "" if width == "n64" else width
    lines, all_rows, all_stats = [], [], {}
    for order in orders:
        cells = {reward: step8_name(order, 3, reward, w) for reward in REWARDS}
        arms = [step8_name(order, r, reward, w) for reward in REWARDS for r in (2, 3)] + [step8_name(order, 1, None, w)]
        stats = load_stats(arms, seeds, episode_tags)
        rows = [r for r in cell_rows(list(cells.values()), seeds, stats)]
        scores = {reward: next((r["score"] for r in rows if r["arm"] == arm), float("nan")) for reward, arm in cells.items()}
        best = max((r for r in scores if not math.isnan(scores[r])), key=lambda r: scores[r])
        comparisons = []
        from baseline_dynamic_reach.config import CRASH_LEVELS, PROGRESS_LEVELS
        for p in PROGRESS_LEVELS:  # Crash cost: neighbouring levels at each progress level.
            levels = list(CRASH_LEVELS)
            for low, high in zip(levels, levels[1:]):
                comparisons.append(compare(cells[f"{high}{p}"], cells[f"{low}{p}"], "crash cost", seeds, stats))
        for c in CRASH_LEVELS:  # Progress on vs off at each crash level.
            comparisons.append(compare(cells[f"{c}w01"], cells[f"{c}w0"], "progress", seeds, stats))
        best, chosen = choose_reward(cells, scores, seeds, stats)
        top = sorted((r for r in scores if not math.isnan(scores[r])), key=lambda r: -scores[r])[:3]
        finalists = list(dict.fromkeys([chosen, *top, P6_REWARD]))
        lines += [f"# Step 8c: reward grid, order {order} (width {width}, seeds " + ", ".join(map(str, seeds)) + ")", "",
                  "Reward from round 2 on: crash cost (× goal) and potential-based progress (per m). Simplicity: fewer "
                  "non-zero terms first. The simplest reward that the best-scoring reward does not beat by the rule "
                  "(paired t p < 0.05, ahead on ≥ 2 of 3 seeds) is chosen.", "",
                  *cells_table(rows, references), "", "## Main effects (neighbouring cells)", "",
                  *comparisons_table(comparisons), "",
                  f"**Best score: {best} ({scores[best]:+.3f}). Rule choice: {chosen}. Finalists for 8d: {', '.join(finalists)}.**",
                  "", f"![heatmap](stage_8c_{order}.png)", ""]
        heatmap(rows, order, w, f"stage_8c_{order}.png")
        all_rows += rows
        all_stats.update(stats)
    return lines, all_rows, all_stats


# --- 8d: confirmation on new seeds, then the final test -------------------------------------------------------
FINAL_SET, FINAL_FIRST, FINAL_EPISODES = "s50000", 50_000, 500
P6_SEEDS = (7, 8, 9)
# The factor that separates two orders at the same reward (one-factor pairs, as in 8b; O vs N and O vs H differ
# in rounds 1 and 2, the "order" factor).
ORDER_FACTOR = {frozenset("HN"): "practice", frozenset("EH"): "easy starts", frozenset("EN"): "practice",
                frozenset("OE"): "round-2 task", frozenset("ON"): "order", frozenset("OH"): "order"}


def parse_cell(cell):
    order, reward = cell.split("_", 1)
    return order, reward


def cell_arm(cell, width=""):
    order, reward = parse_cell(cell)
    return step8_name(order, 3, reward, width)


def factor_between(cell_a, cell_b):
    (order_a, reward_a), (order_b, reward_b) = parse_cell(cell_a), parse_cell(cell_b)
    if order_a == order_b:
        return "reward"
    if reward_a == reward_b:
        return ORDER_FACTOR[frozenset(order_a + order_b)]
    return "recipe"  # Both differ: a choice between whole recipes, not a measured effect of either setting.


def simplicity(cell):
    """Fewer special training rounds first (decided in 8b), then fewer reward terms (8c)."""
    order, reward = parse_cell(cell)
    return ORDER_SIMPLICITY.index(order), reward_terms(reward)


def welch(a, b):
    """Two independent groups of per-seed scores (P6's Mac seeds are not paired with Step 8 seeds)."""
    mean = statistics.fmean(a) - statistics.fmean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else float(len(a) + len(b) - 2)
    t = mean / se if se > 0 else (math.inf if mean else 0.0)
    half = t_quantile(0.975, df) * se
    return {"mean": mean, "ci95": [mean - half, mean + half], "p": t_two_sided_p(abs(t), df), "df": df}


def final_tags(set_name=FINAL_SET, first=FINAL_FIRST, count=FINAL_EPISODES):
    from baseline_dynamic_reach.blocked import tag_episodes
    path = GRID_DIR / ("final_tags.json" if set_name == FINAL_SET else f"{set_name}_tags.json")
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    found = {row["seed"]: {"slot": row["slot"], "blocked": row["blocked"]} for row in tag_episodes(count, first)}
    path.write_text(json.dumps(found), encoding="utf-8")
    return found


def counts_as_stats(counts):
    """group_counts of one evaluation in run_stats' shape, so cell_rows and compare can use it."""
    stats = {group: {"n": c["n"], "score": c["score"], **{f"{k}_rate": c[k] / c["n"] if c["n"] else float("nan") for k in "SCT"}}
             for group, c in counts.items()}
    return {**stats, "snapshots": ["final"], "final_score": counts["all"]["score"]}


def final_test_stats(runs, episode_tags, set_name=FINAL_SET, count=FINAL_EPISODES):
    from baseline_dynamic_reach.blocked import saved_outcomes
    stats = {}
    for run in runs:
        outcomes = saved_outcomes(run, "ppo", f"eval8/{set_name}_final/evaluation_episodes.csv")
        if outcomes is not None:
            if len(outcomes) != count or set(outcomes) != set(episode_tags):
                raise ValueError(f"{run}: {len(outcomes)} {set_name} episodes, expected the {count} tagged ones")
            stats[run] = counts_as_stats(group_counts(outcomes, episode_tags))
    return stats


def final_references(episode_tags, set_name=FINAL_SET):
    from baseline_dynamic_reach.blocked import saved_outcomes
    found = {}
    for policy in ("scripted", "random"):
        outcomes = saved_outcomes(f"grid8/references/{set_name}", policy)
        if outcomes is not None:
            found[policy] = group_counts(outcomes, episode_tags)
    return found


def tests_table(results):
    lines = ["| Comparison (a − b) | Test | Per-seed values | Mean difference | 95% interval | p | a ahead |",
             "|---|---|---|---:|---|---:|---:|"]
    for r in results:
        values = ", ".join(f"{d:+.3f}" for d in r.get("differences", []))
        ahead = f"{r['ahead']} of {r['n']}" if "ahead" in r else "–"
        lines.append(f"| {r['a']} − {r['b']} | {r['test']} | {values} | {r['mean']:+.3f} | "
                     f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.3f} | {ahead} |")
    return lines


def stage_8d(cells, width, episode_tags, references):
    """Rank the finalists on the 5 confirmation seeds only (the screening seeds chose them, so their scores there
    are biased upward), apply the simplicity rule, then report the final test if it has been run."""
    w = "" if width == "n64" else width
    confirm, screen = list(STEP8_CONFIRM_SEEDS), list(STEP8_SCREEN_SEEDS)
    arms = {cell: cell_arm(cell, w) for cell in cells}
    stats = load_stats(arms.values(), screen + confirm, episode_tags)
    rows_confirm = cell_rows(arms.values(), confirm, stats)
    rows_all = cell_rows(arms.values(), screen + confirm, stats)
    scores = {cell: next((r["score"] for r in rows_confirm if r["arm"] == arm), float("nan")) for cell, arm in arms.items()}
    ready = [cell for cell in cells if not math.isnan(scores[cell])]
    lines = [f"# Step 8d: confirmation of the finalists (width {width})", "",
             "Finalists from 8c, retrained from fresh weights on 5 new seeds (" + ", ".join(map(str, confirm)) + "). "
             "They are ranked on these seeds only: the screening seeds (" + ", ".join(map(str, screen)) + ") chose "
             "them, so scores there are biased upward. Rule: the simplest finalist (fewer special training rounds, "
             "then fewer reward terms) that the best-scoring one does not beat (paired t p < 0.05 and ahead on at "
             "least 4 of 5 seeds).", "", "## Confirmation seeds", "", *cells_table(rows_confirm, references), "",
             "## All 8 seeds (screening + confirmation), for reference", "", *cells_table(rows_all, {}), ""]
    if not ready:
        return lines + ["No finalist has been evaluated on the confirmation seeds yet."], rows_confirm, stats, None
    best = max(ready, key=lambda cell: scores[cell])
    comparisons = [compare(arms[a], arms[b], factor_between(a, b), confirm, stats)
                   for i, a in enumerate(ready) for b in ready[i + 1:]]
    chosen = best
    for cell in sorted(ready, key=lambda c: (simplicity(c), -scores[c])):
        if cell == best:
            break
        if not beats(compare(arms[best], arms[cell], factor_between(best, cell), confirm, stats)):
            chosen = cell
            break
    recipe_note = (["Factor `recipe`: the two finalists differ in both training order and reward, so the comparison "
                    "chooses between whole recipes and does not measure either setting's effect.", ""]
                   if any(r and r["factor"] == "recipe" for r in comparisons) else [])
    lines += ["## Comparisons on the confirmation seeds", "", *recipe_note,
              *comparisons_table(comparisons), "",
              f"**Best on the confirmation seeds: {best} ({scores[best]:+.3f}). Rule choice (the winner): {chosen} "
              f"({scores[chosen]:+.3f}).**", ""]
    figure(rows_confirm, references, "stage_8d.png", "Step 8d: score on the confirmation seeds (dots: seeds; bar: mean)")
    lines += ["![scores](stage_8d.png)", ""]

    # Final test: fresh episodes, every seed of each tested cell, P6 itself and the two reference drivers.
    tags_final = final_tags()
    all_seeds = screen + confirm
    final_stats = final_test_stats([run_name(arm, s) for arm in arms.values() for s in all_seeds]
                                   + [run_name("P6", s) for s in P6_SEEDS], tags_final)
    tested = [cell for cell in cells if any(run_name(arms[cell], s) in final_stats for s in all_seeds)]
    if not tested:
        return lines + ["Final test (seeds 50,000-50,499) not run yet."], rows_confirm, stats, chosen
    final_refs = final_references(tags_final)
    final_rows = cell_rows([arms[c] for c in tested], all_seeds, final_stats) + cell_rows(["P6"], P6_SEEDS, final_stats)
    winner_runs = {s: run_name(arms[chosen], s) for s in all_seeds if run_name(arms[chosen], s) in final_stats}
    winner_scores = [final_stats[run]["all"]["score"] for run in winner_runs.values()]
    tests = []
    for policy in ("scripted", "random"):
        if policy in final_refs:
            differences = [score - final_refs[policy]["all"]["score"] for score in winner_scores]
            tests.append({**paired(differences), "a": arms[chosen], "b": f"{policy} driver", "differences": differences,
                          "test": f"one-sample t over the winner's {len(differences)} seeds (same {FINAL_EPISODES} episodes)"})
    p6_scores = [final_stats[run_name("P6", s)]["all"]["score"] for s in P6_SEEDS if run_name("P6", s) in final_stats]
    if len(p6_scores) >= 2 and len(winner_scores) >= 2:
        tests.append({**welch(winner_scores, p6_scores), "a": arms[chosen], "b": "P6 (Mac)",
                      "test": f"Welch t, {len(winner_scores)} vs {len(p6_scores)} seeds (not paired)"})
    for cell in tested:
        if cell != chosen:
            result = compare(arms[chosen], arms[cell], factor_between(chosen, cell), all_seeds, final_stats)
            if result:
                tests.append({**result, "test": f"paired t over seeds ({result['factor']})"})
    lines += [f"## Final test: {FINAL_EPISODES} fresh episodes (seeds {FINAL_FIRST:,}-{FINAL_FIRST + FINAL_EPISODES - 1:,}), "
              "final models, every seed", "", *cells_table(final_rows, final_refs), "", *tests_table(tests), ""]
    figure(final_rows, final_refs, "stage_8d_final.png", "Step 8d: final test, score by seed (dots) and mean (bar)")
    lines += ["![final test](stage_8d_final.png)", ""]
    return lines, rows_confirm, stats, chosen


def front_stuck(arm, seeds, stats, limit=0.30):
    """Seeds whose scored snapshots time out on at least `limit` of the FRONT episodes (the give-up)."""
    return [s for s in seeds if stats.get(run_name(arm, s)) and stats[run_name(arm, s)]["front"]["T_rate"] >= limit]


def width_bar(result):
    """Step 8e's bar for 256-256 on one recipe (written before the runs): +0.10 or more, p < 0.05, ahead on at
    least 6 of 8 seeds, crash rate not higher."""
    return bool(result and result["mean"] >= WIDTH_BAR and result["p"] < ALPHA
                and result["ahead"] >= math.ceil(0.75 * result["n"]) and result["crash_change"] <= 0)


def longer(arm_short, arm_long, seeds, stats):
    """Round 4 - round 3 on the same chains: the same runs trained 3M longer (checked with check_round)."""
    from baseline_dynamic_reach.ivcheck import check_round, load_sidecar
    differences, used = [], []
    for seed in seeds:
        short, long_ = run_name(arm_short, seed), run_name(arm_long, seed)
        if stats.get(short) is None or stats.get(long_) is None:
            continue
        check_round(load_sidecar(short), load_sidecar(long_))
        differences.append(stats[long_]["all"]["score"] - stats[short]["all"]["score"])
        used.append(seed)
    if not differences:
        return None
    return {**paired(differences), "a": arm_long, "b": arm_short, "factor": "3M more training", "seeds": used,
            "differences": differences,
            "crash_change": statistics.fmean(stats[run_name(arm_long, s)]["all"]["C_rate"]
                                             - stats[run_name(arm_short, s)]["all"]["C_rate"] for s in used)}


def stage_8e(cells, episode_tags, references):
    """Width (64 vs 256) and training length (round 3 at 6M vs round 4 at 9M) for the given cells, all 8 seeds."""
    seeds = list(STEP8_SCREEN_SEEDS + STEP8_CONFIRM_SEEDS)
    arm = {(cell, w, r): step8_name(*parse_cell(cell)[:1], r, parse_cell(cell)[1], w)
           for cell in cells for w in ("", "n256") for r in (3, 4)}
    stats = load_stats(arm.values(), seeds, episode_tags)
    order_rows = [arm[(cell, w, r)] for r in (3, 4) for w in ("", "n256") for cell in cells]
    rows = cell_rows(order_rows, seeds, stats)
    for row in rows:
        row["front_timeout"] = 1 - row["front_success"] - row["front_crash"]
        row["front_stuck"] = front_stuck(row["arm"], seeds, stats)
    lines = ["# Step 8e: 256 wide and 3M more training (all 8 seeds)", "",
             "Cells " + ", ".join(cells) + ". Round 3 ends at 6M transitions, round 4 (3M more of round 3's training) at 9M. "
             "Score = reached − crash on the 400 selection episodes, mean of the last 3 snapshots of the round "
             "(final model and the checkpoints 200k and 400k before it). Paired over seeds 7, 10, 13, 16, 19, 22, 25, 28.", "",
             "Bars, written before the runs: **256-256** is adopted only if, at 9M, on at least 2 of the 3 cells, it scores "
             f"+{WIDTH_BAR:.2f} or more above 64-64 with p < 0.05, ahead on at least 6 of 8 seeds and the crash rate not higher. "
             "**Longer training** is adopted if round 4 beats round 3 (p < 0.05) on at least 2 of 3 cells at the chosen width.", "",
             "## Cells", "", *cells_table(rows, references), "",
             "| Cell | FRONT timed out | Seeds with ≥ 30% FRONT timeouts |", "|---|---:|---|"]
    lines += [f"| {r['arm']} | {r['front_timeout']:.1%} | {', '.join(map(str, r['front_stuck'])) or 'none'} |" for r in rows]

    width_9 = {cell: compare(arm[(cell, "n256", 4)], arm[(cell, "", 4)], "width", seeds, stats) for cell in cells}
    width_6 = {cell: compare(arm[(cell, "n256", 3)], arm[(cell, "", 3)], "width", seeds, stats) for cell in cells}
    length = {(cell, w): longer(arm[(cell, w, 3)], arm[(cell, w, 4)], seeds, stats) for w in ("", "n256") for cell in cells}
    passed = [cell for cell in cells if width_bar(width_9[cell])]
    adopt_256 = len(passed) >= 2
    w = "n256" if adopt_256 else ""
    longer_wins = [cell for cell in cells if length[(cell, w)] and length[(cell, w)]["mean"] > 0
                   and length[(cell, w)]["p"] < ALPHA]
    adopt_long = len(longer_wins) >= 2
    r = 4 if adopt_long else 3
    lines += ["", "## Width at the same training", "", "At 9M (primary):", "", *comparisons_table(width_9.values()), "",
              "At 6M (secondary):", "", *comparisons_table(width_6.values()), "",
              f"Bar met on: {', '.join(passed) or 'none'}. **" + ("Adopt 256-256." if adopt_256 else "Keep 64-64.") + "**",
              "", "## 3M more training (round 4 − round 3, same runs)", "", *comparisons_table(length.values()), "",
              f"At the chosen width ({'256' if w else '64'}), round 4 beats round 3 beyond noise on: {', '.join(longer_wins) or 'none'}. **"
              + ("Adopt the longer training (9M)." if adopt_long else "Keep 6M.") + "**", ""]

    others = [compare(arm[("O_c1w0t4", width, rr)], arm[("N_c1w0t4", width, rr)], "order", seeds, stats)
              for width in ("n256", "") for rr in (4, 3) if "O_c1w0t4" in cells and "N_c1w0t4" in cells]
    others += [compare(arm[("N_c05w0t1", width, rr)], arm[("N_c1w0t4", width, rr)], "reward", seeds, stats)
               for width in ("n256", "") for rr in (4, 3) if "N_c05w0t1" in cells and "N_c1w0t4" in cells]
    lines += ["## Order and reward at each width and length", "", *comparisons_table(others), ""]

    chosen_arms = {cell: arm[(cell, w, r)] for cell in cells}
    scores = {cell: statistics.fmean(stats[run_name(a, s)]["all"]["score"] for s in seeds) for cell, a in chosen_arms.items()}
    best = max(cells, key=lambda cell: scores[cell])
    chosen = best
    for cell in sorted(cells, key=lambda c: (simplicity(c), -scores[c])):
        if cell == best:
            break
        if not beats(compare(chosen_arms[best], chosen_arms[cell], factor_between(best, cell), seeds, stats)):
            chosen = cell
            break
    lines += [f"**Recipe at the chosen width and length ({chosen_arms[best]} scores highest, {scores[best]:+.3f}): "
              f"rule choice {chosen_arms[chosen]} ({scores[chosen]:+.3f}).**", ""]
    figure(rows, references, "stage_8e.png", "Step 8e: score by run (dots: seeds; bar: mean); rounds 3 (6M) and 4 (9M)")
    lines += ["![scores](stage_8e.png)", ""]
    return lines, rows, stats


# --- 8f: the 256-wide recipe against the 64-wide winner on fresh episodes ---------------------------------------
FRESH_SET, FRESH_FIRST, FRESH_EPISODES = "s60000", 60_000, 500
F_CANDIDATE, F_WINNER = "O3_c1w0t4_n256", "N3_c05w0t1"
F_REPORTED = ("O3_c1w0t4", "N3_c05w0t1_n256", "O4_c1w0t4_n256")


def stage_8f():
    """Bar written before the evaluation (EXPERIMENT_NOTES.md, Step 8f): adopt the candidate only if it beats the
    winner by +0.10 or more (p < 0.05, ahead on >= 6 of 8 seeds), crashes no more beyond noise, and at most one of its
    seeds gives up on FRONT (>= 30% FRONT timeouts). Final models, 500 fresh episodes, all 8 seeds."""
    seeds = list(STEP8_SCREEN_SEEDS + STEP8_CONFIRM_SEEDS)
    arms = [F_WINNER, F_REPORTED[0], F_REPORTED[1], F_CANDIDATE, F_REPORTED[2]]
    tags_fresh = final_tags(FRESH_SET, FRESH_FIRST, FRESH_EPISODES)
    stats = final_test_stats([run_name(a, s) for a in arms for s in seeds], tags_fresh, FRESH_SET, FRESH_EPISODES)
    refs = final_references(tags_fresh, FRESH_SET)
    missing = [run_name(a, s) for a in arms for s in seeds if run_name(a, s) not in stats]
    if missing or set(refs) != {"scripted", "random"}:
        raise SystemExit(f"8f is not fully evaluated: {len(missing)} runs missing, references {sorted(refs)}")
    rows = cell_rows(arms, seeds, stats)
    for row in rows:
        row["front_timeout"] = statistics.fmean(stats[run_name(row["arm"], s)]["front"]["T_rate"] for s in seeds)
        row["front_stuck"] = front_stuck(row["arm"], seeds, stats)

    main = compare(F_CANDIDATE, F_WINNER, "recipe and width", seeds, stats)
    crash = paired([stats[run_name(F_CANDIDATE, s)]["all"]["C_rate"] - stats[run_name(F_WINNER, s)]["all"]["C_rate"]
                    for s in seeds])
    stuck = front_stuck(F_CANDIDATE, seeds, stats)
    bar = {"gain": main["mean"] >= WIDTH_BAR and main["p"] < ALPHA and main["ahead"] >= math.ceil(0.75 * main["n"]),
           "crash": not (crash["mean"] > 0 and crash["p"] < ALPHA),
           "reliable": len(stuck) <= 1}
    adopt = all(bar.values())

    tests = []
    for arm in (F_CANDIDATE, F_WINNER, F_REPORTED[2]):
        for policy in ("scripted", "random"):
            differences = [stats[run_name(arm, s)]["all"]["score"] - refs[policy]["all"]["score"] for s in seeds]
            tests.append({**paired(differences), "a": arm, "b": f"{policy} driver", "differences": differences,
                          "test": f"one-sample t over {len(seeds)} seeds (same {FRESH_EPISODES} episodes)"})
    reported = [compare(F_CANDIDATE, F_REPORTED[0], "width", seeds, stats),
                compare(F_REPORTED[1], F_WINNER, "width", seeds, stats),
                longer(F_CANDIDATE, F_REPORTED[2], seeds, stats)]
    check = lambda ok: "**met**" if ok else "**not met**"  # noqa: E731
    lines = [f"# Step 8f: the 256-wide recipe on {FRESH_EPISODES} fresh episodes (seeds {FRESH_FIRST:,}-{FRESH_FIRST + FRESH_EPISODES - 1:,})", "",
             f"Final models, seeds {', '.join(map(str, seeds))}. Score = reached − crash per episode, mean over the episodes. "
             f"Candidate {F_CANDIDATE} (original order, crash 1 + time cost 0.004, 256 wide, 6M); current winner {F_WINNER} "
             "(no practice, crash 0.5 + time cost 0.001, 64 wide, 6M). Bar and predictions written before the evaluation "
             "(EXPERIMENT_NOTES.md, Step 8f).", "", "## Cells", "", *cells_table(rows, refs), "",
             "| Cell | FRONT timed out | Seeds with ≥ 30% FRONT timeouts |", "|---|---:|---|"]
    lines += [f"| {r['arm']} | {r['front_timeout']:.1%} | {', '.join(map(str, r['front_stuck'])) or 'none'} |" for r in rows]
    lines += ["", "## The bar: candidate against the winner", "",
              "Factor `recipe and width`: order, reward and width all differ, so this chooses between whole recipes and does "
              "not measure any one setting's effect.", "", *comparisons_table([main]), "",
              "| Condition | Measured | Result |", "|---|---|---|",
              f"| 1. Gain ≥ +{WIDTH_BAR:.2f}, p < {ALPHA}, ahead on ≥ {math.ceil(0.75 * main['n'])} of {main['n']} | "
              f"{main['mean']:+.3f} (95% [{main['ci95'][0]:+.3f}, {main['ci95'][1]:+.3f}]), p {main['p']:.4f}, "
              f"ahead on {main['ahead']} of {main['n']} | {check(bar['gain'])} |",
              f"| 2. Crash rate not higher beyond noise | {crash['mean']:+.1%} (95% [{crash['ci95'][0]:+.1%}, {crash['ci95'][1]:+.1%}]), "
              f"p {crash['p']:.3f} | {check(bar['crash'])} (8e's stricter version, any increase fails: "
              f"{'met' if crash['mean'] <= 0 else 'not met'}) |",
              f"| 3. At most 1 of {len(seeds)} seeds with ≥ 30% FRONT timeouts | {len(stuck)} ({', '.join(map(str, stuck)) or 'none'}) | "
              f"{check(bar['reliable'])} |", "",
              "**" + (f"All three met: adopt {F_CANDIDATE}'s recipe (O_c1w0t4 at 256 wide, 6M)." if adopt
                      else f"Not all met: the 64-wide winner {F_WINNER} stays.") + "**", "",
              "## Against the reference drivers (reported, no decision)", "", *tests_table(tests), "",
              "## One-factor rows (reported, no decision)", "", *comparisons_table(reported), ""]
    figure(rows, refs, "stage_8f.png", f"Step 8f: {FRESH_EPISODES} fresh episodes, score by seed (dots) and mean (bar)")
    lines += ["![fresh test](stage_8f.png)", ""]
    return lines, rows, stats


def write_rows(rows, filename):
    if not rows:
        return
    with (GRID_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in rows[0] if key != "score_by_seed"] + ["score_by_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "score_by_seed": json.dumps(row["score_by_seed"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("8a", "8b", "8c", "8c2", "8d", "8e", "8f", "registry"))
    parser.add_argument("--width", choices=("n64", "n256"), default="n64")
    parser.add_argument("--order", nargs="+", choices=ORDER_SIMPLICITY, default=["E"])
    parser.add_argument("--cells", nargs="+", help="8d: finalist cells as <order>_<reward>, e.g. O_c05w0.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STEP8_SCREEN_SEEDS))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252.
    episode_tags = tags()
    if args.stage == "registry":
        rows, manifest = registry(episode_tags)
        print(f"registry.csv: {len(rows)} runs; checkpoints_manifest.csv: {len(manifest)} checkpoints")
        return
    references = {**reference_stats(episode_tags), **p6_reference(episode_tags)}
    if args.stage == "8a":
        lines, rows, stats = stage_8a(args.seeds, episode_tags, references)
    elif args.stage == "8b":
        lines, rows, stats = stage_8b(args.seeds, args.width, episode_tags, references)
    elif args.stage == "8d":
        if not args.cells:
            raise SystemExit("8d needs --cells.")
        lines, rows, stats, _ = stage_8d(args.cells, args.width, episode_tags, references)
    elif args.stage == "8e":
        lines, rows, stats = stage_8e(args.cells or ["N_c05w0t1", "N_c1w0t4", "O_c1w0t4"], episode_tags, references)
    elif args.stage == "8f":
        lines, rows, stats = stage_8f()
    elif args.stage == "8c2":
        lines, rows, stats, _ = stage_8c2(args.seeds, args.order, args.width, episode_tags, references)
    else:
        lines, rows, stats = stage_8c(args.seeds, args.order, args.width, episode_tags, references)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    (GRID_DIR / f"summary_{args.stage}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows(rows, f"cells_{args.stage}.csv")
    registry(episode_tags)  # Every Step 8 run on disk, whatever its stage.
    print("\n".join(lines))


if __name__ == "__main__":
    main()
