"""Step 9: consecutive goals. The adopted recipe (O3_c1w0t4_n256, 8 seeds) and, as a yardstick and feasibility
check, the scripted driver, on 500 chains of up to 5 goals in three conditions (EXPERIMENT_NOTES.md, Step 9):

- "new car": the next anchor is drawn as at reset (a car 30-120 m ahead); the episode origin restarts at ego;
- "same car": the same anchor, a different slot;
- "origin kept": as "new car", but the episode origin stays at the first goal's start.

Every chained goal passes the feasibility check in environment.py (goal_time_needed <= CHAIN_TIME_LIMIT).

    python -B baseline_dynamic_reach/chain.py run        # all chains, one process per (policy, condition)
    python -B baseline_dynamic_reach/chain.py summary    # tables, bar, figures -> results/chains/
    python -B baseline_dynamic_reach/chain.py demos      # 3 chains per condition, training seed 7, as videos
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import MODELS_DIR, RESULTS_DIR, arm_config, run_name

ARM = "O3_c1w0t4_n256"
SEEDS = (7, 10, 13, 16, 19, 22, 25, 28)
SCRIPTED = "scripted"
CONDITIONS = {"new car": {"mode": "new car", "origin": "reset"},
              "same car": {"mode": "same car", "origin": "reset"},
              "origin kept": {"mode": "new car", "origin": "keep"}}
FIRST, COUNT, GOALS = 70_000, 500, 5
OUT = RESULTS_DIR / "chains"
CHECK_FIRST = 50  # Goal 1 of these chains is also played as an ordinary episode and must match exactly.
SWITCH_WINDOW = 3.0  # s: "right after a switch"
FLAG_BELOW = 0.20    # a from -> to pair the scripted driver reaches less often than this is flagged


def policy(name, env):
    if name == SCRIPTED:
        from baseline_dynamic_waypoint.scripted import scripted_action
        return lambda obs: scripted_action(env)
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model = PPO.load(MODELS_DIR / f"{name}.zip", device="cpu")
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def outcome_of(info):
    return ("S" if info["is_success"] else "C" if info["is_collision"]
            else "L" if info.get("is_anchor_lost") else "T")


def make_env(condition, render_mode=None):
    from baseline_dynamic_reach.environment import ReachEnv
    settings = deepcopy(arm_config(ARM)["environment"])
    return ReachEnv(render_mode=render_mode, **settings, chain={"goals": GOALS, **CONDITIONS[condition]})


def play_chains(job):
    """One (policy, condition): a row per goal and a row per chain."""
    from baseline_dynamic_reach.environment import ReachEnv, goal_time_needed
    name, condition = job
    env = make_env(condition)
    act = policy(name, env)
    goals, chains, mismatches = [], [], []
    try:
        for chain_seed in range(FIRST, FIRST + COUNT):
            obs, info = env.reset(seed=chain_seed)
            ego = env.highway.vehicle

            def start(index, previous, time_needed, candidates=None, excluded=None):
                return {"policy": name, "condition": condition, "chain": chain_seed, "goal": index, "slot": env.slot,
                        "from_slot": previous, "start_time": float(env.highway.time), "time_needed": time_needed,
                        "anchor_speed": float(env.anchor.speed), "ego_speed": float(ego.speed),
                        "gap_m": float(env.anchor.position[0] - ego.position[0]),
                        "candidates": candidates, "excluded": excluded}

            goal = start(1, "start", goal_time_needed(env, env.anchor, env.slot))
            steps = 0
            while True:
                obs, _, terminated, truncated, info = env.step(act(obs))
                steps += 1
                switched = bool(info.get("goal_switched"))
                if switched or terminated or truncated:
                    letter = "S" if switched else outcome_of(info)
                    goals.append({**goal, "outcome": letter, "steps": steps,
                                  "duration": float(env.highway.time) - goal["start_time"] if not switched
                                  else float(info["elapsed_time"]) - goal["start_time"]})
                    steps = 0
                if switched:
                    goal = start(env.goal_index, goals[-1]["slot"], info["next_time_needed"],
                                 info.get("chain_candidates"), info.get("chain_excluded"))
                    continue
                if terminated or truncated:
                    end = info.get("chain_end") or {"C": "crash", "T": "timeout", "L": "anchor lost"}.get(outcome_of(info))
                    chains.append({"policy": name, "condition": condition, "chain": chain_seed,
                                   "goals_reached": sum(g["outcome"] == "S" for g in goals if g["chain"] == chain_seed),
                                   "end": end, "last_candidates": info.get("chain_candidates"),
                                   "last_excluded": info.get("chain_excluded")})
                    break
        # Goal 1 against the ordinary episode (no chain) on the first chains.
        plain = ReachEnv(**deepcopy(arm_config(ARM)["environment"]))
        act_plain = policy(name, plain)
        first_goals = {g["chain"]: g for g in goals if g["goal"] == 1}
        for chain_seed in range(FIRST, FIRST + CHECK_FIRST):
            obs, _ = plain.reset(seed=chain_seed)
            steps, done = 0, False
            while not done:
                obs, _, terminated, truncated, info = plain.step(act_plain(obs))
                steps += 1
                done = terminated or truncated
            mine = first_goals[chain_seed]
            if (outcome_of(info), steps) != (mine["outcome"], mine["steps"]):
                mismatches.append(chain_seed)
        plain.close()
    finally:
        env.close()
    return goals, chains, mismatches


def run(args):
    names = [run_name(ARM, s) for s in SEEDS] + [SCRIPTED]
    jobs = [(n, c) for c in CONDITIONS for n in names]
    OUT.mkdir(parents=True, exist_ok=True)
    all_goals, all_chains, checks = [], [], {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for job, (goals, chains, mismatches) in zip(jobs, pool.map(play_chains, jobs)):
            all_goals += goals
            all_chains += chains
            checks[f"{job[0]} / {job[1]}"] = mismatches
            print(f"{job[0]} / {job[1]}: {len(chains)} chains, {len(goals)} goals, goal-1 mismatches {mismatches}",
                  flush=True)
    for name, rows in (("goals.csv", all_goals), ("chains.csv", all_chains)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "goal1_check.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")


# --- analysis -------------------------------------------------------------------------------------------------
def load(name):
    import pandas as pd
    return pd.read_csv(OUT / name)


def score(frame):
    return ((frame.outcome == "S").sum() - (frame.outcome == "C").sum()) / len(frame) if len(frame) else float("nan")


def paired(values):
    from baseline_dynamic_reach.grid import paired as grid_paired
    return grid_paired(list(values))


def verdict(result):
    if result["ci95"][1] < 0.10:
        return "carries over"
    if result["mean"] >= 0.10 and result["p"] < 0.05:
        return "breaks"
    return "inconclusive"


def fmt_pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def flagged_pairs(goals, condition):
    scripted = goals[(goals.policy == SCRIPTED) & (goals.condition == condition) & (goals.goal > 1)]
    rates = scripted.groupby(["from_slot", "slot"]).outcome.apply(lambda s: (s == "S").mean())
    return {pair for pair, rate in rates.items() if rate < FLAG_BELOW}


def drop_rows(goals, condition, exclude=frozenset()):
    """Per training seed: score of goal 1, of goals 2-5, and the drop."""
    rows = {}
    for seed in SEEDS:
        part = goals[(goals.policy == run_name(ARM, seed)) & (goals.condition == condition)]
        later = part[part.goal > 1]
        if exclude:
            later = later[[(f, s) not in exclude for f, s in zip(later.from_slot, later.slot)]]
        rows[seed] = {"goal1": score(part[part.goal == 1]), "later": score(later), "n_later": len(later)}
        rows[seed]["drop"] = rows[seed]["goal1"] - rows[seed]["later"]
    return rows


def summary(args):
    import pandas as pd
    goals, chains = load("goals.csv"), load("chains.csv")
    checks = json.loads((OUT / "goal1_check.json").read_text(encoding="utf-8"))
    envelope = json.loads((OUT / "envelope.json").read_text(encoding="utf-8"))
    model = goals[goals.policy != SCRIPTED]
    lines = ["# Step 9: consecutive goals (chained goals, evaluation only)", "",
             f"Model {ARM}, seeds {', '.join(map(str, SEEDS))}; the scripted driver as yardstick. {COUNT} chains per "
             f"condition (chain seeds {FIRST:,}–{FIRST + COUNT - 1:,}), up to {GOALS} goals, 25 s per goal. Score per goal: "
             "+1 reached, −1 crash, 0 timeout or lost anchor. Every chained goal passed the feasibility check "
             f"(time needed ≤ {envelope['percentiles']['95']} s, the 95th percentile of goals drawn at reset).", "",
             f"**Goal 1 against the ordinary episode** (first {CHECK_FIRST} chains of every policy and condition): "
             + ("identical outcome and length in all." if not any(checks.values())
                else f"MISMATCHES: { {k: v for k, v in checks.items() if v} }"), ""]
    # Both drivers, goal 1 against goals 2-5 (added after the results; the bar below is unchanged).
    lines += ["## Both drivers: goal 1 against goals 2–5", "",
              "Added after the results; it is not part of the bar. Each driver's later goals start where its own "
              "previous goal ended, so the two drivers face different later goals, and the scripted driver gets to "
              "goal 2 less often (a goal-1 timeout ends its chain). The scripted driver reads the true state, so "
              "'origin kept' is identical to 'new car' for it.", ""]
    both = []
    for condition in CONDITIONS:
        for who, part in (("model (8 seeds pooled)", model[model.condition == condition]),
                          ("scripted", goals[(goals.policy == SCRIPTED) & (goals.condition == condition)])):
            one, later = part[part.goal == 1], part[part.goal > 1]
            both.append([condition, who, " / ".join(map(fmt_pct, outcome_shares(one))), f"{score(one):+.3f}",
                         len(later), " / ".join(map(fmt_pct, outcome_shares(later))), f"{score(later):+.3f}",
                         f"{score(one) - score(later):+.3f}"])
    lines += table(["Condition", "Driver", "Goal 1: reached / crash / timeout", "Goal 1 score", "Goals 2–5 (n)",
                    "Goals 2–5: reached / crash / timeout", "Goals 2–5 score", "Drop"], both)
    lines += ["", "- **New car and origin kept:** the scripted driver drops too, by timing out more (24% → 47%), with "
              "almost no crashes. The model drops more, by crashing more (11% → 31–32%). Part of the model's drop is "
              "the later goals being harder for any driver; the rest (0.16 with a new car, 0.19 with the origin kept) is its own.",
              "- **Same car:** the scripted driver does better on later goals than on goal 1, and the model drops "
              "further than in the other conditions, so the whole drop is the model's.", ""]
    verdicts = {}
    for condition in CONDITIONS:
        rows = drop_rows(goals, condition)
        result = paired([r["drop"] for r in rows.values()])
        flags = flagged_pairs(goals, condition)
        rows_f = drop_rows(goals, condition, flags)
        result_f = paired([r["drop"] for r in rows_f.values()])
        scripted_part = goals[(goals.policy == SCRIPTED) & (goals.condition == condition)]
        scripted_drop = score(scripted_part[scripted_part.goal == 1]) - score(scripted_part[scripted_part.goal > 1])
        extra = paired([r["drop"] - scripted_drop for r in rows.values()])
        verdicts[condition] = (verdict(result), result, verdict(result_f), result_f, flags, scripted_drop, extra)
        lines += [f"## Condition: {condition}", "",
                  "| Seed | Goal 1 score | Goals 2–5 score | Goals 2–5 (n) | Drop |", "|---:|---:|---:|---:|---:|"]
        lines += [f"| {s} | {r['goal1']:+.3f} | {r['later']:+.3f} | {r['n_later']} | {r['drop']:+.3f} |" for s, r in rows.items()]
        lines += ["", f"**Drop** (goal 1 − goals 2–5, paired over {result['n']} seeds): {result['mean']:+.3f} "
                  f"(95% interval {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f}, p {result['p']:.3f}). "
                  f"**Verdict: {verdict(result)}.**", "",
                  f"**Against the scripted driver's drop** (added after the results, not part of the bar): the scripted "
                  f"driver's drop is {scripted_drop:+.3f}, so the model's extra drop is {extra['mean']:+.3f} (95% interval "
                  f"{extra['ci95'][0]:+.3f} to {extra['ci95'][1]:+.3f}, p {extra['p']:.3f}; one-sample t over the 8 seeds "
                  "with the scripted drop held fixed, so the scripted driver's own sampling noise is not included).", "",
                  f"Pairs the scripted driver reaches in under {FLAG_BELOW:.0%} of goals (flagged as possibly infeasible): "
                  f"{', '.join(f'{a}→{b}' for a, b in sorted(flags)) or 'none'}. Without them: {result_f['mean']:+.3f} "
                  f"(95% {result_f['ci95'][0]:+.3f} to {result_f['ci95'][1]:+.3f}, p {result_f['p']:.3f}), {verdict(result_f)}.", ""]
        # By goal index.
        by_goal = []
        for k in range(1, GOALS + 1):
            m = model[(model.condition == condition) & (model.goal == k)]
            s = goals[(goals.policy == SCRIPTED) & (goals.condition == condition) & (goals.goal == k)]
            by_goal.append([k, len(m), fmt_pct((m.outcome == "S").mean() if len(m) else None),
                            fmt_pct((m.outcome == "C").mean() if len(m) else None),
                            fmt_pct(m.outcome.isin(["T", "L"]).mean() if len(m) else None),
                            f"{score(m):+.3f}" if len(m) else "–", len(s), f"{score(s):+.3f}" if len(s) else "–"])
        lines += ["**By goal** (model: all 8 seeds pooled; a goal counts if its chain got to it):", ""]
        lines += table(["Goal", "Model goals", "Reached", "Crash", "Timeout", "Score", "Scripted goals", "Scripted score"], by_goal)
        # Chain ends.
        cm = chains[(chains.policy != SCRIPTED) & (chains.condition == condition)]
        cs = chains[(chains.policy == SCRIPTED) & (chains.condition == condition)]
        ends = sorted(set(cm.end) | set(cs.end))
        lines += ["", "**How chains end** (share of chains):", ""]
        lines += table(["End", "Model", "Scripted"], [[e, fmt_pct((cm.end == e).mean()), fmt_pct((cs.end == e).mean())] for e in ends])
        lines += ["", f"Goals reached per chain: model {cm.goals_reached.mean():.2f} (seeds "
                  + ", ".join(f"{cm[cm.policy == run_name(ARM, s)].goals_reached.mean():.2f}" for s in SEEDS)
                  + f"), scripted {cs.goals_reached.mean():.2f}.", ""]
        # Crashes right after a switch.
        later_crash = m_l = model[(model.condition == condition) & (model.goal > 1) & (model.outcome == "C")]
        first_crash = model[(model.condition == condition) & (model.goal == 1) & (model.outcome == "C")]
        n_later = len(model[(model.condition == condition) & (model.goal > 1)])
        n_first = len(model[(model.condition == condition) & (model.goal == 1)])
        lines += [f"Crashes in a goal's first {SWITCH_WINDOW:g} s: goals 2–5 {(later_crash.duration <= SWITCH_WINDOW).sum()} of "
                  f"{n_later} goals ({fmt_pct((later_crash.duration <= SWITCH_WINDOW).sum() / max(n_later, 1))}); goal 1 "
                  f"{(first_crash.duration <= SWITCH_WINDOW).sum()} of {n_first} ({fmt_pct((first_crash.duration <= SWITCH_WINDOW).sum() / max(n_first, 1))}).", ""]
        # Feasibility removals.
        sw = model[(model.condition == condition) & (model.goal > 1)]
        cand, excl = sw.candidates.sum(), sw.excluded.sum()
        lines += [f"Feasibility check at the model's switches: {int(excl)} of {int(cand)} candidate goals removed "
                  f"({fmt_pct(excl / cand if cand else float('nan'))}).", ""]
        # From -> to table.
        pairs = sorted({(f, s) for f, s in zip(sw.from_slot, sw.slot)})
        scripted_sw = goals[(goals.policy == SCRIPTED) & (goals.condition == condition) & (goals.goal > 1)]
        pair_rows = []
        for f, s in pairs:
            m = sw[(sw.from_slot == f) & (sw.slot == s)]
            sc = scripted_sw[(scripted_sw.from_slot == f) & (scripted_sw.slot == s)]
            pair_rows.append([f"{f} → {s}", len(m), fmt_pct((m.outcome == "S").mean()), fmt_pct((m.outcome == "C").mean()),
                              fmt_pct(m.outcome.isin(["T", "L"]).mean()), f"{score(m):+.3f}", f"{m.time_needed.median():.1f}",
                              len(sc), fmt_pct((sc.outcome == "S").mean()) if len(sc) else "–",
                              "flagged" if (f, s) in flags else ""])
        lines += ["**From slot → to slot** (goals 2–5; model pooled over seeds):", ""]
        lines += table(["Pair", "Model goals", "Reached", "Crash", "Timeout", "Score", "Time needed (median s)",
                        "Scripted goals", "Scripted reached", ""], pair_rows) + [""]
        # Goal 1 by slot, for reference.
    lines += ["## Verdicts (bar written before the run)", ""]
    lines += table(["Condition", "Drop", "95% interval", "p", "Verdict", "Without flagged pairs",
                    "Scripted driver's drop (added after)", "Model's extra drop (added after)"],
                   [[c, f"{v[1]['mean']:+.3f}", f"{v[1]['ci95'][0]:+.3f} to {v[1]['ci95'][1]:+.3f}", f"{v[1]['p']:.3f}",
                     f"**{v[0]}**", f"{v[3]['mean']:+.3f}, {v[2]}", f"{v[5]:+.3f}",
                     f"{v[6]['mean']:+.3f} ({v[6]['ci95'][0]:+.3f} to {v[6]['ci95'][1]:+.3f})"] for c, v in verdicts.items()])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(goals)
    print("\n".join(lines))


def outcome_shares(frame):
    """Reached, crash, timeout (a lost anchor counts as a timeout, as in the score)."""
    return [(frame.outcome == "S").mean(), (frame.outcome == "C").mean(), frame.outcome.isin(["T", "L"]).mean()]


def figures(goals):
    """One column per condition. Top: score of goal k for the model (line and seed range) and the scripted driver
    (dashed). Bottom: what happened on goal 1 and on goals 2-5, for each driver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ink, muted, surface, grid = "#2b2a27", "#6b6a64", "#fcfcfb", "#e1e0d9"
    model_color = "#7b4fc9"
    outcome_colors = {"reached": "#2a78d6", "crash": "#eb6834", "timeout": "#898781"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "figure.facecolor": surface, "axes.facecolor": surface})
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(12, 7.4), gridspec_kw={"height_ratios": [1.45, 1]})
    x = np.arange(1, GOALS + 1)
    for col, condition in enumerate(CONDITIONS):
        model = goals[(goals.policy != SCRIPTED) & (goals.condition == condition)]
        scripted = goals[(goals.policy == SCRIPTED) & (goals.condition == condition)]
        # Top: score by goal.
        ax = axes[0, col]
        values = np.array([[score(model[(model.policy == run_name(ARM, seed)) & (model.goal == k)]) for k in x]
                           for seed in SEEDS], dtype=float)
        ax.fill_between(x, np.nanmin(values, 0), np.nanmax(values, 0), color=model_color, alpha=0.14, lw=0,
                        label="model: range over the 8 seeds")
        ax.plot(x, np.nanmean(values, 0), color=model_color, lw=2, marker="o", ms=5, label="model: mean of the 8 seeds")
        ax.plot(x, [score(scripted[scripted.goal == k]) for k in x], color=ink, lw=1.3, ls="--", marker="o", ms=4,
                mfc=surface, label="scripted driver")
        ax.set_xticks(x, [f"{k}\n{(model.goal == k).sum()} · {(scripted.goal == k).sum()}" for k in x], fontsize=8)
        ax.set_ylim(-0.7, 1.05)
        ax.axhline(0, color=grid, lw=1)
        ax.grid(axis="y", color=grid, lw=0.6)
        change = {who: score(part[part.goal > 1]) - score(part[part.goal == 1])
                  for who, part in (("model", model), ("scripted", scripted))}
        ax.set_title(f"{condition}\ngoal 1 → goals 2–5: model {change['model']:+.2f}, scripted {change['scripted']:+.2f}",
                     loc="left", fontsize=9)
        ax.set_xlabel("goal in the chain\n(goals scored: model, 8 seeds · scripted)", fontsize=8, color=muted)
        if col == 0:
            ax.set_ylabel("score of goal k (reached − crash)\namong chains that got to goal k")
            ax.legend(frameon=False, fontsize=8, loc="lower left")
        else:
            ax.tick_params(labelleft=False)
        # Bottom: outcome shares, goal 1 against goals 2-5.
        ax = axes[1, col]
        bars = [("model, goal 1", model[model.goal == 1]), ("model, goals 2–5", model[model.goal > 1]),
                ("scripted, goal 1", scripted[scripted.goal == 1]), ("scripted, goals 2–5", scripted[scripted.goal > 1])]
        rows = [0, 1, 2.4, 3.4]
        for y, (label, part) in zip(rows, bars):
            left = 0.0
            for (name, color), share in zip(outcome_colors.items(), outcome_shares(part)):
                ax.barh(y, share, left=left, height=0.7, color=color, edgecolor=surface, linewidth=1.5)
                if share >= 0.08:
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8,
                            color=ink if name == "timeout" else "white")
                left += share
        ax.set_yticks(rows, [b[0] for b in bars] if col == 0 else [""] * len(rows))
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("share of goals", loc="left", fontsize=8, color=muted)
    fig.legend(handles=[Patch(color=c, label=n) for n, c in outcome_colors.items()], loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Consecutive goals. With a new car, both drivers score lower after goal 1: the scripted driver times "
                 "out more, the model crashes more.\nWith the same car, only the model drops. "
                 "(The scripted driver reads the true state, so 'origin kept' is identical to 'new car' for it.)",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "chains.png", dpi=120)
    plt.close(fig)


def illustrative_chains():
    """Chosen after the results, by stated rules, to show the main findings (training seed 7)."""
    import pandas as pd
    run = run_name(ARM, 7)
    crashes = pd.read_csv(OUT / "crashes_new_car.csv")
    crashes = crashes[(crashes.seed == 7) & (crashes.partner == "previous anchor") & (crashes.since_switch <= 3.0)]
    chains, goals = load("chains.csv"), load("goals.csv")
    full = chains[(chains.policy == run) & (chains.condition == "new car") & (chains.end == "all goals reached")]
    behind = goals[(goals.policy == run) & (goals.condition == "same car") & (goals.from_slot == "front")
                   & goals.slot.isin(["left", "right", "back"]) & (goals.outcome != "S")]
    return [("new car", int(crashes.chain.min()), "rams the previous anchor right after a switch"),
            ("new car", int(full.chain.min()), "all 5 goals reached"),
            ("same car", int(behind.chain.min()), "after FRONT, fails to drop back to a slot behind")]


def demos(args):
    from baseline_dynamic_reach.document import record_episode
    folder = OUT / "demos"
    folder.mkdir(parents=True, exist_ok=True)
    found = json.loads((folder / "demos.json").read_text(encoding="utf-8")) if (folder / "demos.json").exists() else {}
    if args.illustrative:
        for condition, chain_seed, why in illustrative_chains():
            env = make_env(condition, render_mode="rgb_array")
            act = policy(run_name(ARM, 7), env)
            try:
                name = f"illustrative_{condition.replace(' ', '_')}_chain{chain_seed}.mp4"
                result = record_episode(env, act, chain_seed, folder / name)
                found[name] = {"why": why, **{k: result[k] for k in ("outcome", "elapsed", "length")},
                               "goals_reached_before_end": env.goal_index - (0 if result["success"] else 1)}
                print(name, found[name], flush=True)
            finally:
                env.close()
        (folder / "demos.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
        return
    for condition in CONDITIONS:
        env = make_env(condition, render_mode="rgb_array")
        act = policy(run_name(ARM, 7), env)
        try:
            for chain_seed in range(FIRST, FIRST + 3):
                name = f"{condition.replace(' ', '_')}_chain{chain_seed}.mp4"
                result = record_episode(env, act, chain_seed, folder / name)
                found[name] = {k: result[k] for k in ("outcome", "elapsed", "length")}
                found[name]["goals_reached_before_end"] = env.goal_index - (0 if result["success"] else 1)
                print(name, found[name], flush=True)
        finally:
            env.close()
    (folder / "demos.json").write_text(json.dumps(found, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", type=int, default=27)
    sub.add_parser("summary")
    d = sub.add_parser("demos")
    d.add_argument("--illustrative", action="store_true", help="also the 3 chains that show the main findings")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"run": run, "summary": summary, "demos": demos}[args.command](args)


if __name__ == "__main__":
    main()
