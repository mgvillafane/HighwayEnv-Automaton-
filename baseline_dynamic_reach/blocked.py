"""Step 5: which evaluation episodes have a car in the way, and what does a policy do about it?

    python -B baseline_dynamic_reach/blocked.py tag           # tag the 200 episodes, join every saved evaluation
    python -B baseline_dynamic_reach/blocked.py probe B3      # replay an arm: value at reset, first actions, crash partner
    python -B baseline_dynamic_reach/blocked.py passing B3    # blocked episodes: does the blocking car leave, does ego pass it?
    python -B baseline_dynamic_reach/blocked.py brake B3      # what-if: the same policy plus the scripted driver's following rule

An episode is *blocked* when, at reset, at least one car is in the slot's lane, ahead of ego
and behind the slot center. The tag is a property of the episode (its seed), not of a policy.

Everything is written to results/step5_blocked/:
    episodes.csv     one row per evaluation episode: the tag and every arm's saved outcome
    summary.md       outcomes for blocked and clear episodes, per arm and slot
    failures.md      every BACK/LEFT/RIGHT episode that an arm fails, with its replay command
    probe_<ARM>.json / probe_<ARM>.md
    passing_<ARM>.json / passing_<ARM>.md
    brake_<ARM>.json / brake_<ARM>.md
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import ARMS, EVAL_EPISODES, EVAL_SEED, MODELS_DIR, RESULTS_DIR, SEEDS, arm_config, run_name
from baseline_dynamic_reach.diagnose import crash_record
from baseline_dynamic_reach.environment import ReachEnv
from baseline_dynamic_waypoint.scripted import FASTER, IDLE, SLOWER, leader_gap

OUTPUT_DIR = RESULTS_DIR / "step5_blocked"
TARGET_SLOTS = ("back", "left", "right")  # FRONT is set aside (EXPERIMENT_NOTES.md, Step 5).
LEAD_CAR_GAP = 35.0  # The injected 12 m/s car starts this far ahead of ego, in ego's lane.
ACTIONS = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
PROBE_SECONDS = 2.0
LETTER = {"success": "S", "collision": "C", "timeout": "T", "anchor_lost": "L"}


def tag(env):
    """Describe the traffic between ego and the slot, right after env.reset()."""
    ego, road = env.highway.vehicle, env.highway.road
    lane = road.network.get_lane(env.slot_lane_index)
    ego_s, slot_s = lane.local_coordinates(ego.position)[0], lane.local_coordinates(env.goal_position)[0]
    between = sorted((lane.local_coordinates(v.position)[0], road.vehicles.index(v)) for v in road.vehicles
                     if v is not ego and v.lane_index[2] == env.slot_lane_index[2]
                     and ego_s < lane.local_coordinates(v.position)[0] < slot_s)
    row = {"ego_lane": int(ego.lane_index[2]), "slot_lane": int(env.slot_lane_index[2]),
           "lane_changes_required": abs(int(ego.lane_index[2]) - int(env.slot_lane_index[2])),
           "lead_car_anchor": bool(abs(env.anchor.position[0] - ego.position[0] - LEAD_CAR_GAP) < 1e-6),
           "slot_ahead": float(slot_s - ego_s), "anchor_speed": float(env.anchor.speed),
           "blocked": bool(between), "blockers": len(between), "blocker_ids": [index for _, index in between]}
    if between:
        first = road.vehicles[between[0][1]]
        row.update(blocker_gap=float(between[0][0] - ego_s), blocker_speed=float(first.speed),
                   blocker_to_slot=float(slot_s - between[-1][0]),
                   blocker_is_anchor=any(road.vehicles[i] is env.anchor for _, i in between),
                   blocker_is_lead_car=any(road.vehicles[i].lane_index[2] == ego.lane_index[2]
                                           and abs(s - ego_s - LEAD_CAR_GAP) < 1e-6 for s, i in between))
    return row


def tag_episodes(episodes=EVAL_EPISODES, seed=EVAL_SEED):
    env = ReachEnv(**deepcopy(arm_config("B0")["environment"]))  # Every arm has the same task and traffic.
    rows = []
    try:
        for episode_seed in range(seed, seed + episodes):
            _, info = env.reset(seed=episode_seed)
            rows.append({"seed": episode_seed, "slot": info["slot"], **tag(env)})
    finally:
        env.close()
    return rows


def saved_outcomes(run, policy="ppo", folder="evaluation_episodes.csv"):
    """{episode seed: S, C, T or L} from a run's saved evaluation, or None if it has none."""
    path = RESULTS_DIR / run / folder
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == policy]
    outcomes = {}
    for row in rows:
        flags = [letter for key, letter in LETTER.items() if int(row[key])]
        if len(flags) != 1:
            raise ValueError(f"{run}, seed {row['seed']}: {len(flags)} outcomes")
        outcomes[int(row["seed"])] = flags[0]
    return outcomes


def evaluated_runs():
    """{arm: {seed: outcomes}} for every arm with saved evaluations, in chain order."""
    found = {}
    for arm in ARMS:
        runs = {seed: saved_outcomes(run_name(arm, seed)) for seed in SEEDS}
        if all(runs.values()):
            found[arm] = runs
    return found


def scripted_outcomes(runs):
    """The scripted driver's rows are saved with every run; they must agree, since the task never changes."""
    tables = [saved_outcomes(run_name(arm, seed), "scripted") for arm in runs for seed in SEEDS]
    for other in tables[1:]:
        if other != tables[0]:
            raise ValueError("The scripted driver's saved outcomes differ between runs.")
    return tables[0]


def count(outcomes, letter="S"):
    return sum(outcome == letter for outcome in outcomes)


def write_tag(rows, runs, scripted):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for row in rows:
        line = {key: (json.dumps(value) if isinstance(value, list) else value) for key, value in row.items()}
        for key in ("blocker_gap", "blocker_speed", "blocker_to_slot", "blocker_is_anchor", "blocker_is_lead_car"):
            line.setdefault(key, "")
        for arm, seeds in runs.items():
            for seed, outcomes in seeds.items():
                line[run_name(arm, seed)] = outcomes[row["seed"]]
        line["scripted"] = scripted[row["seed"]]
        table.append(line)
    with (OUTPUT_DIR / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    def cells(subset, outcomes_by_seed):
        got = [outcomes[row["seed"]] for outcomes in outcomes_by_seed for row in subset]
        return f"{count(got)}/{len(got)} | {count(got, 'C')} | {count(got, 'T')}"

    lines = ["# Blocked and clear evaluation episodes", "",
             "Blocked: at reset, a car is in the slot's lane, ahead of ego and behind the slot center.",
             "Arm cells add up the three seeds. Columns: success / crashes / timeouts.", ""]
    for arm, seeds in runs.items():
        lines += [f"## {arm}", "", "| Slot | Tag | Episodes | PPO success | crash | timeout | Scripted success | crash | timeout |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        groups = [(slot, flag, [r for r in rows if r["slot"] == slot and r["blocked"] == flag])
                  for slot in ("back", "left", "right", "front") for flag in (False, True)]
        for flag in (False, True):
            groups.append(("back+left+right", flag, [r for r in rows if r["slot"] in TARGET_SLOTS and r["blocked"] == flag]))
        for slot, flag, subset in groups:
            if subset:
                lines.append(f"| {slot} | {'blocked' if flag else 'clear'} | {len(subset)} | "
                             f"{cells(subset, list(seeds.values()))} | {cells(subset, [scripted])} |")
        target = [r for r in rows if r["slot"] in TARGET_SLOTS]
        all_fail = [r for r in target if all(outcomes[r["seed"]] != "S" for outcomes in seeds.values())]
        lines += ["", f"Episodes of back+left+right that fail on all three seeds: {len(all_fail)}, "
                      f"of which blocked: {sum(r['blocked'] for r in all_fail)}.", ""]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    arm = list(runs)[-1]
    lines = [f"# BACK/LEFT/RIGHT episodes that {arm} fails on at least one seed", "",
             "Outcomes: S reached, C crash, T timeout, L anchor lost. Distances in meters, speeds in m/s.",
             "Replay any of them (from the repository root):", "",
             f"    .venv/bin/python -B baseline_dynamic_reach/examples.py show {arm}_seed7 --episode <episode>",
             f"    .venv/bin/python -B baseline_dynamic_reach/examples.py record {arm}_seed7 --episodes <episode> <episode>",
             "", f"| Episode | Slot | Lane changes | Slot ahead | Tag | Blocker gap | Blocker speed | "
                 f"{arm} seeds {'/'.join(map(str, SEEDS))} | Scripted |", "|---:|---|---:|---:|---|---:|---:|---|---|"]
    for row in rows:
        got = "".join(runs[arm][seed][row["seed"]] for seed in SEEDS)
        if row["slot"] in TARGET_SLOTS and got != "S" * len(SEEDS):
            gap = f"{row['blocker_gap']:.0f}" if row["blocked"] else ""
            speed = f"{row['blocker_speed']:.1f}" if row["blocked"] else ""
            lines.append(f"| {row['seed']} | {row['slot']} | {row['lane_changes_required']} | {row['slot_ahead']:.0f} | "
                         f"{'blocked' if row['blocked'] else 'clear'} | {gap} | {speed} | {got} | {scripted[row['seed']]} |")
    (OUTPUT_DIR / "failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUTPUT_DIR / "summary.md").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- probe

def probe_run(run, tags):
    """Replay one trained run on the evaluation episodes; deterministic actions, as in evaluate.py."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    spec = sidecar["environment"]["reward_weights"].get("potential_progress")
    shaping_weight = spec["weight"] if isinstance(spec, dict) else 0.0
    env = ReachEnv(**deepcopy(sidecar["environment"]))
    steps = round(PROBE_SECONDS * sidecar["environment"]["env_config"]["policy_frequency"])
    rows = []
    try:
        for episode in tags:
            obs, info = env.reset(seed=episode["seed"])
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(obs)
                value = float(model.policy.predict_values(tensor)[0, 0])
                probabilities = model.policy.get_distribution(tensor).distribution.probs[0].tolist()
            # Potential shaping adds weight x distance to every value; removing it leaves the
            # critic's estimate of (discounted) goal x success - crash weight x crash.
            row = {"seed": episode["seed"], "value_at_reset": value,
                   "value_without_shaping": value - shaping_weight * float(info["distance_to_goal"]),
                   "action_probabilities_at_reset": probabilities, "first_actions": []}
            first_lane_change = first_slower = None
            done = False
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                time = float(env.highway.time)
                if len(row["first_actions"]) < steps:
                    row["first_actions"].append(action)
                if first_lane_change is None and action in (0, 2):
                    first_lane_change = time
                if first_slower is None and action == 4:
                    first_slower = time
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            row.update(outcome=info["termination_reason"], time=float(info["elapsed_time"]),
                       first_lane_change=first_lane_change, first_slower=first_slower)
            if info["is_collision"]:
                ego = env.highway.vehicle
                partners = [v for v in env.highway.road.vehicles if v is not ego and v.crashed]
                partner = min(partners, key=lambda v: np.linalg.norm(v.position - ego.position), default=None)
                row.update(crash_record(env))
                row["hit_a_blocker"] = partner is not None and env.highway.road.vehicles.index(partner) in episode["blocker_ids"]
            rows.append(row)
    finally:
        env.close()
    return rows


def auc(positive, negative):
    """Probability that a random positive value is above a random negative one (ties count half)."""
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def crash_weights(arm):
    weights = arm_config(arm)["environment"]["reward_weights"]
    return f"{weights['goal']:g} x success - {weights['collision']:g} x crash"


def probe(arm):
    tags = tag_episodes()
    by_seed = {row["seed"]: row for row in tags}
    result, lines = {}, [f"# Probe of {arm} on the evaluation episodes", "",
                         f"Deterministic replay, as in evaluate.py. \"First actions\" are the first {PROBE_SECONDS:.0f} s.",
                         "\"Value\" is the critic at reset with the shaping constant removed: its estimate of the",
                         f"discounted {crash_weights(arm)}. BACK/LEFT/RIGHT episodes only.", ""]
    for seed in SEEDS:
        run = run_name(arm, seed)
        rows = probe_run(run, tags)
        saved = saved_outcomes(run)
        mismatched = [r["seed"] for r in rows if (r["outcome"] == "GOAL REACHED") != (saved[r["seed"]] == "S")]
        result[run] = {"replay_matches_evaluation": not mismatched, "mismatched": mismatched, "episodes": rows}
        target = [r for r in rows if by_seed[r["seed"]]["slot"] in TARGET_SLOTS]
        lines += [f"## {run}", "", f"Replay matches the saved evaluation: {not mismatched} {mismatched[:5] or ''}", "",
                  "| Tag | Episodes | Reached | Crashed | Mean value | " + " | ".join(ACTIONS)
                  + " | First lane change (median s) | Ever SLOWER before the end |", "|---|---:|---:|---:|---:|" + "---:|" * 7]
        groups = {}
        for flag in (False, True):
            subset = groups[flag] = [r for r in target if by_seed[r["seed"]]["blocked"] == flag]
            actions = Counter(a for r in subset for a in r["first_actions"])
            total = sum(actions.values())
            changes = [r["first_lane_change"] for r in subset if r["first_lane_change"] is not None]
            lines.append(f"| {'blocked' if flag else 'clear'} | {len(subset)} | {sum(r['outcome'] == 'GOAL REACHED' for r in subset)} | "
                         f"{sum(r['outcome'] == 'COLLISION' for r in subset)} | "
                         f"{np.mean([r['value_without_shaping'] for r in subset]):.1f} | "
                         + " | ".join(f"{actions[i] / total:.0%}" for i in range(len(ACTIONS)))
                         + f" | {np.median(changes):.1f} ({len(changes)}/{len(subset)}) | "
                           f"{sum(r['first_slower'] is not None for r in subset)}/{len(subset)} |")
        values = lambda rows_: [r["value_without_shaping"] for r in rows_]
        failed = [r for r in target if r["outcome"] != "GOAL REACHED"]
        reached = [r for r in target if r["outcome"] == "GOAL REACHED"]
        crashes = [r for r in target if r["outcome"] == "COLLISION"]
        blocked_crashes = [r for r in crashes if by_seed[r["seed"]]["blocked"]]
        kinds = Counter(f"{r['partner']} {r['where']}" + (" during lane change" if r["changing_lane"] else "") for r in crashes)
        lines += ["", f"- Value separates clear from blocked episodes with AUC {auc(values(groups[False]), values(groups[True])):.2f}, "
                      f"and reached from failed episodes with AUC {auc(values(reached), values(failed)):.2f} (0.5 = no information).",
                  f"- Mean value: reached {np.mean(values(reached)):.1f}, failed {np.mean(values(failed)):.1f}.",
                  f"- Crashes in blocked episodes that hit a car tagged as a blocker at reset: "
                  f"{sum(r['hit_a_blocker'] for r in blocked_crashes)}/{len(blocked_crashes)}.",
                  f"- All {len(crashes)} crashes by kind: {dict(kinds.most_common())}.", ""]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"probe_{arm}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUTPUT_DIR / f"probe_{arm}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def passing_run(run, tags):
    """Replay one run on the blocked BACK/LEFT/RIGHT episodes and follow the nearest blocking car."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    env = ReachEnv(**deepcopy(sidecar["environment"]))
    spec = sidecar["environment"]["reward_weights"].get("potential_progress")
    shaping_weight = spec["weight"] if isinstance(spec, dict) else 0.0
    rows = []
    try:
        for episode in tags:
            if episode["slot"] not in TARGET_SLOTS or not episode["blocked"]:
                continue
            obs, info = env.reset(seed=episode["seed"])
            ego, vehicles = env.highway.vehicle, env.highway.road.vehicles
            blocker = vehicles[episode["blocker_ids"][0]]  # The nearest one: tag() sorts them along the lane.
            lane = blocker.lane_index[2]
            left_at = passed_at = None
            passed_in_lane = ego_left_lane_behind = braked_behind = False
            values, behind_probabilities = [], []
            done = False
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                behind_in_lane = left_at is None and passed_at is None
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(obs)
                    values.append(float(model.policy.predict_values(tensor)[0, 0])
                                  - shaping_weight * float(env.distance_to_goal()))
                    if behind_in_lane:  # What the sampling policy used in training would do here.
                        behind_probabilities.append(model.policy.get_distribution(tensor).distribution.probs[0].tolist())
                if behind_in_lane and action == 4:
                    braked_behind = True
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                time = float(env.highway.time)
                if left_at is None and blocker.target_lane_index[2] != lane:
                    left_at = time  # The blocking car starts a lane change of its own (MOBIL).
                if behind_in_lane and left_at is None and episode["lane_changes_required"] == 0 \
                        and ego.target_lane_index[2] != lane:
                    ego_left_lane_behind = True
                if passed_at is None and ego.position[0] > blocker.position[0]:
                    passed_at, passed_in_lane = time, left_at is None
            partner = None
            if info["is_collision"]:
                hit = [v for v in vehicles if v is not ego and v.crashed]
                partner = "blocker" if blocker in hit else "anchor" if env.anchor in hit else "other car"
            rows.append({"seed": episode["seed"], "slot": episode["slot"],
                         "lane_changes_required": episode["lane_changes_required"],
                         "outcome": info["termination_reason"], "time": float(info["elapsed_time"]),
                         "blocker_left_at": left_at, "ego_passed_at": passed_at,
                         "passed_while_blocker_in_lane": passed_in_lane,
                         "ego_left_blocker_lane_while_behind": ego_left_lane_behind,
                         "braked_while_behind": braked_behind, "crash_partner": partner,
                         "value_at_reset": values[0], "value_at_last_decision": values[-1],
                         "mean_action_probabilities_while_behind": np.mean(behind_probabilities, axis=0).tolist()})
    finally:
        env.close()
    return rows


def passing(arm):
    """In blocked episodes, is success explained by what the blocking car does or by what ego does?"""
    tags = tag_episodes()
    result = {}
    lines = [f"# {arm} on the blocked BACK/LEFT/RIGHT episodes: who gets out of whose way?", "",
             "Deterministic replay. The *blocking car* is the nearest car between ego and the slot, in the slot's",
             "lane, at reset. It *leaves* when it starts a lane change of its own (its MOBIL model decides this;",
             "the anchor never changes lane, other cars may). Ego *passes it in its lane* when ego gets ahead of it",
             "while it is still in its starting lane.", ""]
    for seed in SEEDS:
        run = run_name(arm, seed)
        rows = result[run] = passing_run(run, tags)
        lines += [f"## {run}", "",
                  "| Blocking car | Episodes | Reached | Crashed into it | Crashed into another car | Timeout |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, subset in (("stays in its lane", [r for r in rows if r["blocker_left_at"] is None]),
                              ("leaves its lane by itself", [r for r in rows if r["blocker_left_at"] is not None])):
            lines.append(f"| {label} | {len(subset)} | {sum(r['outcome'] == 'GOAL REACHED' for r in subset)} | "
                         f"{sum(r['crash_partner'] == 'blocker' for r in subset)} | "
                         f"{sum(r['crash_partner'] not in (None, 'blocker') for r in subset)} | "
                         f"{sum(r['outcome'] == 'TIMED OUT' for r in subset)} |")
        stays_reached = [r["seed"] for r in rows if r["blocker_left_at"] is None and r["outcome"] == "GOAL REACHED"]
        own_lane = [r for r in rows if r["lane_changes_required"] == 0]
        lines += ["",
                  f"- Ego passed a blocking car that was still in its lane: "
                  f"{sum(r['passed_while_blocker_in_lane'] for r in rows)}/{len(rows)} episodes.",
                  f"- Slot in ego's own lane ({len(own_lane)} episodes): ego started a lane change while behind the "
                  f"blocking car in {sum(r['ego_left_blocker_lane_while_behind'] for r in own_lane)}, and chose SLOWER "
                  f"while behind it in {sum(r['braked_while_behind'] for r in own_lane)}.",
                  f"- Reached although the blocking car stayed: {stays_reached or 'none'}."]
        rammed = [r for r in rows if r["crash_partner"] == "blocker"]
        if rammed:
            probabilities = np.mean([r["mean_action_probabilities_while_behind"] for r in rammed], axis=0)
            lines += [f"- The {len(rammed)} crashes into the blocking car. Critic value (shaping removed; a crash is "
                      f"-{arm_config(arm)['environment']['reward_weights']['collision']:g}): {np.mean([r['value_at_reset'] for r in rammed]):.1f} at reset, "
                      f"{np.mean([r['value_at_last_decision'] for r in rammed]):.1f} at the last decision, 0.1 s before "
                      f"the crash ({sum(r['value_at_last_decision'] > 0 for r in rammed)} of them above 0).",
                      "- Same crashes, sampling probabilities per decision while behind the blocking car (what "
                      "training explores): " + ", ".join(f"{name} {p:.1%}" for name, p in zip(ACTIONS, probabilities))
                      + "."]
        lines.append("")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"passing_{arm}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUTPUT_DIR / f"passing_{arm}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def following_limit(env):
    """The scripted driver's following rule (scripted.py, the lines on leader_gap): the highest target speed
    allowed behind the car ahead in ego's lane, or None when that car is far enough away."""
    ego = env.highway.vehicle
    gap = leader_gap(env, ego.lane_index[2])
    if gap < 12 + 0.8 * ego.speed:
        return float(np.clip(env.anchor.speed if gap > 9 else 10.0, 10.0, 25.0))
    return None


def brake_run(run, tags):
    """Replay one run on the BACK/LEFT/RIGHT episodes with the following rule laid over the policy's speed choice."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    env = ReachEnv(**deepcopy(sidecar["environment"]))
    rows = []
    try:
        for episode in tags:
            if episode["slot"] not in TARGET_SLOTS:
                continue
            obs, info = env.reset(seed=episode["seed"])
            ego = env.highway.vehicle
            blocker = env.highway.road.vehicles[episode["blocker_ids"][0]] if episode["blocked"] else None
            lane = None if blocker is None else blocker.lane_index[2]
            blocker_left, overridden, decisions, done = False, 0, 0, False
            while not done:
                action = chosen = int(model.predict(obs, deterministic=True)[0])
                limit = following_limit(env)
                if limit is not None:
                    if ego.target_speed > limit + 0.5:
                        action = SLOWER
                    elif chosen == FASTER and ego.target_speed > limit - 0.5:
                        action = IDLE
                overridden += action != chosen
                decisions += 1
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if blocker is not None and blocker.target_lane_index[2] != lane:
                    blocker_left = True
            rows.append({"seed": episode["seed"], "slot": episode["slot"], "blocked": episode["blocked"],
                         "blocker_left": blocker_left, "outcome": info["termination_reason"],
                         "time": float(info["elapsed_time"]), "decisions": decisions, "overridden": overridden})
    finally:
        env.close()
    return rows


def brake(arm):
    """A what-if, not an arm: how much would braking behind the car ahead be worth to this policy?"""
    tags = tag_episodes()
    result = {}
    lines = [f"# What-if: {arm} plus the scripted driver's following rule", "",
             "Nothing is trained. The policy is replayed deterministically on the BACK/LEFT/RIGHT episodes, and one",
             "hand-written rule is laid over its action: when the gap to the car ahead in ego's lane is below",
             "12 m + 0.8 s x ego speed, the target speed may not exceed the anchor's speed (10 m/s below a 9 m gap).",
             "Then SLOWER replaces the action while the target speed is above that, and IDLE replaces FASTER at it.",
             "Lane changes are the policy's own. \"Without the rule\" is the saved evaluation.", ""]
    for seed in SEEDS:
        run = run_name(arm, seed)
        rows = result[run] = brake_run(run, tags)
        saved = saved_outcomes(run)
        lines += [f"## {run}", "",
                  "| Episodes | Count | Without the rule: reached | crash | timeout | With the rule: reached | crash | timeout |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        groups = (("clear", [r for r in rows if not r["blocked"]]),
                  ("blocked, blocking car stays", [r for r in rows if r["blocked"] and not r["blocker_left"]]),
                  ("blocked, blocking car leaves by itself", [r for r in rows if r["blocked"] and r["blocker_left"]]),
                  ("all BACK/LEFT/RIGHT", rows))
        for label, subset in groups:
            before = [saved[r["seed"]] for r in subset]
            lines.append(f"| {label} | {len(subset)} | " + " | ".join(str(count(before, x)) for x in "SCT") + " | "
                         + " | ".join(str(sum(r["outcome"] == name for r in subset))
                                      for name in ("GOAL REACHED", "COLLISION", "TIMED OUT")) + " |")
        gained = [r["seed"] for r in rows if r["outcome"] == "GOAL REACHED" and saved[r["seed"]] != "S"]
        lost = [r["seed"] for r in rows if r["outcome"] != "GOAL REACHED" and saved[r["seed"]] == "S"]
        crashes = [r["seed"] for r in rows if r["outcome"] == "COLLISION"]
        lines += ["",
                  f"- The rule changed {sum(r['overridden'] for r in rows):,} of {sum(r['decisions'] for r in rows):,} "
                  f"decisions, in {sum(r['overridden'] > 0 for r in rows)} of {len(rows)} episodes.",
                  f"- Episodes gained: {gained or 'none'}.", f"- Episodes lost: {lost or 'none'}.",
                  f"- Still crashing: {crashes or 'none'}.",
                  "- \"Stays\" and \"leaves\" are measured in this replay, so they can differ from passing_"
                  f"{arm}.md: a blocking car has more time to leave when ego waits behind it.", ""]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"brake_{arm}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUTPUT_DIR / f"brake_{arm}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tag")
    sub.add_parser("probe").add_argument("arm", choices=list(ARMS))
    sub.add_parser("passing").add_argument("arm", choices=list(ARMS))
    sub.add_parser("brake").add_argument("arm", choices=list(ARMS))
    args = parser.parse_args()
    if args.command == "tag":
        runs = evaluated_runs()
        write_tag(tag_episodes(), runs, scripted_outcomes(runs))
    elif args.command == "probe":
        probe(args.arm)
    elif args.command == "passing":
        passing(args.arm)
    else:
        brake(args.arm)


if __name__ == "__main__":
    main()
