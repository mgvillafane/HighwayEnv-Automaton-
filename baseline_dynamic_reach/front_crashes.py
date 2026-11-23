"""Step 8g diagnosis (no training): how the adopted recipe crashes on FRONT.

Replays the adopted model O3_c1w0t4_n256 (all 8 seeds) and the scripted driver on the FRONT episodes of
the two selection sets (seeds 10,000-10,199 and 40,000-40,199: 133 episodes), with deterministic actions
as in evaluate.py. Every replayed outcome and length is checked against the saved evaluation rows.

For each crash it records:
- the car hit (the anchor, the lead car that starts 35 m ahead of ego, or another car), and where it was
  (ahead, behind or beside ego), and whether ego was changing lanes;
- the phase of the FRONT maneuver: ego behind the anchor, beside it, or past it (in another lane, merging
  into the anchor's lane, or already in it);
- speeds, ego's position relative to the anchor and the slot, and the last 2 s of actions.

    python -B baseline_dynamic_reach/front_crashes.py          # about 5 min with 9 processes
Outputs: results/front_crashes/ (episodes.csv, crashes.csv, summary.md).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import MODELS_DIR, RESULTS_DIR, STEP8_CONFIRM_SEEDS, STEP8_SCREEN_SEEDS

ARM = "O3_c1w0t4_n256"
SCRIPTED = "scripted"
SETS = {"s10000": 10_000, "s40000": 40_000}
OUT = RESULTS_DIR / "front_crashes"
LETTERS = "LIRFS"  # LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
HISTORY = 20       # decisions kept before a crash (2 s)
BESIDE = 5.0       # m: |ego - anchor| along the road within this counts as beside the anchor


def front_episodes():
    from baseline_dynamic_reach.grid import tags
    return sorted(seed for seed, tag in tags().items() if tag["slot"] == "front")


def saved_rows(run):
    """{episode seed: (outcome letter, length)} from the saved evaluations of both selection sets."""
    found = {}
    for name in SETS:
        if run == SCRIPTED:
            path, policy = RESULTS_DIR / "grid8" / "references" / name / "evaluation_episodes.csv", "scripted"
        else:
            path, policy = RESULTS_DIR / run / "eval8" / f"{name}_final" / "evaluation_episodes.csv", "ppo"
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["policy"] == policy:
                    letter = "S" if int(row["success"]) else "C" if int(row["collision"]) else "T"
                    found[int(row["seed"])] = (letter, int(row["length"]))
    return found


def phase(env, ego_dx_anchor):
    ego = env.highway.vehicle
    slot_lane = env.slot_lane_index[2]
    if ego_dx_anchor < -BESIDE:
        return "behind the anchor"
    if ego_dx_anchor <= BESIDE:
        return "beside the anchor"
    if ego.lane_index[2] == slot_lane and ego.target_lane_index[2] == slot_lane:
        return "past the anchor, in its lane"
    if ego.target_lane_index[2] == slot_lane or ego.lane_index[2] == slot_lane:
        return "past the anchor, merging into its lane"
    return "past the anchor, other lane"


def crash_detail(env, lead, history):
    ego, anchor = env.highway.vehicle, env.anchor
    partners = [v for v in env.highway.road.vehicles if v is not ego and v.crashed]
    partner = min(partners, key=lambda v: np.linalg.norm(v.position - ego.position), default=None)
    dx_anchor = float(ego.position[0] - anchor.position[0])
    detail = {"phase": phase(env, dx_anchor), "ego_minus_anchor_m": dx_anchor,
              "slot_minus_ego_m": float(env.slot_error()[0]), "ego_speed": float(ego.speed),
              "anchor_speed": float(anchor.speed), "ego_lane": int(ego.lane_index[2]),
              "slot_lane": int(env.slot_lane_index[2]), "changing_lane": bool(ego.lane_index != ego.target_lane_index),
              "last_2s_actions": "".join(LETTERS[a] for a in history),
              "slower_in_last_2s": sum(a == 4 for a in history), "faster_in_last_2s": sum(a == 3 for a in history)}
    if partner is None:
        return {**detail, "partner": "none", "where": "unknown"}
    dx, dy = partner.position - ego.position
    where = "ahead" if dx > 2.5 else "behind" if dx < -2.5 else "beside"
    name = "anchor" if partner is anchor else "lead car" if partner is lead else "other car"
    return {**detail, "partner": name, "where": where, "partner_speed": float(partner.speed),
            "partner_lane": int(partner.lane_index[2]), "dx_m": float(dx), "dy_m": float(dy),
            "closing_speed": float(ego.speed - partner.speed) if where == "ahead" else float(partner.speed - ego.speed)}


def replay(run, episodes):
    """One row per episode; crash rows carry crash_detail. Runs in its own process."""
    import torch
    from baseline_dynamic_reach.config import arm_config
    from baseline_dynamic_reach.environment import ReachEnv
    torch.set_num_threads(1)
    if run == SCRIPTED:
        from baseline_dynamic_waypoint.scripted import scripted_action
        env = ReachEnv(**deepcopy(arm_config(ARM)["environment"]))  # Dynamics do not depend on the reward.
        act = lambda obs: scripted_action(env)  # noqa: E731
    else:
        from stable_baselines3 import PPO
        sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
        model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
        env = ReachEnv(**deepcopy(sidecar["environment"]))
        act = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    rows = []
    try:
        for episode in episodes:
            obs, info = env.reset(seed=episode)
            assert info["slot"] == "front", (run, episode, info["slot"])
            ego = env.highway.vehicle
            lead = next((v for v in env.highway.road.vehicles if v is not ego and v.lane_index == ego.lane_index
                         and abs(v.position[0] - ego.position[0] - 35.0) < 1.0), None)
            history, passed_at, length, done = [], None, 0, False
            while not done:
                action = int(act(obs))
                history = (history + [action])[-HISTORY:]
                obs, _, terminated, truncated, info = env.step(action)
                length += 1
                if passed_at is None and ego.position[0] - env.anchor.position[0] > BESIDE:
                    passed_at = length / 10
                done = terminated or truncated
            letter = ("S" if info["is_success"] else "C" if info["is_collision"]
                      else "L" if info.get("is_anchor_lost") else "T")
            row = {"run": run, "episode": episode, "outcome": letter, "length": length, "time_s": length / 10,
                   "passed_anchor_at_s": passed_at, "lead_is_anchor": lead is env.anchor}
            if letter == "C":
                row.update(crash_detail(env, lead, history))
            rows.append(row)
    finally:
        env.close()
    return rows


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def summary(rows, seeds, saved_ok):
    ppo = [r for r in rows if r["run"] != SCRIPTED]
    scripted = {r["episode"]: r for r in rows if r["run"] == SCRIPTED}
    crashes = [r for r in ppo if r["outcome"] == "C"]
    n_ep = len(scripted)
    lines = [f"# Step 8g: how {ARM} crashes on FRONT (diagnosis, no training)", "",
             f"{n_ep} FRONT episodes of the selection sets (10,000-10,199 and 40,000-40,199), final models of seeds "
             f"{', '.join(map(str, seeds))}, deterministic actions. Replayed outcomes and lengths match the saved "
             f"evaluations: {saved_ok}.", "", "## Outcomes", ""]
    out_rows = []
    for run in [f"{ARM}_seed{s}" for s in seeds] + [SCRIPTED]:
        part = [r for r in rows if r["run"] == run]
        c = Counter(r["outcome"] for r in part)
        out_rows.append([run, f"{c['S']} ({c['S'] / len(part):.0%})", f"{c['C']} ({c['C'] / len(part):.0%})",
                         c["T"] + c["L"]])
    c = Counter(r["outcome"] for r in ppo)
    out_rows.append([f"all {len(seeds)} seeds", f"{c['S']} ({c['S'] / len(ppo):.1%})", f"{c['C']} ({c['C'] / len(ppo):.1%})",
                     c["T"] + c["L"]])
    lines += table(["Run", "Reached", "Crash", "Timeout"], out_rows)

    def share(counter, total):
        return [[k, v, f"{v / total:.0%}"] for k, v in counter.most_common()]

    lines += ["", f"## The {len(crashes)} crashes of the {len(seeds)} seeds", "", "**Car hit and where it was:**", ""]
    kinds = Counter(f"{r['partner']}, {r['where']}" + (", ego changing lanes" if r["changing_lane"] else "")
                    for r in crashes)
    lines += table(["Crash", "Count", "Share"], share(kinds, len(crashes)))
    lines += ["", "**Phase of the FRONT maneuver at the crash:**", ""]
    lines += table(["Phase", "Count", "Share"], share(Counter(r["phase"] for r in crashes), len(crashes)))
    lines += ["", "**Phase × car hit:**", ""]
    lines += table(["Phase, car hit", "Count", "Share"],
                   share(Counter(f"{r['phase']}: {r['partner']} {r['where']}" for r in crashes), len(crashes)))

    def med(key, part):
        values = [r[key] for r in part if r.get(key) is not None]
        return f"{statistics.median(values):.1f}" if values else "–"

    lines += ["", "**Numbers at the crash (medians):**", ""]
    num_rows = []
    for name, part in [("all crashes", crashes)] + [(p, [r for r in crashes if r["phase"] == p])
                                                     for p, _ in Counter(r["phase"] for r in crashes).most_common()]:
        num_rows.append([name, len(part), med("time_s", part), med("ego_speed", part), med("anchor_speed", part),
                         med("closing_speed", part), med("ego_minus_anchor_m", part), med("slot_minus_ego_m", part),
                         f"{sum(r['slower_in_last_2s'] > 0 for r in part) / len(part):.0%}",
                         f"{sum(r['faster_in_last_2s'] > 0 for r in part) / len(part):.0%}"])
    lines += table(["Group", "n", "Time (s)", "Ego speed", "Anchor speed", "Closing speed", "Ego − anchor (m)",
                    "Slot − ego (m)", "SLOWER in last 2 s", "FASTER in last 2 s"], num_rows)

    per_episode = Counter(r["episode"] for r in crashes)
    spread = Counter(per_episode.get(e, 0) for e in scripted)
    lines += ["", "## Are the crashes tied to episodes or to seeds?", "",
              f"Number of the {len(seeds)} seeds that crash on an episode, over the {n_ep} FRONT episodes:", ""]
    lines += table(["Seeds crashing", "Episodes"], [[k, spread[k]] for k in sorted(spread)])
    hard = sorted(e for e, k in per_episode.items() if k >= len(seeds) / 2)
    hard_scripted = Counter(scripted[e]["outcome"] for e in hard)
    lines += ["", f"Episodes where at least half the seeds crash: {len(hard)}, with "
              f"{sum(per_episode[e] for e in hard)} of the {len(crashes)} crashes. The scripted driver on them: "
              f"{hard_scripted['S']} reached, {hard_scripted['C']} crash, {hard_scripted['T'] + hard_scripted['L']} timeout.",
              "", "| Episode | Seeds crashing | Lead car is the anchor | Scripted | Typical crash |", "|---|---:|---|---|---|"]
    for e in hard:
        kinds_e = Counter(f"{r['phase']}: {r['partner']} {r['where']}" for r in crashes if r["episode"] == e)
        lead = next(r["lead_is_anchor"] for r in ppo if r["episode"] == e)
        lines.append(f"| {e} | {per_episode[e]} | {'yes' if lead else 'no'} | {scripted[e]['outcome']} | "
                     f"{kinds_e.most_common(1)[0][0]} |")
    lines += ["", "## Scripted driver's outcome on each PPO outcome", "",
              "Each PPO episode (seed × episode) against the scripted driver's outcome on the same episode:", ""]
    cross = Counter((r["outcome"], scripted[r["episode"]]["outcome"]) for r in ppo)
    lines += table(["PPO \\ scripted", "S", "C", "T"],
                   [[p, cross[(p, "S")], cross[(p, "C")], cross[(p, "T")] + cross[(p, "L")]] for p in "SCT"])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STEP8_SCREEN_SEEDS + STEP8_CONFIRM_SEEDS))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    episodes = front_episodes()
    runs = [f"{ARM}_seed{s}" for s in args.seeds] + [SCRIPTED]
    with ProcessPoolExecutor(max_workers=len(runs)) as pool:
        results = dict(zip(runs, pool.map(replay, runs, [episodes] * len(runs))))
    mismatches = []
    for run, rows in results.items():
        saved = saved_rows(run)
        mismatches += [(run, r["episode"]) for r in rows
                       if saved[r["episode"]] != (r["outcome"] if r["outcome"] != "L" else "T", r["length"])]
    rows = [r for run in runs for r in results[run]]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("episodes.csv", rows), ("crashes.csv", [r for r in rows if r["outcome"] == "C"])):
        fields = list(dict.fromkeys(k for r in part for k in r))
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(part)
    saved_ok = "yes, all" if not mismatches else f"NO, {len(mismatches)} differ: {mismatches[:5]}"
    lines = summary(rows, args.seeds, saved_ok)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
