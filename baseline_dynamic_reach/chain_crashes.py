"""Step 9 follow-up (no training): what does the adopted recipe hit when a chained goal ends in a crash?

Replays the first 200 "new car" chains of every seed (chain seeds 70,000-70,199, the same as chain.py) and, for
every crash on goals 2-5, records the car hit (the previous goal's anchor, the new anchor, or another car), where it
was, the time since the switch and the speeds. Outcomes are checked against chain.py's saved goal rows.

    python -B baseline_dynamic_reach/chain_crashes.py
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.chain import ARM, FIRST, OUT, SEEDS, make_env, outcome_of, policy
from baseline_dynamic_reach.config import run_name

CHAINS = 200


def replay(seed):
    name = run_name(ARM, seed)
    env = make_env("new car")
    act = policy(name, env)
    crashes, outcomes = [], {}
    try:
        for chain_seed in range(FIRST, FIRST + CHAINS):
            obs, info = env.reset(seed=chain_seed)
            previous_anchor, previous_slot, switch_time, index = None, "start", 0.0, 1
            while True:
                anchor_before, slot_before = env.anchor, env.slot
                obs, _, terminated, truncated, info = env.step(act(obs))
                if info.get("goal_switched"):
                    outcomes[(chain_seed, index)] = "S"
                    previous_anchor, previous_slot, switch_time = anchor_before, slot_before, float(env.goal_start_time)
                    index = env.goal_index
                    continue
                if terminated or truncated:
                    letter = outcome_of(info)
                    outcomes[(chain_seed, index)] = letter
                    if letter == "C" and index > 1:
                        ego = env.highway.vehicle
                        partners = [v for v in env.highway.road.vehicles if v is not ego and v.crashed]
                        partner = min(partners, key=lambda v: np.linalg.norm(v.position - ego.position), default=None)
                        dx = None if partner is None else float(partner.position[0] - ego.position[0])
                        crashes.append({
                            "seed": seed, "chain": chain_seed, "goal": index, "from_slot": previous_slot, "slot": env.slot,
                            "since_switch": float(env.highway.time) - switch_time,
                            "partner": "none" if partner is None else "previous anchor" if partner is previous_anchor
                            else "new anchor" if partner is env.anchor else "other car",
                            "where": "unknown" if dx is None else "ahead" if dx > 2.5 else "behind" if dx < -2.5 else "beside",
                            "ego_speed": float(ego.speed), "changing_lane": bool(ego.lane_index != ego.target_lane_index)})
                    break
    finally:
        env.close()
    return seed, crashes, outcomes


def main():
    import pandas as pd
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    saved = pd.read_csv(OUT / "goals.csv")
    saved = saved[saved.condition == "new car"]
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        results = list(pool.map(replay, SEEDS))
    crashes, mismatches = [], 0
    for seed, rows, outcomes in results:
        crashes += rows
        mine = saved[(saved.policy == run_name(ARM, seed)) & (saved.chain < FIRST + CHAINS)]
        for row in mine.itertuples():
            if outcomes.get((row.chain, row.goal)) != row.outcome:
                mismatches += 1
    frame = pd.DataFrame(crashes)
    early = frame[frame.since_switch <= 3.0]
    lines = [f"# What the adopted recipe hits on chained goals (new car; first {CHAINS} chains of each of the 8 seeds)", "",
             f"Replayed goal outcomes against chain.py's saved rows: {mismatches} mismatches.", "",
             f"{len(frame)} crashes on goals 2–5; {len(early)} ({len(early) / len(frame):.0%}) within 3 s of the switch.", "",
             "| Car hit, where | All crashes on goals 2–5 | Within 3 s of the switch |", "|---|---:|---:|"]
    kinds = Counter(f"{r.partner}, {r.where}" for r in frame.itertuples())
    kinds_early = Counter(f"{r.partner}, {r.where}" for r in early.itertuples())
    lines += [f"| {k} | {v} ({v / len(frame):.0%}) | {kinds_early.get(k, 0)} |" for k, v in kinds.most_common()]
    lines += ["", "| Previous slot | Crashes | Into the previous anchor | Median time since switch (s) | Median ego speed |",
              "|---|---:|---:|---:|---:|"]
    for slot, part in frame.groupby("from_slot"):
        lines.append(f"| {slot} | {len(part)} | {(part.partner == 'previous anchor').mean():.0%} | "
                     f"{part.since_switch.median():.1f} | {part.ego_speed.median():.1f} |")
    out = OUT / "crashes_new_car.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frame.to_csv(OUT / "crashes_new_car.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
