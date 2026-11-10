"""Why do episodes fail? Replay a policy on the evaluation episodes and describe each crash.

For every crash: which car was hit (the anchor or another car), where it was relative to ego
(ahead, behind or beside, in ego's lane frame), whether ego was changing lanes, and the
speeds. The replay uses the same seeds and deterministic actions as evaluate.py, so its
outcomes match the evaluation rows (checked).

    python -B baseline_dynamic_reach/diagnose.py B0_seed7
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

from baseline_dynamic_reach.config import EVAL_EPISODES, EVAL_SEED, MODELS_DIR, RESULTS_DIR, SLOTS
from baseline_dynamic_reach.environment import ReachEnv


def crash_record(env):
    ego, anchor = env.highway.vehicle, env.anchor
    partners = [v for v in env.highway.road.vehicles if v is not ego and v.crashed]
    # The car whose footprint is nearest is the one ego hit (others may have crashed earlier).
    partner = min(partners, key=lambda v: np.linalg.norm(v.position - ego.position), default=None)
    record = {"changing_lane": bool(ego.lane_index != ego.target_lane_index), "ego_speed": float(ego.speed),
              "ego_lane": int(ego.lane_index[2])}
    if partner is None:
        return {**record, "partner": "none", "where": "unknown"}
    dx, dy = partner.position - ego.position
    where = "ahead" if dx > 2.5 else "behind" if dx < -2.5 else "beside"
    return {**record, "partner": "anchor" if partner is anchor else "other car", "where": where,
            "partner_speed": float(partner.speed), "closing_speed": float(ego.speed - partner.speed) if where == "ahead"
            else float(partner.speed - ego.speed), "dx": float(dx), "dy": float(dy),
            "partner_lane": int(partner.lane_index[2])}


def diagnose(run, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
    model = PPO.load(MODELS_DIR / f"{run}.zip", device="cpu")
    env = ReachEnv(**deepcopy(sidecar["environment"]))
    rows = []
    try:
        for episode_seed in range(seed, seed + episodes):
            obs, info = env.reset(seed=episode_seed)
            slot = info["slot"]
            lead_anchor = bool(abs(info["anchor_position"][0] - env.highway.vehicle.position[0] - 35.0) < 1e-6)
            lane_changes = 0
            done = False
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                before = env.highway.vehicle.target_lane_index
                obs, _, terminated, truncated, info = env.step(action)
                lane_changes += env.highway.vehicle.target_lane_index != before
                done = terminated or truncated
            row = {"seed": episode_seed, "slot": slot, "lead_car_anchor": lead_anchor,
                   "outcome": info["termination_reason"], "time": float(info["elapsed_time"]),
                   "lane_changes": lane_changes, "final_distance": float(info["distance_to_goal"])}
            if info["is_collision"]:
                row.update(crash_record(env))
            rows.append(row)
    finally:
        env.close()
    return rows


def report(rows):
    lines = []
    for slot in SLOTS:
        subset = [r for r in rows if r["slot"] == slot]
        crashes = [r for r in subset if r["outcome"] == "COLLISION"]
        kinds = Counter(f"{r['partner']} {r['where']}" + (" during lane change" if r["changing_lane"] else "")
                        for r in crashes)
        lines.append(f"{slot.upper()}: {len(subset)} episodes, {Counter(r['outcome'] for r in subset)}")
        for kind, count in kinds.most_common():
            lines.append(f"    crash into {kind}: {count}")
        if crashes:
            lines.append(f"    crash time median {np.median([r['time'] for r in crashes]):.1f} s, "
                         f"closing speed median {np.median([r.get('closing_speed', np.nan) for r in crashes]):.1f} m/s")
    lead = [r for r in rows if r["lead_car_anchor"]]
    other = [r for r in rows if not r["lead_car_anchor"]]
    lines.append(f"Lead-car anchor: {sum(r['outcome'] == 'GOAL REACHED' for r in lead)}/{len(lead)} reached; "
                 f"other anchors: {sum(r['outcome'] == 'GOAL REACHED' for r in other)}/{len(other)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()
    rows = diagnose(args.run, args.episodes)
    evaluation = RESULTS_DIR / args.run / "evaluation_episodes.csv"
    if evaluation.exists():
        with evaluation.open(encoding="utf-8") as handle:
            ppo = [r for r in csv.DictReader(handle) if r["policy"] == "ppo"][:args.episodes]
        mismatched = [r["seed"] for r, d in zip(ppo, rows) if int(r["success"]) != (d["outcome"] == "GOAL REACHED")]
        print(f"Replay matches evaluation outcomes: {not mismatched} {mismatched[:5]}")
    out = RESULTS_DIR / args.run / "crash_diagnosis.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    text = report(rows)
    (RESULTS_DIR / args.run / "crash_diagnosis.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
