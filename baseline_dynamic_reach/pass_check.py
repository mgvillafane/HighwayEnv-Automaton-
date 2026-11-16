"""Replay models on an arm's task (deterministic actions) and check that each success is a real pass.

    python -B baseline_dynamic_reach/pass_check.py P2 baseline_dynamic_reach/results/step7g_P6/final_on_P2.csv \
        baseline_dynamic_reach/models/P6_seed7.zip baseline_dynamic_reach/models/P6_seed8.zip ...

One row per model and evaluation episode (the fixed 200): slot, start type (mixed-start arms), outcome,
whether ego passed, and the crash partner. A pass: ego more than 5 m ahead of every car that blocked the
slot at reset (`ReachEnv.blocking_cars()`), at any point before the episode ends. Crash partner "blocker"
means one of those cars. Used for Steps 7f and 7g (results/step7f_P4, step7f_P5, step7g_P6).
"""
import csv
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def replay(job):
    import torch
    from stable_baselines3 import PPO
    from baseline_dynamic_reach.config import arm_config, EVAL_EPISODES, EVAL_SEED
    from baseline_dynamic_reach.environment import ReachEnv
    torch.set_num_threads(1)
    env_arm, path = job
    model = PPO.load(path, device="cpu")
    env = ReachEnv(**deepcopy(arm_config(env_arm)["environment"]))
    rows = []
    for seed in range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES):
        obs, info = env.reset(seed=seed)
        ego, vehicles = env.highway.vehicle, env.highway.road.vehicles
        blockers = [vehicles[i] for _, i in env.blocking_cars()]
        start = info.get("start", "natural")
        passed_at = None
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(int(model.predict(obs, deterministic=True)[0]))
            done = terminated or truncated
            if passed_at is None and all(ego.position[0] > b.position[0] + 5.0 for b in blockers):
                passed_at = float(env.highway.time)
        partner = ""
        if info["is_collision"]:
            hit = [v for v in vehicles if v is not ego and v.crashed]
            partner = "blocker" if any(b in hit for b in blockers) else "anchor" if env.anchor in hit else "other"
        outcome = "reached" if info["is_success"] else "crash" if info["is_collision"] else "timeout"
        rows.append({"model": Path(path).stem, "seed": seed, "slot": info["slot"], "start": start, "outcome": outcome,
                     "passed": passed_at is not None, "passed_at": passed_at, "partner": partner,
                     "time": float(info["elapsed_time"])})
    env.close()
    return rows


if __name__ == "__main__":
    env_arm, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    with Pool(min(10, len(paths))) as pool:
        rows = [r for part in pool.map(replay, [(env_arm, p) for p in paths]) for r in part]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out, len(rows))
