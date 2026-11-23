"""How hard are the goals the task itself poses? The envelope behind environment.CHAIN_TIME_LIMIT.

For 2,000 resets of the full task (seeds 90,000-91,999, used for nothing else) it computes
`goal_time_needed` for the goal drawn at reset, and prints its percentiles. A chained goal (Step 9) is used
only if its time needed is at most the 95th percentile: no harder, by this measure, than the goals of the task.

    python -B baseline_dynamic_reach/chain_envelope.py
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import RESULTS_DIR, arm_config
from baseline_dynamic_reach.environment import ReachEnv, goal_time_needed

FIRST, COUNT = 90_000, 2_000


def main():
    env = ReachEnv(**deepcopy(arm_config("O3_c1w0t4_n256")["environment"]))
    needed, by_slot = [], {}
    try:
        for seed in range(FIRST, FIRST + COUNT):
            env.reset(seed=seed)
            value = goal_time_needed(env, env.anchor, env.slot)
            needed.append(value)
            by_slot.setdefault(env.slot, []).append(value)
    finally:
        env.close()
    needed = np.array(needed)
    result = {"seeds": [FIRST, FIRST + COUNT - 1],
              "percentiles": {str(p): round(float(np.percentile(needed, p)), 2) for p in (5, 25, 50, 75, 90, 95, 99, 100)},
              "by_slot_median": {slot: round(float(np.median(v)), 2) for slot, v in sorted(by_slot.items())},
              "by_slot_95": {slot: round(float(np.percentile(v, 95)), 2) for slot, v in sorted(by_slot.items())}}
    out = RESULTS_DIR / "chains"
    out.mkdir(parents=True, exist_ok=True)
    (out / "envelope.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
