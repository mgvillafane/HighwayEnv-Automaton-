"""Replay regression test: saved evaluation episodes must come out exactly the same with today's code.

The study's environment code is shared with other experiments in this repository (baseline_dynamic_waypoint,
baseline_DQN's observation encoder). If a change there alters what a saved model does, earlier results can
no longer be reproduced or compared. This test replays the first episodes of saved evaluations and compares
every column of every row:
- evaluations saved on this kind of machine must replay bit for bit;
- evaluations saved on another machine (P6: the Mac) must match exactly in every outcome and count, and in
  real-valued columns to 1e-9 relative. Distances there differ in the last bits (measured: at most 3e-14
  relative) because NumPy's vector norm goes through each platform's own linear-algebra library.
The 8d winner (N3_c05w0t1: no practice, crash 0.5, time cost 0.001) is in REPLAYS since 2026-11-19, and the
recipe adopted in Step 8f (O3_c1w0t4_n256: original order, crash 1, time cost 0.004, 256 wide) since 2026-11-22.

    python -B -m unittest baseline_dynamic_reach.test_replay
"""
import csv
import json
import math
import platform
import unittest

from baseline_dynamic_reach.config import MODELS_DIR, RESULTS_DIR
from baseline_dynamic_reach.evaluate import run_policies

EPISODES = 10
# (run, evaluation file inside results/<run>/, first episode seed of that file)
REPLAYS = [
    ("P6_seed7", "evaluation_episodes.csv", 10_000),  # Trained on the Mac; replays identically on the PC.
    ("B3_seed7", "evaluation_episodes.csv", 10_000),  # Trained on the PC before the NumPy observation encoder.
    ("N3_c05w0t1_seed16", "eval8/s10000_final/evaluation_episodes.csv", 10_000),  # Step 8 winner until 8f (PC).
    ("O3_c1w0t4_n256_seed16", "eval8/s10000_final/evaluation_episodes.csv", 10_000),  # Adopted in 8f (256 wide).
]


class ReplayTests(unittest.TestCase):
    def test_saved_evaluations_replay_exactly(self):
        from stable_baselines3 import PPO
        checked = 0
        for run, filename, first in REPLAYS:
            saved_path = RESULTS_DIR / run / filename
            model_path = MODELS_DIR / f"{run}.zip"
            if not (saved_path.exists() and model_path.exists()):
                continue
            with saved_path.open(newline="", encoding="utf-8") as handle:
                saved = [row for row in csv.DictReader(handle) if row["policy"] == "ppo"][:EPISODES]
            sidecar = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            same_machine = sidecar["platform"]["system"] == platform.system()
            rows = run_policies(sidecar["environment"], PPO.load(model_path, device="cpu"), EPISODES, first,
                                references=False)["ppo"]
            self.assertEqual(len(rows), len(saved))
            for old, new in zip(saved, rows):
                for key, value in old.items():
                    where = f"{run}, episode seed {old['seed']}, column {key}"
                    if same_machine or not isinstance(new[key], float):
                        self.assertEqual(str(new[key]), value, where)
                    else:
                        self.assertTrue(math.isclose(new[key], float(value), rel_tol=1e-9, abs_tol=1e-12), where)
            checked += 1
        self.assertGreater(checked, 0, "No saved evaluation to replay.")


if __name__ == "__main__":
    unittest.main()
