"""Missing-success diagnostics and best-policy selection priorities."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from baseline_dynamic_waypoint.evaluate import summarize
from baseline_dynamic_waypoint.train_single_car import evaluation_scalars, save_best_evaluation


class StabilityDiagnosticTests(unittest.TestCase):
    def test_steps_include_only_successes_and_missing_is_not_zero(self):
        rows = [{"slot": "front", "success": success, "length": length,
                 "anchor_lost": 0, "max_hold_seconds": 0}
                for success, length in ((1, 12), (0, 300), (1, 18))]
        with patch("baseline_dynamic_waypoint.evaluate.summarize_static", return_value={}):
            self.assertEqual(summarize(rows)["average_steps_to_success"], 15)
            stats = summarize(rows[1:2])
        self.assertIsNone(stats["average_steps_to_success"])
        self.assertEqual(stats["successful_episodes"], 0)
        stats.update(success_rate=0, collision_rate=0, timeout_rate=1, average_episode_reward=-25)
        metrics = evaluation_scalars({"ppo": stats, "ppo_by_slot": {"front": stats}})
        self.assertNotIn("evaluation/average_steps_to_success", metrics)
        self.assertEqual(metrics["evaluation/successful_episodes"], 0)

    def test_best_prefers_success_then_safety_and_preserves_checkpoint(self):
        model = Mock(num_timesteps=16)
        model.save.side_effect = lambda path: Path(path).write_bytes(b"checkpoint")
        with TemporaryDirectory() as directory:
            run = Path(directory)
            def evaluate(success, collision, reward):
                save_best_evaluation(model, {"ppo": {"success_rate": success, "collision_rate": collision,
                    "average_episode_reward": reward}}, run, run / "policy.zip", {"initialization": None})
            evaluate(.3, .2, -20)
            evaluate(.2, 0, 100)
            self.assertEqual(model.save.call_count, 1)
            evaluate(.3, .1, -30)
            self.assertEqual(model.save.call_count, 2)
            evaluate(.3, .1, -30)
            self.assertEqual(model.save.call_count, 2)
            saved = json.loads((run / "policy_best.json").read_text())
            self.assertEqual(saved["total_lifetime_timesteps"], 16)
            self.assertTrue((run / "policy_best.zip").exists())
            save_best_evaluation(model, {"ppo": {"success_rate": .4, "collision_rate": 0,
                "average_episode_reward": 10}}, run, run / "policy.zip",
                {"initialization": {"lifetime_timesteps": 200}}, step=0)
            saved = json.loads((run / "policy_best.json").read_text())
            self.assertEqual(saved["actual_timesteps"], 0)
            self.assertEqual(saved["total_lifetime_timesteps"], 200)


if __name__ == "__main__":
    unittest.main()
