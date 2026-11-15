"""Verify that training produces readable checkpoints and live TensorBoard metrics."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from baseline_dynamic_waypoint.config import ENV_CONFIG, GOAL_CONFIG, PPO_CONFIG, REWARD_WEIGHTS
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.train import SlotEpisodeMetrics


class TrainingLoggingTests(unittest.TestCase):
    def test_checkpoint_reloads_task_and_tensorboard_contains_training_metrics(self):
        torch.set_num_threads(1)
        env_config = deepcopy(ENV_CONFIG)
        env_config["duration"] = GOAL_CONFIG["hold_seconds"] + 1.0
        configuration = {"env_config": env_config, "goal_config": deepcopy(GOAL_CONFIG),
                         "reward_weights": deepcopy(REWARD_WEIGHTS)}
        parameters = {**deepcopy(PPO_CONFIG), "n_steps": 16, "batch_size": 16, "n_epochs": 1}
        env = Monitor(DynamicGoalHighwayEnv(**configuration))
        self.addCleanup(env.close)
        with TemporaryDirectory(prefix="dynamic-ppo-logging-") as directory:
            run_dir = Path(directory)
            model = PPO("MultiInputPolicy", env, seed=7, verbose=0, **parameters)
            logger = configure(str(run_dir), ["csv", "tensorboard"])
            model.set_logger(logger)
            metadata = {"environment": configuration, "ppo_config": parameters, "status": "running"}
            callback = SlotEpisodeMetrics(run_dir, run_dir / "smoke.zip", metadata, checkpoint_timesteps=16)
            try:
                model.learn(total_timesteps=32, callback=callback)
            finally:
                logger.close()

            checkpoint = run_dir / "smoke_32.zip"
            saved = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["actual_timesteps"], 32)
            self.assertEqual(saved["status"], "checkpoint")
            self.assertEqual(saved["environment"], configuration)
            self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "running")
            reloaded, replay_env, _ = load_experiment(checkpoint)
            try:
                observation, _ = replay_env.reset(seed=7)
                replay_env.step(int(reloaded.predict(observation, deterministic=True)[0]))
            finally:
                replay_env.close()

            events = EventAccumulator(str(run_dir)).Reload()
            for tag in ("episodes/reward_mean_100", "episodes/collision_mean_100",
                        "episodes/success_mean_100", "reward_components/progress"):
                self.assertIn(tag, events.Tags()["scalars"])
                self.assertEqual(events.Scalars(tag)[-1].step, 32)


if __name__ == "__main__":
    unittest.main()
