"""Terminal timeout costs, actor/critic dropout, training and checkpoint replay."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torch import nn

from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
from baseline_dynamic_waypoint.config import COMMITTED_LANE_ENV_CONFIG, SINGLE_CAR_GOAL_CONFIG, REWARD_PROFILES, RESULTS_DIR
from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.train_single_car import SampledScenarioMetrics


class TimeoutPenaltyTests(unittest.TestCase):
    def make_env(self, profile="timeout25"):
        env = CommittedLaneEnv(env_config={**deepcopy(COMMITTED_LANE_ENV_CONFIG), "duration": .6},
                               reward_weights=REWARD_PROFILES[profile])
        self.addCleanup(env.close)
        env.reset(seed=10002, options={"slot": "front"})
        return env

    def test_timeout_is_charged_once_across_committed_physics_ticks(self):
        penalized, baseline = self.make_env(), self.make_env("baseline")
        ticks = []
        penalized.physics_step_callback = lambda obs, reward, t, tr, info: ticks.append(info["reward_components"]["timeout"])
        _, reward, terminated, truncated, info = penalized.step(0)
        _, original_reward, _, _, old_info = baseline.step(0)
        self.assertTrue(truncated and not terminated and info["is_timeout"])
        self.assertGreater(len(ticks), 1)
        self.assertEqual(sum(value != 0 for value in ticks), 1)
        self.assertEqual(sum(ticks), -25)
        self.assertEqual(info["episode_reward_components"]["timeout"], -25)
        self.assertAlmostEqual(reward, original_reward - 25)
        self.assertAlmostEqual(reward, sum(info["episode_reward_components"].values()))
        self.assertNotIn("timeout", old_info["reward_components"])
        np.testing.assert_array_equal(penalized.highway.vehicle.position, baseline.highway.vehicle.position)

    def test_collision_and_anchor_loss_at_deadline_are_not_timeouts(self):
        for lost in (False, True):
            with self.subTest(anchor_lost=lost):
                env = self.make_env()
                env.highway.time = .5
                (env.anchor if lost else env.highway.vehicle).crashed = True
                _, _, _, _, info = env.step(1)
                self.assertFalse(info["is_timeout"])
                self.assertEqual(info["reward_components"]["timeout"], 0)

    def test_timeout10_profile_preserves_other_rewards_and_charges_once(self):
        self.assertEqual({key: value for key, value in REWARD_PROFILES["timeout10"].items() if key != "timeout"},
                         REWARD_PROFILES["baseline"])
        env = self.make_env("timeout10")
        ticks = []
        env.physics_step_callback = lambda obs, reward, t, tr, info: ticks.append(info["reward_components"]["timeout"])
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated and not terminated and info["is_timeout"])
        self.assertEqual(sum(value != 0 for value in ticks), 1)
        self.assertEqual(sum(ticks), -10)
        self.assertEqual(info["episode_reward_components"]["timeout"], -10)
        self.assertAlmostEqual(reward, sum(info["episode_reward_components"].values()))

    def test_success_at_deadline_does_not_receive_timeout_cost(self):
        env = self.make_env()
        ego = env.highway.vehicle
        ego.position = env.goal_position.copy()
        ego.heading = 0.0
        ego.speed = ego.target_speed = env.anchor.speed
        ego.on_state_update()
        ego.target_lane_index = env.slot_lane_index
        env.hold_steps = env.hold_required - 1
        env.highway.time = .5
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated and info["is_success"])
        self.assertFalse(truncated or info["is_timeout"])
        self.assertEqual(info["reward_components"]["timeout"], 0)


class DropoutPolicyTests(unittest.TestCase):
    def make_env(self):
        goal = {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "lane_goal_v2",
                "enabled_slots": ["front", "left", "back", "right"]}
        env = CommittedLaneEnv(env_config={**deepcopy(COMMITTED_LANE_ENV_CONFIG), "duration": .6},
                               goal_config=goal, reward_weights=REWARD_PROFILES["timeout25"])
        self.addCleanup(env.close)
        return env

    def test_actor_and_critic_have_train_only_dropout(self):
        torch.set_num_threads(1)
        model = PPO(DropoutMultiInputPolicy, self.make_env(), seed=7, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}, "dropout_rate": .1})
        features = torch.ones((32, 54))
        for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
            layers = [layer for layer in network.modules() if isinstance(layer, nn.Dropout)]
            self.assertEqual(len(layers), 2)
            self.assertTrue(all(layer.p == .1 for layer in layers))
            model.policy.set_training_mode(True)
            self.assertFalse(torch.equal(network(features), network(features)))
            model.policy.set_training_mode(False)
            self.assertTrue(torch.equal(network(features), network(features)))
        # Existing hidden Linear parameter names remain compatible with checkpoints.
        self.assertIn("mlp_extractor.policy_net.2.weight", model.policy.state_dict())

    def test_training_periodic_evaluation_reward_logs_and_save_reload(self):
        torch.set_num_threads(1)
        env = self.make_env()
        parameters = {"n_steps": 8, "batch_size": 8, "n_epochs": 1, "device": "cpu",
                      "policy_kwargs": {"net_arch": {"pi": [16, 16], "vf": [16, 16]}, "dropout_rate": .1}}
        model = PPO(DropoutMultiInputPolicy, Monitor(env), seed=7, **parameters)
        observed_modes = []
        for layer in model.policy.modules():
            if isinstance(layer, nn.Dropout):
                layer.register_forward_pre_hook(lambda layer, inputs: observed_modes.append(layer.training))
        configuration = {"env_config": env.env_config, "goal_config": env.goal_config, "reward_weights": env.reward_weights}
        metadata = {"task": "single_car_committed_lane", "requested_timesteps": 32,
                    "training_slots": ["front", "left", "back", "right"], "environment": configuration,
                    "evaluation_environment": configuration, "ppo_config": parameters, "dropout_rate": .1}
        with TemporaryDirectory(prefix="dropout-test-", dir=RESULTS_DIR) as directory:
            run = Path(directory)
            self.assertTrue(run.resolve().is_relative_to(RESULTS_DIR.resolve()))
            logger = configure(str(run), ["csv", "tensorboard"])
            model.set_logger(logger)
            callback = SampledScenarioMetrics(run, run / "model.zip", metadata,
                                             checkpoint_timesteps=16, eval_every=16, eval_episodes=4)
            try:
                model.learn(total_timesteps=32, callback=callback)
            finally:
                logger.close()
            self.assertIn(True, observed_modes)
            self.assertIn(False, observed_modes)
            self.assertTrue((run / "evaluation_16.json").exists())
            self.assertTrue((run / "model_best.zip").exists())
            best, best_env, best_meta = load_experiment(run / "model_best.zip")
            self.addCleanup(best_env.close)
            self.assertEqual(best_meta["actual_timesteps"], 16)
            self.assertEqual(best.policy.features_extractor.features_dim, model.policy.features_extractor.features_dim)
            events = EventAccumulator(str(run)).Reload()
            self.assertEqual(events.Scalars("evaluation/success_rate")[0].step, 16)
            self.assertIn("reward_components/timeout", events.Tags()["scalars"])
            for tag in ("episodes/timeout_mean_100", "episodes/success_count_100",
                        "evaluation/average_episode_reward", "evaluation/successful_episodes",
                        "train/approx_kl", "train/clip_fraction", "train/explained_variance"):
                self.assertIn(tag, events.Tags()["scalars"])
            with (run / "training_episodes.csv").open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertTrue(records)
            self.assertTrue(any(row["reward_timeout"] == "-25.0" for row in records))
            for row in records:
                self.assertEqual(float(row["reward_timeout"]), -25 * int(row["timeout"]))
            model.save(run / "final.zip")
            (run / "final.json").write_text(json.dumps(metadata))
            reloaded, replay, _ = load_experiment(run / "final.zip")
            self.addCleanup(replay.close)
            self.assertIsInstance(reloaded.policy, DropoutMultiInputPolicy)
            self.assertEqual(reloaded.policy.dropout_rate, .1)
            obs, _ = replay.reset(seed=7)
            model.policy.set_training_mode(False)
            reloaded.policy.set_training_mode(False)
            tensors, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                np.testing.assert_array_equal(model.policy.predict_values(tensors).numpy(),
                                              reloaded.policy.predict_values(tensors).numpy())
                np.testing.assert_array_equal(model.policy.get_distribution(tensors).distribution.probs.numpy(),
                                              reloaded.policy.get_distribution(tensors).distribution.probs.numpy())


if __name__ == "__main__":
    unittest.main()
