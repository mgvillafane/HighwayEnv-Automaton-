"""Explicit target lanes retain 57 inputs and survive training/checkpoint replay."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
from baseline_dynamic_waypoint.config import SINGLE_CAR_GOAL_CONFIG, REWARD_PROFILES
from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
from baseline_dynamic_waypoint.evaluate import load_experiment


class ExplicitTargetLaneTests(unittest.TestCase):
    def make_env(self, schema="legacy_target_lanes"):
        env = CommittedLaneEnv(
            goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": schema},
            reward_weights=deepcopy(REWARD_PROFILES["timeout10"]))
        self.addCleanup(env.close)
        return env

    def test_all_slots_and_lane_combinations_expose_actual_target_lanes(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        for slot in ("front", "back", "left", "right"):
            valid_anchor_lanes = {"left": (1, 2), "right": (0, 1)}.get(slot, (0, 1, 2))
            for anchor_lane in valid_anchor_lanes:
                for ego_lane in (0, 1, 2):
                    with self.subTest(slot=slot, anchor_lane=anchor_lane, ego_lane=ego_lane):
                        env.reset(seed=7, options={"slot": slot})
                        for car, lane_id in ((env.anchor, anchor_lane), (env.highway.vehicle, ego_lane)):
                            car.position[1] = 4 * lane_id
                            car.on_state_update()
                        env._update_goal()
                        obs = env._observation(env.highway.observation_type.observe())
                        expected_goal_lane = anchor_lane + {"left": -1, "right": 1}.get(slot, 0)
                        np.testing.assert_array_equal(obs["target_lanes"], [anchor_lane, expected_goal_lane])
                        self.assertEqual(sum(value.size for value in obs.values()), 57)
                        self.assertNotIn("achieved_goal", obs)
                        self.assertTrue(env.observation_space.contains(obs))

    def test_retained_inputs_physics_and_rewards_equal_legacy(self):
        legacy, current = self.make_env("legacy"), self.make_env()
        for seed in (7, 10002):
            old, _ = legacy.reset(seed=seed)
            new, _ = current.reset(seed=seed)
            for action in (0, 3, 1, 2, 4):
                self.assertEqual(set(new), set(old) - {"achieved_goal"} | {"target_lanes"})
                for key in set(old) - {"achieved_goal"}:
                    np.testing.assert_array_equal(new[key], old[key])
                old, reward, terminated, truncated, info = legacy.step(action)
                new, other_reward, other_terminated, other_truncated, other_info = current.step(action)
                self.assertEqual((reward, terminated, truncated, info["internal_steps"]),
                                 (other_reward, other_terminated, other_truncated, other_info["internal_steps"]))
                np.testing.assert_array_equal(new["target_lanes"],
                                              [current.anchor.lane_index[2], current.goal_lane])
                if terminated or truncated:
                    break

    def test_fresh_dropout_training_and_checkpoint_restore_lane_contract(self):
        torch.set_num_threads(1)
        env = self.make_env()
        model = PPO(DropoutMultiInputPolicy, env, n_steps=16, batch_size=16, n_epochs=1,
                    seed=7, device="cpu", policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]},
                                                        "dropout_rate": .1})
        self.assertEqual(model.policy.features_extractor.features_dim, 57)
        model.learn(total_timesteps=32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "explicit_lanes.zip"
            model.save(path)
            path.with_suffix(".json").write_text(json.dumps({"task": "single_car_committed_lane", "environment": {
                "env_config": env.env_config, "goal_config": env.goal_config, "reward_weights": env.reward_weights}}))
            reloaded, replay, _ = load_experiment(path)
            self.addCleanup(replay.close)
            self.assertEqual(reloaded.policy.features_extractor.features_dim, 57)
            for slot in ("front", "back", "left", "right"):
                obs, _ = replay.reset(seed=10000, options={"slot": slot})
                np.testing.assert_array_equal(obs["target_lanes"], [replay.anchor.lane_index[2], replay.goal_lane])
                replay.step(int(reloaded.predict(obs, deterministic=True)[0]))


if __name__ == "__main__":
    unittest.main()
