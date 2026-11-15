"""Exact task-equivalence checks for faster kinematics array assembly."""
from copy import deepcopy
import unittest

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from baseline_DQN.environment import TrafficHighwayEnv
from baseline_DQN.fast_observation import enable_fast_observation, FastKinematicObservation
from highway_env.vehicle.objects import Obstacle


class FastObservationTests(unittest.TestCase):
    def compare_task(self, observation_config=None, vehicles_count=20, obstacle=False):
        config = {"vehicles_count": vehicles_count}
        if observation_config:
            config["observation"] = {"type": "Kinematics", **observation_config}
        reference = gym.make("highway-fast-v0", config=deepcopy(config))
        optimized = gym.make("highway-fast-v0", config=deepcopy(config))
        self.addCleanup(reference.close)
        self.addCleanup(optimized.close)
        original_config = deepcopy(optimized.unwrapped.config)
        enable_fast_observation(optimized)
        enable_fast_observation(optimized)  # Enabling twice must not stack hooks.
        for seed in (2, 7, 42):
            expected, expected_info = reference.reset(seed=seed)
            actual, actual_info = optimized.reset(seed=seed)
            self.assertIsInstance(optimized.unwrapped.observation_type, FastKinematicObservation)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, expected.dtype)
            if obstacle:
                for env in (reference, optimized):
                    ego = env.unwrapped.vehicle
                    env.unwrapped.road.objects.append(Obstacle(env.unwrapped.road, ego.position + [8, 4]))
            for action in np.random.default_rng(seed).integers(5, size=35):
                expected, reward, terminated, truncated, _ = reference.step(int(action))
                actual, actual_reward, actual_terminated, actual_truncated, _ = optimized.step(int(action))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual((actual_reward, actual_terminated, actual_truncated),
                                 (reward, terminated, truncated))
                np.testing.assert_array_equal(optimized.unwrapped.np_random.bit_generator.state,
                                              reference.unwrapped.np_random.bit_generator.state)
                for actual_car, expected_car in zip(optimized.unwrapped.road.vehicles,
                                                    reference.unwrapped.road.vehicles):
                    np.testing.assert_array_equal(actual_car.position, expected_car.position)
                    self.assertEqual(actual_car.speed, expected_car.speed)
                if terminated or truncated:
                    break
        self.assertEqual(optimized.unwrapped.config, original_config)

    def test_native_quickstart_observations_rewards_and_physics_are_exact(self):
        self.compare_task()

    def test_custom_ranges_absolute_coordinates_and_reordered_features(self):
        self.compare_task({"absolute": True, "clip": False, "see_behind": True,
                           "vehicles_count": 7, "features": ["vy", "presence", "vx", "y", "x"],
                           "features_range": {"x": [-200, 200], "y": [-12, 12],
                                              "vx": [-30, 30], "vy": [-10, 10]}})

    def test_shuffling_padding_and_obstacles_preserve_rng(self):
        self.compare_task({"order": "shuffled", "vehicles_count": 7}, vehicles_count=1, obstacle=True)

    def test_unnormalized_feature_subset_and_missing_vehicles(self):
        self.compare_task({"normalize": False, "features": ["presence", "x", "vx"],
                           "vehicles_count": 7}, vehicles_count=0)

    def test_extended_features_use_native_fallback(self):
        self.compare_task({"features": ["presence", "x", "y", "vx", "vy", "heading"],
                           "observe_intentions": True})

    def test_wrapper_reset_keeps_optimization_and_interaction_metrics(self):
        plain = TrafficHighwayEnv(scenario="quickstart")
        fast = enable_fast_observation(TrafficHighwayEnv(scenario="quickstart"))
        self.addCleanup(plain.close)
        self.addCleanup(fast.close)
        for seed in (2, 3):
            plain_obs, _ = plain.reset(seed=seed)
            fast_obs, _ = fast.reset(seed=seed)
            np.testing.assert_array_equal(fast_obs, plain_obs)
            for action in (1, 3, 4, 0, 2, 1):
                plain_obs, reward, terminated, truncated, info = plain.step(action)
                fast_obs, fast_reward, fast_terminated, fast_truncated, fast_info = fast.step(action)
                np.testing.assert_array_equal(fast_obs, plain_obs)
                self.assertEqual((fast_reward, fast_terminated, fast_truncated), (reward, terminated, truncated))
                self.assertEqual(fast_info["interaction_metrics"], info["interaction_metrics"])
                if terminated or truncated:
                    break


if __name__ == "__main__":
    unittest.main()
