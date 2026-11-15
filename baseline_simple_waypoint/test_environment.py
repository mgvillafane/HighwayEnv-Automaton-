"""Check two-car dynamics, goal behavior, isolation and future PPO compatibility."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

from baseline_dynamic_waypoint import config as previous
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_simple_waypoint.config import ENV_CONFIG, RANDOMIZED_ENV_CONFIG, GOAL_CONFIG
from baseline_simple_waypoint.environment import SimpleFollowingEnv
from baseline_simple_waypoint.demo import load_experiment
from baseline_simple_waypoint.scripted import scripted_action


class SimpleFollowingTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = SimpleFollowingEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_exactly_two_cars_and_correct_observation(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        obs, info = env.reset(seed=7, options={"slot": "back"})
        self.assertEqual(len(env.highway.road.vehicles), 2)
        self.assertEqual(env.anchor.lane_index[2], env.highway.vehicle.lane_index[2])
        self.assertEqual(info["slot"], "back")
        self.assertFalse(env.anchor.enable_lane_change)
        self.assertAlmostEqual(env.highway.vehicle.speed, 20.0)
        self.assertAlmostEqual(env.anchor.speed, 15.0)
        np.testing.assert_allclose(obs["vehicles"][1], [1, 35/200, 0, -5/30, 0], atol=1e-6)
        np.testing.assert_array_equal(obs["vehicles"][2:], 0)
        np.testing.assert_allclose(obs["goal_delta"], [20/200, 0], atol=1e-6)

    def test_scripted_controller_reaches_all_four_slots_with_speed_match(self):
        env = self.make_env()
        for slot in previous.SLOTS:
            with self.subTest(slot=slot):
                env.reset(seed=7, options={"slot": slot})
                visited_lanes = {env.highway.vehicle.lane_index[2]}
                for _ in range(160):
                    _, _, terminated, truncated, info = env.step(scripted_action(env))
                    visited_lanes.add(env.highway.vehicle.lane_index[2])
                    self.assertEqual(len(env.highway.road.vehicles), 2)
                    self.assertAlmostEqual(env.anchor.speed, 15.0)
                    if terminated or truncated:
                        break
                self.assertTrue(info["is_success"], info["termination_reason"])
                self.assertFalse(info["is_collision"])
                self.assertEqual(env.hold_steps, 10)
                self.assertLessEqual(abs(env.highway.vehicle.speed - env.anchor.speed), 0.5)
                self.assertEqual(env.highway.vehicle.lane_index[2], env.goal_lane)
                if slot == "front":
                    self.assertGreater(len(visited_lanes), 1)  # It passed beside the car.
                    self.assertGreater(env.highway.vehicle.position[0], env.anchor.position[0])

    def test_four_slot_geometry_observations_and_moving_goals(self):
        env = self.make_env()
        offsets = {"front": [20, 0], "back": [-15, 0], "left": [0, -4], "right": [0, 4]}
        for slot, offset in offsets.items():
            with self.subTest(slot=slot):
                obs, info = env.reset(seed=7, options={"slot": slot})
                np.testing.assert_allclose(env.goal_position - env.anchor.position, offset, atol=1e-6)
                self.assertEqual(previous.SLOTS[int(obs["slot"].argmax())], slot)
                np.testing.assert_allclose(obs["goal_delta"] * env.position_scale,
                                           env.goal_position - env.highway.vehicle.position, atol=1e-5)
                goal_before = env.goal_position.copy()
                env.step(1)
                np.testing.assert_allclose(env.goal_position - goal_before, [1.5, 0], atol=1e-6)

    def test_seeded_sampling_visits_all_four_slots_and_fixed_back_still_works(self):
        env = self.make_env()
        first, info = env.reset(seed=7)
        second, again = env.reset(seed=7)
        self.assertEqual(info["slot"], again["slot"])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        slots = {env.reset(seed=seed)[1]["slot"] for seed in range(40)}
        self.assertEqual(slots, set(previous.SLOTS))
        goal = deepcopy(GOAL_CONFIG)
        goal["default_slot"] = "back"
        fixed = self.make_env(goal_config=goal)
        for seed in range(4):
            self.assertEqual(fixed.reset(seed=seed)[1]["slot"], "back")
        self.assertEqual(fixed.reset(seed=7, options={"slot": "front"})[1]["slot"], "front")

    def test_ignoring_slow_car_causes_collision(self):
        env = self.make_env()
        env.reset(seed=7, options={"slot": "back"})
        for _ in range(150):
            _, _, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertTrue(info["is_collision"])
        self.assertFalse(info["is_success"])

    def test_deadline_without_arrival_is_a_timeout(self):
        settings = deepcopy(ENV_CONFIG)
        settings["duration"] = 2.0
        env = self.make_env(env_config=settings)
        env.reset(seed=7)
        for _ in range(25):
            _, _, terminated, truncated, info = env.step(4)
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertTrue(info["is_timeout"])

    def test_previous_environment_is_still_available_and_unchanged(self):
        settings = deepcopy(previous.ENV_CONFIG)
        env = self.make_env()
        env.reset(seed=7)
        env.env_config["observation"]["vehicles_count"] = 2
        self.assertEqual(previous.ENV_CONFIG, settings)
        original = DynamicGoalHighwayEnv()
        self.addCleanup(original.close)
        original.reset(seed=7)
        self.assertEqual(len(original.highway.road.vehicles), 9)

    def test_randomized_velocity_lanes_and_goal_inputs_are_seeded_and_valid(self):
        env = self.make_env(env_config=RANDOMIZED_ENV_CONFIG)
        first, initial = env.reset(seed=7)
        second, repeated = env.reset(seed=7)
        self.assertEqual(initial["initial_scenario"], repeated["initial_scenario"])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        lanes_ego, lanes_traffic, slots, speeds = set(), set(), set(), set()
        for seed in range(200):
            obs, info = env.reset(seed=seed)
            sample = info["initial_scenario"]
            lanes_ego.add(sample["ego_lane"])
            lanes_traffic.add(sample["traffic_lane"])
            slots.add(sample["slot"])
            speeds.add(round(sample["traffic_speed"], 4))
            self.assertEqual(len(env.highway.road.vehicles), 2)
            self.assertGreaterEqual(sample["traffic_speed"], 12.0)
            self.assertLessEqual(sample["traffic_speed"], 18.0)
            self.assertLess(sample["traffic_speed"], max(env.env_config["action"]["target_speeds"]))
            if sample["slot"] == "left":
                self.assertGreater(sample["traffic_lane"], 0)
            if sample["slot"] == "right":
                self.assertLess(sample["traffic_lane"], 2)
            actual = np.r_[env.anchor.position - env.highway.vehicle.position,
                           env.anchor.velocity - env.highway.vehicle.velocity]
            np.testing.assert_allclose(obs["vehicles"][1, 1:], actual / [200, 12, 30, 10], atol=1e-6)
            np.testing.assert_allclose(obs["goal_delta"] * env.position_scale,
                                       env.goal_position - env.highway.vehicle.position, atol=1e-5)
            _, _, _, _, stepped = env.step(1)
            self.assertEqual(stepped["initial_scenario"], sample)
        self.assertEqual(lanes_ego, {0, 1, 2})
        self.assertEqual(lanes_traffic, {0, 1, 2})
        self.assertEqual(slots, set(previous.SLOTS))
        self.assertGreater(len(speeds), 100)
        self.assertEqual(ENV_CONFIG["traffic_speed_range"], [15.0, 15.0])
        self.assertEqual(ENV_CONFIG["initial_lane_id"], 1)

    def test_traffic_maximum_cannot_reach_ego_maximum(self):
        invalid = deepcopy(RANDOMIZED_ENV_CONFIG)
        invalid["traffic_speed_range"] = [12.0, 25.0]
        with self.assertRaises(ValueError):
            self.make_env(env_config=invalid)

    def test_ppo_can_consume_the_observation_without_training(self):
        torch.set_num_threads(1)
        env = self.make_env()
        model = PPO("MultiInputPolicy", env, n_steps=16, batch_size=16, seed=7, device="cpu")
        obs, initial = env.reset(seed=7)
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, _, _, _ = env.step(int(action))
        self.assertTrue(env.observation_space.contains(next_obs))
        self.assertEqual(model.policy.features_extractor.features_dim, 53)
        self.assertEqual(model.num_timesteps, 0)
        # Check future checkpoint playback without running any learning steps.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "untrained.zip"
            model.save(path)
            metadata = {"task": "single_traffic_following", "environment": {
                "env_config": env.env_config, "goal_config": env.goal_config,
                "reward_weights": env.reward_weights}}
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            restored, replay, _ = load_experiment(path)
            self.addCleanup(replay.close)
            replay_obs, replay_info = replay.reset(seed=7)
            self.assertEqual(len(replay.highway.road.vehicles), 2)
            self.assertEqual(replay_info["slot"], initial["slot"])
            self.assertEqual(replay.goal_config["default_slot"], "random")
            self.assertEqual(restored.num_timesteps, 0)
            replay.step(int(restored.predict(replay_obs, deterministic=True)[0]))


if __name__ == "__main__":
    unittest.main()
