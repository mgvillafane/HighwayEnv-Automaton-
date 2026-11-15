"""Behavioral checks for relaxed holding and saved two-car experiments."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
import torch

from baseline_dynamic_waypoint.config import SINGLE_CAR_ENV_CONFIG, SINGLE_CAR_GOAL_CONFIG
from baseline_dynamic_waypoint.single_car import SingleCarWaypointEnv
from baseline_dynamic_waypoint.evaluate import load_experiment


class FlexibleHoldTests(unittest.TestCase):
    def make_env(self):
        config = deepcopy(SINGLE_CAR_ENV_CONFIG)
        config.update(initial_lane_id=1, traffic_speed_range=[15.0, 15.0],
                      randomize_traffic_speed=False, randomize_traffic_lane=False)
        env = SingleCarWaypointEnv(env_config=config,
                                  goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "legacy"})
        self.addCleanup(env.close)
        env.reset(seed=7, options={"slot": "back"})
        return env

    def place(self, env, along=0.0, lateral=0.0, speed_error=0.0):
        lane = env.highway.road.network.get_lane(env.slot_lane_index)
        slot_s, _ = lane.local_coordinates(env.goal_position)
        ego = env.highway.vehicle
        ego.position = lane.position(slot_s + along, lateral).astype(float)
        ego.heading = 0.0
        ego.speed = ego.target_speed = env.anchor.speed + speed_error
        ego.on_state_update()
        ego.target_lane_index = env.slot_lane_index

    def test_relaxed_tolerances_accept_a_nearby_speed_matched_goal(self):
        env = self.make_env()
        for step in range(5):
            self.place(env, along=4.0, lateral=1.6, speed_error=1.0)
            _, _, terminated, _, info = env.step(1)
            self.assertEqual(info["hold_steps"], step + 1)
            self.assertEqual(terminated, step == 4)
        self.assertTrue(info["is_success"])

    def test_short_exit_pauses_without_awarding_hold_or_success(self):
        env = self.make_env()
        for _ in range(4):
            self.place(env)
            env.step(1)
        for gap in range(1, 4):
            self.place(env, along=6.0)
            _, _, terminated, _, info = env.step(1)
            self.assertEqual(info["hold_steps"], 4)
            self.assertEqual(info["hold_gap_steps"], gap)
            self.assertFalse(terminated or info["is_success"])
            self.assertEqual(info["reward_components"]["hold"], 0)
        self.place(env)
        _, _, terminated, _, info = env.step(1)
        self.assertTrue(terminated and info["is_success"])

    def test_long_exit_clears_previous_progress(self):
        env = self.make_env()
        for _ in range(4):
            self.place(env)
            env.step(1)
        for _ in range(4):
            self.place(env, along=6.0)
            env.step(1)
        self.assertEqual(env.hold_steps, 0)
        self.place(env)
        _, _, terminated, _, info = env.step(1)
        self.assertEqual(info["hold_steps"], 1)
        self.assertFalse(terminated)

    def test_collision_or_anchor_loss_cannot_use_grace_to_succeed(self):
        for lost in (False, True):
            with self.subTest(anchor_lost=lost):
                env = self.make_env()
                for _ in range(4):
                    self.place(env)
                    env.step(1)
                (env.anchor if lost else env.highway.vehicle).crashed = True
                _, _, terminated, truncated, info = env.step(1)
                self.assertEqual(info["hold_steps"], 0)
                self.assertFalse(info["is_success"])
                self.assertTrue(truncated if lost else terminated)

    def test_episode_runs_for_thirty_seconds_without_success(self):
        env = self.make_env()
        self.assertEqual(env.env_config["duration"], 30.0)
        self.assertEqual(env.highway.config["duration"], 30.0)
        ego = env.highway.vehicle
        ego.position[1] = 0.0
        ego.speed = ego.target_speed = 10.0
        ego.on_state_update()
        ego.target_lane_index = ego.lane_index
        for _ in range(310):
            _, _, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        self.assertTrue(truncated and info["is_timeout"])
        self.assertAlmostEqual(info["elapsed_time"], 30.0)

    def test_checkpoint_dispatch_restores_flexible_profile(self):
        torch.set_num_threads(1)
        env = self.make_env()
        model = PPO("MultiInputPolicy", env, n_steps=8, batch_size=8, device="cpu", seed=7)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "untrained.zip"
            model.save(path)
            metadata = {"task": "single_car_waypoint", "environment": {
                "env_config": env.env_config, "goal_config": env.goal_config,
                "reward_weights": env.reward_weights}}
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            restored, replay, _ = load_experiment(path)
            self.addCleanup(replay.close)
            self.assertIsInstance(replay, SingleCarWaypointEnv)
            obs, _ = replay.reset(seed=7)
            self.assertEqual(len(replay.highway.road.vehicles), 2)
            self.assertEqual(replay.hold_grace_steps, 3)
            self.assertEqual(replay.hold_required, 5)
            self.assertEqual(restored.policy.features_extractor.features_dim, 53)
            replay.step(int(restored.predict(obs, deterministic=True)[0]))


class ScenarioSubsetTests(unittest.TestCase):
    def test_disabled_goals_cannot_be_sampled_or_explicitly_requested(self):
        from baseline_dynamic_waypoint.train_single_car import evaluation_environment_for
        goal = {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "enabled_slots": ["right", "front"]}
        env = SingleCarWaypointEnv(goal_config=goal)
        self.addCleanup(env.close)
        observed = {env.reset(seed=seed)[1]["slot"] for seed in range(60)}
        self.assertEqual(observed, {"front", "right"})
        for slot in ("back", "left"):
            with self.subTest(slot=slot), self.assertRaisesRegex(ValueError, "disabled"):
                env.reset(seed=7, options={"slot": slot})
        configuration = {"env_config": env.env_config, "goal_config": env.goal_config,
                         "reward_weights": env.reward_weights}
        evaluation = evaluation_environment_for(configuration)
        self.assertEqual(evaluation, configuration)
        replay = SingleCarWaypointEnv(**evaluation)
        self.addCleanup(replay.close)
        self.assertEqual({replay.reset(seed=seed)[1]["slot"] for seed in range(30)}, {"front", "right"})

    def test_training_only_subset_still_allows_four_goal_evaluation(self):
        from baseline_dynamic_waypoint.train_single_car import evaluation_environment_for
        configuration = {"goal_config": {"sampled_slots": ["right", "front"]}}
        self.assertNotIn("sampled_slots", evaluation_environment_for(configuration)["goal_config"])
        self.assertIn("sampled_slots", configuration["goal_config"])

    def test_disabled_goals_cannot_appear_in_sampling_or_fixed_defaults(self):
        for extra in ({"sampled_slots": ["left"]}, {"default_slot": "back"}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "disabled"):
                SingleCarWaypointEnv(goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG),
                    "enabled_slots": ["right", "front"], **extra})

    def test_subset_samples_both_goals_with_valid_lanes_and_repeatable_seeds(self):
        goal = {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "sampled_slots": ["right", "front"]}
        env = SingleCarWaypointEnv(goal_config=goal)
        self.addCleanup(env.close)
        counts = {"front": 0, "right": 0}
        for seed in range(100):
            _, info = env.reset(seed=seed)
            scenario = info["initial_scenario"]
            counts[info["slot"]] += 1
            self.assertIn(info["goal_lane"], range(3))
            if info["slot"] == "right":
                self.assertEqual(info["goal_lane"], scenario["traffic_lane"] + 1)
            _, repeated = env.reset(seed=seed)
            self.assertEqual(scenario, repeated["initial_scenario"])
        self.assertTrue(all(30 < count < 70 for count in counts.values()), counts)
        # Explicit held-out resets can still inspect goals excluded from training.
        _, info = env.reset(seed=7, options={"slot": "left"})
        self.assertEqual(info["slot"], "left")

    def test_invalid_subsets_are_rejected(self):
        for slots in ([], ["right", "right"], ["up"], "right", None):
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "sampled_slots"):
                SingleCarWaypointEnv(goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG), "sampled_slots": slots})


if __name__ == "__main__":
    unittest.main()
