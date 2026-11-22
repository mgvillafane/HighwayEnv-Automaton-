"""Verify navigation direction, staged passing, reward accounting and PPO replay."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from baseline_dynamic_waypoint.config import (
    REWARD_GOAL_OVERRIDES, REWARD_PROFILES, SINGLE_CAR_ENV_CONFIG, SINGLE_CAR_GOAL_CONFIG,
)
from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.maneuver_guidance import ManeuverGuidanceEnv, NAVIGATION_FEATURES
from baseline_dynamic_waypoint.single_car import make_single_car_env
from baseline_dynamic_waypoint.train_single_car import SampledScenarioMetrics


class ManeuverGuidanceTests(unittest.TestCase):
    def make_env(self, guided=True, duration=30.0, profile=None):
        config = deepcopy(SINGLE_CAR_ENV_CONFIG)
        config.update(initial_lane_id=1, traffic_speed_range=[15.0, 15.0],
                      randomize_traffic_speed=False, randomize_traffic_lane=False,
                      duration=duration)
        goal = deepcopy(SINGLE_CAR_GOAL_CONFIG)
        profile = profile or ("maneuver10" if guided else "timeout10")
        goal.update(deepcopy(REWARD_GOAL_OVERRIDES[profile]))
        env = make_single_car_env(env_config=config, goal_config=goal,
                                  reward_weights=deepcopy(REWARD_PROFILES[profile]))
        self.addCleanup(env.close)
        return env

    def place(self, env, *, lead=-35.0, lane=1, speed=20.0):
        ego = env.highway.vehicle
        ego.position = np.array([env.anchor.position[0] + lead, 4.0 * lane])
        ego.heading = 0.0
        ego.speed = ego.target_speed = speed
        ego.speed_index = ego.speed_to_index(speed)
        ego.on_state_update()
        ego.target_lane_index = ego.lane_index
        env._update_goal()

    def test_added_inputs_leave_physics_sampling_and_success_unchanged(self):
        guided, baseline = self.make_env(), self.make_env(guided=False)
        for slot in ("front", "right", "back", "left"):
            a, _ = guided.reset(seed=7, options={"slot": slot})
            b, _ = baseline.reset(seed=7, options={"slot": slot})
            self.assertTrue(guided.observation_space.contains(a))
            self.assertEqual(a["navigation"].shape, (16,))
            self.assertEqual(sum(value.size for value in a.values()), 69)
            for key in b:
                np.testing.assert_array_equal(a[key], b[key])
            for action in [2] + [1] * 15:
                a, reward, terminated, truncated, info = guided.step(action)
                b, base_reward, bt, btr, base_info = baseline.step(action)
                for key in b:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual((terminated, truncated), (bt, btr))
                self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
                extra = sum(info["reward_components"][key] for key in ("route_progress", "tracking"))
                self.assertAlmostEqual(reward - extra, base_reward)
                self.assertAlmostEqual(sum(info["episode_reward_components"].values()),
                                       sum(guided.episode_totals.values()))
                if terminated or truncated:
                    break

    def test_front_exposes_passing_lane_then_safe_merge_and_speed_matching(self):
        env = self.make_env()
        env.reset(seed=7, options={"slot": "front"})
        for lead, fraction, lateral in [(-35, 0, -4), (10, 0, -4), (15, .5, -2), (20, 1, 0)]:
            self.place(env, lead=lead, lane=1)
            state = env.navigation_state()
            self.assertAlmostEqual(state["merge_fraction"], fraction)
            self.assertAlmostEqual(state["lateral"], lateral)
        self.assertAlmostEqual(env.navigation_state()["preferred_speed"], env.anchor.speed)
        self.place(env, lead=-35, lane=1, speed=15)
        self.assertGreater(env.navigation_state()["preferred_speed"], env.anchor.speed)

    def test_mirrored_left_right_have_opposite_directions_and_equal_guidance(self):
        left, right = self.make_env(), self.make_env()
        left.reset(seed=7, options={"slot": "left"})
        right.reset(seed=7, options={"slot": "right"})
        self.place(left, lead=-35, lane=1)
        self.place(right, lead=-35, lane=1)
        for env, expected in [(left, -1), (right, 1)]:
            obs = env._observation(env.highway.observation_type.observe())
            values = dict(zip(NAVIGATION_FEATURES, obs["navigation"]))
            self.assertEqual(np.sign(values["goal_lane_minus_ego_lane"]), expected)
            self.assertEqual(np.sign(values["navigation_lateral_error"]), expected)
        for index in range(15):
            la, ra = ((0, 2) if index == 0 else (1, 1))
            _, lr, _, _, li = left.step(la)
            _, rr, _, _, ri = right.step(ra)
            self.assertAlmostEqual(lr, rr, places=8)
            self.assertAlmostEqual(li["reward_components"]["route_progress"],
                                   ri["reward_components"]["route_progress"], places=8)

    def test_collision_risk_detects_closing_in_anchor_lane(self):
        env = self.make_env()
        env.reset(seed=7, options={"slot": "front"})
        self.place(env, lead=-10, lane=1, speed=20)
        self.assertGreater(env.navigation_state()["collision_risk"], .5)
        self.place(env, lead=-10, lane=0, speed=20)
        self.assertEqual(env.navigation_state()["collision_risk"], 0)
        self.place(env, lead=-10, lane=1, speed=15)
        self.assertEqual(env.navigation_state()["collision_risk"], 0)

    def test_tracking_is_bounded_timeout_is_ten_and_returns_reconcile(self):
        env = self.make_env(duration=1.0)
        env.reset(seed=7, options={"slot": "front"})
        total, timeouts = 0.0, []
        for _ in range(20):
            _, reward, terminated, truncated, info = env.step(1)
            total += reward
            tracking = info["reward_components"]["tracking"]
            self.assertGreaterEqual(tracking, -.05)
            self.assertLessEqual(tracking, 0)
            timeouts.append(info["reward_components"]["timeout"])
            if terminated or truncated:
                break
        self.assertTrue(info["is_timeout"])
        self.assertEqual(sum(timeouts), -10)
        self.assertEqual(timeouts.count(-10), 1)
        self.assertAlmostEqual(total, sum(info["episode_reward_components"].values()))

    def test_original_front_success_box_and_hold_remain_required(self):
        env = self.make_env()
        env.reset(seed=7, options={"slot": "front"})
        self.place(env, lead=20, lane=1, speed=15)
        for tick in range(5):
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(truncated)
            self.assertEqual(terminated, tick == 4)
            self.assertAlmostEqual(info["reward_components"]["route_progress"], 0, places=8)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["reward_components"]["goal"], 50)

    def test_lateral_only_route_ignores_longitudinal_and_speed_errors(self):
        env = self.make_env(profile="maneuver_lateral_side10")
        for slot in ("front", "back", "left", "right"):
            env.reset(seed=7, options={"slot": slot})
            expected_lateral = 4.0 * (env.goal_lane - 1)
            for lead, speed in ((-35, 10), (0, 20), (15, 25), (30, 15)):
                self.place(env, lead=lead, lane=1, speed=speed)
                state = env.navigation_state()
                self.assertAlmostEqual(state["lateral"], expected_lateral)
                self.assertAlmostEqual(state["route_cost"], 2 * abs(expected_lateral))
            self.place(env, lead=-35, lane=env.goal_lane, speed=20)
            self.assertAlmostEqual(env.navigation_state()["route_cost"], 0)

    def test_lateral_only_front_milestone_and_reward_accounting(self):
        env = self.make_env(profile="maneuver_lateral_side10")
        env.reset(seed=7, options={"slot": "front"})
        self.place(env, lead=0, lane=0, speed=20)
        previous = env.navigation_state()["route_cost"]
        _, reward, _, _, info = env.step(1)
        current = info["navigation_guidance"]
        self.assertAlmostEqual(current["route_cost"], 2 * abs(current["lateral"]))
        self.assertAlmostEqual(info["reward_components"]["route_progress"],
                               .5 * (previous - current["route_cost"]))
        self.assertEqual(info["reward_components"]["tracking"], 0)
        self.assertEqual(info["reward_components"]["front_side"], 10)
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))

    def test_side_profile_disables_only_tracking_and_adds_front_bonus(self):
        old, new = self.make_env(), self.make_env(profile="maneuver_side10")
        self.assertEqual(new.reward_weights, {**old.reward_weights, "tracking": 0.0, "front_side": 10.0})
        a, _ = old.reset(seed=7, options={"slot": "right"})
        b, info = new.reset(seed=7, options={"slot": "right"})
        self.assertFalse(info["front_side_reached"])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for action in [2] + [1] * 10:
            a, old_reward, _, _, old_info = old.step(action)
            b, new_reward, _, _, new_info = new.step(action)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(new_info["reward_components"]["tracking"], 0.0)
            self.assertEqual(new_info["reward_components"]["front_side"], 0.0)
            self.assertAlmostEqual(new_reward, old_reward - old_info["reward_components"]["tracking"])

    def test_front_both_sides_pay_once_without_speed_match_or_hold(self):
        env = self.make_env(profile="maneuver_side10")
        for lane in (0, 2):
            env.reset(seed=7, options={"slot": "front"})
            self.place(env, lead=0, lane=lane, speed=20)
            _, reward, terminated, truncated, info = env.step(1)
            self.assertEqual(info["reward_components"]["front_side"], 10)
            self.assertTrue(info["front_side_new"] and info["front_side_reached"])
            self.assertFalse(terminated or truncated or info["is_success"])
            self.assertEqual(info["hold_steps"], 0)
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            # Leaving and revisiting the other side cannot farm another bonus.
            self.place(env, lead=-20, lane=lane)
            env.step(1)
            self.place(env, lead=0, lane=2-lane)
            _, _, _, _, info = env.step(1)
            self.assertFalse(info["front_side_new"])
            self.assertEqual(info["reward_components"]["front_side"], 0)
            self.assertEqual(info["episode_reward_components"]["front_side"], 10)

    def test_side_milestone_requires_actual_alignment_and_front_objective(self):
        env = self.make_env(profile="maneuver_side10")
        env.reset(seed=7, options={"slot": "front"})
        for lead, lane in [(-35, 0), (6, 2), (0, 1), (-5, 0), (5, 2)]:
            self.place(env, lead=lead, lane=lane)
            self.assertFalse(env.in_front_side_milestone())
        self.place(env, lead=0, lane=0)
        ego = env.highway.vehicle
        ego.position[1] = 1.81
        ego.on_state_update()
        self.assertFalse(env.in_front_side_milestone())
        for slot in ("back", "left", "right"):
            env.reset(seed=7, options={"slot": slot})
            self.place(env, lead=0, lane=0)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["reward_components"]["front_side"], 0)
            self.assertFalse(info["front_side_reached"])

    def test_collision_or_anchor_loss_blocks_side_bonus(self):
        env = self.make_env(profile="maneuver_side10")
        for anchor_lost in (False, True):
            env.reset(seed=7, options={"slot": "front"})
            self.place(env, lead=0, lane=0)
            (env.anchor if anchor_lost else env.highway.vehicle).crashed = True
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["reward_components"]["front_side"], 0)
            self.assertFalse(info["front_side_reached"])

    def test_side_profile_trains_logs_and_reloads_with_same_69_inputs(self):
        torch.set_num_threads(1)
        env = self.make_env(profile="maneuver_side10", duration=1.0)
        model = PPO(DropoutMultiInputPolicy, Monitor(env), seed=7, device="cpu",
                    n_steps=16, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}, "dropout_rate": .05})
        configuration = {"env_config": env.env_config, "goal_config": env.goal_config,
                         "reward_weights": env.reward_weights}
        metadata = {"task": "single_car_waypoint", "requested_timesteps": 32,
                    "training_slots": ["front", "back", "left", "right"],
                    "environment": configuration, "evaluation_environment": configuration}
        with TemporaryDirectory(prefix="side-milestone-") as directory:
            run = Path(directory)
            logger = configure(str(run), ["csv"])
            model.set_logger(logger)
            try:
                model.learn(total_timesteps=32, callback=SampledScenarioMetrics(
                    run, run / "model.zip", metadata, checkpoint_timesteps=16))
            finally:
                logger.close()
            with (run / "training_episodes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["reward_front_side"] for row in rows))
            self.assertTrue(all(float(row["reward_tracking"]) == 0 for row in rows))
            restored, replay, _ = load_experiment(run / "model_32.zip")
            self.addCleanup(replay.close)
            self.assertEqual(restored.policy.features_extractor.features_dim, 69)
            self.assertEqual(replay.reward_weights["front_side"], 10)
            replay.reset(seed=7, options={"slot": "front"})
            self.place(replay, lead=0, lane=0)
            _, _, _, _, info = replay.step(1)
            self.assertEqual(info["reward_components"]["front_side"], 10)

    def test_dropout_training_logs_new_rewards_and_reloads_navigation_contract(self):
        torch.set_num_threads(1)
        env = self.make_env(duration=1.0)
        check_env(env, warn=True, skip_render_check=True)
        model = PPO(DropoutMultiInputPolicy, Monitor(env), seed=7, device="cpu",
                    n_steps=16, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}, "dropout_rate": .05})
        configuration = {"env_config": env.env_config, "goal_config": env.goal_config,
                         "reward_weights": env.reward_weights}
        metadata = {"task": "single_car_waypoint", "requested_timesteps": 32,
                    "training_slots": ["front", "back", "left", "right"],
                    "environment": configuration, "evaluation_environment": configuration}
        with TemporaryDirectory(prefix="maneuver-guidance-") as directory:
            run = Path(directory)
            logger = configure(str(run), ["csv"])
            model.set_logger(logger)
            callback = SampledScenarioMetrics(run, run / "model.zip", metadata,
                                             checkpoint_timesteps=16)
            try:
                model.learn(total_timesteps=32, callback=callback)
            finally:
                logger.close()
            with (run / "training_episodes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["reward_route_progress"] and row["reward_tracking"] for row in rows))
            # Callback checkpoints precede the rollout's optimizer update;
            # compare the final weights with a final save, not that earlier snapshot.
            model.save(run / "final.zip")
            (run / "final.json").write_text(json.dumps(metadata), encoding="utf-8")
            restored, replay, _ = load_experiment(run / "final.zip")
            self.addCleanup(replay.close)
            self.assertIsInstance(replay, ManeuverGuidanceEnv)
            self.assertEqual(restored.policy.features_extractor.features_dim, 69)
            self.assertEqual(restored.policy.dropout_rate, .05)
            obs, _ = replay.reset(seed=7)
            self.assertEqual(int(model.predict(obs, deterministic=True)[0]),
                             int(restored.predict(obs, deterministic=True)[0]))


if __name__ == "__main__":
    unittest.main()
