"""Check lane commitment, state inputs, reward aggregation and model transfer."""
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
from baseline_dynamic_waypoint.config import COMMITTED_LANE_ENV_CONFIG, SINGLE_CAR_GOAL_CONFIG
from baseline_dynamic_waypoint.single_car import SingleCarWaypointEnv
from baseline_dynamic_waypoint.policy_transfer import transfer_observation_weights
from baseline_dynamic_waypoint.evaluate import load_experiment


class CommittedLaneTests(unittest.TestCase):
    def make_env(self, **kwargs):
        kwargs.setdefault("goal_config", {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "legacy"})
        env = CommittedLaneEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_one_request_finishes_one_lane_before_returning(self):
        env = self.make_env()
        env.reset(seed=10002)
        self.assertEqual(env.highway.vehicle.lane_index[2], 2)
        ticks = []
        env.physics_step_callback = lambda obs, r, t, tr, info: ticks.append((obs, r, info))
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["lane_change_completed"])
        self.assertGreater(info["internal_steps"], 1)
        self.assertEqual(env.highway.vehicle.lane_index[2], 1)
        self.assertEqual(env.highway.vehicle.target_lane_index[2], 1)
        self.assertTrue(all(tick[2]["requested_lane"] == 1 for tick in ticks))
        self.assertEqual([tick[2]["applied_action"] for tick in ticks], [0] + [1] * (len(ticks) - 1))
        self.assertEqual(obs["lane_change_in_progress"][0], 0)
        np.testing.assert_array_equal(obs["requested_lane"], [0, 1, 0])
        self.assertTrue(any(tick[0]["lane_change_in_progress"][0] == 1 for tick in ticks[:-1]))
        self.assertAlmostEqual(reward, sum(tick[1] for tick in ticks))
        self.assertAlmostEqual(info["episode_reward_components"]["time"], -0.05 * len(ticks))
        # A second request is accepted only after the first returned settled.
        obs, _, _, _, info = env.step(0)
        self.assertEqual(env.highway.vehicle.target_lane_index[2], 0)
        self.assertFalse(env.lane_change_in_progress)

    def test_pending_maneuver_ignores_other_actions(self):
        env = self.make_env()
        env.reset(seed=10002)
        ego = env.highway.vehicle
        ego.target_lane_index = ("0", "1", 1)
        speed = ego.target_speed
        _, _, _, _, info = env.step(3)
        self.assertEqual(ego.target_speed, speed)
        self.assertEqual(ego.target_lane_index[2], 1)
        self.assertTrue(info["lane_change_completed"])

    def test_success_waits_until_maneuver_finishes(self):
        env = self.make_env()
        env.reset(seed=10002)
        ego = env.highway.vehicle
        ego.position[0] = env.goal_position[0]
        ego.speed = ego.target_speed = env.anchor.speed
        ego.on_state_update()
        ticks = []
        env.physics_step_callback = lambda o, r, t, tr, info: ticks.append(info)
        obs, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated and info["is_success"])
        self.assertFalse(env.lane_change_in_progress)
        self.assertTrue(any(t["hold_steps"] >= env.hold_required and t["lane_change_in_progress"]
                            and not t["is_success"] for t in ticks))
        self.assertTrue(info["lane_change_completed"])

    def test_collision_and_deadline_can_interrupt(self):
        env = self.make_env()
        env.reset(seed=10002)
        env.highway.vehicle.crashed = True
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated and info["is_collision"])
        self.assertTrue(info["lane_change_interrupted"])
        settings = deepcopy(COMMITTED_LANE_ENV_CONFIG)
        settings["duration"] = 0.6
        short = self.make_env(env_config=settings)
        short.reset(seed=10002)
        _, _, terminated, truncated, info = short.step(0)
        self.assertTrue(truncated and not terminated)
        self.assertTrue(info["lane_change_interrupted"])

    def test_boundaries_and_observation_contract(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        env.reset(seed=10002)
        env.step(2)  # Already in rightmost lane.
        self.assertEqual(env.episode_physics_steps, 1)
        self.assertEqual(env.episode_policy_decisions, 1)
        self.assertEqual(sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values()), 57)

    def test_fast_encoder_preserves_observations_and_dynamics_exactly(self):
        settings = deepcopy(COMMITTED_LANE_ENV_CONFIG)
        settings["fast_observation"] = False
        slow = self.make_env(env_config=settings)
        fast = self.make_env()
        for seed in (7, 10002):
            a, _ = slow.reset(seed=seed)
            b, _ = fast.reset(seed=seed)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            for action in (0, 3, 1, 2, 4):
                a, r, t, tr, info = slow.step(action)
                b, rb, tb, trb, ib = fast.step(action)
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual((r, t, tr, info["internal_steps"]), (rb, tb, trb, ib["internal_steps"]))
                np.testing.assert_array_equal(slow.highway.vehicle.position, fast.highway.vehicle.position)
                if t or tr:
                    break

    def test_network_transfer_preserves_outputs_and_checkpoint_schema(self):
        torch.set_num_threads(1)
        old = SingleCarWaypointEnv(goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "legacy"})
        self.addCleanup(old.close)
        source = PPO("MultiInputPolicy", old, n_steps=8, batch_size=8, seed=7, device="cpu")
        env = self.make_env()
        target = PPO("MultiInputPolicy", env, n_steps=8, batch_size=8, seed=8, device="cpu")
        transfer = transfer_observation_weights(source, target)
        self.assertEqual((transfer["old_features"], transfer["new_features"]), (53, 57))
        for seed in (7, 10002, 10009):
            obs, _ = env.reset(seed=seed)
            old_obs = {k: obs[k] for k in source.observation_space.spaces}
            tensor, _ = target.policy.obs_to_tensor(obs)
            previous, _ = source.policy.obs_to_tensor(old_obs)
            with torch.no_grad():
                np.testing.assert_allclose(target.policy.get_distribution(tensor).distribution.probs.numpy(),
                                           source.policy.get_distribution(previous).distribution.probs.numpy(), atol=1e-7)
                np.testing.assert_allclose(target.policy.predict_values(tensor).numpy(),
                                           source.policy.predict_values(previous).numpy(), atol=1e-6)
        self.assertEqual(len(target.policy.optimizer.state), 0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.zip"
            target.save(path)
            data = {"task": "single_car_committed_lane", "environment": {
                "env_config": env.env_config, "goal_config": env.goal_config, "reward_weights": env.reward_weights}}
            path.with_suffix(".json").write_text(json.dumps(data))
            restored, replay, _ = load_experiment(path)
            self.addCleanup(replay.close)
            self.assertIsInstance(replay, CommittedLaneEnv)
            obs, _ = replay.reset(seed=7)
            self.assertEqual(restored.policy.features_extractor.features_dim, 57)
            replay.step(int(restored.predict(obs, deterministic=True)[0]))


if __name__ == "__main__":
    unittest.main()
