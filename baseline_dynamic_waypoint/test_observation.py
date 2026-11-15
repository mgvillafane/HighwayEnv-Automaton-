"""Lane-aware inputs, moving goal coordinates, and checkpoint compatibility."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

from baseline_dynamic_waypoint.config import COMMITTED_LANE_ENV_CONFIG, SINGLE_CAR_GOAL_CONFIG
from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.policy_transfer import transfer_observation_weights


class LaneGoalObservationTests(unittest.TestCase):
    def make_env(self, **kwargs):
        kwargs.setdefault("goal_config", {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "lane_goal_v2"})
        env = CommittedLaneEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_default_contract_and_padded_vehicle_lanes(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        obs, _ = env.reset(seed=10000)
        expected = {"vehicles": (7, 6), "goal_delta": (2,), "anchor": (4,),
                    "lane_difference": (2,), "desired_goal": (3,), "hold_progress": (1,)}
        self.assertEqual({key: value.shape for key, value in obs.items()}, expected)
        self.assertEqual(sum(value.size for value in obs.values()), 54)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["vehicles"][0, 5], env.highway.vehicle.lane_index[2])
        self.assertEqual(obs["vehicles"][1, 5], env.anchor.lane_index[2])
        np.testing.assert_array_equal(obs["vehicles"][2:], np.zeros((5, 6)))

    def test_goal_offsets_are_relative_to_target_car_for_every_slot(self):
        env = self.make_env()
        for slot, offset in (("front", [20, 0]), ("back", [-15, 0]),
                             ("left", [0, -4]), ("right", [0, 4])):
            with self.subTest(slot=slot):
                obs, info = env.reset(seed=7, options={"slot": slot})
                self.assertEqual(obs["desired_goal"][0], info["anchor_id"])
                np.testing.assert_allclose(obs["desired_goal"][1:] * env.position_scale, offset, atol=1e-6)
                np.testing.assert_allclose(obs["goal_delta"],
                                           (env.goal_position - env.highway.vehicle.position) / env.position_scale, atol=1e-6)
                # Both vehicles can move without changing the target-car-relative command.
                env.highway.vehicle.position[0] += 2
                env.anchor.position[0] += 7
                env._update_goal()
                moved = env._observation(env.highway.observation_type.observe())
                np.testing.assert_allclose(moved["desired_goal"], obs["desired_goal"], atol=1e-6)
                np.testing.assert_allclose(moved["goal_delta"][0] - obs["goal_delta"][0], 5 / 200, atol=1e-6)

    def test_signed_lane_counts_and_hold_progress(self):
        env = self.make_env()
        env.reset(seed=7, options={"slot": "right"})
        for ego_lane, anchor_lane in ((0, 0), (0, 1), (2, 0), (2, 1)):
            with self.subTest(ego_lane=ego_lane, anchor_lane=anchor_lane):
                for car, lane_id in ((env.highway.vehicle, ego_lane), (env.anchor, anchor_lane)):
                    lane_index = ("0", "1", lane_id)
                    car.position[1] = env.highway.road.network.get_lane(lane_index).position(0, 0)[1]
                    car.on_state_update()
                env._update_goal()
                env.hold_steps = 3
                obs = env._observation(env.highway.observation_type.observe())
                np.testing.assert_array_equal(obs["lane_difference"],
                                              [anchor_lane - ego_lane, anchor_lane + 1 - ego_lane])
                self.assertAlmostEqual(float(obs["hold_progress"][0]), .6)

    def test_multi_car_lane_column_tracks_sorted_rows_and_stable_target_id(self):
        from baseline_dynamic_waypoint.config import GOAL_CONFIG
        env = DynamicGoalHighwayEnv(goal_config={**deepcopy(GOAL_CONFIG), "observation_schema": "lane_goal_v2"})
        self.addCleanup(env.close)
        env.reset(seed=7, options={"slot": "front"})
        for reverse in (False, True):
            ego = env.highway.vehicle
            traffic = [car for car in env.highway.road.vehicles if car is not ego]
            for index, car in enumerate(traffic):
                rank = len(traffic) - index if reverse else index + 1
                lane_id = index % 3
                car.position = env.highway.road.network.get_lane(("0", "1", lane_id)).position(
                    ego.position[0] + 10 * rank, 0).astype(float)
                car.on_state_update()
            env._update_goal()
            obs = env._observation(env.highway.observation_type.observe())
            ordered = env.highway.road.close_objects_to(ego, env.highway.PERCEPTION_DISTANCE,
                                                        count=6, see_behind=True, sort=True, vehicles_only=True)
            for row, car in enumerate([ego, *ordered]):
                self.assertEqual(obs["vehicles"][row, 5], car.lane_index[2])
                if row:
                    self.assertAlmostEqual(float(obs["vehicles"][row, 1]),
                                           (car.position[0] - ego.position[0]) / 200, places=6)
            self.assertEqual(obs["desired_goal"][0], env.highway.road.vehicles.index(env.anchor))

    def test_new_observation_preserves_physics_rewards_and_retained_inputs(self):
        legacy_goal = {**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "legacy"}
        legacy = self.make_env(goal_config=legacy_goal)
        current = self.make_env()
        for seed in (7, 10002):
            old, _ = legacy.reset(seed=seed)
            new, _ = current.reset(seed=seed)
            for action in (0, 3, 1, 2, 4):
                np.testing.assert_array_equal(new["vehicles"][:, :5], old["vehicles"])
                for key in ("anchor", "goal_delta", "hold_progress"):
                    np.testing.assert_array_equal(new[key], old[key])
                old, reward, terminated, truncated, info = legacy.step(action)
                new, new_reward, new_terminated, new_truncated, new_info = current.step(action)
                self.assertEqual((reward, terminated, truncated, info["internal_steps"]),
                                 (new_reward, new_terminated, new_truncated, new_info["internal_steps"]))
                self.assertEqual(info["is_success"], new_info["is_success"])
                if terminated or truncated:
                    break

    def test_fast_and_native_encoder_match_new_schema(self):
        fast = self.make_env()
        native = self.make_env(env_config={**deepcopy(COMMITTED_LANE_ENV_CONFIG), "fast_observation": False})
        for seed in (7, 10002):
            a, _ = fast.reset(seed=seed)
            b, _ = native.reset(seed=seed)
            for action in (0, 3, 2):
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                a, reward, terminated, truncated, _ = fast.step(action)
                b, other_reward, other_terminated, other_truncated, _ = native.step(action)
                self.assertEqual((reward, terminated, truncated), (other_reward, other_terminated, other_truncated))
                if terminated or truncated:
                    break

    def test_checkpoint_round_trip_for_both_schemas_and_partial_transfer(self):
        torch.set_num_threads(1)
        goal = deepcopy(SINGLE_CAR_GOAL_CONFIG)
        goal.pop("observation_schema")  # Exactly how historical sidecars select their layout.
        legacy = self.make_env(goal_config=goal)
        current = self.make_env()
        source = PPO("MultiInputPolicy", legacy, n_steps=8, batch_size=8, seed=7, device="cpu")
        target = PPO("MultiInputPolicy", current, n_steps=8, batch_size=8, seed=8, device="cpu")
        transfer = transfer_observation_weights(source, target)
        self.assertEqual((transfer["old_features"], transfer["new_features"]), (57, 54))
        self.assertFalse(transfer["policy_outputs_preserved"])
        self.assertTrue(transfer["optimizer_reset"])

        def columns(model):
            mapping, offset = {}, 0
            for key in model.policy.features_extractor.extractors:
                size = int(np.prod(model.observation_space[key].shape))
                mapping[key] = slice(offset, offset + size)
                offset += size
            return mapping

        old_cols, new_cols = columns(source), columns(target)
        for layer in ("policy_net", "value_net"):
            old_weights = getattr(source.policy.mlp_extractor, layer)[0].weight.detach().numpy()
            new_weights = getattr(target.policy.mlp_extractor, layer)[0].weight.detach().numpy()
            old_vehicles = old_weights[:, old_cols["vehicles"]].reshape(-1, 7, 5)
            new_vehicles = new_weights[:, new_cols["vehicles"]].reshape(-1, 7, 6)
            np.testing.assert_array_equal(new_vehicles[:, :, :5], old_vehicles)
            np.testing.assert_array_equal(new_vehicles[:, :, 5], 0)
            for key in ("anchor", "goal_delta", "hold_progress"):
                np.testing.assert_array_equal(new_weights[:, new_cols[key]], old_weights[:, old_cols[key]])
            for key in ("desired_goal", "lane_difference"):
                np.testing.assert_array_equal(new_weights[:, new_cols[key]], 0)
        with TemporaryDirectory() as directory:
            for label, model, env, expected in (("legacy", source, legacy, 57), ("current", target, current, 54)):
                path = Path(directory) / f"{label}.zip"
                model.save(path)
                path.with_suffix(".json").write_text(json.dumps({"task": "single_car_committed_lane", "environment": {
                    "env_config": env.env_config, "goal_config": env.goal_config, "reward_weights": env.reward_weights}}))
                reloaded, replay, _ = load_experiment(path)
                self.addCleanup(replay.close)
                self.assertEqual(reloaded.policy.features_extractor.features_dim, expected)
                obs, _ = replay.reset(seed=7)
                self.assertTrue(replay.observation_space.contains(obs))
                replay.step(int(reloaded.predict(obs, deterministic=True)[0]))


if __name__ == "__main__":
    unittest.main()
