"""Checks for reusing LEFT behavior only in RIGHT-goal observations."""
from copy import deepcopy
import unittest

from gymnasium import spaces
import numpy as np

from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
from baseline_dynamic_waypoint.config import SINGLE_CAR_GOAL_CONFIG
from baseline_dynamic_waypoint.reflection_policy import RightCanonicalPolicy, reflect_observation


class RecordingModel:
    """Record the actual model inputs without depending on learned predictions."""

    def __init__(self, observation_space, action=0):
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(5)
        self.action = action
        self.calls = []
        self.returned_state = (np.array([[17.0]], dtype=np.float32),)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.calls.append((deepcopy(observation), state, episode_start, deterministic))
        return np.asarray(self.action, dtype=np.int64), self.returned_state


class ReflectionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = CommittedLaneEnv(goal_config={**deepcopy(SINGLE_CAR_GOAL_CONFIG), "observation_schema": "legacy"})
        cls.base_observation, _ = cls.env.reset(seed=10000)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def observation(self, slot=3):
        result = deepcopy(type(self).base_observation)
        result["slot"] = np.eye(4, dtype=np.float32)[slot]
        return result

    def assert_observations_equal(self, actual, expected, exact=True):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            if exact:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            else:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-7, rtol=0, err_msg=key)

    def test_reflection_preserves_geometry_and_input(self):
        obs = self.observation()
        obs["achieved_goal"][1] = 0.0
        obs["desired_goal"][1] = 1.0 / 3.0
        obs["goal_delta"][1] = 1.0 / 3.0
        obs["vehicles"][0, 2] = 0.0
        obs["vehicles"][1, 2] = 1.0 / 3.0
        obs["vehicles"][0, 4] = 0.15
        obs["vehicles"][1, 4] = -0.1
        obs["anchor"][[1, 3]] = [1.0 / 3.0, -0.1]
        obs["slot_error"][1] = 1.0
        obs["requested_lane"] = np.array([1, 0, 0], dtype=np.float32)
        saved = deepcopy(obs)
        result = reflect_observation(obs)
        self.assertAlmostEqual(result["achieved_goal"][1], 2.0 / 3.0)
        self.assertAlmostEqual(result["desired_goal"][1], 1.0 / 3.0)
        self.assertAlmostEqual(result["goal_delta"][1], -1.0 / 3.0)
        self.assertAlmostEqual(result["vehicles"][0, 2], 2.0 / 3.0)
        self.assertAlmostEqual(result["vehicles"][1, 2], -1.0 / 3.0)
        self.assertAlmostEqual(result["vehicles"][0, 4], -0.15)
        self.assertAlmostEqual(result["vehicles"][1, 4], 0.1)
        np.testing.assert_allclose(result["anchor"][[1, 3]], [-1.0 / 3.0, 0.1])
        np.testing.assert_array_equal(result["requested_lane"], [0, 0, 1])
        np.testing.assert_array_equal(result["slot"], [0, 0, 1, 0])
        self.assertEqual(result["slot_error"][1], -1)
        self.assert_observations_equal(obs, saved)
        # Forward position, speed mismatch, hold credit and progress flag retain meaning.
        for key in ("achieved_goal", "desired_goal", "goal_delta"):
            self.assertEqual(result[key][0], obs[key][0])
        for key in ("hold_progress", "lane_change_in_progress"):
            np.testing.assert_array_equal(result[key], obs[key])
        np.testing.assert_array_equal(result["vehicles"][2:], obs["vehicles"][2:])

    def test_reflection_is_an_involution_for_single_and_batched_observations(self):
        single = self.observation()
        batched = {key: np.stack([single[key], single[key]]) for key in single}
        for observation in (single, batched):
            result = reflect_observation(reflect_observation(observation))
            self.assert_observations_equal(result, observation, exact=False)
            for key in observation:
                self.assertEqual(result[key].dtype, observation[key].dtype)

    def test_nonright_tasks_are_forwarded_without_reflection(self):
        for slot in (0, 1, 2):
            model = RecordingModel(self.env.observation_space, action=0)
            policy = RightCanonicalPolicy(model)
            obs = self.observation(slot)
            saved = deepcopy(obs)
            state = (np.ones((1, 1), dtype=np.float32),)
            episode_start = np.array([True])
            action, returned_state = policy.predict(
                obs, state=state, episode_start=episode_start, deterministic=True)
            self.assertEqual(int(action), 0)
            self.assertIs(returned_state, model.returned_state)
            self.assert_observations_equal(model.calls[-1][0], saved)
            self.assertIs(model.calls[-1][1], state)
            self.assertIs(model.calls[-1][2], episode_start)
            self.assertTrue(model.calls[-1][3])
            self.assert_observations_equal(obs, saved)

    def test_right_swaps_only_lane_actions_and_leaves_model_unchanged(self):
        for source_action, expected_action in enumerate((2, 1, 0, 3, 4)):
            model = RecordingModel(self.env.observation_space, action=source_action)
            policy = RightCanonicalPolicy(model)
            obs = self.observation()
            saved = deepcopy(obs)
            action, state = policy.predict(obs, deterministic=True)
            self.assertEqual(int(action), expected_action)
            self.assertEqual(action.dtype, np.int64)
            self.assertIs(state, model.returned_state)
            self.assert_observations_equal(model.calls[-1][0], reflect_observation(obs))
            self.assertEqual(model.action, source_action)
            self.assert_observations_equal(obs, saved)

    def test_mixed_batch_reflects_only_right_rows(self):
        observations = [self.observation(slot) for slot in range(4)]
        obs = {key: np.stack([item[key] for item in observations]) for key in observations[0]}
        saved = deepcopy(obs)
        model = RecordingModel(self.env.observation_space, action=[0, 0, 2, 0])
        policy = RightCanonicalPolicy(model)
        action, _ = policy.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action, [0, 0, 2, 2])
        observed = model.calls[-1][0]
        for row in range(4):
            actual = {key: value[row] for key, value in observed.items()}
            expected = reflect_observation(observations[row]) if row == 3 else observations[row]
            self.assert_observations_equal(actual, expected)
        self.assert_observations_equal(obs, saved)

    def test_unsupported_road_or_encoding_is_rejected(self):
        model = RecordingModel(self.env.observation_space)
        for parameters in ({"lanes_count": 4}, {"lane_width": 3.5},
                           {"position_y_scale": 8.0}, {"vehicle_y_scale": 8.0}):
            with self.assertRaises(ValueError, msg=str(parameters)):
                RightCanonicalPolicy(model, **parameters)

    def test_incompatible_model_contract_is_rejected(self):
        model = RecordingModel(self.env.observation_space)
        altered = deepcopy(model.observation_space.spaces)
        altered["requested_lane"] = spaces.Box(0.0, 1.0, (4,), np.float32)
        model.observation_space = spaces.Dict(altered)
        with self.assertRaises(ValueError):
            RightCanonicalPolicy(model)
        model = RecordingModel(self.env.observation_space)
        model.action_space = spaces.Discrete(4)
        with self.assertRaises(ValueError):
            RightCanonicalPolicy(model)


if __name__ == "__main__":
    unittest.main()
