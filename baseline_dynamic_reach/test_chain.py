"""Tests of chained goals (Step 9): goal 1 is the ordinary episode, a switch changes only the goal, every
chained goal is feasible, and the goal draws never touch the simulator's random stream.

    python -B -m unittest baseline_dynamic_reach.test_chain
"""
from copy import deepcopy
import unittest

import numpy as np

from baseline_dynamic_reach.config import arm_config
from baseline_dynamic_reach.environment import CHAIN_TIME_LIMIT, ReachEnv, goal_time_needed
from baseline_dynamic_waypoint.scripted import scripted_action

SEEDS = range(21_000, 21_012)  # Test-only seeds, outside every evaluation range.
GOAL_KEYS = {"desired_goal", "achieved_goal", "goal_delta", "anchor", "slot", "slot_error", "hold_progress"}


def environment(**chain):
    settings = deepcopy(arm_config("O3_c1w0t4_n256")["environment"])
    return ReachEnv(**settings, chain=chain or None) if chain else ReachEnv(**settings)


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


class ChainTests(unittest.TestCase):
    def test_goal_one_is_the_ordinary_episode(self):
        """Same seed, same driver: identical observations, rewards and ends until the first goal ends."""
        plain, chained = environment(), environment(goals=3, mode="new car")
        try:
            for seed in SEEDS:
                obs_a, info_a = plain.reset(seed=seed)
                obs_b, info_b = chained.reset(seed=seed)
                self.assertTrue(same(obs_a, obs_b), seed)
                while True:
                    action = scripted_action(plain)
                    self.assertEqual(action, scripted_action(chained))
                    obs_a, r_a, term_a, trunc_a, info_a = plain.step(action)
                    obs_b, r_b, term_b, trunc_b, info_b = chained.step(action)
                    self.assertEqual(r_a, r_b)
                    for key in ("is_success", "is_collision", "is_timeout", "is_anchor_lost", "elapsed_time"):
                        self.assertEqual(info_a[key], info_b[key], (seed, key))
                    if term_a or trunc_a:
                        if not info_a["is_success"]:  # A crash or timeout ends goal 1, and the chain, the same way.
                            self.assertEqual((term_a, trunc_a), (term_b, trunc_b), seed)
                        break
                    self.assertTrue(same(obs_a, obs_b), seed)
                    self.assertFalse(term_b or trunc_b)
        finally:
            plain.close()
            chained.close()

    def check_switches(self, mode, origin):
        env = environment(goals=5, mode=mode, origin=origin)
        switches = 0
        try:
            for seed in SEEDS:
                obs, _ = env.reset(seed=seed)
                done = False
                while not done:
                    before_anchor, before_slot, before_origin = env.anchor, env.slot, env.episode_origin.copy()
                    ego_before = env.highway.vehicle
                    obs, _, term, trunc, info = env.step(scripted_action(env))
                    done = term or trunc
                    if not info.get("goal_switched"):
                        continue
                    switches += 1
                    self.assertFalse(done)
                    # The same state seen under the old goal: only goal-dependent inputs may differ.
                    self.assertLessEqual(info["next_time_needed"], CHAIN_TIME_LIMIT)
                    self.assertAlmostEqual(info["next_time_needed"], goal_time_needed(env, env.anchor, env.slot))
                    if mode == "same car":
                        self.assertIs(env.anchor, before_anchor)
                        self.assertNotEqual(env.slot, before_slot)
                    else:
                        self.assertIsNot(env.anchor, before_anchor)
                        gap = env.anchor.position[0] - env.highway.vehicle.position[0]
                        self.assertTrue(30.0 <= gap <= 120.0, gap)
                        self.assertTrue(before_anchor.enable_lane_change)
                    self.assertFalse(env.anchor.enable_lane_change)
                    self.assertEqual(env.hold_steps, 0)
                    if origin == "keep":
                        np.testing.assert_array_equal(env.episode_origin, before_origin)
                    else:
                        self.assertEqual(env.episode_origin[0], env.highway.vehicle.position[0])
                    self.assertIs(env.highway.vehicle, ego_before)
                    # Rebuild the observation under the new goal from scratch: it must equal what step returned.
                    fresh = env._observation(env.highway.observation_type.observe())
                    self.assertTrue(same(obs, fresh))
        finally:
            env.close()
        return switches

    # The counts only make sure the checks ran on some switches (the scripted driver reaches fewer same-car goals).
    def test_new_car_switches(self):
        self.assertGreaterEqual(self.check_switches("new car", "reset"), 3)

    def test_same_car_switches(self):
        self.assertGreaterEqual(self.check_switches("same car", "reset"), 3)

    def test_origin_kept(self):
        self.assertGreaterEqual(self.check_switches("new car", "keep"), 3)

    def test_switch_changes_only_goal_inputs(self):
        """At a switch the car rows are unchanged, except ego's progress when the origin restarts."""
        for origin in ("reset", "keep"):
            env = environment(goals=2, mode="new car", origin=origin)
            try:
                for seed in SEEDS:
                    env.reset(seed=seed)
                    done = False
                    while not done:
                        action = scripted_action(env)
                        obs, _, term, trunc, info = env.step(action)
                        done = term or trunc
                        if info.get("goal_switched"):
                            # Every car row equals the simulator's own Kinematics observation; only ego's x entry,
                            # which is progress from the episode origin, is replaced (0 m when the origin restarts).
                            raw = np.asarray(env.highway.observation_type.observe(), dtype=np.float32)
                            np.testing.assert_array_equal(obs["vehicles"][1:], raw[1:])
                            np.testing.assert_array_equal(obs["vehicles"][0, [0, 2, 3, 4]], raw[0, [0, 2, 3, 4]])
                            if origin == "reset":
                                self.assertAlmostEqual(float(obs["vehicles"][0, 1]), 0.0, places=6)
                            self.assertEqual(set(obs), GOAL_KEYS | {"vehicles"})
                            break
            finally:
                env.close()

    def test_draws_leave_the_simulator_stream_alone(self):
        env = environment(goals=5, mode="new car")
        try:
            for seed in SEEDS:
                env.reset(seed=seed)
                done = False
                while not done:
                    action = scripted_action(env)
                    state = deepcopy(env.highway.np_random.bit_generator.state)
                    obs, _, term, trunc, info = env.step(action)
                    done = term or trunc
                    if info.get("goal_switched"):
                        # The goal draw uses the chain's own stream: the simulator's stream did not move.
                        self.assertEqual(state, env.highway.np_random.bit_generator.state)
        finally:
            env.close()

    def test_each_goal_has_its_own_time_limit(self):
        env = environment(goals=5, mode="same car")
        try:
            for seed in SEEDS:
                env.reset(seed=seed)
                done, start = False, 0.0
                while not done:
                    obs, _, term, trunc, info = env.step(scripted_action(env))
                    done = term or trunc
                    if info.get("goal_switched"):
                        start = env.goal_start_time
                    if trunc and info["is_timeout"]:
                        self.assertAlmostEqual(env.highway.time - start, 25.0, places=6)
        finally:
            env.close()

    def test_off_by_default(self):
        env = environment()
        try:
            self.assertIsNone(env.chain)
            self.assertEqual(env.env_config["duration"], 25.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
