"""Physical checks for lane and safe-overtaking reward guidance."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
from baseline_dynamic_waypoint.config import REWARD_PROFILES, REWARD_GOAL_OVERRIDES, SINGLE_CAR_GOAL_CONFIG


class RewardProfileTests(unittest.TestCase):
    def make_env(self, profile="guided", slot="right", ego_lane=0, anchor_lane=0,
                 gap=35.0, ego_speed=15.0, anchor_speed=15.0):
        goal_config = deepcopy(SINGLE_CAR_GOAL_CONFIG)
        goal_config.update(REWARD_GOAL_OVERRIDES.get(profile, {}))
        env = CommittedLaneEnv(reward_weights=REWARD_PROFILES[profile], goal_config=goal_config)
        self.addCleanup(env.close)
        env.reset(seed=10000, options={"slot": slot})
        anchor_x = float(env.anchor.position[0])
        for vehicle, lane_id, x, speed in (
            (env.anchor, anchor_lane, anchor_x, anchor_speed),
            (env.highway.vehicle, ego_lane, anchor_x - gap, ego_speed),
        ):
            lane_index = ("0", "1", lane_id)
            vehicle.position = env.highway.road.network.get_lane(lane_index).position(x, 0).astype(float)
            vehicle.heading = 0.0
            vehicle.speed = vehicle.target_speed = speed
            vehicle.on_state_update()
            vehicle.target_lane_index = lane_index
        env._update_goal()
        env.previous_distance = env.distance_to_goal()
        env.previous_shaping_errors = env.shaping_errors()
        return env

    def test_target_lane_maneuver_has_stronger_feedback_in_both_directions(self):
        for profile in (name for name in REWARD_PROFILES if REWARD_PROFILES[name]["lane"] > 0):
            rewards = []
            for slot, lane, action in (("right", 0, 2), ("left", 2, 0)):
                with self.subTest(profile=profile, slot=slot):
                    baseline = self.make_env("baseline", slot, lane, lane)
                    guided = self.make_env(profile, slot, lane, lane)
                    _, before, _, _, original = baseline.step(action)
                    _, after, terminated, truncated, info = guided.step(action)
                    self.assertFalse(terminated or truncated or info["is_collision"])
                    self.assertTrue(info["lane_change_completed"])
                    self.assertGreater(info["episode_reward_components"]["lane"], 0)
                    self.assertGreater(after, before)
                    np.testing.assert_array_equal(guided.highway.vehicle.position,
                                                  baseline.highway.vehicle.position)
                    self.assertEqual(info["internal_steps"], original["internal_steps"])
                    rewards.append(after)
            self.assertAlmostEqual(*rewards, places=10)

    def test_right_command_away_from_the_goal_has_negative_lane_reward(self):
        env = self.make_env(slot="left", ego_lane=1, anchor_lane=2)
        # LEFT objective is lane 1 here; move right to lane 2, away from it.
        _, _, _, _, info = env.step(2)
        self.assertLess(info["episode_reward_components"]["lane"], 0)
        self.assertFalse(info["is_success"])

    def test_front_rewards_safe_passing_lane_and_forward_overtake_progress(self):
        env = self.make_env(slot="front", ego_lane=1, anchor_lane=1)
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated or truncated or info["is_collision"])
        self.assertGreater(info["episode_reward_components"]["lane"], 0)
        before = info["episode_reward_components"]["overtake"]
        # Ego now travels safely beside the anchor and accelerates toward passing.
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(3)
            self.assertFalse(terminated or truncated or info["is_collision"])
        self.assertGreater(info["episode_reward_components"]["overtake"], before)

    def test_front_rewards_return_to_goal_lane_after_safe_clearance(self):
        env = self.make_env(slot="front", ego_lane=0, anchor_lane=1, gap=-12.0)
        _, _, terminated, truncated, info = env.step(2)
        self.assertFalse(terminated or truncated or info["is_collision"])
        self.assertTrue(info["lane_change_completed"])
        self.assertGreater(info["episode_reward_components"]["lane"], 0)
        self.assertEqual(env.highway.vehicle.target_lane_index, env.slot_lane_index)

    def test_speed_matching_is_rewarded_near_the_goal(self):
        slowing = self.make_env(ego_lane=1, gap=10, ego_speed=20)
        speeding = self.make_env(ego_lane=1, gap=10, ego_speed=20)
        _, _, _, _, slower = slowing.step(4)
        _, _, _, _, faster = speeding.step(3)
        self.assertGreater(slower["reward_components"]["speed_match"], 0)
        self.assertLess(faster["reward_components"]["speed_match"], 0)

    def test_front_lane_guidance_blends_continuously_after_clearance(self):
        epsilon = 1e-5
        before = self.make_env(slot="front", ego_lane=0, anchor_lane=1, gap=-(10 - epsilon))
        after = self.make_env(slot="front", ego_lane=0, anchor_lane=1, gap=-(10 + epsilon))
        self.assertLess(abs(before.shaping_errors()[0] - after.shaping_errors()[0]), 1e-4)
        # Before the blend, a passing lane is preferable to merging. After it,
        # the slot lane is preferable to remaining beside the anchor.
        for lead, preferred in ((5.0, "passing"), (15.0, "goal")):
            passing = self.make_env(slot="front", ego_lane=0, anchor_lane=1, gap=-lead)
            goal = self.make_env(slot="front", ego_lane=1, anchor_lane=1, gap=-lead)
            if preferred == "passing":
                self.assertLess(passing.shaping_errors()[0], goal.shaping_errors()[0])
            else:
                self.assertLess(goal.shaping_errors()[0], passing.shaping_errors()[0])

    def test_macro_reward_is_exact_sum_of_physics_components(self):
        env = self.make_env()
        ticks = []
        env.physics_step_callback = lambda obs, reward, terminated, truncated, info: ticks.append((reward, info))
        _, reward, _, _, info = env.step(2)
        self.assertGreater(len(ticks), 1)
        self.assertAlmostEqual(reward, sum(item[0] for item in ticks))
        for tick_reward, tick_info in ticks:
            self.assertAlmostEqual(tick_reward, sum(tick_info["reward_components"].values()))
        self.assertAlmostEqual(reward, sum(info["episode_reward_components"].values()))

    def test_previous_checkpoint_reward_and_profile_defaults_are_preserved(self):
        profiles = deepcopy(REWARD_PROFILES)
        path = Path(__file__).resolve().parent / "models" / "ppo_singlecar_committed57_seed7_200k.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        env = CommittedLaneEnv(**metadata["environment"])
        self.addCleanup(env.close)
        env.reset(seed=10000)
        _, _, _, _, info = env.step(2)
        self.assertEqual(env.reward_weights, metadata["environment"]["reward_weights"])
        for name in ("lane", "speed_match", "overtake"):
            self.assertEqual(env.reward_weights[name], 0.0)
            self.assertEqual(info["episode_reward_components"][name], 0.0)
        self.assertEqual(REWARD_PROFILES, profiles)


if __name__ == "__main__":
    unittest.main()
