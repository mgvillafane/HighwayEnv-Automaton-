"""Checks for the reach chain: B0 is reach with R0, arms change one IV, ivcheck enforces it."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from baseline_dynamic_reach import config as chain
from baseline_dynamic_reach.config import ARMS, MACHINE_IV, R0, arm_config
from baseline_dynamic_reach.environment import ReachEnv
from baseline_dynamic_reach.evaluate import check_paired, check_rows, wilson
from baseline_dynamic_reach.ivcheck import IVError, check, check_replicates, differences, flatten
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv

REPO = Path(__file__).resolve().parent.parent
LAST_OLD_RUN = REPO / "baseline_dynamic_waypoint" / "results" / "ppo_hold1s_1m_overtake_seed7" / "run.json"


def sidecar(config, seed=7, **extra):
    return {"seed": seed, "run_name": f"x_seed{seed}", "created_utc": "now", **deepcopy(config),
            "versions": {"torch": "2.15.0"}, "platform": {"node": "here"}, **extra}


class ReachTaskTests(unittest.TestCase):
    def make_env(self):
        env = ReachEnv(**arm_config("B0")["environment"])
        self.addCleanup(env.close)
        return env

    def reset_with_slot(self, env, slot):
        for seed in range(50):
            try:
                return env.reset(seed=seed, options={"slot": slot, "anchor_index": 0})
            except ValueError:
                continue
        self.fail(f"No seed could host slot {slot}")

    def place_in_slot(self, env, longitudinal=0.0):
        lane = env.highway.road.network.get_lane(env.slot_lane_index)
        slot_s, _ = lane.local_coordinates(env.goal_position)
        ego = env.highway.vehicle
        ego.position = lane.position(slot_s + longitudinal, 0).astype(float)
        ego.heading = 0.0
        ego.speed = ego.target_speed = env.anchor.speed
        ego.lane_index = ego.target_lane_index = env.slot_lane_index

    def test_b0_is_reach_one_decision_in_the_box_succeeds_with_50(self):
        env = self.make_env()
        for slot in chain.SLOTS:
            self.reset_with_slot(env, slot)
            self.assertEqual(env.hold_required, 1)
            self.place_in_slot(env)
            _, reward, terminated, truncated, info = env.step(1)
            self.assertTrue(terminated and info["is_success"] and not truncated, slot)
            self.assertEqual(reward, 50.0)
            self.assertEqual(info["reward_components"], {"goal": 50.0, "collision": 0.0, "progress": 0.0,
                                                         "time": -0.0, "hold": 0.0})

    def test_outside_the_box_nothing_is_paid(self):
        env = self.make_env()
        self.reset_with_slot(env, "back")
        self.place_in_slot(env, longitudinal=env.goal_config["radius"] + 2.0)
        _, reward, terminated, _, info = env.step(1)
        self.assertFalse(terminated or info["is_success"])
        self.assertEqual(reward, 0.0)

    def test_a_crash_beats_success_and_earns_nothing(self):
        env = self.make_env()
        self.reset_with_slot(env, "back")
        self.place_in_slot(env)
        env.highway.vehicle.crashed = True
        _, reward, terminated, _, info = env.step(1)
        self.assertTrue(terminated and info["is_collision"] and not info["is_success"])
        self.assertEqual(reward, 0.0)

    def test_r0_return_is_exactly_50_times_success(self):
        env = self.make_env()
        rng = np.random.default_rng(0)
        outcomes = set()
        for seed in range(12):
            env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                _, reward, terminated, truncated, info = env.step(int(rng.integers(5)))
                total += reward
                done = terminated or truncated
            self.assertEqual(total, 50.0 * info["is_success"])
            outcomes.add(info["termination_reason"])
        self.assertGreater(len(outcomes), 1)  # The check covered more than one kind of ending.

    def test_b0_keeps_the_inherited_53_input_observation(self):
        env = self.make_env()
        obs, _ = env.reset(seed=7)
        self.assertEqual(set(obs), {"vehicles", "achieved_goal", "desired_goal", "goal_delta",
                                    "anchor", "slot", "hold_progress", "slot_error"})
        self.assertEqual(sum(value.size for value in obs.values()), 53)
        self.assertEqual(float(obs["hold_progress"][0]), 0.0)


class ChainTests(unittest.TestCase):
    def test_b0_differs_from_the_last_old_run_only_in_task_and_reward(self):
        with LAST_OLD_RUN.open(encoding="utf-8") as handle:
            old = json.load(handle)
        diffs = differences(old, sidecar(arm_config("B0"), versions=old["versions"]) | {"platform": None})
        diffs.pop("platform", None)  # Old sidecars predate the platform record.
        changed = {path for path in diffs if not path.startswith("environment.reward_weights.")}
        self.assertEqual(changed, {"environment.goal_config.hold_seconds"})
        self.assertEqual(diffs["environment.goal_config.hold_seconds"], (1.0, 0.1))
        self.assertEqual(arm_config("B0")["environment"]["reward_weights"], R0)

    def test_an_arm_is_its_parent_plus_one_setting(self):
        ARMS["T1"] = {"parent": "B0", "iv": "environment.reward_weights.collision", "value": 50.0, "reason": "test"}
        ARMS["T2"] = {"parent": "T1", "iv": "environment.reward_weights.lane", "value": 1.0, "reason": "test"}
        try:
            diffs = check(sidecar(arm_config("B0")), sidecar(arm_config("T1"), seed=8), ARMS["T1"]["iv"])
            self.assertEqual(diffs, {"environment.reward_weights.collision": (0.0, 50.0)})
            # A new key (not in the parent) is still one setting.
            diffs = check(sidecar(arm_config("T1")), sidecar(arm_config("T2")))
            self.assertEqual(diffs, {"environment.reward_weights.lane": ("<missing>", 1.0)})
            # Skipping a generation would compare two changes at once.
            with self.assertRaises(IVError):
                check(sidecar(arm_config("B0")), sidecar(arm_config("T2")))
        finally:
            ARMS.pop("T1"), ARMS.pop("T2")

    def test_ivcheck_rejects_zero_or_two_differences(self):
        base = sidecar(arm_config("B0"))
        with self.assertRaises(IVError):
            check(base, sidecar(arm_config("B0"), seed=8))  # Only the seed differs: nothing to compare.
        two = sidecar(arm_config("B0"))
        two["num_envs"] = 4
        two["ppo_config"]["n_steps"] = 128
        with self.assertRaisesRegex(IVError, "2 settings differ"):
            check(base, two)
        one = sidecar(arm_config("B0"))
        one["num_envs"] = 4
        self.assertEqual(check(base, one), {"num_envs": (3, 4)})

    def test_hidden_changes_in_packages_or_machine_are_caught(self):
        base = sidecar(arm_config("B0"))
        for section, key in (("versions", "torch"), ("platform", "node")):
            child = sidecar(arm_config("B0"))
            child["environment"]["reward_weights"]["collision"] = 50.0
            child[section][key] = "other"
            with self.assertRaises(IVError):
                check(base, child, "environment.reward_weights.collision")

    def test_a_declared_compound_iv_must_contain_every_difference(self):
        base = sidecar(arm_config("B0"))
        child = sidecar(arm_config("B0"))
        child["ppo_config"]["policy_kwargs"]["net_arch"] = {"pi": [128, 128], "vf": [128, 128]}
        self.assertEqual(len(check(base, child, "ppo_config.policy_kwargs")), 2)
        child["num_envs"] = 4
        with self.assertRaises(IVError):
            check(base, child, "ppo_config.policy_kwargs")

    def test_a_machine_arm_changes_the_machine_and_nothing_else(self):
        self.assertEqual(ARMS["B3m"]["iv"], MACHINE_IV)
        self.assertEqual(arm_config("B3m"), arm_config("B3"))
        pc = sidecar(arm_config("B3"), platform={"node": "pc", "python": "3.13.1"})
        mac = sidecar(arm_config("B3m"), platform={"node": "mac", "python": "3.12.13"})
        self.assertEqual(set(check(pc, mac, MACHINE_IV)), {"platform.node", "platform.python"})
        # Trained on the parent's machine, a machine arm has no IV at all.
        with self.assertRaises(IVError):
            check(pc, sidecar(arm_config("B3m"), platform=pc["platform"]), MACHINE_IV)
        # A package or a setting that changes along with the machine is still refused.
        for section, key, value in (("versions", "torch", "other"), ("environment", "reward_weights", {})):
            mixed = deepcopy(mac)
            mixed[section][key] = value
            with self.assertRaises(IVError):
                check(pc, mixed, MACHINE_IV)

    def test_b4_is_b3m_with_a_triple_crash_weight(self):
        parent, child = arm_config("B3m"), arm_config("B4")
        self.assertEqual(child["environment"]["reward_weights"]["collision"], 150.0)
        self.assertEqual(parent["environment"]["reward_weights"]["collision"], 50.0)
        self.assertEqual(child["init_policy"], "B0")
        diffs = check(sidecar(parent), sidecar(child, seed=8), ARMS["B4"]["iv"])
        self.assertEqual(diffs, {"environment.reward_weights.collision": (50.0, 150.0)})

    def test_replicates_must_be_identical(self):
        check_replicates([sidecar(arm_config("B0"), seed=s) for s in (7, 8, 9)])
        odd = sidecar(arm_config("B0"), seed=9)
        odd["requested_timesteps"] = 500_000
        with self.assertRaises(IVError):
            check_replicates([sidecar(arm_config("B0"), seed=7), odd])

    def test_a_setting_older_sidecars_lack_counts_as_its_default(self):
        old = sidecar(arm_config("B1"))
        old.pop("init_policy")  # B0 and B1 were saved before init_policy existed.
        self.assertEqual(differences(old, sidecar(arm_config("B1"))), {})
        diffs = check(old, sidecar(arm_config("B2")), ARMS["B2"]["iv"])
        self.assertEqual(diffs, {"init_policy": ("<missing>", "B0")})
        # Only None is a default; a real value is still a difference.
        with self.assertRaises(IVError):
            check(sidecar(arm_config("B2")), sidecar(arm_config("B2")) | {"init_policy": "B1"}, "num_envs")

    def test_lists_are_single_settings(self):
        leaves = flatten({"a": {"b": [1, 2], "c": 3}})
        self.assertEqual(leaves, {"a.b": [1, 2], "a.c": 3})


class MeasurementTests(unittest.TestCase):
    def test_wilson_interval(self):
        np.testing.assert_allclose(wilson(0, 10), [0.0, 0.2775], atol=1e-4)
        np.testing.assert_allclose(wilson(5, 10), [0.2366, 0.7634], atol=1e-4)
        self.assertAlmostEqual(wilson(10, 10)[1], 1.0)

    def test_rows_need_one_outcome_and_consistent_reward(self):
        row = {"policy": "p", "seed": 1, "success": 1, "collision": 0, "timeout": 0, "anchor_lost": 0,
               "reward": 50.0, "reward_goal": 50.0, "reward_time": 0.0}
        check_rows([row])
        with self.assertRaises(AssertionError):
            check_rows([{**row, "collision": 1}])
        with self.assertRaises(AssertionError):
            check_rows([{**row, "reward": 49.0}])

    def test_policies_must_share_initial_episodes(self):
        row = {"policy": "a", "seed": 1, "slot": "back", "goal_lane": 1, "initial_lane": 1,
               "anchor_initial_gap": 40.0, "initial_distance": 25.0}
        check_paired([[row], [{**row, "policy": "b"}]])
        with self.assertRaises(AssertionError):
            check_paired([[row], [{**row, "policy": "b", "slot": "front"}]])


class ShapingTests(unittest.TestCase):
    def run_episode(self, env, seed, actions):
        obs, info = env.reset(seed=seed)
        trace, done, step = [], False, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(int(actions[step % len(actions)]))
            trace.append((reward, terminated, truncated, info["termination_reason"], info["distance_to_goal"],
                          np.concatenate([np.ravel(value) for value in obs.values()])))
            done, step = terminated or truncated, step + 1
        return trace, info

    def test_reach_env_without_new_terms_is_the_inherited_env(self):
        """B0 and B1 were trained with DynamicGoalHighwayEnv; ReachEnv must not change them."""
        actions = np.random.default_rng(3).integers(5, size=400)
        for arm_name in ("B0", "B1"):
            environment = arm_config(arm_name)["environment"]
            old, new = DynamicGoalHighwayEnv(**deepcopy(environment)), ReachEnv(**deepcopy(environment))
            self.addCleanup(old.close)
            self.addCleanup(new.close)
            for seed in (10_000, 10_001, 10_005):
                a, info_a = self.run_episode(old, seed, actions)
                b, info_b = self.run_episode(new, seed, actions)
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    self.assertEqual(x[:5], y[:5])
                    np.testing.assert_array_equal(x[5], y[5])
                self.assertEqual(info_a["episode_reward_components"], info_b["episode_reward_components"])

    def test_potential_shaping_telescopes_to_the_initial_distance(self):
        environment = arm_config("B1")["environment"]
        environment["reward_weights"]["potential_progress"] = {"weight": 0.1, "gamma": 0.99}
        env = ReachEnv(**environment)
        self.addCleanup(env.close)
        rng = np.random.default_rng(5)
        endings = set()
        for seed in range(10_000, 10_012):
            _, info = env.reset(seed=seed)
            start = info["distance_to_goal"]
            discounted, total, t, done = 0.0, 0.0, 0, False
            while not done:
                _, reward, terminated, truncated, info = env.step(int(rng.integers(5)))
                shaping = info["reward_components"]["potential_progress"]
                discounted += 0.99 ** t * shaping
                total += shaping
                t, done = t + 1, terminated or truncated
            endings.add("terminated" if terminated else "truncated")
            final = 0.0 if terminated else -info["distance_to_goal"]
            # sum_t g^t F_t = g^T Phi(s_T) - Phi(s_0), with Phi(terminal) = 0.
            self.assertAlmostEqual(discounted, 0.1 * (0.99 ** t * final + start), places=6)
            self.assertAlmostEqual(info["episode_reward_components"]["potential_progress"], total, places=9)
            # The other terms are untouched: the episode return minus shaping is still R1's.
            outcome = 50.0 * info["is_success"] - 50.0 * info["is_collision"]
            self.assertAlmostEqual(sum(info["episode_reward_components"].values()) - total, outcome, places=9)
        self.assertIn("terminated", endings)

    def test_shaping_gamma_must_match_ppo(self):
        from baseline_dynamic_reach.train import check_shaping
        config = arm_config("B1")
        config["environment"]["reward_weights"]["potential_progress"] = {"weight": 0.1, "gamma": 0.99}
        check_shaping(config)
        config["environment"]["reward_weights"]["potential_progress"]["gamma"] = 1.0
        with self.assertRaises(ValueError):
            check_shaping(config)


if __name__ == "__main__":
    unittest.main()
