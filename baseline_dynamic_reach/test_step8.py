"""Step 8: generated arms, the round and pipeline checks, the statistics and the winner rules."""
from copy import deepcopy
import math
import unittest

from baseline_dynamic_reach.config import (ARMS, P6_REWARD, REWARDS, STEP8_ARMS, arm_config, parse_step8,
                                           step8_name)
from baseline_dynamic_reach.grid import (beats, betainc, episode_score, factor_between, group_counts, paired,
                                         simplicity, t_quantile, t_two_sided_p, welch)
from baseline_dynamic_reach.ivcheck import IVError, check, check_pipeline, check_round, differences


def sidecar(arm, seed=7):
    """What train.py would save for this arm (configuration plus machine and packages)."""
    return {"arm": arm, "seed": seed, "run_name": f"{arm}_seed{seed}", **deepcopy(arm_config(arm)),
            "versions": {"torch": "2.15.0"}, "platform": {"system": "Windows", "node": "pc"}}


def loader(seed=7):
    cache = {}

    def load(run):
        arm = run.rsplit("_seed", 1)[0]
        cache.setdefault(arm, sidecar(arm, seed))
        return cache[arm]
    return load


class GeneratedArmTests(unittest.TestCase):
    def test_names_round_trip_and_bad_names_are_refused(self):
        for name in STEP8_ARMS:
            order, round_number, reward, width = parse_step8(name)
            self.assertEqual(step8_name(order, round_number, reward, width), name)
        for bad in ("O1", "E5_c1w01", "E4", "E2", "E1_c1w01", "E2_c3w01", "X1", "B5", "P6", "E2_c1w01_n128"):
            self.assertIsNone(parse_step8(bad), bad)

    def test_old_arms_keep_their_configurations(self):
        self.assertNotIn("B5", STEP8_ARMS)
        self.assertEqual(ARMS["P6"]["parent"], "B5")

    def test_e1_is_p4_run_for_two_million_transitions(self):
        diffs = differences(sidecar("P4"), sidecar("E1"))
        self.assertEqual(set(diffs), {"requested_timesteps"})
        self.assertEqual(diffs["requested_timesteps"], (1_000_000, 2_000_000))

    def test_round_one_arms_are_one_iv_from_e1(self):
        for arm in ("H1", "N1"):
            check(sidecar("E1"), sidecar(arm), ARMS[arm]["iv"])
        check(sidecar("E1"), sidecar("E1_n256"), "ppo_config.policy_kwargs.net_arch")
        self.assertIsNone(arm_config("N1")["environment"]["scenario"])
        self.assertNotIn("start_offsets", arm_config("H1")["environment"]["scenario"])

    def test_round_one_is_success_only_from_fresh_weights(self):
        for arm in ("E1", "H1", "N1", "E1_n256"):
            config = arm_config(arm)
            self.assertIsNone(config["init_policy"])
            self.assertEqual(config["environment"]["reward_weights"]["collision"], 0.0)
            self.assertEqual(config["environment"]["reward_weights"]["potential_progress"]["weight"], 0.0)
            self.assertEqual(config["environment"]["reward_scale"], 0.02)

    def test_reward_levels(self):
        weights = arm_config("E2_c05w01")["environment"]["reward_weights"]
        self.assertEqual(weights["collision"] * 0.02, 0.5)  # Crash cost 0.5 x goal (+1).
        self.assertEqual(weights["potential_progress"], {"weight": 0.1, "gamma": 0.99})
        self.assertEqual(arm_config("E3_c2w0")["environment"]["reward_weights"]["collision"] * 0.02, 2.0)
        self.assertEqual(len(REWARDS), 8)
        self.assertIn(P6_REWARD, REWARDS)

    def test_time_cost_arms(self):
        weights = arm_config("O3_c1w0t4")["environment"]["reward_weights"]
        self.assertAlmostEqual(weights["time"] * 0.02, 0.004)                # Per decision, as a multiple of the goal.
        self.assertAlmostEqual(arm_config("N2_c05w0t1")["environment"]["reward_weights"]["time"], 0.05)  # Marcos's weight.
        self.assertEqual(arm_config("O3_c1w0")["environment"]["reward_weights"]["time"], 0.0)
        self.assertEqual(arm_config("O2_c1w0t4")["init_policy"], "E1")
        self.assertEqual(differences(sidecar("O3_c1w0"), sidecar("O3_c1w0t1"))["environment.reward_weights.time"], (0.0, 0.05))
        for bad in ("O3_c0w0t1", "O3_c1w01t1", "O3_c1w0t2", "N1_t1"):
            self.assertIsNone(parse_step8(bad), bad)
        check_round(sidecar("O2_c1w0t4"), sidecar("O3_c1w0t4"))
        check_round(sidecar("E1"), sidecar("O2_c1w0t4"))
        check_pipeline("O3_c1w0t4_seed7", "O3_c1w0_seed7", "time cost", loader())
        with self.assertRaises(IVError):  # Crash cost differs as well.
            check_pipeline("O3_c1w0t4_seed7", "O3_c05w0_seed7", "time cost", loader())

    def test_rounds_chain_through_init_policy(self):
        self.assertEqual(arm_config("O2_c1w01")["init_policy"], "E1")  # O shares E's first round.
        self.assertEqual(arm_config("O3_c1w01")["init_policy"], "O2_c1w01")
        self.assertEqual(arm_config("N2_c2w0_n256")["init_policy"], "N1_n256")
        self.assertIsNotNone(arm_config("O2_c1w01")["environment"]["scenario"])  # Practice in round 2.
        self.assertIsNone(arm_config("E2_c1w01")["environment"]["scenario"])     # Full task in round 2.
        for arm in STEP8_ARMS:
            expected = 3_000_000 if STEP8_ARMS[arm]["round"] == 4 else 2_000_000
            self.assertEqual(arm_config(arm)["requested_timesteps"], expected, arm)

    def test_round_four_is_three_million_more_of_round_three(self):
        # Step 8e: same task and reward as round 3, from round 3's policy; only the start and the length differ.
        for arm in ("N4_c05w0t1", "N4_c1w0t4_n256", "O4_c1w0t4", "O4_c1w0t4_n256"):
            config, before = arm_config(arm), arm_config(arm.replace("4_", "3_", 1))
            self.assertEqual(config["init_policy"], arm.replace("4_", "3_", 1))
            diffs = differences(sidecar(arm.replace("4_", "3_", 1)), sidecar(arm))
            self.assertEqual(set(diffs), {"init_policy", "requested_timesteps"}, arm)
            self.assertEqual(config["requested_timesteps"], 3_000_000)
            self.assertIsNone(config["environment"]["scenario"])
            self.assertEqual(config["environment"]["reward_weights"], before["environment"]["reward_weights"])
            check_round(sidecar(arm.replace("4_", "3_", 1)), sidecar(arm))
        # The 8e comparisons: width at every round, more training within one chain, order and reward at 256.
        check_pipeline("N4_c05w0t1_n256_seed7", "N4_c05w0t1_seed7", "width", loader())
        check_pipeline("O4_c1w0t4_n256_seed7", "O4_c1w0t4_seed7", "width", loader())
        check_pipeline("O4_c1w0t4_n256_seed7", "N4_c1w0t4_n256_seed7", "order", loader())
        check_pipeline("N4_c05w0t1_n256_seed7", "N4_c1w0t4_n256_seed7", "reward", loader())
        with self.assertRaises(IVError):  # Width and reward both differ.
            check_pipeline("N4_c05w0t1_n256_seed7", "N4_c1w0t4_seed7", "width", loader())

    def test_a_round_changes_only_task_reward_and_weights(self):
        check_round(sidecar("E1"), sidecar("E2_c1w01"))
        check_round(sidecar("E2_c2w0"), sidecar("E3_c2w0"))
        changed = sidecar("E3_c2w0")
        changed["ppo_config"]["learning_rate"] = 1e-4
        with self.assertRaises(IVError):
            check_round(sidecar("E2_c2w0"), changed)
        other_machine = sidecar("E3_c2w0")
        other_machine["platform"]["system"] = "Darwin"
        with self.assertRaises(IVError):
            check_round(sidecar("E2_c2w0"), other_machine)
        with self.assertRaises(IVError):  # Not its own previous round.
            check_round(sidecar("E2_c1w01"), sidecar("E3_c2w0"))


class PipelineTests(unittest.TestCase):
    def test_one_factor_pipelines_pass(self):
        load = loader()
        cases = [("O3_c1w01", "E3_c1w01", "round-2 task"), ("E3_c1w01", "H3_c1w01", "easy starts"),
                 ("H3_c1w01", "N3_c1w01", "practice"), ("E3_c1w01", "N3_c1w01", "practice"),
                 ("E3_c2w0", "E3_c1w0", "crash cost"), ("E3_c1w01", "E3_c1w0", "progress"),
                 ("E3_c2w01", "E3_c05w0", "reward"), ("E3_c1w01_n256", "E3_c1w01", "width")]
        for a, b, factor in cases:
            rounds = check_pipeline(f"{a}_seed7", f"{b}_seed7", factor, load)
            self.assertGreaterEqual(len(rounds), 1, (a, b))

    def test_8d_order_and_recipe_comparisons(self):
        load = loader()
        check_pipeline("O3_c1w01_seed7", "N3_c1w01_seed7", "order", load)    # Rounds 1 and 2 differ.
        check_pipeline("O3_c05w0_seed7", "N3_c1w01_seed7", "recipe", load)   # Order and reward differ.
        with self.assertRaises(IVError):  # "order" may not hide a reward difference.
            check_pipeline("O3_c05w0_seed7", "N3_c1w01_seed7", "order", load)
        with self.assertRaises(IVError):  # "recipe" may not hide a width difference.
            check_pipeline("O3_c05w0_seed7", "N3_c1w01_n256_seed7", "recipe", load)
        check_pipeline("O3_c1w0t4_n256_seed7", "N3_c05w0t1_seed7", "recipe and width", load)  # 8f: all three differ.
        other_ppo = loader()
        changed = other_ppo("O3_c1w0t4_n256_seed7")
        changed["ppo_config"]["learning_rate"] = 1e-4
        with self.assertRaises(IVError):  # "recipe and width" may not hide any other setting.
            check_pipeline("O3_c1w0t4_n256_seed7", "N3_c05w0t1_seed7", "recipe and width",
                           lambda run: changed if run == "O3_c1w0t4_n256_seed7" else other_ppo(run))
        self.assertEqual(factor_between("O_c05w0", "O_c1w01"), "reward")
        self.assertEqual(factor_between("O_c1w01", "N_c1w01"), "order")
        self.assertEqual(factor_between("E_c1w01", "O_c1w01"), "round-2 task")
        self.assertEqual(factor_between("O_c05w0", "N_c1w01"), "recipe")
        self.assertLess(simplicity("N_c1w01"), simplicity("O_c0w0"))   # Order first, then reward terms.
        self.assertLess(simplicity("O_c0w0"), simplicity("O_c05w0"))
        self.assertLess(simplicity("O_c05w0"), simplicity("O_c1w01"))

    def test_walks_back_to_round_one(self):
        rounds = check_pipeline("E3_c1w01_seed7", "H3_c1w01_seed7", "easy starts", loader())
        self.assertEqual([pair[0] for pair in rounds], ["E3_c1w01_seed7", "E2_c1w01_seed7", "E1_seed7"])

    def test_two_factors_or_none_are_refused(self):
        load = loader()
        for a, b, factor in [("E3_c2w01", "E3_c1w0", "crash cost"),   # progress differs too
                             ("O3_c1w01", "N3_c1w01", "round-2 task"),  # round 1 differs as well
                             ("E3_c1w01", "E3_c1w01_n256", "reward"),
                             ("E3_c1w01", "E3_c1w01", "reward")]:
            with self.assertRaises(IVError, msg=(a, b, factor)):
                check_pipeline(f"{a}_seed7", f"{b}_seed7", factor, load)

    def test_seeds_must_match(self):
        load7, load10 = loader(7), loader(10)
        with self.assertRaises(IVError):
            check_pipeline("E3_c1w01_seed7", "E3_c2w01_seed10", "crash cost",
                           lambda run: load7(run) if run.endswith("seed7") else load10(run))


class StatisticsTests(unittest.TestCase):
    def test_t_distribution_against_tables(self):
        self.assertAlmostEqual(t_two_sided_p(4.302653, 2), 0.05, places=5)
        self.assertAlmostEqual(t_two_sided_p(2.776445, 4), 0.05, places=5)
        self.assertAlmostEqual(t_two_sided_p(2.364624, 7), 0.05, places=5)
        self.assertAlmostEqual(t_two_sided_p(0.0, 5), 1.0, places=9)
        self.assertAlmostEqual(t_quantile(0.975, 2), 4.302653, places=4)
        self.assertAlmostEqual(betainc(2, 3, 0.4), 0.5248, places=4)

    def test_paired_summary(self):
        result = paired([0.1, 0.12, 0.08])
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertEqual(result["ahead"], 3)
        self.assertLess(result["p"], 0.05)
        self.assertLess(result["ci95"][0], 0.1)
        self.assertGreater(result["ci95"][1], 0.1)

    def test_welch_against_a_worked_example(self):
        # Two groups, means 0.7 and 0.5; Welch-Satterthwaite df and p checked by hand.
        result = welch([0.6, 0.7, 0.8, 0.7, 0.7], [0.4, 0.5, 0.6])
        self.assertAlmostEqual(result["mean"], 0.2)
        va, vb = 0.005 / 5, 0.01 / 3
        self.assertAlmostEqual(result["df"], (va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 2))
        self.assertAlmostEqual(result["p"], t_two_sided_p(0.2 / math.sqrt(va + vb), result["df"]))
        self.assertLess(result["ci95"][0], 0.2)
        self.assertGreater(result["ci95"][1], 0.2)

    def test_winner_rule_needs_significance_and_consistency(self):
        self.assertTrue(beats(paired([0.1, 0.12, 0.08])))
        self.assertFalse(beats(paired([0.3, -0.05, 0.2])))   # Large mean, inconsistent: noise.
        self.assertFalse(beats(paired([-0.1, -0.12, -0.08])))  # Consistently worse.
        self.assertFalse(beats(None))

    def test_episode_scores_and_groups(self):
        self.assertEqual([episode_score(x) for x in "SCTL"], [1, -1, 0, 0])
        tags = {1: {"slot": "back", "blocked": True}, 2: {"slot": "front", "blocked": False},
                3: {"slot": "left", "blocked": False}, 4: {"slot": "front", "blocked": False}}
        counts = group_counts({1: "S", 2: "C", 3: "T", 4: "S"}, tags)
        self.assertEqual(counts["all"]["score"], (1 - 1 + 0 + 1) / 4)
        self.assertEqual(counts["blocked"], {"n": 1, "S": 1, "C": 0, "T": 0, "score": 1.0})
        self.assertEqual(counts["front"]["C"], 1)
        self.assertTrue(math.isclose(counts["blr"]["score"], 0.5))


if __name__ == "__main__":
    unittest.main()
