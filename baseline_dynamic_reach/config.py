"""Dynamic-waypoint *reach*: the experiment chain, one independent variable (IV) per arm.

An arm is its parent's full configuration with exactly one setting replaced, so every
comparison between an arm and its parent isolates one effect. B0 is the root.

B0 inherits everything unchanged from the last dynamic-waypoint runs in
baseline_dynamic_waypoint/ (their sidecars record 3 environments and 1,000,000 requested
transitions), except the two settings that define this study:

- the task is *reach*: hold_seconds = 0.1, i.e. one decision inside the slot box;
- the reward is R0: 50 x success, with every other inherited term at weight 0.

The recorded B0 sidecar freezes these historical settings. Later changes to the
waypoint defaults must not change this study's arms or checkpoint contracts.
EXPERIMENT_NOTES.md gives the reason behind every value.
"""
from copy import deepcopy
import json
from pathlib import Path

from baseline_dynamic_waypoint.config import (EVALUATION_SEED, SLOTS,  # noqa: F401
                                              experiment_name)  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
SMOKE_DIR = BASE_DIR / "smoke"

# Paired replicates: every arm trains on these seeds. Training workers use seed + 0..2,
# so they never overlap the evaluation seeds below.
SEEDS = (7, 8, 9)
# Fixed measurement for the whole chain: the same 200 naturally sampled episodes for every
# policy and every arm. The first 50 are the seeds the earlier studies evaluated on.
EVAL_EPISODES = 200
EVAL_SEED = EVALUATION_SEED  # 10,000

# R0: the scored outcome and nothing else. The goal keeps Marcos's magnitude (50), so a later
# arm that switches on one of his terms changes that one term, not also the overall scale.
# The other inherited terms stay listed at weight 0 because the inherited environment reads them.
R0 = {"goal": 50.0, "collision": 0.0, "progress": 0.0, "time": 0.0, "hold": 0.0}

# Use the archived experiment specification rather than today's waypoint defaults.
# In particular, the reach study used a 35 m FRONT slot, LR 3e-4, and 64-unit
# networks; the independent single-car study now uses 20 m, 1e-4, and 256 units.
# Reading the tracked JSON does not load or execute model weights.
ROOT_SIDECAR = MODELS_DIR / "B0_seed7.json"
with ROOT_SIDECAR.open(encoding="utf-8") as handle:
    _recorded_root = json.load(handle)
ROOT = {key: deepcopy(_recorded_root[key])
        for key in ("environment", "ppo_config", "num_envs", "requested_timesteps")}
# Older B0/B1 sidecars predate this setting; absent means fresh initialization.
ROOT["init_policy"] = _recorded_root.get("init_policy")

# The one IV that is not a setting: the machine a run trains on. train.py records it in every
# sidecar ("platform": operating system, processor, host name and Python version, which change
# together), and ivcheck compares it like any other setting. An arm with this IV has exactly its
# parent's configuration and must be trained on a different machine than its parent.
MACHINE_IV = "platform"

# Each arm: parent arm, the single IV as a dotted path into the configuration, its new value,
# and the reason. A value may be a dict (one compound setting); it is still one IV, and every
# changed leaf must sit under that path (ivcheck enforces this on the saved sidecars).
ARMS = {
    "B0": {"parent": None, "iv": None, "value": None,
           "reason": "Root of the chain: reach a slot of a moving car, reward R0 = 50 x success."},
    "B1": {"parent": "B0", "iv": "environment.reward_weights.collision", "value": 50.0,
           "reason": "B0 fails only by crashing (42-47%), almost always rear-ending a slower car directly ahead "
                     "(FRONT: the anchor itself). R0 scores a crash like a timeout, so nothing teaches avoiding it. "
                     "50 is Marcos's crash weight, symmetric with the goal."},
    "B2": {"parent": "B1", "iv": "init_policy", "value": "B0",
           "reason": "B1 trained from scratch collapses to never driving toward the goal (100% timeouts from "
                     "~40k transitions): from a near-random start a crash is ~7x likelier than success, so with "
                     "+/-50 giving up (0) beats trying (~-34). B0's policy already reaches 55%, so under R1 trying "
                     "is worth ~+5 from the start; the crash penalty can then teach it to avoid crashes."},
    "B3": {"parent": "B2", "iv": "environment.reward_weights.potential_progress",
           "value": {"weight": 0.1, "gamma": 0.99},
           "reason": "B2 abandons FRONT (62/62 timeouts, 0 successes vs scripted 40/62): nothing pays for the "
                     "long pass before the goal. Potential-based progress (Phi = -distance, gamma = PPO's) gives "
                     "per-step credit for closing on the slot without changing the best policy. 0.1 per meter is "
                     "Marcos's progress scale."},
    "B3m": {"parent": "B3", "iv": MACHINE_IV, "value": None,
            "reason": "The study moves from the Windows PC to the Mac. B3 is repeated here with nothing else "
                      "changed, so that later arms have a parent trained on the same machine and the size of the "
                      "machine effect is measured instead of assumed. It starts from the same B0 weight files as B3."},
    "B4": {"parent": "B3m", "iv": "environment.reward_weights.collision", "value": 150.0,
           "reason": "Step 5: on BACK/LEFT/RIGHT the only failure is driving into a slower car in the slot's lane "
                     "(25-28 crashes of 138 per seed; the scripted driver 3). The critic tells blocked from clear "
                     "episodes, but at crash weight 50 ramming (wins about half the time) and waiting are worth about "
                     "the same, and the training crash rate stays near 20% for 1M transitions. At 150 a crash costs "
                     "3x the goal, so ramming at even odds is clearly worse than waiting. Approved 2026-11-09."},
    "B5": {"parent": "B3m", "iv": "environment.reward_scale", "value": 0.02,
           "reason": "Step 7a audit: SB3 clips one gradient norm (0.5) over the policy and value networks together. "
                     "With rewards of +/-50 the value gradient is 30-120x the policy gradient, so every minibatch is "
                     "scaled by 0.01-0.04 and one PPO update moves the policy ~5x less (KL 0.002 vs 0.011 without "
                     "the clip, B3m). Scaling the whole reward by 1/50 (goal +1, crash -1) cannot change the best "
                     "policy and shrinks value gradients ~50x. The inherited critic's output layer is scaled by the "
                     "same factor, so the starting policy and its value estimates are unchanged in the new units. "
                     "Approved 2026-11-15."},
    "P1": {"parent": "B5", "iv": "environment.scenario", "value": {"require_blocked": True, "blocker_stays": True},
           "reason": "User's hypothesis (2026-11-15): the training set is too diverse. In most episodes 'go to the slot's "
                     "lane and accelerate' is right, and that habit traps PPO in the few that need a pass; every arm from B0 "
                     "to B5 reaches the same ~113 BACK/LEFT/RIGHT episodes as the scripted driver and fails the rest. "
                     "P1 trains only on blocked BACK/LEFT/RIGHT episodes, with the blocking car held in its lane, so a pass "
                     "is the only way to succeed. Parent B5: its updates are not throttled by the gradient clip."},
    "P1s": {"parent": "P1", "iv": "init_policy", "value": None,
            "reason": "The user asked to compare both starts: P1 begins from B0's policy and its 'floor it in the slot's "
                      "lane' habit; P1s begins from random weights, with no habit but with B1's risk of giving up."},
    "P2": {"parent": "P1s", "iv": "environment.reward_weights.potential_progress", "value": {"weight": 0.0, "gamma": 0.99},
           "reason": "User's question (2026-11-15): is the progress reward necessary? Test removing it, one IV, on the "
                     "passing task with fresh weights (no B0 habit). Weight 0 switches the term off; the earlier "
                     "full-task test (B2 -> B3) showed only a small effect."},
    "P3": {"parent": "P2", "iv": "environment.scenario.start_offsets", "value": ["past", "beside", "behind", "natural"],
           "reason": "P1, P1s and P2 all end by waiting. With gamma 0.99 at 10 Hz a failed pass crashes at ~2-3 s "
                     "(weight 0.74) and a pass succeeds at ~8 s (0.45), so trying beats waiting only above ~62% success, "
                     "and learners start far below that. Mixed practice starts (just past, beside, behind the slow car, "
                     "or the normal start; chosen by the user over a schedule) make the end of a pass easy and frequent, "
                     "so its value can chain back to the normal start. No reward term added. Approved 2026-11-15."},
    "P4": {"parent": "P3", "iv": "environment.reward_weights.collision", "value": 0.0,
           "reason": "P3 also gave up: from fresh weights about 70% of early attempts crash, even from the easy starts, "
                     "so with crash = -goal stopping is learned before passing (B1's collapse). The reach chain's working "
                     "pattern is success first (B0), crash cost afterwards from that policy (B2). P4 is the first half: "
                     "success-only reward on P3's task, so waiting is never better than trying. Approved 2026-11-15."},
    "P5": {"parent": "P3", "iv": "init_policy", "value": "P4",
           "reason": "Second half of Step 7f, as B1 -> B2: P3's task and reward (crash = -goal), but training starts "
                     "from P4's policy, which passes (34-50% of P2's natural-start episodes, 70-80% on its own mixed "
                     "starts) and crashes in the rest. One IV against P3; approved with Step 7f on 2026-11-15."},
    "P6": {"parent": "B5", "iv": "init_policy", "value": "P5",
           "reason": "P5 was trained only on passing practice, yet on the full task (B5's settings) seeds 7 and 8 reach "
                     "119 and 121 of 138 BACK/LEFT/RIGHT (B0-B5 and scripted: 113) and 21-27 of 62 FRONT (B5: 0). P6 "
                     "trains on the full task itself, starting from P5's policy, so it learns when a pass is needed. "
                     "One IV against B5, the full-task baseline. Approved by the user 2026-11-15."},
}


def set_path(tree, path, value):
    """Replace one setting. Intermediate sections must exist; the last key may be new."""
    keys = path.split(".")
    node = tree
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"{path}: no section {key!r} in the parent configuration")
        node = node[key]
    node[keys[-1]] = deepcopy(value)


def arm_config(name):
    """The full configuration of an arm: its parent's, with its one IV applied."""
    if name not in ARMS:
        raise KeyError(f"Unknown arm {name!r}; defined arms: {', '.join(ARMS)}")
    if name in STEP8_ARMS:  # Generated from the Step 8 tables below.
        return step8_config(name)
    arm = ARMS[name]
    if arm["parent"] is None:
        return deepcopy(ROOT)
    config = arm_config(arm["parent"])
    if arm["iv"] != MACHINE_IV:
        set_path(config, arm["iv"], arm["value"])
    return config


def run_name(arm, seed):
    return experiment_name(f"{arm}_seed{seed}")


# ---------------------------------------------------------------------------------------------
# Step 8: settle the recipe (network width, training order, reward). EXPERIMENT_NOTES.md, Step 8.
#
# A new chain on the Windows PC. Every pipeline has three rounds of 2,000,000 transitions; round 1
# starts from fresh weights with the success-only reward, and each later round starts from the
# previous round's policy (same seed). Arms are generated from the tables below, not written one by
# one. Names: <order><round>[_<reward>][_<width>], e.g. E1, O2_c1w01, E3_c05w0, E1_n256.
# Comparisons between pipelines are checked with ivcheck.check_pipeline: two runs may differ only in
# the factor being compared, at every round.
# Step 8e (after 8d) adds round 4: 3,000,000 more transitions of round 3's training (same task and
# reward, from round 3's final policy), so the last stage on the full task is 5M instead of 2M.
# ---------------------------------------------------------------------------------------------
STEP8_ROUND_TIMESTEPS = 2_000_000
STEP8_ROUNDS = (1, 2, 3, 4)
ROUND_LENGTHS = {1: STEP8_ROUND_TIMESTEPS, 2: STEP8_ROUND_TIMESTEPS, 3: STEP8_ROUND_TIMESTEPS, 4: 3_000_000}
STEP8_SCREEN_SEEDS = (7, 10, 13)  # Spaced by 3: a run's workers use seed, seed+1, seed+2, so no two share a layout stream.
STEP8_CONFIRM_SEEDS = (16, 19, 22, 25, 28)
STEP8_CHECKPOINT_EVERY = 200_000

PRACTICE = {"require_blocked": True, "blocker_stays": True}  # Blocked BACK/LEFT/RIGHT episodes, natural start.
EASY_STARTS = {**PRACTICE, "start_offsets": ["past", "beside", "behind", "natural"]}  # P3/P4's practice task.
# Task of each round (None = the full task). Round 1 is always the success-only reward; round 4 repeats round 3.
ORDERS = {
    "N": (None, None, None, None),                # No practice: the "one more B".
    "H": (PRACTICE, None, None, None),            # Practice from the natural start only.
    "E": (EASY_STARTS, None, None, None),         # Practice with easy starts, then the full task.
    "O": (EASY_STARTS, EASY_STARTS, None, None),  # The original order: P4 -> P5 -> P6.
}
CRASH_LEVELS = {"c0": 0.0, "c05": 0.5, "c1": 1.0, "c2": 2.0}  # Crash cost as a multiple of the goal.
PROGRESS_LEVELS = {"w0": 0.0, "w01": 0.1}                     # Potential-based progress, per meter before scaling.
REWARDS = [f"{c}{w}" for c in CRASH_LEVELS for w in PROGRESS_LEVELS]
P6_REWARD = "c1w01"                                            # P6's reward: crash = goal, progress 0.1.
# Step 8c2 (added after 8c): a time cost per decision (0.1 s), as a multiple of the goal; the inherited
# "time" term (Marcos's weight 0.05 before scaling is t1). Only with a crash cost: with crash 0, any time
# cost makes crashing cheaper than waiting. t4 keeps a whole timed-out episode (250 decisions, discounted:
# 0.004 x 91.9 = 0.37) below the smallest crash cost tested (0.5), so a crash never pays better than waiting.
TIME_LEVELS = {"t1": 0.001, "t4": 0.004}
TIME_REWARDS = [f"{c}w0{t}" for c in ("c05", "c1") for t in TIME_LEVELS]
ALL_REWARDS = REWARDS + TIME_REWARDS


def reward_parts(reward):
    """('c1', 'w0', 't1') for 'c1w0t1'; the time part is '' when there is no time cost."""
    crash, rest = reward.split("w", 1)
    progress, _, time = rest.partition("t")
    return crash, "w" + progress, f"t{time}" if time else ""
WIDTHS = {"": [64, 64], "n256": [256, 256]}
GOAL_WEIGHT = 50.0  # Unscaled goal; reward_scale 0.02 makes it +1.


def step8_name(order, round_number, reward=None, width=""):
    if round_number == 1:
        order = "E" if order == "O" else order  # O's first round is E's first round: the same run.
        name = f"{order}1"
    else:
        name = f"{order}{round_number}_{reward}"
    return f"{name}_{width}" if width else name


def parse_step8(name):
    """(order, round, reward or None, width) of a Step 8 arm name; None if the name is not one."""
    parts = name.split("_")
    head = parts[0]
    if len(head) != 2 or head[0] not in ORDERS or head[1] not in "".join(map(str, STEP8_ROUNDS)):
        return None
    order, round_number = head[0], int(head[1])
    rest = parts[1:]
    width = rest.pop() if rest and rest[-1] in WIDTHS and rest[-1] else ""
    reward = rest.pop(0) if rest else None
    if rest or (round_number == 1) != (reward is None) or (reward is not None and reward not in ALL_REWARDS):
        return None
    if round_number == 1 and order == "O":
        return None
    return order, round_number, reward, width


def step8_config(name):
    """The full configuration of a Step 8 arm, built from P4's recorded configuration."""
    order, round_number, reward, width = parse_step8(name)
    config = arm_config("P4")  # Practice with easy starts, success only, fresh weights, reward scale 1/50.
    config["requested_timesteps"] = ROUND_LENGTHS[round_number]
    config["environment"]["scenario"] = deepcopy(ORDERS[order][round_number - 1])
    config["ppo_config"]["policy_kwargs"]["net_arch"] = {"pi": list(WIDTHS[width]), "vf": list(WIDTHS[width])}
    weights = config["environment"]["reward_weights"]
    if reward is None:
        weights["collision"] = 0.0
        weights["potential_progress"] = {"weight": 0.0, "gamma": config["ppo_config"]["gamma"]}
        config["init_policy"] = None
    else:
        crash, progress, time = reward_parts(reward)
        weights["collision"] = CRASH_LEVELS[crash] * GOAL_WEIGHT
        weights["potential_progress"] = {"weight": PROGRESS_LEVELS[progress], "gamma": config["ppo_config"]["gamma"]}
        if time:  # Rewards without a time cost keep P4's recorded time weight (0), untouched.
            weights["time"] = TIME_LEVELS[time] * GOAL_WEIGHT
        config["init_policy"] = step8_name(order, round_number - 1, reward, width)
    return config


def _step8_arms():
    arms = {}
    for width in WIDTHS:
        for order in ORDERS:
            for round_number in STEP8_ROUNDS:
                for reward in ([None] if round_number == 1 else ALL_REWARDS):
                    name = step8_name(order, round_number, reward, width)
                    if name in arms:
                        continue
                    task = ORDERS[order][round_number - 1]
                    task_text = "the full task" if task is None else (
                        "practice with easy starts" if "start_offsets" in task else "practice from the natural start")
                    if round_number == 1:
                        # Round-1 arms are one IV from E1 (the root), so train.py can check them like any arm.
                        parent, iv, value = {
                            "E": (None, None, None), "H": ("E1", "environment.scenario.start_offsets", None),
                            "N": ("E1", "environment.scenario", None)}[order]
                        if width:
                            parent, iv, value = step8_name(order, 1), "ppo_config.policy_kwargs.net_arch", None
                        reason = f"Step 8, round 1 of order {order}: {task_text}, success-only reward, fresh weights."
                    else:
                        parent, iv, value = step8_name(order, round_number - 1, reward, width), "round", None
                        reason = (f"Step 8, round {round_number} of order {order}: {task_text}, reward {reward}, "
                                  f"from the previous round's policy"
                                  + (", 3M more of round 3's training (Step 8e)." if round_number == 4 else "."))
                    arms[name] = {"parent": parent, "iv": iv, "value": value, "reason": reason, "step": 8,
                                  "order": order, "round": round_number, "reward": reward, "width": width or "n64"}
    return arms


STEP8_ARMS = _step8_arms()
ARMS.update(STEP8_ARMS)
