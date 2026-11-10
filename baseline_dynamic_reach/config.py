"""Dynamic-waypoint *reach*: the experiment chain, one independent variable (IV) per arm.

An arm is its parent's full configuration with exactly one setting replaced, so every
comparison between an arm and its parent isolates one effect. B0 is the root.

B0 inherits everything unchanged from the last dynamic-waypoint runs in
baseline_dynamic_waypoint/ (their sidecars record 3 environments and 1,000,000 requested
transitions), except the two settings that define this study:

- the task is *reach*: hold_seconds = 0.1, i.e. one decision inside the slot box;
- the reward is R0: 50 x success, with every other inherited term at weight 0.

EXPERIMENT_NOTES.md gives the reason behind every value.
"""
from copy import deepcopy
from pathlib import Path

from baseline_dynamic_waypoint.config import (ENV_CONFIG as DYNAMIC_ENV_CONFIG, GOAL_CONFIG as DYNAMIC_GOAL_CONFIG,
                                              PPO_CONFIG as DYNAMIC_PPO_CONFIG, EVALUATION_SEED, SLOTS,  # noqa: F401
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

ROOT = {
    "environment": {
        "env_config": deepcopy(DYNAMIC_ENV_CONFIG),
        "goal_config": {**deepcopy(DYNAMIC_GOAL_CONFIG), "hold_seconds": 0.1},  # reach: 1 decision
        "reward_weights": dict(R0),
    },
    "ppo_config": deepcopy(DYNAMIC_PPO_CONFIG),
    "num_envs": 3,  # As in the earlier dynamic-waypoint runs: 3 x 256 = 768 transitions per PPO update.
    "requested_timesteps": 1_000_000,  # Rounds up to 1,000,704 with whole rollouts.
    # None: random initial weights. An arm name: start from that arm's trained policy on the same seed.
    "init_policy": None,
}

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
    arm = ARMS[name]
    if arm["parent"] is None:
        return deepcopy(ROOT)
    config = arm_config(arm["parent"])
    if arm["iv"] != MACHINE_IV:
        set_path(config, arm["iv"], arm["value"])
    return config


def run_name(arm, seed):
    return experiment_name(f"{arm}_seed{seed}")
