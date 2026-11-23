"""Enforce one independent variable per comparison by diffing two runs' saved sidecars.

A sidecar holds a run's complete configuration. Two runs may be compared only when exactly
one setting differs (or, for a declared compound IV, when every difference lies under that
one path). Bookkeeping such as seed, run name, timestamps and runtime is not configuration.

    python -m baseline_dynamic_reach.ivcheck B0_seed7 B1_seed7
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dynamic_reach.config import ARMS, MODELS_DIR, run_name

# Everything that can change what is trained or measured. "platform" pins the machine.
COMPARED = ("environment", "ppo_config", "num_envs", "requested_timesteps", "init_policy", "versions", "platform")
MISSING = "<missing>"
# Settings added to the configuration later (such as init_policy) are absent from older sidecars.
# Absent means the setting's default, which is None, so the two compare equal.
EQUIVALENT = ({MISSING, None},)
# Recorded but not compared. The host name is set by the network on this Mac (DHCP over Wi-Fi), so
# it changes between sessions on the same machine; the operating system, its release, the processor
# and the Python version still pin the machine. (Decision log, 2026-11-15.)
BOOKKEEPING = {"platform.node"}


def same(old, new):
    return old == new or any(old in group and new in group for group in EQUIVALENT
                             if isinstance(old, (str, type(None))) and isinstance(new, (str, type(None))))


class IVError(ValueError):
    """Two runs differ in a way that would mix more than one effect (or in no way at all)."""


def flatten(tree, prefix=""):
    """Dotted-path leaves. Lists (speed grids, network sizes, ranges) are single settings."""
    if not isinstance(tree, dict):
        return {prefix: tree}
    leaves = {}
    for key, value in tree.items():
        leaves.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return leaves


def differences(parent, child):
    """{path: (parent value, child value)} over the compared sections."""
    old = flatten({key: parent[key] for key in COMPARED if key in parent})
    new = flatten({key: child[key] for key in COMPARED if key in child})
    return {path: (old.get(path, MISSING), new.get(path, MISSING))
            for path in sorted((set(old) | set(new)) - BOOKKEEPING)
            if not same(old.get(path, MISSING), new.get(path, MISSING))}


def describe(diffs):
    return "\n".join(f"  {path}: {old!r} -> {new!r}" for path, (old, new) in diffs.items()) or "  (no differences)"


def check(parent, child, iv=None):
    """Return the differences if they form exactly one IV; raise IVError otherwise."""
    diffs = differences(parent, child)
    if not diffs:
        raise IVError("The runs have identical configurations: there is no IV to compare.")
    if iv is None:
        if len(diffs) != 1:
            raise IVError(f"{len(diffs)} settings differ; a comparison may change exactly one:\n{describe(diffs)}")
    else:
        outside = {path: change for path, change in diffs.items() if path != iv and not path.startswith(iv + ".")}
        if outside:
            raise IVError(f"Declared IV is {iv!r}, but other settings also differ:\n{describe(outside)}")
    return diffs


def check_replicates(sidecars):
    """Seeds of one arm are replicates: their configurations must be identical."""
    first = sidecars[0]
    for other in sidecars[1:]:
        diffs = differences(first, other)
        if diffs:
            raise IVError(f"Replicates seed {first['seed']} and seed {other['seed']} differ:\n{describe(diffs)}")


# Step 8: a round continues its init run (the previous round, same seed). It may change only these
# settings: the task, the reward terms under test, its starting weights, and its length (round 4 of Step 8e
# is 3M; train.py's launch check still requires every run to match its generated configuration exactly).
ROUND_SETTINGS = ("environment.scenario", "environment.reward_weights.collision",
                  "environment.reward_weights.potential_progress", "environment.reward_weights.time", "init_policy",
                  "requested_timesteps")
REWARD_TERMS = ("environment.reward_weights.collision", "environment.reward_weights.potential_progress",
                "environment.reward_weights.time")
# The factors Step 8 compares: the settings each may change, and in which rounds (None: any round).
FACTORS = {
    "crash cost": (("environment.reward_weights.collision",), {2, 3, 4}),
    "progress": (("environment.reward_weights.potential_progress",), {2, 3, 4}),
    "time cost": (("environment.reward_weights.time",), {2, 3, 4}),
    "reward": (REWARD_TERMS, {2, 3, 4}),
    "width": (("ppo_config.policy_kwargs.net_arch",), None),
    "round-2 task": (("environment.scenario",), {2}),            # O vs E: practice or the full task in round 2.
    "easy starts": (("environment.scenario.start_offsets",), {1}),  # E vs H.
    "practice": (("environment.scenario",), {1}),                # H vs N, E vs N: practice or the full task in round 1.
    "order": (("environment.scenario",), {1, 2}),                # Any two training orders at the same reward (O vs N, O vs H).
    # 8d only: two finalists that differ in both order and reward. This compares whole recipes to choose one;
    # it cannot say which of the two settings caused a difference. Everything else must still be identical.
    "recipe": (("environment.scenario", *REWARD_TERMS), None),
    # 8f only: the 256-wide candidate recipe against the 64-wide winner (order, reward and width may all differ).
    # Also a choice between whole recipes; PPO settings, machine, packages and seed must still be identical.
    "recipe and width": (("environment.scenario", *REWARD_TERMS, "ppo_config.policy_kwargs.net_arch"), None),
}


def within(path, prefixes):
    return any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)


def check_round(init_sidecar, child):
    """A Step 8 round may differ from the run it starts from only in ROUND_SETTINGS (same machine, packages, PPO)."""
    if child.get("init_policy") != init_sidecar.get("arm") or child.get("seed") != init_sidecar.get("seed"):
        raise IVError(f"A round must start from its init run on the same seed: init_policy {child.get('init_policy')!r} "
                      f"seed {child.get('seed')}, init run {init_sidecar.get('arm')!r} seed {init_sidecar.get('seed')}.")
    diffs = differences(init_sidecar, child)
    outside = {path: change for path, change in diffs.items() if not within(path, ROUND_SETTINGS)}
    if outside:
        raise IVError("A round may change only the task, the reward terms and its starting weights; also differs:\n"
                      f"{describe(outside)}")
    return diffs


def check_pipeline(run_a, run_b, factor, load=None):
    """Two runs, each the last round of a chain linked by init_policy, differ only in `factor` at every round.

    Walks both chains back one round at a time (same seed) until they start from the same run or both
    from fresh weights. At each round the two sidecars may differ only in the factor's settings and in
    init_policy. Returns [(run a, run b, differences)] from the last round back; raises IVError otherwise.
    """
    load = load or load_sidecar
    prefixes, allowed_rounds = FACTORS.get(factor, ((factor,), None))
    a, b = load(run_a), load(run_b)
    if a["seed"] != b["seed"]:
        raise IVError(f"Pipelines are compared on the same seed: {a['seed']} vs {b['seed']}.")
    rounds, changed = [], False
    while True:
        diffs = differences(a, b)
        outside = {path: change for path, change in diffs.items() if path != "init_policy" and not within(path, prefixes)}
        if outside:
            raise IVError(f"{a['run_name']} vs {b['run_name']}: besides {factor!r}, these differ:\n{describe(outside)}")
        setting_changed = any(path != "init_policy" for path in diffs)
        round_number = ARMS.get(a.get("arm"), {}).get("round")
        if setting_changed and allowed_rounds is not None and round_number not in allowed_rounds:
            raise IVError(f"{a['run_name']} vs {b['run_name']}: {factor!r} may differ only in round(s) "
                          f"{sorted(allowed_rounds)}, but differs in round {round_number}:\n{describe(diffs)}")
        rounds.append((a["run_name"], b["run_name"], diffs))
        changed |= setting_changed
        init_a, init_b = a.get("init_policy"), b.get("init_policy")
        if init_a == init_b:  # The chains join here (the same starting run, or both fresh).
            break
        if init_a is None or init_b is None:
            raise IVError(f"{a['run_name']} and {b['run_name']}: one chain starts from fresh weights, the other does not.")
        a, b = load(run_name(init_a, a["seed"])), load(run_name(init_b, b["seed"]))
    if not changed:
        raise IVError(f"{run_a} and {run_b} are the same pipeline: {factor!r} never differs.")
    return rounds


def load_sidecar(run):
    path = Path(run) if str(run).endswith(".json") else MODELS_DIR / f"{run}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", help="Parent run name (or path to its .json sidecar).")
    parser.add_argument("child", help="Child run name (or path to its .json sidecar).")
    parser.add_argument("--iv", help="Declared compound IV path; defaults to the child's own declaration.")
    args = parser.parse_args()
    parent, child = load_sidecar(args.parent), load_sidecar(args.child)
    diffs = check(parent, child, args.iv or child.get("iv"))
    print(f"One IV between {args.parent} and {args.child}:\n{describe(diffs)}")


if __name__ == "__main__":
    main()
