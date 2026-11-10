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

from baseline_dynamic_reach.config import MODELS_DIR

# Everything that can change what is trained or measured. "platform" pins the machine.
COMPARED = ("environment", "ppo_config", "num_envs", "requested_timesteps", "init_policy", "versions", "platform")
MISSING = "<missing>"
# Settings added to the configuration later (such as init_policy) are absent from older sidecars.
# Absent means the setting's default, which is None, so the two compare equal.
EQUIVALENT = ({MISSING, None},)


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
            for path in sorted(set(old) | set(new)) if not same(old.get(path, MISSING), new.get(path, MISSING))}


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
