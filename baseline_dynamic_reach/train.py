"""Train one arm of the reach chain on one seed, then evaluate it with the fixed measurement.

    python -B baseline_dynamic_reach/train.py --arm B0 --seed 7

Before training, the arm's configuration is checked against its parent's saved runs (exactly
one IV may differ) and against its own other seeds (replicates must be identical).
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_dynamic_reach.config import ARMS, EVAL_EPISODES, MODELS_DIR, RESULTS_DIR, SMOKE_DIR, arm_config, run_name
from baseline_dynamic_reach.evaluate import evaluate_run
from baseline_dynamic_reach.ivcheck import IVError, check, check_replicates, describe
from baseline_dynamic_reach.environment import SHAPING_TERMS, ReachEnv
from baseline_dynamic_waypoint.train import SlotEpisodeMetrics
from baseline_waypoint.plot_results import plot_training

PACKAGES = ("highway-env", "gymnasium", "stable-baselines3", "torch", "numpy")


def ignore_console_interrupts():
    """Runs are detached; a stray console Ctrl+C/Ctrl+Break must not kill them.

    On 2026-11-08 two batches of runs died seconds after a Windows session transition
    (Kernel-Power event 566) delivered a console interrupt: the env workers exited, and the
    learner lost its pipes. Stop a run with taskkill instead. Nothing computed changes.
    """
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)


def make_env(seed, environment):
    """Every setting comes from the arm's configuration, including env_config."""
    ignore_console_interrupts()  # This runs inside each worker process.
    env = ReachEnv(**deepcopy(environment))
    env.reset(seed=seed)
    return Monitor(env)


class ReachEpisodeMetrics(SlotEpisodeMetrics):
    """The inherited per-episode CSV, with one reward column per term this arm actually has."""
    def __init__(self, run_dir, reward_names):
        super().__init__(run_dir)
        self.FIELDS = ([field for field in SlotEpisodeMetrics.FIELDS if not field.startswith("reward_")]
                       + [f"reward_{name}" for name in reward_names])


def platform_info():
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "processor": platform.processor(), "node": platform.node(), "python": platform.python_version()}


def sidecars(models_dir, arm):
    found = []
    for path in sorted(Path(models_dir).glob(f"{arm}_seed*.json")):
        with path.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
        if sidecar.get("arm") == arm and sidecar.get("status") == "trained":
            found.append(sidecar)
    return found


def check_shaping(config):
    """Potential-based shaping keeps the best policy only if its gamma is PPO's discount."""
    for name in SHAPING_TERMS:
        spec = config["environment"]["reward_weights"].get(name)
        if isinstance(spec, dict) and spec.get("gamma") != config["ppo_config"]["gamma"]:
            raise ValueError(f"{name}: gamma {spec.get('gamma')} must equal PPO's gamma {config['ppo_config']['gamma']}.")


def check_chain(metadata, models_dir, parent_in_progress=False):
    """Refuse to train an arm that would not be a one-IV comparison with its parent.

    Normally the parent's saved sidecars are the reference. With parent_in_progress, a parent
    that is still training is represented by its defined configuration on this machine; the
    saved sidecars must then be checked with ivcheck once the parent finishes.
    """
    arm = ARMS[metadata["arm"]]
    if arm["parent"] is not None:
        parents = sidecars(models_dir, arm["parent"])
        if not parents and parent_in_progress:
            parents = [{**metadata, **arm_config(arm["parent"]), "seed": metadata["seed"]}]
            metadata["parent_check"] = "defined configuration; parent still training (re-check with ivcheck)"
        elif not parents:
            raise IVError(f"Train parent arm {arm['parent']} before {metadata['arm']}.")
        else:
            metadata["parent_check"] = "saved sidecar"
        same_seed = [p for p in parents if p["seed"] == metadata["seed"]]
        diffs = check((same_seed or parents)[0], metadata, arm["iv"])
        print(f"IV against {arm['parent']} ({metadata['parent_check']}):\n{describe(diffs)}", flush=True)
    siblings = sidecars(models_dir, metadata["arm"])
    if siblings:
        check_replicates([metadata, *siblings])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="4,096 transitions and 8 evaluation episodes, saved under smoke/ and outside the chain.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--parent-in-progress", action="store_true",
                        help="The parent arm is still training: check against its defined configuration.")
    args = parser.parse_args()
    ignore_console_interrupts()

    config = arm_config(args.arm)
    eval_episodes = EVAL_EPISODES
    models_dir, results_dir = MODELS_DIR, RESULTS_DIR
    if args.smoke:
        config["requested_timesteps"] = 4096
        eval_episodes = 8
        models_dir, results_dir = SMOKE_DIR / "models", SMOKE_DIR / "results"
    name = run_name(args.arm, args.seed)
    run_dir, model_path = results_dir / name, models_dir / f"{name}.zip"
    if model_path.exists() or (run_dir / "training_episodes.csv").exists():
        parser.error(f"{name} already exists; results are never overwritten.")
    arm = ARMS[args.arm]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_name": name, "arm": args.arm,
        "parent_arm": arm["parent"], "iv": arm["iv"], "iv_value": arm["value"], "reason": arm["reason"],
        "seed": args.seed, "smoke": args.smoke, **config,
        "versions": {package: importlib.metadata.version(package) for package in PACKAGES},
        "platform": platform_info(), "status": "running",
    }
    check_shaping(config)
    if not args.smoke:
        check_chain(metadata, models_dir, args.parent_in_progress)
    init_path = None
    if config.get("init_policy"):
        init_path = models_dir / f"{run_name(config['init_policy'], args.seed)}.zip"
        init_sidecar = init_path.with_suffix(".json")
        if not init_path.exists() or json.loads(init_sidecar.read_text(encoding="utf-8")).get("status") != "trained":
            parser.error(f"init_policy {config['init_policy']}: no trained model at {init_path}")
        metadata["init_policy_file"] = init_path.name
        metadata["init_policy_sha256"] = hashlib.sha256(init_path.read_bytes()).hexdigest()
    environment = config["environment"]
    ReachEnv(**deepcopy(environment)).close()  # Reject an invalid task before spawning workers.

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Tiny MLPs are faster without a thread pool per inference.
    set_random_seed(args.seed)
    factories = [partial(make_env, args.seed + index, environment) for index in range(config["num_envs"])]
    vec_env = DummyVecEnv(factories) if config["num_envs"] == 1 else SubprocVecEnv(factories, start_method="spawn")
    vec_env.seed(args.seed)
    print(f"{name}: {config['requested_timesteps']:,} transitions on {config['num_envs']} environments; "
          f"reward {environment['reward_weights']}; hold {environment['goal_config']['hold_seconds']} s", flush=True)
    model = PPO("MultiInputPolicy", vec_env, seed=args.seed, verbose=0, **config["ppo_config"])
    if init_path is not None:
        # Policy and value networks start from the parent arm's trained weights (same seed);
        # the Adam optimizer starts fresh, as in any new PPO run.
        model.policy.load_state_dict(PPO.load(init_path, device="cpu").policy.state_dict())
        print(f"Initial weights: {init_path.name}", flush=True)
    model.set_logger(configure(str(run_dir), ["csv"] + ([] if args.no_tensorboard else ["tensorboard"])))
    callback = ReachEpisodeMetrics(run_dir, list(environment["reward_weights"]))
    started = time.perf_counter()
    status = "trained"
    try:
        model.learn(total_timesteps=config["requested_timesteps"], callback=callback)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted; saving the current policy and the completed episodes.", flush=True)
    except Exception:
        status = "failed"
        raise
    finally:
        model.save(model_path)
        metadata.update(actual_timesteps=model.num_timesteps, training_seconds=time.perf_counter() - started,
                        status=status)
        for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        if callback.handle is not None and not callback.handle.closed:
            callback.handle.close()
        vec_env.close()
    plot_training(run_dir)
    print(f"Saved {model_path}: {model.num_timesteps:,} transitions in {metadata['training_seconds'] / 60:.1f} min.", flush=True)
    if status == "trained":
        evaluate_run(model_path, run_dir, episodes=eval_episodes)


if __name__ == "__main__":
    main()
