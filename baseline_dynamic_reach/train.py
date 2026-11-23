"""Train one arm of the reach chain on one seed, then evaluate it with the fixed measurement.

    python -B baseline_dynamic_reach/train.py --arm B0 --seed 7

Before training, the arm's configuration is checked against its parent's saved runs (exactly
one IV may differ) and against its own other seeds (replicates must be identical).
"""
import argparse
import csv
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

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_dynamic_reach.config import ARMS, EVAL_EPISODES, MODELS_DIR, RESULTS_DIR, SMOKE_DIR, arm_config, run_name
from baseline_dynamic_reach.evaluate import evaluate_run
from baseline_dynamic_reach.ivcheck import IVError, check, check_replicates, check_round, describe
from baseline_dynamic_reach.provenance import code_provenance
from baseline_dynamic_reach.environment import SHAPING_TERMS, ReachEnv
from baseline_waypoint.train import EpisodeMetrics
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


class ReachEpisodeMetrics(EpisodeMetrics):
    """Per-episode CSV and rolling rates, with one reward column per term this arm actually has.

    A frozen copy of baseline_dynamic_waypoint.train.SlotEpisodeMetrics as of commit 8b2c911, the version
    every arm up to B4 was trained with. That class has since gained required arguments and its own
    checkpointing for the single-car study; copying it keeps this study's logs and checkpoints unchanged
    whatever happens there (the same reason config.py reads its settings from the recorded B0 sidecar).
    It only records what happened; it does not influence training.
    """
    BASE_FIELDS = ["timesteps", "reward", "length", "success", "collision", "timeout", "anchor_lost",
                   "elapsed_time", "final_distance", "slot", "goal_lane", "max_hold_steps"]

    def __init__(self, run_dir, reward_names):
        super().__init__(run_dir)
        self.FIELDS = self.BASE_FIELDS + [f"reward_{name}" for name in reward_names]

    def _on_training_start(self):
        self.handle = (self.run_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.max_hold = [0] * self.training_env.num_envs

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            self.max_hold[index] = max(self.max_hold[index], info["hold_steps"])
            if "episode" not in info:
                continue
            row = {
                "timesteps": self.num_timesteps, "reward": info["episode"]["r"],
                "length": info["episode"]["l"], "success": int(info["is_success"]),
                "collision": int(info["is_collision"]), "timeout": int(info["is_timeout"]),
                "anchor_lost": int(info["is_anchor_lost"]),
                "elapsed_time": info["elapsed_time"], "final_distance": info["distance_to_goal"],
                "slot": info["slot"], "goal_lane": info["goal_lane"], "max_hold_steps": self.max_hold[index],
                **{f"reward_{name}": value for name, value in info["episode_reward_components"].items()},
            }
            self.max_hold[index] = 0
            self.writer.writerow(row)
            self.handle.flush()
            self.recent.append(row)
            for key in ("success", "collision", "reward", "final_distance", "max_hold_steps"):
                self.logger.record(f"episodes/{key}_mean_100", np.mean([item[key] for item in self.recent]))
            for name, value in info["episode_reward_components"].items():
                self.logger.record(f"reward_components/{name}", value)
        if self.num_timesteps - self.last_report >= 10_000:
            success = np.mean([row["success"] for row in self.recent]) if self.recent else 0
            print(f"{self.num_timesteps:,} transitions; recent success: {success:.1%}", flush=True)
            self.last_report = self.num_timesteps
        return True


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
    if arm.get("iv") == "round":
        # Step 8: a later round continues its init run; only the task, reward terms and weights may change.
        inits = [s for s in sidecars(models_dir, arm["parent"]) if s["seed"] == metadata["seed"]]
        if not inits:
            raise IVError(f"Train round {arm['parent']} seed {metadata['seed']} before {metadata['arm']}.")
        diffs = check_round(inits[0], metadata)
        metadata["parent_check"] = "saved sidecar of the previous round"
        print(f"Round change from {arm['parent']}:\n{describe(diffs)}", flush=True)
    elif arm["parent"] is not None:
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


def load_initial_weights(model, config, seed, models_dir=MODELS_DIR):
    """Start from the init_policy arm's trained weights on the same seed (policy and value networks).

    If this arm's reward_scale differs from the source run's, the value network's output layer is
    multiplied by the ratio: the policy is unchanged and its value estimates are expressed in the new
    reward units, as they would have been had the source been trained at this scale. Returns the path.
    """
    init_path = models_dir / f"{run_name(config['init_policy'], seed)}.zip"
    source = json.loads(init_path.with_suffix(".json").read_text(encoding="utf-8"))
    model.policy.load_state_dict(PPO.load(init_path, device="cpu").policy.state_dict())
    ratio = config["environment"].get("reward_scale", 1.0) / source["environment"].get("reward_scale", 1.0)
    if ratio != 1.0:
        with torch.no_grad():
            model.policy.value_net.weight.mul_(ratio)
            model.policy.value_net.bias.mul_(ratio)
    return init_path, ratio


def make_vec_env(config, seed, vec="subproc"):
    """The run's environments: one per worker seed (seed, seed+1, ...).

    vec="subproc" runs each environment in its own process (every arm up to Step 7); vec="dummy" steps
    them one after another inside the learner's process. Both are required to give identical training
    (speed.py replay); "dummy" avoids the inter-process transfer and lets more runs share the CPU.
    """
    factories = [partial(make_env, seed + index, config["environment"]) for index in range(config["num_envs"])]
    if vec == "dummy" or config["num_envs"] == 1:
        vec_env = DummyVecEnv(factories)
    elif vec == "subproc":
        vec_env = SubprocVecEnv(factories, start_method="spawn")
    else:
        raise ValueError(f"vec must be 'subproc' or 'dummy', not {vec!r}")
    vec_env.seed(seed)
    return vec_env


def build_model(config, seed, run_dir, models_dir=MODELS_DIR, vec="subproc", tensorboard=True):
    """Everything train.py does before model.learn, in the same order. Returns (model, vec_env, metrics, init).

    init is (path, value scale) of the starting weights, or None for fresh weights.
    """
    torch.set_num_threads(1)  # Tiny MLPs are faster without a thread pool per inference.
    set_random_seed(seed)
    vec_env = make_vec_env(config, seed, vec)
    model = PPO("MultiInputPolicy", vec_env, seed=seed, verbose=0, **config["ppo_config"])
    init = None
    if config.get("init_policy"):
        # Policy and value networks start from the init_policy arm's trained weights (same seed);
        # the Adam optimizer starts fresh, as in any new PPO run.
        init = load_initial_weights(model, config, seed, models_dir)
    model.set_logger(configure(str(run_dir), ["csv"] + (["tensorboard"] if tensorboard else [])))
    metrics = ReachEpisodeMetrics(run_dir, list(config["environment"]["reward_weights"]))
    return model, vec_env, metrics, init


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="4,096 transitions and 8 evaluation episodes, saved under smoke/ and outside the chain.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--vec", choices=("subproc", "dummy"), default="subproc",
                        help="Environments in worker processes (default) or inside this process; training is identical.")
    parser.add_argument("--parent-in-progress", action="store_true",
                        help="The parent arm is still training: check against its defined configuration.")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Do not run the usual 200-episode evaluation after training (Step 8 evaluates separately).")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Also save the policy every N transitions to <run>/checkpoints/ (does not affect training).")
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
        "platform": platform_info(), "status": "running", "vec": args.vec, "code": code_provenance(),
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
    print(f"{name}: {config['requested_timesteps']:,} transitions on {config['num_envs']} environments; "
          f"reward {environment['reward_weights']}; hold {environment['goal_config']['hold_seconds']} s", flush=True)
    model, vec_env, metrics, init = build_model(config, args.seed, run_dir, models_dir, vec=args.vec,
                                                tensorboard=not args.no_tensorboard)
    if init is not None:
        metadata["init_value_scale"] = ratio = init[1]
        print(f"Initial weights: {init_path.name}" + (f" (value output x {ratio:g})" if ratio != 1.0 else ""), flush=True)
    callback = metrics
    if args.checkpoint_every:
        metadata["checkpoint_every"] = args.checkpoint_every
        callback = CallbackList([callback, CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // config["num_envs"]),
            save_path=str(run_dir / "checkpoints"), name_prefix=name)])
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
        if metrics.handle is not None and not metrics.handle.closed:
            metrics.handle.close()
        vec_env.close()
    plot_training(run_dir)
    print(f"Saved {model_path}: {model.num_timesteps:,} transitions in {metadata['training_seconds'] / 60:.1f} min.", flush=True)
    if status == "trained" and not args.skip_evaluation:
        evaluate_run(model_path, run_dir, episodes=eval_episodes)


if __name__ == "__main__":
    main()
