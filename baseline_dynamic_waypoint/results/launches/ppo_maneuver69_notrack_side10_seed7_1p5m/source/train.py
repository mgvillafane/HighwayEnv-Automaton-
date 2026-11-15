"""Train PPO to reach and hold a slot of a moving car, then evaluate it against random actions."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_dynamic_waypoint.config import (ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS, PPO_CONFIG, SEED,
                                              TRAINING_TIMESTEPS, CHECKPOINT_TIMESTEPS, NUM_ENVS, EVALUATION_EPISODES,
                                              MODELS_DIR, RESULTS_DIR, experiment_name)
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_dynamic_waypoint.evaluate import compare
from baseline_waypoint.plot_results import plot_training
from baseline_waypoint.train import EpisodeMetrics


def make_training_env(seed, goal_config, reward_weights):
    env = DynamicGoalHighwayEnv(goal_config=goal_config, reward_weights=reward_weights)
    env.reset(seed=seed)
    return Monitor(env)


class SlotEpisodeMetrics(EpisodeMetrics):
    """Marcos's per-episode CSV and rolling rates, plus hold reward, slot and anchor-lost columns."""
    FIELDS = ["timesteps", "reward", "length", "success", "collision", "timeout", "anchor_lost",
              "elapsed_time", "final_distance", "slot", "goal_lane", "max_hold_steps",
              "reward_goal", "reward_collision", "reward_progress", "reward_time", "reward_hold",
              "reward_lane", "reward_speed_match", "reward_overtake", "reward_timeout",
              "reward_route_progress", "reward_tracking", "reward_front_side"]

    def __init__(self, run_dir, model_path, metadata, checkpoint_timesteps=CHECKPOINT_TIMESTEPS):
        super().__init__(run_dir)
        self.model_path = model_path
        self.metadata = metadata
        self.checkpoint_timesteps = checkpoint_timesteps
        self.last_checkpoint = 0

    def _on_training_start(self):
        self.handle = (self.run_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.max_hold = [0] * self.training_env.num_envs

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            self.max_hold[index] = max(self.max_hold[index], info.get("internal_max_hold_steps", info["hold_steps"]))
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
                "reward_timeout": info["episode_reward_components"].get("timeout", 0.0),
            }
            self.max_hold[index] = 0
            self.writer.writerow(row)
            self.handle.flush()
            self.recent.append(row)
            for key in ("success", "collision", "timeout", "reward", "final_distance", "max_hold_steps"):
                self.logger.record(f"episodes/{key}_mean_100", np.mean([item[key] for item in self.recent]))
            successful = [item for item in self.recent if item["success"]]
            self.logger.record("episodes/success_count_100", len(successful))
            if successful:
                self.logger.record("episodes/steps_to_success_mean_100", np.mean([item["length"] for item in successful]))
            for name in info["episode_reward_components"]:
                self.logger.record(f"reward_components_mean_100/{name}",
                                   np.mean([item.get(f"reward_{name}", 0.0) for item in self.recent]))
            for name, value in info["episode_reward_components"].items():
                self.logger.record(f"reward_components/{name}", value)
        if self.num_timesteps - self.last_report >= 10_000:
            success = np.mean([row["success"] for row in self.recent]) if self.recent else 0
            print(f"{self.num_timesteps:,} transitions; recent success: {success:.1%}", flush=True)
            self.last_report = self.num_timesteps
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_timesteps:
            checkpoint = self.model_path.with_name(f"{self.model_path.stem}_{self.num_timesteps}.zip")
            self.model.save(checkpoint)
            snapshot = dict(self.metadata, actual_timesteps=self.num_timesteps, status="checkpoint")
            checkpoint.with_suffix(".json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            (self.run_dir / "run.json").write_text(
                json.dumps(dict(snapshot, status="running"), indent=2), encoding="utf-8")
            self.last_checkpoint = self.num_timesteps
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-seconds", type=float, default=GOAL_CONFIG["hold_seconds"],
                        help="Uninterrupted time ego must stay in the slot (e.g. 1, 3 or 5).")
    parser.add_argument("--no-slot-error", action="store_true",
                        help="Leave out the local slot-error input (the original 50-input observation).")
    parser.add_argument("--observation-schema", choices=("legacy", "legacy_target_lanes", "lane_goal_v2"),
                        default=GOAL_CONFIG["observation_schema"], help="Observation layout for this run.")
    parser.add_argument("--no-shaping", action="store_true",
                        help="Drop the lane, speed_match and overtake terms (the reward used before shaping).")
    parser.add_argument("--no-overtake", action="store_true",
                        help="Drop only the FRONT overtake term (the shaping used before it existed).")
    parser.add_argument("--timesteps", type=int, default=TRAINING_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", help="Defaults to ppo_hold<seconds>s_seed<seed>.")
    parser.add_argument("--eval-episodes", type=int, default=EVALUATION_EPISODES,
                        help="0 skips evaluation; full runs should use at least 50.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.num_envs <= 0 or args.eval_episodes < 0 or args.hold_seconds <= 0:
        parser.error("Timesteps, num-envs and hold-seconds must be positive; eval-episodes must be nonnegative.")
    name = experiment_name(args.run_name or f"ppo_hold{args.hold_seconds:g}s_seed{args.seed}".replace(".", "p"))
    run_dir = RESULTS_DIR / name
    model_path = MODELS_DIR / f"{name}.zip"
    if (run_dir / "training_episodes.csv").exists() or model_path.exists():
        parser.error("This run already exists. Choose a new --run-name to preserve its results.")
    goal_config = deepcopy(GOAL_CONFIG)
    goal_config["observation_schema"] = args.observation_schema
    goal_config["hold_seconds"] = args.hold_seconds
    if args.no_slot_error:
        goal_config["observation_schema"] = "legacy"
        goal_config["slot_error_features"] = False
    reward_weights = {name: weight for name, weight in REWARD_WEIGHTS.items()
                      if not (args.no_shaping and name in ("lane", "speed_match", "overtake"))
                      and not (args.no_overtake and name == "overtake")}
    DynamicGoalHighwayEnv(goal_config=goal_config, reward_weights=reward_weights).close()  # Reject an invalid hold time before spawning workers.
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Tiny MLPs are faster without a thread pool per inference.
    set_random_seed(args.seed)
    factories = [partial(make_training_env, args.seed + index, goal_config, reward_weights)
                 for index in range(args.num_envs)]
    vec_env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    vec_env.seed(args.seed)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
        "checkpoint_timesteps": CHECKPOINT_TIMESTEPS,
        "environment": {"env_config": ENV_CONFIG, "goal_config": goal_config, "reward_weights": reward_weights},
        "ppo_config": PPO_CONFIG,
        "versions": {name: importlib.metadata.version(name) for name in ("highway-env", "gymnasium", "stable-baselines3", "torch", "numpy")},
        "status": "running",
    }
    print(f"Hold time: {args.hold_seconds:g} s; observation schema: {goal_config['observation_schema']}; "
          f"reward weights: {reward_weights}", flush=True)
    print(f"Training {args.timesteps:,} transitions on {args.num_envs} environment(s) as {name}.", flush=True)
    model = PPO("MultiInputPolicy", vec_env, seed=args.seed, verbose=0, **PPO_CONFIG)
    formats = ["csv"] + ([] if args.quiet else ["stdout"]) + ([] if args.no_tensorboard else ["tensorboard"])
    model.set_logger(configure(str(run_dir), formats))
    callback = SlotEpisodeMetrics(run_dir, model_path, metadata)
    for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    interrupted = False
    failed = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving the current policy and completed episode metrics.", flush=True)
    except Exception:
        failed = True
        raise
    finally:
        model.save(model_path)
        metadata.update(actual_timesteps=model.num_timesteps, training_seconds=time.perf_counter() - started,
                        status="failed" if failed else "interrupted" if interrupted else "trained")
        for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        if callback.handle is not None and not callback.handle.closed:
            callback.handle.close()
        vec_env.close()
    plot_training(run_dir)
    print(f"Saved {model_path}; actual transitions: {model.num_timesteps:,}.", flush=True)
    if args.eval_episodes and not interrupted:
        compare(model_path, run_dir, episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
