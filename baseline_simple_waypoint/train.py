"""Train or fine-tune PPO on the fixed or randomized one-traffic-car task."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_dynamic_waypoint.config import SLOTS, PPO_CONFIG as DYNAMIC_PPO_CONFIG, experiment_name
from baseline_dynamic_waypoint.train import SlotEpisodeMetrics
from baseline_simple_waypoint.config import (ENV_CONFIG, RANDOMIZED_ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS,
                                             MODELS_DIR, RESULTS_DIR, SEED)
from baseline_simple_waypoint.environment import SimpleFollowingEnv
from baseline_simple_waypoint.evaluate import evaluate


def make_training_env(env_config, goal_config):
    return Monitor(SimpleFollowingEnv(env_config=env_config, goal_config=goal_config))


class SampledScenarioMetrics(SlotEpisodeMetrics):
    """Keep a separate record of the actual sampled lanes and velocities."""
    scenario_handle = None

    def _on_training_start(self):
        super()._on_training_start()
        self.scenario_handle = (self.run_dir / "sampled_scenarios.csv").open("w", newline="", encoding="utf-8")
        self.scenario_writer = csv.DictWriter(self.scenario_handle, fieldnames=[
            "timesteps", "env_index", "slot", "ego_lane", "traffic_lane", "traffic_speed", "ego_speed"])
        self.scenario_writer.writeheader()

    def _on_step(self):
        result = super()._on_step()
        for index, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                self.scenario_writer.writerow({"timesteps": self.num_timesteps, "env_index": index,
                                              **info["initial_scenario"]})
                self.scenario_handle.flush()
        return result

    def _on_training_end(self):
        super()._on_training_end()
        if self.scenario_handle is not None:
            self.scenario_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default="ppo_slots_seed7")
    parser.add_argument("--slot", choices=("random", *SLOTS),
                        help="Train all four goals by default, or fix one goal.")
    parser.add_argument("--init-model", type=Path, help="Continue a saved PPO policy and optimizer in a new run.")
    parser.add_argument("--randomize", action="store_true", help="Sample traffic speed and both initial lanes.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs, args.rollout_steps) < 1 or args.eval_episodes < 0:
        parser.error("Training counts must be positive and eval-episodes nonnegative.")
    rollout_size = args.num_envs * args.rollout_steps
    if rollout_size < 2 or args.timesteps % rollout_size:
        parser.error("timesteps must be a multiple of num-envs * rollout-steps for an exact transition budget.")
    name = experiment_name(args.run_name)
    directory = RESULTS_DIR / name
    model_path = MODELS_DIR / f"{name}.zip"
    if directory.exists() or model_path.exists() or model_path.with_suffix(".json").exists():
        parser.error("This run already exists. Choose a fresh --run-name to keep its results.")
    torch.set_num_threads(1)
    set_random_seed(args.seed)
    env_config = deepcopy(RANDOMIZED_ENV_CONFIG if args.randomize else ENV_CONFIG)
    goal_config = deepcopy(GOAL_CONFIG)
    if args.slot is not None:
        goal_config["default_slot"] = args.slot
    SimpleFollowingEnv(env_config=env_config, goal_config=goal_config).close()
    parameters = deepcopy(DYNAMIC_PPO_CONFIG)
    source = None
    if args.init_model:
        source_path = args.init_model.resolve()
        source_metadata = json.loads(source_path.with_suffix(".json").read_text(encoding="utf-8"))
        parameters = deepcopy(source_metadata["ppo_config"])
        source = {"model": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  "training_timesteps": source_metadata["actual_timesteps"]}
    # 4 * 250 = 1,000 transitions per rollout, so 200 rollouts are exactly 200,000.
    parameters.update(n_steps=args.rollout_steps, batch_size=min(250, rollout_size), device="cpu")
    factories = [partial(make_training_env, env_config, goal_config) for _ in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    try:
        model = (PPO.load(source["model"], env=env, seed=args.seed, **parameters) if source else
                 PPO("MultiInputPolicy", env, seed=args.seed, verbose=0, **parameters))
    except Exception:
        env.close()
        raise
    directory.mkdir(parents=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"task": "single_traffic_following", "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed, "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
                "initialization": source, "counter": "Additional transitions in this run; source weights and optimizer retained.",
                "randomized": args.randomize,
                "environment": deepcopy({"env_config": env_config, "goal_config": goal_config,
                                          "reward_weights": REWARD_WEIGHTS}),
                "ppo_config": parameters, "status": "running"}
    for path in (directory / "run.json", model_path.with_suffix(".json")):
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Training {args.timesteps:,} additional transitions on {args.num_envs} environments; "
          f"traffic speeds {env_config['traffic_speed_range']}, ego maximum {max(env_config['action']['target_speeds'])} m/s.", flush=True)
    if source:
        print(f"Loaded policy and optimizer from {source['model']} ({model.num_timesteps:,} prior transitions).", flush=True)
    if args.eval_episodes:
        evaluate(model, metadata["environment"], directory, args.eval_episodes, prefix="evaluation_before")
    logger = configure(str(directory), (["stdout"] if not args.quiet else []) + ["csv", "tensorboard"])
    callback = SampledScenarioMetrics(directory, model_path, metadata, checkpoint_timesteps=25_000)
    model.set_logger(logger)
    status = "trained"
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, reset_num_timesteps=True)
        if model.num_timesteps != args.timesteps:
            raise RuntimeError(f"Expected {args.timesteps} transitions, got {model.num_timesteps}.")
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        model.save(model_path)
        metadata.update(status=status, actual_timesteps=model.num_timesteps,
                        total_lifetime_timesteps=(source["training_timesteps"] if source else 0) + model.num_timesteps,
                        training_seconds=time.perf_counter() - started)
        for path in (directory / "run.json", model_path.with_suffix(".json")):
            path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if callback.handle is not None and not callback.handle.closed:
            callback.handle.close()
        if callback.scenario_handle is not None and not callback.scenario_handle.closed:
            callback.scenario_handle.close()
        logger.close()
        env.close()
    print(f"Saved {model_path}")
    if status == "trained" and args.eval_episodes:
        evaluate(model, metadata["environment"], directory, args.eval_episodes)


if __name__ == "__main__":
    main()
