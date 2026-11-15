"""Rerun PPO on randomized two-car scenarios with flexible holding and 30 s episodes."""
import argparse
from collections import deque
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
from baseline_dynamic_waypoint.config import (SINGLE_CAR_ENV_CONFIG as ENV_CONFIG,
    SINGLE_CAR_GOAL_CONFIG as GOAL_CONFIG, COMMITTED_LANE_ENV_CONFIG, REWARD_PROFILES,
    REWARD_GOAL_OVERRIDES, MODELS_DIR, RESULTS_DIR, SEED)
from baseline_dynamic_waypoint.single_car import make_single_car_env
from baseline_dynamic_waypoint.evaluate import evaluate_single_car as evaluate


def make_training_env(env_config, goal_config, reward_weights=None):
    return Monitor(make_single_car_env(env_config=env_config, goal_config=goal_config,
                                      reward_weights=reward_weights))


def evaluation_environment_for(environment):
    configuration = deepcopy(environment)
    if "enabled_slots" not in configuration["goal_config"]:
        configuration["goal_config"].pop("sampled_slots", None)
    return configuration


def evaluation_scalars(result):
    metrics = {f"evaluation/{key}": result["ppo"][key]
               for key in ("success_rate", "collision_rate", "timeout_rate")}
    for slot, stats in result["ppo_by_slot"].items():
        for key in ("success_rate", "collision_rate", "timeout_rate", "average_episode_reward"):
            metrics[f"evaluation/{slot}/{key}"] = stats[key]
    for prefix, stats in [("evaluation", result["ppo"]),
                          *[(f"evaluation/{slot}", stats) for slot, stats in result["ppo_by_slot"].items()]]:
        for key in ("average_episode_reward", "average_steps_to_success", "successful_episodes",
                    "average_time_to_goal_seconds"):
            if stats.get(key) is not None:
                metrics[f"{prefix}/{key}"] = stats[key]
    return metrics


def save_best_evaluation(model, result, run_dir, model_path, metadata, step=None):
    """Rank fixed monitoring cases by success, then safety, then reward."""
    record = run_dir / "best_evaluation.json"
    stats = result["ppo"]
    score = [stats["success_rate"], -stats["collision_rate"], stats["average_episode_reward"]]
    if record.exists() and score <= json.loads(record.read_text())["score"]:
        return
    path = model_path.with_name(model_path.stem + "_best.zip")
    model.save(path)
    step = model.num_timesteps if step is None else step
    source_steps = (metadata.get("initialization") or {}).get("lifetime_timesteps", 0)
    snapshot = dict(metadata, actual_timesteps=step,
                    total_lifetime_timesteps=source_steps + step, status="best_evaluation")
    path.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    record.write_text(json.dumps({"step": step, "model": str(path), "score": score,
                                 "summary": stats,
                                 "selection": "Success, fewer collisions, then reward on fixed monitoring seeds; not an independent test."},
                                indent=2) + "\n", encoding="utf-8")


class SampledScenarioMetrics(SlotEpisodeMetrics):
    """Keep a separate record of the actual sampled lanes and velocities."""
    scenario_handle = None
    physics_steps = 0
    lane_changes_started = 0
    lane_changes_completed = 0
    lane_changes_interrupted = 0

    def __init__(self, *args, eval_every=0, eval_episodes=100, eval_seed=10_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.last_evaluation = 0

    def _on_training_start(self):
        super()._on_training_start()
        self.by_slot = {slot: deque(maxlen=100) for slot in self.metadata["training_slots"]}
        self.scenario_counts = {slot: 0 for slot in self.metadata["training_slots"]}
        self.scenario_handle = (self.run_dir / "sampled_scenarios.csv").open("w", newline="", encoding="utf-8")
        self.scenario_writer = csv.DictWriter(self.scenario_handle, fieldnames=[
            "timesteps", "env_index", "slot", "ego_lane", "traffic_lane", "traffic_speed", "ego_speed"])
        self.scenario_writer.writeheader()

    def _on_step(self):
        self.physics_steps += sum(info.get("internal_steps", 1) for info in self.locals["infos"])
        self.lane_changes_started += sum(info.get("lane_change_started", False) for info in self.locals["infos"])
        self.lane_changes_completed += sum(info.get("lane_change_completed", False) for info in self.locals["infos"])
        self.lane_changes_interrupted += sum(info.get("lane_change_interrupted", False) for info in self.locals["infos"])
        self.logger.record("time/physics_steps", self.physics_steps)
        self.logger.record("lane_changes/started", self.lane_changes_started)
        self.logger.record("lane_changes/completed", self.lane_changes_completed)
        self.logger.record("lane_changes/interrupted", self.lane_changes_interrupted)
        result = super()._on_step()
        for index, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                slot = info["initial_scenario"]["slot"]
                self.scenario_counts[slot] += 1
                values = {"success": float(info["is_success"]), "collision": float(info["is_collision"]),
                          "timeout": float(info["is_timeout"]), "reward": info["episode"]["r"]}
                self.by_slot[slot].append(values)
                for key in values:
                    average = sum(row[key] for row in self.by_slot[slot]) / len(self.by_slot[slot])
                    self.logger.record(f"episodes/{slot}/{key}_mean_100", average)
                self.logger.record(f"sampling/{slot}/episodes", self.scenario_counts[slot])
                self.scenario_writer.writerow({"timesteps": self.num_timesteps, "env_index": index,
                                              **info["initial_scenario"]})
                self.scenario_handle.flush()
        return result

    def _on_training_end(self):
        super()._on_training_end()
        if self.scenario_handle is not None:
            self.scenario_handle.close()

    def _on_rollout_end(self):
        if (self.eval_every and self.eval_episodes
                and self.num_timesteps - self.last_evaluation >= self.eval_every
                and self.num_timesteps < self.metadata["requested_timesteps"]):
            result = evaluate(self.model, self.metadata["evaluation_environment"], self.run_dir,
                              self.eval_episodes, seed=self.eval_seed,
                              prefix=f"evaluation_{self.num_timesteps}")
            for tag, value in evaluation_scalars(result).items():
                self.logger.record(tag, value)
            save_best_evaluation(self.model, result, self.run_dir, self.model_path, self.metadata)
            self.last_evaluation = self.num_timesteps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default="ppo_singlecar_flexible30s_seed7_200k")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--slot", choices=("random", *SLOTS),
                        help="Train all four goals by default, or fix one goal.")
    sampling.add_argument("--training-slots", nargs="+", choices=SLOTS,
                          help="Sample uniformly from these goals during training; evaluate all four goals.")
    sampling.add_argument("--enabled-slots", nargs="+", choices=SLOTS,
                          help="Enable only these goals in both training and evaluation; reject other reset goals.")
    parser.add_argument("--init-model", type=Path, help="Continue a saved PPO policy and optimizer in a new run.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000,
                        help="First held-out evaluation seed, reused before and after training.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Evaluate at rollout boundaries every N training steps; 0 evaluates only before/after.")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        help="Dropout after each actor/critic hidden activation; 0 uses the historical policy.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--time-penalty", type=float, default=None,
                        help="Nonnegative cost per physics tick; overrides the reward profile's time weight.")
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="baseline",
                        help="Named reward weights, saved in each checkpoint's environment metadata.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--committed-lanes", action="store_true",
                        help="Finish lane changes before the next decision.")
    parser.add_argument("--observation-schema", choices=("legacy", "legacy_target_lanes", "lane_goal_v2"),
                        default=GOAL_CONFIG["observation_schema"], help="Observation layout for this run.")
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs, args.rollout_steps) < 1 or args.eval_episodes < 0:
        parser.error("Training counts must be positive and eval-episodes nonnegative.")
    if args.eval_every < 0 or not 0 <= args.dropout_rate < 1:
        parser.error("eval-every must be nonnegative; dropout-rate must be in [0, 1).")
    if args.learning_rate is not None and not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be finite and positive.")
    if args.gamma is not None and not 0 <= args.gamma <= 1:
        parser.error("gamma must be in [0, 1].")
    if args.time_penalty is not None and not 0 <= args.time_penalty < float("inf"):
        parser.error("time-penalty must be finite and nonnegative.")
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
    env_config = deepcopy(COMMITTED_LANE_ENV_CONFIG if args.committed_lanes else ENV_CONFIG)
    goal_config = deepcopy(GOAL_CONFIG)
    goal_config["observation_schema"] = args.observation_schema
    reward_goal_overrides = deepcopy(REWARD_GOAL_OVERRIDES[args.reward_profile])
    goal_config.update(reward_goal_overrides)
    reward_weights = deepcopy(REWARD_PROFILES[args.reward_profile])
    if args.time_penalty is not None:
        reward_weights["time"] = args.time_penalty
    if args.slot is not None:
        goal_config["default_slot"] = args.slot
    if args.training_slots is not None:
        goal_config["sampled_slots"] = args.training_slots
    if args.enabled_slots is not None:
        goal_config["enabled_slots"] = args.enabled_slots
        goal_config["sampled_slots"] = args.enabled_slots
    make_single_car_env(env_config=env_config, goal_config=goal_config,
                        reward_weights=reward_weights).close()
    parameters = deepcopy(DYNAMIC_PPO_CONFIG)
    source = None
    if args.init_model:
        source_path = args.init_model.resolve()
        source_metadata = json.loads(source_path.with_suffix(".json").read_text(encoding="utf-8"))
        parameters = deepcopy(source_metadata["ppo_config"])
        source = {"model": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  "training_timesteps": source_metadata["actual_timesteps"],
                  "lifetime_timesteps": source_metadata.get("total_lifetime_timesteps", source_metadata["actual_timesteps"])}
    # 4 * 250 = 1,000 transitions per rollout, so 200 rollouts are exactly 200,000.
    parameters.update(n_steps=args.rollout_steps, batch_size=min(250, rollout_size), device="cpu")
    if args.learning_rate is not None:
        parameters["learning_rate"] = args.learning_rate
    if args.gamma is not None:
        parameters["gamma"] = args.gamma
    if args.dropout_rate:
        from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
        policy_type = DropoutMultiInputPolicy
        parameters.setdefault("policy_kwargs", {})["dropout_rate"] = args.dropout_rate
    else:
        policy_type = "MultiInputPolicy"
        parameters.setdefault("policy_kwargs", {}).pop("dropout_rate", None)
    factories = [partial(make_training_env, env_config, goal_config, reward_weights)
                 for _ in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    try:
        if source:
            original = PPO.load(source["model"], device="cpu")
            policy_changed = (getattr(original.policy, "dropout_rate", 0.0) != args.dropout_rate)
            if original.observation_space != env.observation_space or policy_changed:
                from baseline_dynamic_waypoint.policy_transfer import transfer_observation_weights
                model = PPO(policy_type, env, seed=args.seed, verbose=0, **parameters)
                source.update(transfer_observation_weights(original, model))
            else:
                model = PPO.load(source["model"], env=env, seed=args.seed, **parameters)
                source.update(kind="policy_and_optimizer", optimizer_reset=False)
            del original
        else:
            model = PPO(policy_type, env, seed=args.seed, verbose=0, **parameters)
    except Exception:
        env.close()
        raise
    directory.mkdir(parents=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"task": "single_car_committed_lane" if args.committed_lanes else "single_car_waypoint", "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed, "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
                "initialization": source, "counter": "Additional PPO decisions in this run; internal physics ticks are recorded separately.",
                "randomized": True, "reward_profile": args.reward_profile,
                "reward_goal_overrides": reward_goal_overrides,
                "evaluation_first_seed": args.eval_seed,
                "evaluation_every_timesteps": args.eval_every,
                "dropout_rate": args.dropout_rate,
                "policy_class": f"{type(model.policy).__module__}.{type(model.policy).__name__}",
                "environment": deepcopy({"env_config": env_config, "goal_config": goal_config,
                                          "reward_weights": reward_weights}),
                "ppo_config": parameters, "status": "running"}
    evaluation_environment = evaluation_environment_for(metadata["environment"])
    metadata["evaluation_environment"] = evaluation_environment
    metadata["training_slots"] = list(args.enabled_slots or args.training_slots or ([args.slot] if args.slot and args.slot != "random" else SLOTS))
    metadata["enabled_slots"] = list(args.enabled_slots or SLOTS)
    metadata["evaluation_slots"] = list(args.enabled_slots or ([args.slot] if args.slot and args.slot != "random" else SLOTS))
    for path in (directory / "run.json", model_path.with_suffix(".json")):
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Training {args.timesteps:,} additional transitions on {args.num_envs} environments; "
          f"traffic speeds {env_config['traffic_speed_range']}, ego maximum {max(env_config['action']['target_speeds'])} m/s.", flush=True)
    if source:
        print(f"Initialized from {source['model']}; transfer={source['kind']}, optimizer_reset={source['optimizer_reset']}.", flush=True)
        if source.get("policy_outputs_preserved") is False:
            print(f"Partial observation transfer; dropped/redefined inputs: {source['dropped_or_redefined_inputs']}.", flush=True)
    print(f"Observation inputs: {model.policy.features_extractor.features_dim}; committed lane changes: {args.committed_lanes}.", flush=True)
    print(f"Reward profile: {args.reward_profile}; weights={reward_weights}.", flush=True)
    print(f"Actor/critic dropout: {args.dropout_rate}; intermediate evaluation every {args.eval_every:,} steps.", flush=True)
    print(f"Enabled slots: {metadata['enabled_slots']}; training: {metadata['training_slots']}; "
          f"evaluation: {metadata['evaluation_slots']}.", flush=True)
    logger = configure(str(directory), (["stdout"] if not args.quiet else []) + ["csv", "tensorboard"])
    callback = SampledScenarioMetrics(directory, model_path, metadata, checkpoint_timesteps=25_000,
                                      eval_every=args.eval_every, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed)
    model.set_logger(logger)
    for key in ("hold_seconds", "hold_grace_seconds", "radius", "lateral_tolerance", "speed_tolerance"):
        logger.record(f"setup/{key}", goal_config[key])
    logger.record("setup/episode_duration", env_config["duration"])
    logger.record("setup/dropout_rate", args.dropout_rate)
    for slot in SLOTS:
        logger.record(f"setup/enabled_{slot}", int(slot in metadata["enabled_slots"]))
    for key, value in reward_weights.items():
        logger.record(f"setup/reward_weight_{key}", value)
    logger.dump(step=0)
    if args.eval_episodes:
        baseline = evaluate(model, evaluation_environment, directory, args.eval_episodes,
                            seed=args.eval_seed, prefix="evaluation_before")
        save_best_evaluation(model, baseline, directory, model_path, metadata, step=0)
        for tag, value in evaluation_scalars(baseline).items():
            logger.record(tag, value)
        logger.dump(step=0)
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
                        total_lifetime_timesteps=(source["lifetime_timesteps"] if source else 0) + model.num_timesteps,
                        actual_physics_steps=callback.physics_steps,
                        lane_changes={"started": callback.lane_changes_started, "completed": callback.lane_changes_completed,
                                      "interrupted": callback.lane_changes_interrupted},
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
        result = evaluate(model, evaluation_environment, directory, args.eval_episodes, seed=args.eval_seed)
        save_best_evaluation(model, result, directory, model_path, metadata)
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(log_dir=str(directory)) as writer:
            for tag, value in evaluation_scalars(result).items():
                writer.add_scalar(tag, value, model.num_timesteps)


if __name__ == "__main__":
    main()
