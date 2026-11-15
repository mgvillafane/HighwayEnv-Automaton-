"""Compare native and optimized observations on identical seeded workloads."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import DQN

from baseline_DQN.config import training_settings
from baseline_DQN.environment import TrafficHighwayEnv
from baseline_DQN.fast_observation import enable_fast_observation


def environment_trial(steps, fast):
    env = TrafficHighwayEnv(scenario="quickstart")
    if fast:
        enable_fast_observation(env)
    seed = 2
    actions = np.random.default_rng(seed).integers(5, size=steps)
    started = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    checksum = 0.0
    try:
        for action in actions:
            observation, reward, terminated, truncated, _ = env.step(int(action))
            checksum += reward + float(observation.sum())
            if terminated or truncated:
                seed += 1
                observation, _ = env.reset(seed=seed)
    finally:
        env.close()
    return {"seconds": time.perf_counter() - started, "checksum": checksum}


def training_trial(steps, fast):
    parameters, configuration, _ = training_settings("quickstart")
    env = TrafficHighwayEnv(**configuration)
    if fast:
        enable_fast_observation(env)
    model = DQN("MlpPolicy", env, seed=2, verbose=0, **parameters)
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=steps)
        elapsed = time.perf_counter() - started
        state = {key: value.detach().cpu().numpy().copy()
                 for key, value in model.policy.state_dict().items()}
        replay = {key: getattr(model.replay_buffer, key).copy()
                  for key in ("observations", "next_observations", "actions", "rewards", "dones")}
        # Ignore unused, uninitialized replay capacity.
        for key in replay:
            replay[key] = replay[key][:model.replay_buffer.pos]
        if fast:
            artifact = BytesIO()
            model.save(artifact)
            artifact.seek(0)
            reloaded = DQN.load(artifact, device="cpu")
            native_playback = TrafficHighwayEnv(**configuration)
            try:
                expected, _ = native_playback.reset(seed=10000)
                actual, _ = env.reset(seed=10000)
                for _ in range(20):
                    np.testing.assert_array_equal(actual, expected)
                    action = int(reloaded.predict(expected, deterministic=True)[0])
                    expected, reward, terminated, truncated, _ = native_playback.step(action)
                    actual, actual_reward, actual_terminated, actual_truncated, _ = env.step(action)
                    assert (actual_reward, actual_terminated, actual_truncated) == (reward, terminated, truncated)
                    if terminated or truncated:
                        break
            finally:
                native_playback.close()
        return elapsed, state, replay
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.training_steps < 0:
        parser.error("steps/repeats must be positive and training-steps nonnegative.")
    torch.set_num_threads(1)
    trials = {"native": [], "fast": []}
    for repeat in range(args.repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for fast in order:
            trials["fast" if fast else "native"].append(environment_trial(args.steps, fast))
    for plain, fast in zip(trials["native"], trials["fast"]):
        assert plain["checksum"] == fast["checksum"], (plain, fast)
    native_seconds = float(np.median([row["seconds"] for row in trials["native"]]))
    fast_seconds = float(np.median([row["seconds"] for row in trials["fast"]]))
    result = {"steps_per_trial": args.steps, "repeats": args.repeats, "environment": {
        "native_transitions_per_second": args.steps / native_seconds,
        "fast_transitions_per_second": args.steps / fast_seconds,
        "speedup": native_seconds / fast_seconds, "trials": trials}}
    if args.training_steps:
        native_time, native_state, native_replay = training_trial(args.training_steps, False)
        fast_time, fast_state, fast_replay = training_trial(args.training_steps, True)
        for before, after in ((native_state, fast_state), (native_replay, fast_replay)):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        result["training"] = {"steps": args.training_steps, "native_seconds": native_time,
                              "fast_seconds": fast_time, "speedup": native_time / fast_time,
                              "policy_parameters_and_replay_exact": True,
                              "saved_model_native_playback_exact": True}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
