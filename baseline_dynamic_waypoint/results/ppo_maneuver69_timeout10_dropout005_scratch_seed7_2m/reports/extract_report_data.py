"""Verify the saved policy contract and collect simulator-derived examples."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.maneuver_guidance import NAVIGATION_FEATURES

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
MODEL = ROOT / "baseline_dynamic_waypoint/models" / (RUN.name + ".zip")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    torch.set_num_threads(1)
    model, env, metadata = load_experiment(MODEL)
    contract, offset = [], 0
    for key in model.policy.features_extractor.extractors:
        shape = model.observation_space[key].shape
        size = int(np.prod(shape))
        contract.append(dict(key=key, shape=list(shape), size=size,
                             start=offset, end=offset + size - 1))
        offset += size
    assert offset == model.policy.features_extractor.features_dim == 69
    examples = {}
    try:
        for slot in ("front", "right"):
            env.reset(seed=10000, options={"slot": slot})
            for car, x, speed in [(env.highway.vehicle, 100., 20.), (env.anchor, 135., 15.)]:
                car.position = np.array([x, 4.])
                car.heading = 0.
                car.speed = car.target_speed = speed
                if hasattr(car, "speed_to_index"):
                    car.speed_index = car.speed_to_index(speed)
                car.on_state_update()
                car.target_lane_index = car.lane_index
            env.episode_origin = np.array([100., 0.])
            env._update_goal()
            env.previous_distance = env.distance_to_goal()
            env.previous_shaping_errors = env.shaping_errors()
            observation = env._observation(env.highway.observation_type.observe())
            before = env.navigation_state()
            distance = env.distance_to_goal()
            after_obs, reward, terminated, truncated, info = env.step(1)
            assert abs(reward - sum(info["reward_components"].values())) < 1e-10
            examples[slot] = dict(
                description="Constructed simulator state for explanation, not an evaluation rollout.",
                ego_position=[100., 4.], anchor_position=[135., 4.],
                ego_speed=20., anchor_speed=15.,
                initial_observation=observation, initial_navigation=before,
                previous_distance=distance, action=1, next_observation=after_obs,
                next_navigation=info["navigation_guidance"], next_distance=info["distance_to_goal"],
                reward=reward, components=info["reward_components"],
                terminated=terminated, truncated=truncated,
            )
        runtime = dict(perception_distance=env.highway.PERCEPTION_DISTANCE,
                       vehicle_length=env.highway.vehicle.LENGTH,
                       vehicle_width=env.highway.vehicle.WIDTH,
                       extractor=type(model.policy.features_extractor).__name__,
                       optimizer=type(model.policy.optimizer).__name__,
                       policy=str(model.policy.mlp_extractor),
                       normalize_advantage=model.normalize_advantage,
                       target_kl=model.target_kl, clip_range_vf=model.clip_range_vf)
    finally:
        env.close()
    sources = []
    for name in ("config.py", "maneuver_guidance.py", "single_car.py", "environment.py",
                 "train.py", "train_single_car.py", "dropout_policy.py", "evaluate.py"):
        path = ROOT / "baseline_dynamic_waypoint" / name
        snapshot = ROOT / "baseline_dynamic_waypoint/results/launches" / RUN.name / "source" / name
        assert path.read_bytes() == snapshot.read_bytes(), f"Training snapshot differs: {name}"
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), snapshot_matches=True))
    refs = {}
    for rel in ("baseline_dynamic_waypoint/maneuver_guidance.py", "baseline_dynamic_waypoint/environment.py",
                "baseline_dynamic_waypoint/single_car.py", "baseline_waypoint/environment.py",
                "baseline_dynamic_waypoint/train_single_car.py"):
        tree = ast.parse((ROOT / rel).read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for func in node.body:
                    if isinstance(func, ast.FunctionDef):
                        refs[node.name + "." + func.name] = dict(path=rel, line=func.lineno)
    payload = dict(run=metadata, evaluation=json.loads((RUN / "evaluation.json").read_text()),
                   contract=contract, navigation_features=NAVIGATION_FEATURES,
                   examples=examples, runtime=runtime, sources=sources, references=refs,
                   model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                   inspection_versions={p:importlib.metadata.version(p) for p in
                                        ("stable-baselines3", "highway-env", "gymnasium", "torch", "numpy")})
    (OUT / "report_data.json").write_text(json.dumps(payload, indent=2, default=serial) + "\n", encoding="utf-8")
    print(json.dumps(dict(features=offset, examples={slot:ex["reward"] for slot,ex in examples.items()},
                          training_sources_match=True, output=str(OUT / "report_data.json"))))


if __name__ == "__main__":
    main()
