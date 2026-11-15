"""Settings for one ego car reaching slots around exactly one traffic car."""
from copy import deepcopy
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASELINE_DIR / "models"
RESULTS_DIR = BASELINE_DIR / "results"
SEED = 7

ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 1,  # Traffic count: one other car, plus the controlled ego.
    "controlled_vehicles": 1,
    "initial_lane_id": 1,
    "vehicles_density": 0.7,
    "traffic_speed_range": [15.0, 15.0],
    "traffic_behavior": "constant_speed",
    "lead_vehicle_gap": 35.0,
    "ego_spacing": 2.0,
    "duration": 15.0,
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "offroad_terminal": False,
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
    },
    # Keep the previous task's input shape: unused traffic rows are zero padded.
    "observation": {
        "type": "Kinematics", "vehicles_count": 7,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False, "normalize": True, "clip": True,
        "see_behind": True, "include_obstacles": False, "order": "sorted",
        "features_range": {"x": [-200, 200], "y": [-12, 12],
                           "vx": [-30, 30], "vy": [-10, 10]},
    },
    "screen_width": 1200, "screen_height": 360, "scaling": 8.0,
    "centering_position": [0.18, 0.55], "real_time_rendering": False,
}

# An explicit training profile keeps the fixed demonstrations reproducible.
RANDOMIZED_ENV_CONFIG = deepcopy(ENV_CONFIG)
RANDOMIZED_ENV_CONFIG.update({
    "initial_lane_id": None,  # Sample ego's starting lane on every reset.
    "traffic_speed_range": [12.0, 18.0],
    "randomize_traffic_speed": True,
    "randomize_traffic_lane": True,
})

GOAL_CONFIG = {
    "distance_range": [30.0, 40.0],
    "slot_offsets": {"front": 20.0, "back": 15.0},
    "default_slot": "random",  # Sample FRONT/BACK/LEFT/RIGHT each episode.
    "radius": 3.0, "lateral_tolerance": 1.5,
    "hold_seconds": 1.0, "speed_tolerance": 0.5, "initial_speed": 20.0,
    "position_scale": [200.0, 12.0],
    "slot_error_features": True, "slot_error_scale": [10.0, 4.0, 10.0],
    "shaping_range": 30.0, "pass_margin": 10.0,
}

# Keep the previous task's reward formula so traffic complexity is the main change.
REWARD_WEIGHTS = {
    "goal": 50.0, "collision": 50.0, "progress": 0.1, "time": 0.05,
    "hold": 0.2, "lane": 0.0, "speed_match": 0.0, "overtake": 0.0,
}

PPO_CONFIG = {
    "learning_rate": 1e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 5,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
    "device": "cpu",
}
