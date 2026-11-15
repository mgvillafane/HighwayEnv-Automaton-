"""Independent settings for the dynamic-waypoint experiment.

Distances are meters, speeds are m/s. Rewards are evaluated at 10 Hz; committed
lane actions return the sum over their internal physics ticks. Difference terms
use changes in distance or speed error.
"""
from copy import deepcopy
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASELINE_DIR / "models"
RESULTS_DIR = BASELINE_DIR / "results"
SEED = 7
TRAINING_TIMESTEPS = 1_000_000
CHECKPOINT_TIMESTEPS = 50_000
NUM_ENVS = 4
EVALUATION_EPISODES = 50
EVALUATION_SEED = 10_000

SLOTS = ("front", "back", "left", "right")

ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 8,  # Traffic cars, in addition to ego.
    "controlled_vehicles": 1,
    "vehicles_density": 0.7,
    "traffic_speed_range": [12.0, 17.0],  # Sustained IDM desired speeds.
    "lead_vehicle_gap": 35.0,  # Center distance to the initial slower lead car.
    "ego_spacing": 2.0,
    "duration": 25.0,  # Time to reach the moving slot and hold it.
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "offroad_terminal": False,
    # FASTER/SLOWER select one grid step from the current speed. A 2.5 m/s step
    # allows roughly 4 m/s^2 braking and matches the 12-17 m/s traffic closely.
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,  # Ego plus the six nearest traffic cars.
        "features": ["presence", "x", "y", "vx", "vy"],  # lane_goal_v2 appends lane_id in the wrapper.
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "include_obstacles": False,
        "order": "sorted",
        "features_range": {
            "x": [-200, 200], "y": [-12, 12],
            "vx": [-30, 30], "vy": [-10, 10],
        },
    },
    "screen_width": 1200,
    "screen_height": 360,
    "scaling": 5.0,
    "centering_position": [0.18, 0.55],
    "real_time_rendering": False,
}

GOAL_CONFIG = {
    # Wrapper schema; the native Kinematics encoder keeps its five features.
    # Missing in historical sidecars means the original observation layout.
    "observation_schema": "legacy",
    # Center-to-center gap of the anchor car at reset (it is chosen uniformly among eligible cars).
    "distance_range": [30.0, 120.0],
    # Center-to-center distance of the FRONT/BACK slots along the anchor's lane.
    # The anchor's IDM driver follows ego when ego is ahead, so it can slow
    # down while ego holds the FRONT slot.
    # LEFT/RIGHT slots sit beside the anchor, centered in the adjacent lane.
    "slot_offsets": {"front": 20.0, "back": 15.0},
    # Success box in the slot lane's frame. "radius" is the longitudinal half-length; the
    # name is kept so the inherited check (no skipping the box between decisions) applies.
    "radius": 3.0,
    "lateral_tolerance": 1.5,
    # Consecutive in-slot time required for success; leaving the box resets the count.
    "hold_seconds": 1.0,
    "initial_speed": 20.0,
    "position_scale": [200.0, 12.0],
    # Legacy schema only: slot error along/across the slot lane and anchor-minus-ego speed,
    # divided by these scales (m, m, m/s). False reproduces the first 50-input runs.
    "slot_error_features": True,
    "slot_error_scale": [10.0, 4.0, 10.0],
    # Along-lane distance (m) over which the speed_match shaping fades in near the slot.
    "shaping_range": 30.0,
    # With the overtake term, a FRONT slot's lane target is a lane beside the anchor
    # until ego leads it by this many meters (center to center; 5 m bumper clearance).
    "pass_margin": 10.0,
}

REWARD_WEIGHTS = {
    "goal": 50.0,  # Bonus once the configured hold requirement is met.
    "collision": 50.0,  # Subtracted on an ego collision.
    "progress": 0.1,  # Per meter of decrease in distance to the moving slot.
    "time": 0.05,  # Subtracted every 0.1 s tick, including the terminal tick.
    "hold": 0.2,  # Per tick inside the slot without a collision/anchor loss.
    "lane": 0.0,  # Per meter of decrease in lateral error to the target lane.
    "speed_match": 0.0,  # Per decrease in proximity-weighted speed error (m/s).
    "overtake": 0.0,  # FRONT only: per meter of increased, capped lead over anchor.
}

# Explicit profiles preserve historical baselines and checkpoint sidecars.
# Select with train_single_car --reward-profile guided; no action bonuses or reflection.
REWARD_PROFILES = {
    "baseline": deepcopy(REWARD_WEIGHTS),
    # Terminal cost only: deadline reached without success/collision/anchor loss.
    "timeout25": {**REWARD_WEIGHTS, "timeout": 25.0},
    "timeout10": {**REWARD_WEIGHTS, "timeout": 10.0},
    "guided": {**REWARD_WEIGHTS, "lane": 1.0, "speed_match": 0.1, "overtake": 0.5},
    # Follow-up after weak approach/merge learning in guided: amplify geometric
    # feedback while increasing collision cost with the larger shaping scale.
    "guided_approach": {**REWARD_WEIGHTS, "collision": 100.0, "progress": 0.5,
                        "lane": 2.0, "speed_match": 0.5, "overtake": 1.0},
    # Opt-in single-car navigation features and staged FRONT passing guidance.
    # Route progress is a difference; tracking is a bounded, nonpositive tick cost.
    "maneuver10": {**REWARD_WEIGHTS, "timeout": 10.0,
                   "route_progress": 0.5, "tracking": 0.05},
}
REWARD_GOAL_OVERRIDES = {
    "baseline": {},
    "timeout25": {},
    "timeout10": {},
    # Reward guidance only: prefer the passing lane until safely ahead, then
    # smoothly favor the goal lane. Success tolerances and dynamics are unchanged.
    "guided": {"front_lane_blend": [5.0, 15.0]},
    "guided_approach": {"front_lane_blend": [5.0, 15.0]},
    "maneuver10": {"maneuver_guidance": {
        "front_merge_lead": [10.0, 20.0],
        "speed_gain": 0.25,
        "max_relative_speed": 5.0,
        "longitudinal_scale": 60.0,
        "lateral_scale": 8.0,
        "speed_scale": 10.0,
        "lane_width": 4.0,
        "route_lateral_weight": 2.0,
        "route_speed_weight": 1.0,
        "ttc_horizon": 5.0,
    }},
}

# Reward-only follow-up: preserve the 69 observations and route guidance.
REWARD_PROFILES["maneuver_side10"] = {
    **REWARD_PROFILES["maneuver10"], "tracking": 0.0, "front_side": 10.0,
}
REWARD_GOAL_OVERRIDES["maneuver_side10"] = {
    **deepcopy(REWARD_GOAL_OVERRIDES["maneuver10"]),
    "front_side_milestone": {"longitudinal_tolerance": 5.0, "lateral_tolerance": 1.8},
}

PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 256,  # Per environment: 4 * 256 = 1024 transitions per rollout.
    "batch_size": 256,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
    "device": "cpu",
}

# Randomized two-car rerun. Historical defaults and checkpoint sidecars remain valid.
SINGLE_CAR_ENV_CONFIG = deepcopy(ENV_CONFIG)
SINGLE_CAR_ENV_CONFIG.update({
    "vehicles_count": 1, "initial_lane_id": None,
    "traffic_speed_range": [12.0, 18.0], "traffic_behavior": "constant_speed",
    "randomize_traffic_speed": True, "randomize_traffic_lane": True,
    "duration": 30.0, "scaling": 8.0,
})
SINGLE_CAR_GOAL_CONFIG = deepcopy(GOAL_CONFIG)
SINGLE_CAR_GOAL_CONFIG.update({
    "distance_range": [30.0, 40.0], "default_slot": "random",
    "radius": 5.0, "lateral_tolerance": 1.8, "speed_tolerance": 2.0,
    "hold_seconds": 0.5, "hold_grace_seconds": 0.3,
})

COMMITTED_LANE_ENV_CONFIG = deepcopy(SINGLE_CAR_ENV_CONFIG)
COMMITTED_LANE_ENV_CONFIG.update({
    "complete_lane_changes": True,
    "lane_center_tolerance": 0.2,
    "lane_lateral_speed_tolerance": 0.2,
    "fast_observation": True,
})


def experiment_name(value):
    """Restrict run names to a single directory inside this experiment."""
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Run names may contain only letters, numbers, underscores and hyphens.")
    return value
