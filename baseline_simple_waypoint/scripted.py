"""A demonstration controller, with no learning or trained weights."""
import numpy as np


def scripted_action(env):
    ego, anchor = env.highway.vehicle, env.anchor
    target_lane = env.goal_lane
    passing = env.slot == "front" and ego.position[0] - anchor.position[0] < env.goal_config["pass_margin"]
    if passing:
        # Pass in an adjacent lane before merging into the slot ahead of the car.
        target_lane = anchor.lane_index[2] - 1 if anchor.lane_index[2] > 0 else anchor.lane_index[2] + 1
    if ego.target_lane_index[2] != target_lane:
        return 0 if target_lane < ego.target_lane_index[2] else 2
    error, _ = env.slot_error()
    desired = 25.0 if passing else float(np.clip(anchor.speed + 0.5 * error, 10.0, 25.0))
    if ego.target_speed < desired - 0.6:
        return 3  # FASTER
    if ego.target_speed > desired + 0.6:
        return 4  # SLOWER
    return 1  # IDLE: maintain the current target speed.
