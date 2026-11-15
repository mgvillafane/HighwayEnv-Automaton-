"""Opt-in RIGHT-to-LEFT policy canonicalization for the current straight road.

This adapter changes policy inputs/outputs only. It never changes simulator
state, rewards, model parameters, or behavior for the other three objectives.
"""
import numpy as np


ACTION_REFLECTION = np.array([2, 1, 0, 3, 4], dtype=np.int64)
REQUIRED_SHAPES = {
    "vehicles": (7, 5), "desired_goal": (2,), "achieved_goal": (2,),
    "goal_delta": (2,), "anchor": (4,), "slot": (4,), "hold_progress": (1,),
    "slot_error": (3,), "requested_lane": (3,), "lane_change_in_progress": (1,),
}


def reflect_observation(observation):
    """Reflect one observation or a batch, preserving its original arrays."""
    obs = {key: np.asarray(value).copy() for key, value in observation.items()}
    for key in ("desired_goal", "achieved_goal"):
        obs[key][..., 1] = 8.0 / 12.0 - obs[key][..., 1]
    obs["goal_delta"][..., 1] *= -1
    obs["vehicles"][..., 0, 2] = 8.0 / 12.0 - obs["vehicles"][..., 0, 2]
    obs["vehicles"][..., 1:, 2] *= -1
    obs["vehicles"][..., :, 4] *= -1
    obs["anchor"][..., [1, 3]] *= -1
    obs["slot"][..., [2, 3]] = obs["slot"][..., [3, 2]]
    obs["slot_error"][..., 1] *= -1
    obs["requested_lane"] = obs["requested_lane"][..., ::-1].copy()
    return obs


class RightCanonicalPolicy:
    """Use the learned LEFT strategy for RIGHT tasks; leave other tasks alone."""
    def __init__(self, model, *, lanes_count=3, lane_width=4.0,
                 position_y_scale=12.0, vehicle_y_scale=12.0):
        if (lanes_count, lane_width, position_y_scale, vehicle_y_scale) != (3, 4.0, 12.0, 12.0):
            raise ValueError("This adapter requires three straight 4 m lanes and both y scales equal to 12 m.")
        if set(model.observation_space.spaces) != set(REQUIRED_SHAPES):
            raise ValueError("This adapter requires the current 57-input committed-lane observation.")
        for key, shape in REQUIRED_SHAPES.items():
            if model.observation_space[key].shape != shape:
                raise ValueError(f"Unsupported observation shape for {key}: expected {shape}.")
        if getattr(model.action_space, "n", None) != 5:
            raise ValueError("This adapter requires the five DiscreteMetaAction commands.")
        self.model = model
        self.observation_space = model.observation_space
        self.action_space = model.action_space

    @staticmethod
    def reflection_mask(observation):
        return np.asarray(observation["slot"])[..., 3] > 0.5

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        mask = self.reflection_mask(observation)
        policy_observation = observation
        if np.any(mask):
            reflected = reflect_observation(observation)
            if np.ndim(mask) == 0:
                policy_observation = reflected
            else:
                policy_observation = {key: np.asarray(value).copy() for key, value in observation.items()}
                for key in policy_observation:
                    policy_observation[key][mask] = reflected[key][mask]
        actions, next_state = self.model.predict(policy_observation, state=state,
                                                episode_start=episode_start, deterministic=deterministic)
        if np.any(mask):
            actions = np.asarray(actions)
            actions = np.where(mask, ACTION_REFLECTION[actions], actions).astype(actions.dtype)
        return actions, next_state
