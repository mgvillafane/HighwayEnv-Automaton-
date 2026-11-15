"""Compatibility entry point for the original simple-task configurations."""
from baseline_dynamic_waypoint.single_car import SingleCarWaypointEnv
from baseline_simple_waypoint.config import ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS


class SimpleFollowingEnv(SingleCarWaypointEnv):
    """Keep the original simple-task defaults and saved sidecars reproducible."""

    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        super().__init__(render_mode=render_mode,
                         env_config=ENV_CONFIG if env_config is None else env_config,
                         goal_config=GOAL_CONFIG if goal_config is None else goal_config,
                         reward_weights=REWARD_WEIGHTS if reward_weights is None else reward_weights)
