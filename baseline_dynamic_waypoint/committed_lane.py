"""Lane-change options: complete one maneuver before another PPO decision."""
from gymnasium import spaces
import numpy as np

from baseline_dynamic_waypoint.config import COMMITTED_LANE_ENV_CONFIG
from baseline_dynamic_waypoint.single_car import SingleCarWaypointEnv


class CommittedLaneEnv(SingleCarWaypointEnv):
    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        self.episode_physics_steps = 0
        self.episode_policy_decisions = 0
        self.physics_step_callback = None
        super().__init__(render_mode=render_mode,
                         env_config=COMMITTED_LANE_ENV_CONFIG if env_config is None else env_config,
                         goal_config=goal_config, reward_weights=reward_weights)
        if self.observation_schema in ("legacy", "legacy_target_lanes"):
            self.observation_space = spaces.Dict({
                **self.observation_space.spaces,
                "requested_lane": spaces.Box(0.0, 1.0, (self.env_config["lanes_count"],), np.float32),
                "lane_change_in_progress": spaces.Box(0.0, 1.0, (1,), np.float32),
            })
        if self.env_config.get("fast_observation", False):
            from baseline_DQN.fast_observation import enable_fast_observation
            enable_fast_observation(self.env)

    @property
    def lane_change_in_progress(self):
        ego = self.highway.vehicle
        lane = self.highway.road.network.get_lane(ego.target_lane_index)
        _, lateral = lane.local_coordinates(ego.position)
        return bool(abs(lateral) > self.env_config.get("lane_center_tolerance", 0.2)
                    or abs(ego.velocity[1]) > self.env_config.get("lane_lateral_speed_tolerance", 0.2))

    def _observation(self, raw):
        obs = super()._observation(raw)
        if self.observation_schema == "lane_goal_v2":
            return obs
        lane = int(self.highway.vehicle.target_lane_index[2])
        obs["requested_lane"] = np.eye(self.env_config["lanes_count"], dtype=np.float32)[lane]
        obs["lane_change_in_progress"] = np.array([self.lane_change_in_progress], dtype=np.float32)
        return obs

    def _success_allowed(self):
        # Do not end an episode halfway through a successful lane change.
        return not self.lane_change_in_progress

    def _info(self, success, collision, timeout, anchor_lost=False):
        info = super()._info(success, collision, timeout, anchor_lost)
        info.update(requested_lane=int(self.highway.vehicle.target_lane_index[2]),
                    lane_change_in_progress=self.lane_change_in_progress)
        return info

    def reset(self, *, seed=None, options=None):
        self.episode_physics_steps = 0
        self.episode_policy_decisions = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}.")
        self.episode_policy_decisions += 1
        pending = self.lane_change_in_progress
        previous_target = self.highway.vehicle.target_lane_index
        applied = 1 if pending else action
        total_reward, ticks, maximum_hold = 0.0, 0, 0
        started = False
        while True:
            obs, reward, terminated, truncated, info = super().step(applied)
            ticks += 1
            self.episode_physics_steps += 1
            total_reward += float(reward)
            maximum_hold = max(maximum_hold, info["hold_steps"])
            if ticks == 1:
                started = action in (0, 2) and self.highway.vehicle.target_lane_index != previous_target
            info.update(applied_action=applied, policy_action=action,
                        episode_physics_steps=self.episode_physics_steps,
                        episode_policy_decisions=self.episode_policy_decisions)
            if self.physics_step_callback is not None:
                self.physics_step_callback(obs, float(reward), terminated, truncated, dict(info))
            if terminated or truncated or not ((started or pending) and self.lane_change_in_progress):
                break
            # Keep both the requested lane and target speed unchanged. No policy
            # prediction or other high-level command occurs during these ticks.
            applied = 1
        info.update(internal_steps=ticks, internal_max_hold_steps=maximum_hold,
                    lane_change_started=bool(started),
                    lane_change_completed=bool((started or pending) and not self.lane_change_in_progress),
                    lane_change_interrupted=bool((started or pending) and self.lane_change_in_progress
                                                 and (terminated or truncated)))
        return obs, total_reward, terminated, truncated, info
