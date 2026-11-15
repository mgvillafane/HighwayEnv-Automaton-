"""Opt-in state features and reward guidance for approaching and passing one car.

PPO still selects the five native actions. A geometric passing target describes
an adjacent lane until ego is safely ahead, then blends into the FRONT slot lane.
There are no forced actions, policy reflections, or changes to success checks.
"""
from gymnasium import spaces
import numpy as np

from baseline_dynamic_waypoint.single_car import SingleCarWaypointEnv


NAVIGATION_FEATURES = (
    "anchor_lane_minus_ego_lane", "goal_lane_minus_ego_lane",
    "requested_lane_minus_ego_lane", "requested_lane_lateral_error",
    "anchor_gap", "anchor_minus_ego_speed", "slot_longitudinal_error",
    "navigation_lateral_error", "preferred_minus_ego_speed", "preferred_speed",
    "front_merge_fraction", "front_pass_fraction", "front_merge_clearance",
    "left_lane_available", "right_lane_available", "collision_risk",
)


class ManeuverGuidanceEnv(SingleCarWaypointEnv):
    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        super().__init__(render_mode=render_mode, env_config=env_config,
                         goal_config=goal_config, reward_weights=reward_weights)
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "navigation": spaces.Box(-1.0, 1.0, (len(NAVIGATION_FEATURES),), np.float32),
        })

    def _validate_config(self):
        super()._validate_config()
        settings = self.goal_config.get("maneuver_guidance")
        if not isinstance(settings, dict):
            raise ValueError("maneuver_guidance must contain the saved navigation settings.")
        if self.env_config.get("complete_lane_changes", False):
            raise ValueError("Maneuver guidance uses ordinary per-tick lane decisions.")
        if self.env_config["lanes_count"] < 2:
            raise ValueError("FRONT passing guidance requires at least two lanes.")
        for key in ("speed_gain", "max_relative_speed", "longitudinal_scale", "lateral_scale",
                    "speed_scale", "lane_width", "route_lateral_weight", "route_speed_weight", "ttc_horizon"):
            if not np.isfinite(settings.get(key, np.nan)) or settings[key] <= 0:
                raise ValueError(f"maneuver_guidance/{key} must be finite and positive.")
        blend = settings.get("front_merge_lead", [])
        if (len(blend) != 2 or not np.isfinite(blend).all()
                or not self.goal_config["pass_margin"] <= blend[0] < blend[1]):
            raise ValueError("front_merge_lead must increase from at least pass_margin.")
        for key in ("route_progress", "tracking"):
            if not np.isfinite(self.reward_weights.get(key, np.nan)) or self.reward_weights[key] < 0:
                raise ValueError(f"Reward {key} must be finite and nonnegative.")

    def navigation_state(self):
        settings = self.goal_config["maneuver_guidance"]
        ego, anchor = self.highway.vehicle, self.anchor
        network = self.highway.road.network
        anchor_lane = network.get_lane(anchor.lane_index)
        ego_s, ego_across = anchor_lane.local_coordinates(ego.position)
        anchor_s, _ = anchor_lane.local_coordinates(anchor.position)
        lead = ego_s - anchor_s
        longitudinal, goal_lateral = self.slot_error()
        merge_fraction = 1.0
        navigation_lateral = goal_lateral
        if self.slot == "front":
            road_start, road_end, lane_id = anchor.lane_index
            passing_lanes = [(road_start, road_end, lane) for lane in (lane_id - 1, lane_id + 1)
                             if 0 <= lane < self.env_config["lanes_count"]]
            # The nearest adjacent lane is an observation/reward target only.
            passing_lane = min(
                (network.get_lane(index) for index in passing_lanes),
                key=lambda lane: abs(lane.local_coordinates(ego.position)[1]),
            )
            passing_lateral = -passing_lane.local_coordinates(ego.position)[1]
            low, high = settings["front_merge_lead"]
            merge_fraction = float(np.clip((lead - low) / (high - low), 0.0, 1.0))
            navigation_lateral = ((1 - merge_fraction) * passing_lateral
                                  + merge_fraction * goal_lateral)
        relative_speed = np.clip(settings["speed_gain"] * longitudinal,
                                 -settings["max_relative_speed"], settings["max_relative_speed"])
        speeds = self.env_config["action"]["target_speeds"]
        preferred_speed = float(np.clip(anchor.speed + relative_speed, min(speeds), max(speeds)))
        speed_error = preferred_speed - ego.speed
        closing_speed = ego.speed - anchor.speed
        gap = -lead
        collision_risk = 0.0
        lateral_clearance = (ego.WIDTH + anchor.WIDTH) / 2
        if gap > 0 and closing_speed > 0 and abs(ego_across) < lateral_clearance:
            bumper_gap = max(0.0, gap - (ego.LENGTH + anchor.LENGTH) / 2)
            ttc = bumper_gap / closing_speed
            collision_risk = float(np.clip(1 - ttc / settings["ttc_horizon"], 0.0, 1.0))
        route_cost = (abs(longitudinal)
                      + settings["route_lateral_weight"] * abs(navigation_lateral)
                      + settings["route_speed_weight"] * abs(speed_error))
        # Bounded extra cost: at most the tracking weight per physics tick.
        tracking_error = (
            0.4 * min(1.0, abs(navigation_lateral) / settings["lateral_scale"])
            + 0.25 * min(1.0, abs(longitudinal) / settings["longitudinal_scale"])
            + 0.25 * min(1.0, abs(speed_error) / settings["speed_scale"])
            + 0.1 * collision_risk
        )
        return {"longitudinal": longitudinal, "lateral": navigation_lateral,
                "preferred_speed": preferred_speed, "speed_error": speed_error,
                "lead": lead, "merge_fraction": merge_fraction,
                "collision_risk": collision_risk, "route_cost": route_cost,
                "tracking_error": tracking_error}

    def _observation(self, raw):
        observation = super()._observation(raw)
        state = self.navigation_state()
        settings = self.goal_config["maneuver_guidance"]
        ego, anchor = self.highway.vehicle, self.anchor
        last_lane = self.env_config["lanes_count"] - 1
        ego_lane = int(ego.lane_index[2])
        requested = int(ego.target_lane_index[2])
        lane = self.highway.road.network.get_lane(ego.target_lane_index)
        requested_lateral = -lane.local_coordinates(ego.position)[1]
        is_front = self.slot == "front"
        observation["navigation"] = np.clip(np.array([
            (int(anchor.lane_index[2]) - ego_lane) / last_lane,
            (self.goal_lane - ego_lane) / last_lane,
            (requested - ego_lane) / last_lane,
            requested_lateral / settings["lane_width"],
            -state["lead"] / settings["longitudinal_scale"],
            (anchor.speed - ego.speed) / settings["speed_scale"],
            state["longitudinal"] / settings["longitudinal_scale"],
            state["lateral"] / settings["lateral_scale"],
            state["speed_error"] / settings["speed_scale"],
            state["preferred_speed"] / max(self.env_config["action"]["target_speeds"]),
            state["merge_fraction"] if is_front else 0.0,
            1 - state["merge_fraction"] if is_front else 0.0,
            (state["lead"] - self.goal_config["pass_margin"]) / settings["longitudinal_scale"] if is_front else 0.0,
            float(ego_lane > 0), float(ego_lane < last_lane), state["collision_risk"],
        ], dtype=np.float32), -1, 1)
        return observation

    def step(self, action):
        previous = self.navigation_state()
        observation, reward, terminated, truncated, info = super().step(action)
        current = self.navigation_state()
        extra = {
            "route_progress": self.reward_weights["route_progress"] * (previous["route_cost"] - current["route_cost"]),
            "tracking": -self.reward_weights["tracking"] * current["tracking_error"],
        }
        self.last_components.update(extra)
        for key, value in extra.items():
            self.episode_totals[key] += value
        info.update(reward_components=dict(self.last_components),
                    episode_reward_components=dict(self.episode_totals),
                    navigation_guidance=current)
        return observation, float(reward + sum(extra.values())), terminated, truncated, info
