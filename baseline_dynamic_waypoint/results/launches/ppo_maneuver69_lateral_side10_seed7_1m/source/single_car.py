"""The existing four-slot moving-goal task, isolated with one traffic car."""
import numpy as np

from baseline_dynamic_waypoint.config import SLOTS
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from baseline_dynamic_waypoint.config import (SINGLE_CAR_ENV_CONFIG as ENV_CONFIG,
    SINGLE_CAR_GOAL_CONFIG as GOAL_CONFIG, REWARD_WEIGHTS)


class SingleCarWaypointEnv(DynamicGoalHighwayEnv):
    """Exactly two vehicles, with optional sampled lanes and traffic velocity.

    Reach its FRONT, BACK, LEFT or RIGHT slot and hold for the configured time.
    The observation schema selects the policy inputs; native kinematics and
    reward retain the dynamic task's conventions. Unused traffic rows are zero padded.
    """

    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        super().__init__(render_mode=render_mode,
                         env_config=ENV_CONFIG if env_config is None else env_config,
                         goal_config=GOAL_CONFIG if goal_config is None else goal_config,
                         reward_weights=REWARD_WEIGHTS if reward_weights is None else reward_weights)

    def _validate_config(self):
        super()._validate_config()
        if self.env_config["vehicles_count"] != 1:
            raise ValueError("SingleCarWaypointEnv requires exactly one traffic car plus ego.")
        if self.goal_config.get("default_slot", "random") not in ("random", *SLOTS):
            raise ValueError("default_slot must be random, front, back, left or right.")
        enabled_slots = self.goal_config.get("enabled_slots", SLOTS)
        if (not isinstance(enabled_slots, (list, tuple)) or not enabled_slots
                or any(slot not in SLOTS for slot in enabled_slots)
                or len(set(enabled_slots)) != len(enabled_slots)):
            raise ValueError("enabled_slots must be a nonempty list of distinct valid slots.")
        sampled_slots = self.goal_config.get("sampled_slots", enabled_slots)
        if (not isinstance(sampled_slots, (list, tuple)) or not sampled_slots
                or any(slot not in SLOTS for slot in sampled_slots)
                or len(set(sampled_slots)) != len(sampled_slots)):
            raise ValueError("sampled_slots must be a nonempty list of distinct valid slots.")
        if not set(sampled_slots).issubset(enabled_slots):
            raise ValueError("sampled_slots cannot include disabled goals.")
        default = self.goal_config.get("default_slot", "random")
        if default != "random" and default not in enabled_slots:
            raise ValueError("default_slot cannot be a disabled goal.")
        if self.goal_config["speed_tolerance"] <= 0:
            raise ValueError("speed_tolerance must be positive.")
        if self.env_config.get("traffic_behavior", "idm") not in ("constant_speed", "idm"):
            raise ValueError("traffic_behavior must be constant_speed or idm.")
        if max(self.env_config["traffic_speed_range"]) >= max(self.env_config["action"]["target_speeds"]):
            raise ValueError("Maximum traffic speed must be strictly below maximum ego speed.")

    def _configure_traffic(self):
        super()._configure_traffic()
        if self.env_config.get("randomize_traffic_speed", False):
            traffic = next(car for car in self.highway.road.vehicles if car is not self.highway.vehicle)
            traffic.speed = traffic.target_speed = float(self.highway.np_random.uniform(*self.env_config["traffic_speed_range"]))
        if self.env_config.get("traffic_behavior", "idm") != "constant_speed":
            return  # Older BACK-only sidecars retain their original IDM car.
        # A simple reference car holds its lane and speed, including after a pass.
        # Standard vehicle physics still handles collisions with either car.
        ego = self.highway.vehicle
        index = next(i for i, car in enumerate(self.highway.road.vehicles) if car is not ego)
        original = self.highway.road.vehicles[index]
        traffic = ControlledVehicle(self.highway.road, original.position.copy(), original.heading,
                                    original.speed, target_lane_index=original.lane_index,
                                    target_speed=original.target_speed)
        traffic.color = original.color
        self.highway.road.vehicles[index] = traffic

    def _generate_goal(self, options):
        options = dict(options)
        default = self.goal_config.get("default_slot", "random")
        if "slot" not in options and default != "random":
            options["slot"] = default
        slots = self.goal_config.get("sampled_slots", self.goal_config.get("enabled_slots"))
        if "slot" not in options and slots is not None:
            options["slot"] = slots[int(self.highway.np_random.integers(len(slots)))]
        enabled = self.goal_config.get("enabled_slots", SLOTS)
        if "slot" in options and options["slot"] not in enabled:
            raise ValueError(f"Goal {options['slot']!r} is disabled; enabled slots: {enabled}.")
        if self.env_config.get("randomize_traffic_lane", False):
            # Select the slot first so all four have equal sampling probability.
            # LEFT/RIGHT then select a lane that can host that slot on the road.
            rng = self.highway.np_random
            slot = options.get("slot")
            if slot is None:
                slot = SLOTS[int(rng.integers(len(SLOTS)))]
                options["slot"] = slot
            if slot not in SLOTS:
                raise ValueError(f"Invalid slot: {slot!r}.")
            lanes = list(range(self.env_config["lanes_count"]))
            eligible = [lane for lane in lanes
                        if (slot != "left" or lane > 0)
                        and (slot != "right" or lane < lanes[-1])]
            if not eligible:
                raise ValueError(f"No lane can host a {slot} slot.")
            traffic = next(car for car in self.highway.road.vehicles if car is not self.highway.vehicle)
            lane_id = int(rng.choice(eligible))
            lane = self.highway.road.network.get_lane(("0", "1", lane_id))
            traffic.position = lane.position(float(traffic.position[0]), 0).astype(np.float64)
            traffic.on_state_update()
            traffic.target_lane_index = traffic.lane_index
        super()._generate_goal(options)
        self.initial_scenario = {
            "slot": self.slot, "ego_lane": int(self.highway.vehicle.lane_index[2]),
            "traffic_lane": int(self.anchor.lane_index[2]),
            "traffic_speed": float(self.anchor.speed), "ego_speed": float(self.highway.vehicle.speed),
        }

    def _info(self, success, collision, timeout, anchor_lost=False):
        info = super()._info(success, collision, timeout, anchor_lost)
        info["initial_scenario"] = dict(self.initial_scenario)
        return info

    def in_slot(self):
        # Passing through the box at high speed must not count as following.
        speed_error = abs(self.highway.vehicle.speed - self.anchor.speed)
        return super().in_slot() and speed_error <= self.goal_config["speed_tolerance"]

    def render(self):
        if self.render_mode is None:
            return None
        from baseline_dynamic_waypoint.rendering import SingleCarViewer
        if self.highway.viewer is None:
            self.highway.viewer = SingleCarViewer(self.highway, self)
        self.highway.viewer.display()
        if not self.highway.viewer.offscreen:
            self.highway.viewer.handle_events()
        return self.highway.viewer.get_image() if self.render_mode == "rgb_array" else None


def make_single_car_env(render_mode=None, **configuration):
    if configuration.get("goal_config", {}).get("maneuver_guidance") is not None:
        from baseline_dynamic_waypoint.maneuver_guidance import ManeuverGuidanceEnv
        return ManeuverGuidanceEnv(render_mode=render_mode, **configuration)
    if configuration.get("env_config", {}).get("complete_lane_changes", False):
        from baseline_dynamic_waypoint.committed_lane import CommittedLaneEnv
        return CommittedLaneEnv(render_mode=render_mode, **configuration)
    return SingleCarWaypointEnv(render_mode=render_mode, **configuration)
