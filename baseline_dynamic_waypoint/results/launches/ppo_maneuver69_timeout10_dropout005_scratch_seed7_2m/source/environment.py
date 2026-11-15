"""A goal that moves with a traffic car: its front, back, left or right slot."""
from gymnasium import spaces
import numpy as np

from baseline_dynamic_waypoint.config import ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS, SLOTS
from baseline_waypoint.environment import GoalHighwayEnv


class DynamicGoalHighwayEnv(GoalHighwayEnv):
    """GoalHighwayEnv whose target is recomputed every step from an anchor car.

    The slot is defined in the anchor's lane frame: FRONT/BACK are slot_offsets
    meters along its lane, LEFT/RIGHT are beside it in the adjacent lane. Success
    means collecting hold_seconds in the slot, with an optional brief exit grace.
    """

    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        super().__init__(render_mode=render_mode,
                         env_config=ENV_CONFIG if env_config is None else env_config,
                         goal_config=GOAL_CONFIG if goal_config is None else goal_config,
                         reward_weights=REWARD_WEIGHTS if reward_weights is None else reward_weights)
        self.observation_schema = self.goal_config.get("observation_schema", "legacy")
        self.anchor, self.slot, self.anchor_id = None, SLOTS[0], -1
        self.hold_steps, self.hold_required = 0, 1
        self.hold_gap_steps, self.hold_grace_steps = 0, 0
        if self.observation_schema == "lane_goal_v2":
            n = self.env_config["observation"]["vehicles_count"]
            last_lane = self.env_config["lanes_count"] - 1
            low = np.tile(np.array([-1, -1, -1, -1, -1, 0], dtype=np.float32), (n, 1))
            high = np.tile(np.array([1, 1, 1, 1, 1, last_lane], dtype=np.float32), (n, 1))
            self.observation_space = spaces.Dict({
                "vehicles": spaces.Box(low, high, dtype=np.float32),
                "goal_delta": spaces.Box(-np.inf, np.inf, (2,), np.float32),
                "anchor": spaces.Box(-1.0, 1.0, (4,), np.float32),
                # Signed lane counts: [anchor lane - ego lane, goal lane - ego lane].
                "lane_difference": spaces.Box(-last_lane, last_lane, (2,), np.float32),
                # Stable episode car ID, then (goal - anchor) / position_scale.
                "desired_goal": spaces.Box(
                    np.array([0, -np.inf, -np.inf], dtype=np.float32),
                    np.array([self.env_config["vehicles_count"], np.inf, np.inf], dtype=np.float32)),
                "hold_progress": spaces.Box(0.0, 1.0, (1,), np.float32),
            })
            return
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            # Anchor x, y, vx, vy relative to ego, normalized like the Kinematics rows.
            "anchor": spaces.Box(-1.0, 1.0, (4,), np.float32),
            "slot": spaces.Box(0.0, 1.0, (len(SLOTS),), np.float32),
            "hold_progress": spaces.Box(0.0, 1.0, (1,), np.float32),
        })
        # Checkpoints saved before this key existed lack it in their sidecar: keep their 50 inputs.
        if self.goal_config.get("slot_error_features", False):
            self.observation_space = spaces.Dict({
                **self.observation_space.spaces,
                "slot_error": spaces.Box(-1.0, 1.0, (3,), np.float32),
            })
        if self.observation_schema == "legacy_target_lanes":
            retained = dict(self.observation_space.spaces)
            # achieved_goal = desired_goal - goal_delta; use these two redundant
            # inputs for explicit target-car and goal lane IDs.
            retained.pop("achieved_goal")
            retained["target_lanes"] = spaces.Box(
                0, self.env_config["lanes_count"] - 1, (2,), np.float32)
            self.observation_space = spaces.Dict(retained)

    def _validate_config(self):
        super()._validate_config()
        goal = self.goal_config
        timeout_penalty = self.reward_weights.get("timeout", 0.0)
        if not np.isfinite(timeout_penalty) or timeout_penalty < 0:
            raise ValueError("The timeout penalty weight must be finite and nonnegative.")
        schema = goal.get("observation_schema", "legacy")
        if schema not in ("legacy", "legacy_target_lanes", "lane_goal_v2"):
            raise ValueError("observation_schema must be legacy, legacy_target_lanes, or lane_goal_v2.")
        if schema == "lane_goal_v2":
            observation = self.env_config["observation"]
            if observation.get("order", "sorted") != "sorted" or observation.get("include_obstacles", False):
                raise ValueError("lane_goal_v2 requires sorted vehicle observations without obstacles.")
        if goal["hold_seconds"] <= 0 or goal["lateral_tolerance"] <= 0:
            raise ValueError("hold_seconds and lateral_tolerance must be positive.")
        if min(goal["slot_offsets"].values()) - goal["radius"] < 5.0:
            raise ValueError("A FRONT/BACK slot box would overlap the anchor car (5 m long).")
        if goal["hold_seconds"] >= self.env_config["duration"]:
            raise ValueError("The hold time must fit inside the episode.")
        grace = goal.get("hold_grace_seconds", 0.0)
        if not np.isfinite(grace) or grace < 0:
            raise ValueError("hold_grace_seconds must be finite and nonnegative.")
        if "front_lane_blend" in goal:
            blend = goal["front_lane_blend"]
            if (len(blend) != 2 or not np.isfinite(blend).all()
                    or not 0 <= blend[0] < blend[1]):
                raise ValueError("front_lane_blend must be two finite ordered nonnegative lead distances.")

    def _generate_goal(self, options):
        rng = self.highway.np_random
        ego = self.highway.vehicle
        low, high = self.goal_config["distance_range"]
        candidates = sorted((car for car in self.highway.road.vehicles
                             if car is not ego and low <= car.position[0] - ego.position[0] <= high),
                            key=lambda car: car.position[0])
        if not candidates:
            raise ValueError("No traffic car is inside distance_range to anchor the waypoint.")
        index = int(options.get("anchor_index", rng.integers(len(candidates))))
        if not 0 <= index < len(candidates):
            raise ValueError(f"anchor_index must be below {len(candidates)} (eligible cars, nearest first).")
        anchor = candidates[index]
        lane_id = anchor.lane_index[2]
        valid = [slot for slot in SLOTS
                 if (slot != "left" or lane_id > 0) and (slot != "right" or lane_id < self.env_config["lanes_count"] - 1)]
        slot = options.get("slot", valid[rng.integers(len(valid))])
        if slot not in valid:
            raise ValueError(f"Slot {slot!r} is off the road for an anchor in lane {lane_id}; valid: {valid}.")
        # IDM still sets the anchor's speed; it just never moves its LEFT/RIGHT slot off the road.
        anchor.enable_lane_change = False
        anchor.target_lane_index = anchor.lane_index
        self.anchor, self.slot = anchor, slot
        self.anchor_id = self.highway.road.vehicles.index(anchor)
        self.hold_steps = 0
        self.hold_gap_steps = 0
        self.hold_grace_steps = round(self.goal_config.get("hold_grace_seconds", 0.0) * self.env_config["policy_frequency"])
        self.hold_required = max(1, round(self.goal_config["hold_seconds"] * self.env_config["policy_frequency"]))
        self._update_goal()
        self.previous_shaping_errors = self.shaping_errors()

    def _slot_lane(self):
        """The slot's lane and its longitudinal coordinate on that lane."""
        network = self.highway.road.network
        anchor_index = self.anchor.lane_index
        s, _ = network.get_lane(anchor_index).local_coordinates(self.anchor.position)
        if self.slot in ("front", "back"):
            offset = self.goal_config["slot_offsets"][self.slot]
            return anchor_index, s + offset if self.slot == "front" else s - offset
        side = -1 if self.slot == "left" else 1  # LANE_LEFT lowers the lane ID.
        return (anchor_index[0], anchor_index[1], anchor_index[2] + side), s

    def _update_goal(self):
        self.slot_lane_index, slot_s = self._slot_lane()
        self.goal_lane = int(self.slot_lane_index[2])
        self.goal_position = self.highway.road.network.get_lane(self.slot_lane_index).position(slot_s, 0).astype(np.float64)

    def slot_error(self):
        """Slot minus ego in the slot lane's frame: (along the lane, across it) in meters."""
        lane = self.highway.road.network.get_lane(self.slot_lane_index)
        ego_s, ego_lateral = lane.local_coordinates(self.highway.vehicle.position)
        slot_s, _ = lane.local_coordinates(self.goal_position)
        return slot_s - ego_s, -ego_lateral

    def shaping_errors(self):
        """(lateral error m, speed error m/s, lead over the anchor m) for the shaping terms.

        The speed term only matters within shaping_range meters along the lane: far
        away, ego should drive fast to catch up rather than match the anchor's speed.
        With the overtake term enabled, a FRONT slot's lateral target is the nearest
        lane beside the anchor until ego has passed it, so the lane term no longer
        punishes the detour an overtake needs. The lead is 0 for other slots.
        """
        longitudinal, lateral = self.slot_error()
        # Sidecars written before shaping lack the key; their weights ignore this term anyway.
        proximity = max(0.0, 1.0 - abs(longitudinal) / self.goal_config.get("shaping_range", 30.0))
        speed_error = proximity * abs(self.anchor.speed - self.highway.vehicle.speed)
        if self.slot != "front" or "overtake" not in self.reward_weights:
            return abs(lateral), speed_error, 0.0
        network, ego = self.highway.road.network, self.highway.vehicle
        anchor_lane = network.get_lane(self.anchor.lane_index)
        lead = anchor_lane.local_coordinates(ego.position)[0] - anchor_lane.local_coordinates(self.anchor.position)[0]
        if "front_lane_blend" in self.goal_config:
            road, start, lane_id = self.anchor.lane_index
            beside = [(road, start, lane) for lane in (lane_id - 1, lane_id + 1)
                      if 0 <= lane < self.env_config["lanes_count"]]
            passing_error = min(abs(network.get_lane(index).local_coordinates(ego.position)[1])
                                for index in beside)
            low, high = self.goal_config["front_lane_blend"]
            blend = float(np.clip((lead - low) / (high - low), 0.0, 1.0))
            # A continuous potential avoids penalizing a successful pass with
            # an abrupt four-meter error jump at pass_margin. In the 5..15 m
            # profile the passing lane remains preferred until a 10 m lead.
            lateral = (1.0 - blend) * passing_error + blend * abs(lateral)
        elif lead < self.goal_config["pass_margin"]:
            road, start, lane_id = self.anchor.lane_index
            beside = [(road, start, lane) for lane in (lane_id - 1, lane_id + 1)
                      if 0 <= lane < self.env_config["lanes_count"]]
            lateral = min(abs(network.get_lane(index).local_coordinates(ego.position)[1]) for index in beside)
        return abs(lateral), speed_error, float(np.clip(lead, -150.0, self.goal_config["slot_offsets"]["front"]))

    def in_slot(self):
        longitudinal, lateral = self.slot_error()
        return bool(abs(longitudinal) < self.goal_config["radius"]
                    and abs(lateral) < self.goal_config["lateral_tolerance"])

    def anchor_lost(self):
        return bool(self.anchor.crashed or not self.anchor.on_road
                    or self.anchor not in self.highway.road.vehicles)

    def _observation(self, raw):
        observation = super()._observation(raw)
        ego, anchor = self.highway.vehicle, self.anchor
        ranges = self.env_config["observation"]["features_range"]
        relative = {"x": anchor.position[0] - ego.position[0], "y": anchor.position[1] - ego.position[1],
                    "vx": anchor.velocity[0] - ego.velocity[0], "vy": anchor.velocity[1] - ego.velocity[1]}
        observation["anchor"] = np.array([
            np.clip(2 * (value - ranges[key][0]) / (ranges[key][1] - ranges[key][0]) - 1, -1, 1)
            for key, value in relative.items()], dtype=np.float32)
        if self.observation_schema == "lane_goal_v2":
            encoder = self.highway.observation_type
            n = self.env_config["observation"]["vehicles_count"]
            close = self.highway.road.close_objects_to(
                encoder.observer_vehicle, self.highway.PERCEPTION_DISTANCE,
                count=n - 1, see_behind=encoder.see_behind, sort=True, vehicles_only=True)
            cars = [encoder.observer_vehicle, *close[:n - 1]]
            lanes = np.zeros((n, 1), dtype=np.float32)
            for row, car in enumerate(cars):
                lanes[row, 0] = car.lane_index[2]
            ego_lane = int(ego.lane_index[2])
            relative_goal = (self.goal_position - anchor.position) / self.position_scale
            return {
                "vehicles": np.concatenate((observation["vehicles"], lanes), axis=1),
                "goal_delta": observation["goal_delta"],
                "anchor": observation["anchor"],
                "lane_difference": np.array([int(anchor.lane_index[2]) - ego_lane,
                                              self.goal_lane - ego_lane], dtype=np.float32),
                "desired_goal": np.array([self.anchor_id, *relative_goal], dtype=np.float32),
                "hold_progress": np.array([min(1.0, self.hold_steps / self.hold_required)], dtype=np.float32),
            }
        observation["slot"] = np.array([slot == self.slot for slot in SLOTS], dtype=np.float32)
        observation["hold_progress"] = np.array([min(1.0, self.hold_steps / self.hold_required)], dtype=np.float32)
        if "slot_error" in self.observation_space.spaces:
            # goal_delta's 200 m scale shrinks the +/-3 m box to +/-0.015; these local
            # scales make the box +/-0.3 and expose the speed mismatch directly.
            longitudinal, lateral = self.slot_error()
            scales = self.goal_config["slot_error_scale"]
            error = [longitudinal / scales[0], lateral / scales[1], (anchor.speed - ego.speed) / scales[2]]
            observation["slot_error"] = np.clip(error, -1, 1).astype(np.float32)
        if self.observation_schema == "legacy_target_lanes":
            observation.pop("achieved_goal")
            observation["target_lanes"] = np.array(
                [int(anchor.lane_index[2]), self.goal_lane], dtype=np.float32)
        return observation

    def step(self, action):
        raw, _, _, native_truncated, _ = self.env.step(int(action))
        lost = self.anchor_lost()
        if not lost:
            self._update_goal()  # The waypoint moves with the anchor car.
        distance = self.distance_to_goal()
        collision = bool(self.highway.vehicle.crashed)
        inside = not (collision or lost) and self.in_slot()
        if inside:
            self.hold_steps += 1
            self.hold_gap_steps = 0
        else:
            self.hold_gap_steps += 1
            if collision or lost or self.hold_gap_steps > self.hold_grace_steps:
                self.hold_steps = 0
        # Grace preserves previous valid time; it never advances the counter or
        # declares success while outside the slot or after a collision/anchor loss.
        success = bool(inside and self.hold_steps >= self.hold_required and self._success_allowed())
        terminated = success or collision
        # Losing the anchor ends the task without it being ego's failure, like a timeout.
        truncated = bool((native_truncated or lost) and not terminated)
        weights = self.reward_weights
        lateral_error, speed_error, lead = self.shaping_errors()
        previous_lateral, previous_speed, previous_lead = self.previous_shaping_errors
        components = {
            "goal": weights["goal"] * float(success),
            "collision": -weights["collision"] * float(collision),
            # Distance to a moving point: still telescopes, since each step uses the previous distance.
            "progress": weights["progress"] * (self.previous_distance - distance),
            "time": -weights["time"],
            "hold": weights["hold"] * float(inside),
            # State-error differences telescope in undiscounted returns. PPO
            # still applies its usual discount per policy decision.
            "lane": weights.get("lane", 0.0) * (previous_lateral - lateral_error),
            "speed_match": weights.get("speed_match", 0.0) * (previous_speed - speed_error),
            # FRONT only: per meter gained on the anchor, capped at the slot offset.
            "overtake": weights.get("overtake", 0.0) * (lead - previous_lead),
            "timeout": -weights.get("timeout", 0.0) * float(truncated and not lost),
        }
        # Checkpoints trained before shaping keep exactly their five reward terms.
        self.last_components = {name: value for name, value in components.items() if name in weights}
        for name, value in self.last_components.items():
            self.episode_totals[name] += value
        self.previous_distance = distance
        self.previous_shaping_errors = (lateral_error, speed_error, lead)
        self.status = ("GOAL REACHED" if success else "COLLISION" if collision else "ANCHOR LOST" if truncated and lost
                       else "TIMED OUT" if truncated else "HOLDING" if inside
                       else "REACQUIRING" if self.hold_steps else "Driving")
        info = self._info(success, collision, truncated and not lost, anchor_lost=truncated and lost)
        if self.render_mode == "human":
            self.render()
        return self._observation(raw), float(sum(self.last_components.values())), terminated, truncated, info

    def _success_allowed(self):
        return True

    def _info(self, success, collision, timeout, anchor_lost=False):
        info = super()._info(success, collision, timeout)
        info.update({
            "is_anchor_lost": bool(anchor_lost), "slot": self.slot, "anchor_id": self.anchor_id,
            "anchor_position": self.anchor.position.copy(),
            "hold_steps": self.hold_steps, "hold_required": self.hold_required,
            "hold_gap_steps": self.hold_gap_steps, "hold_grace_steps": self.hold_grace_steps,
        })
        return info

    def render(self):
        if self.render_mode is None:
            return None
        from baseline_dynamic_waypoint.rendering import DynamicGoalViewer
        if self.highway.viewer is None:
            self.highway.viewer = DynamicGoalViewer(self.highway, self)
        self.highway.viewer.display()
        if not self.highway.viewer.offscreen:
            self.highway.viewer.handle_events()
        if self.render_mode == "rgb_array":
            return self.highway.viewer.get_image()
        return None
