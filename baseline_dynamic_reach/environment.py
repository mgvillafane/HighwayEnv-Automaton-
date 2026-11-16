"""The reach environment: the inherited DynamicGoalHighwayEnv plus optional reward terms.

Every term here is off unless an arm gives it a weight. With none of them weighted, ReachEnv
behaves exactly like DynamicGoalHighwayEnv (tested), so arms trained before a term existed
are unaffected by its addition.

Terms
-----
``potential_progress``: potential-based shaping toward the slot (Ng, Harada & Russell 1999).
    Configured as ``{"weight": w, "gamma": g}`` in ``reward_weights``. With the potential
    ``Phi(s) = -distance to the slot center (m)``, each decision pays

        F = w * (g * Phi(s') - Phi(s)),   with Phi(s') = 0 when the episode terminates.

    When ``g`` equals PPO's discount (train.py enforces it), the discounted sum of F over any
    terminated episode is exactly ``-w * Phi(s_0) = w * initial distance``: a constant that does
    not depend on what the policy does, so the best policy under the shaped reward is the best
    policy under the unshaped one. The term only changes *when* credit arrives. On a time-limit
    or anchor-lost truncation the episode is not over for the value function, which bootstraps,
    so Phi(s') keeps its real value there.

    The inherited ``progress`` term pays ``w * (d - d')``; it omits g, so its discounted sum
    depends on the path and it can change the best policy.

Reward scale
------------
``reward_scale`` (an ``environment`` setting next to ``reward_weights``; default 1) multiplies the
whole reward and every logged component. A positive rescaling cannot change which policy is best.
It exists because of how Stable-Baselines3 trains: one gradient-norm clip (0.5) covers the policy and
value networks together, and with rewards of +/-50 the value gradient is 30-120x the policy gradient,
so the clip shrinks every policy update (EXPERIMENT_NOTES.md, Step 7a). At scale 1 the code path is
the one used by every earlier arm, unchanged.

Scenario (passing practice)
---------------------------
``scenario`` (an ``environment`` setting; default none) can restrict which episodes are drawn:

- ``require_blocked``: keep resetting (continuing the seeded random stream, so a seed still names one
  episode) until the slot is BACK, LEFT or RIGHT and at least one car sits in the slot's lane between
  ego and the slot: the "blocked" episodes of Step 5 (``blocking_cars``; ``blocked.py`` uses it too).
- ``blocker_stays``: switch off lane changes for those blocking cars, as for the anchor, so waiting
  behind them never clears the way and a pass is the only route to the slot.
- ``start_offsets``: practice starts (Step 7e). Each episode picks one of the listed starts uniformly
  from the seeded stream and moves ego (same speed) relative to the nearest blocking car:
  ``past`` next lane, 10 m ahead of it; ``beside`` next lane, level with it; ``behind`` next lane,
  15 m behind it; ``natural`` no move. "Next lane" is the lane beside the slot's lane nearest to ego
  (a random one if both are equally near). A start is used only with at least 8 m of bumper clearance
  to every car in that lane and at least 10 m before the slot; otherwise the episode keeps its natural
  start. ``info["start"]`` records the start actually used.

Observations are assembled with Marcos's NumPy encoder (baseline_DQN/fast_observation.py). It
reproduces highway-env's Kinematics observation exactly (tested here on seeded episodes); it only
avoids building a pandas DataFrame on every step.
"""
import numpy as np

from baseline_DQN.fast_observation import enable_fast_observation
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv

SHAPING_TERMS = ("potential_progress",)
TARGET_SLOTS = ("back", "left", "right")  # FRONT is set aside (EXPERIMENT_NOTES.md, Step 5).
MAX_SCENARIO_RESETS = 1000
START_OFFSETS = {"past": 10.0, "beside": 0.0, "behind": -15.0}  # Ego center minus blocking car center, m.
START_CLEARANCE = 8.0  # Minimum bumper-to-bumper gap to any car in ego's new lane, m.
START_BEFORE_SLOT = 10.0  # Ego must start at least this far before the slot center, m.


class ReachEnv(DynamicGoalHighwayEnv):
    def __init__(self, render_mode=None, reward_scale=1.0, scenario=None, fast_observation=True, **settings):
        super().__init__(render_mode=render_mode, **settings)
        self.scenario = dict(scenario or {})
        unknown = set(self.scenario) - {"require_blocked", "blocker_stays", "start_offsets"}
        if unknown:
            raise ValueError(f"Unknown scenario settings: {sorted(unknown)}")
        starts = self.scenario.get("start_offsets")
        if starts is not None:
            if not starts or set(starts) - {*START_OFFSETS, "natural"}:
                raise ValueError(f"start_offsets must be a non-empty list from {[*START_OFFSETS, 'natural']}.")
            if not self.scenario.get("require_blocked"):
                raise ValueError("start_offsets needs require_blocked: starts are placed relative to a blocking car.")
        if self.scenario.get("blocker_stays") and not self.scenario.get("require_blocked"):
            raise ValueError("blocker_stays needs require_blocked: without it there may be no blocking car.")
        if not reward_scale > 0:
            raise ValueError("reward_scale must be positive; a negative or zero scale changes the task.")
        self.reward_scale = float(reward_scale)
        if fast_observation:
            enable_fast_observation(self.env)

    def _scaled(self, observation, reward, terminated, truncated, info):
        if self.reward_scale == 1.0:
            return observation, reward, terminated, truncated, info
        scale = self.reward_scale
        info["reward_components"] = {name: scale * value for name, value in info["reward_components"].items()}
        info["episode_reward_components"] = {name: scale * value
                                             for name, value in info["episode_reward_components"].items()}
        return observation, scale * reward, terminated, truncated, info

    def _shaping(self, name):
        spec = self.reward_weights.get(name)
        return spec if isinstance(spec, dict) and spec.get("weight", 0.0) else None

    def potential(self):
        return -self.distance_to_goal()

    def blocking_cars(self):
        """[(position along the slot's lane, index in road.vehicles)] of the cars in the slot's lane that are
        ahead of ego and behind the slot center, nearest first (Step 5's "blocked" test)."""
        ego, road = self.highway.vehicle, self.highway.road
        lane = road.network.get_lane(self.slot_lane_index)
        ego_s, slot_s = lane.local_coordinates(ego.position)[0], lane.local_coordinates(self.goal_position)[0]
        return sorted((lane.local_coordinates(v.position)[0], road.vehicles.index(v)) for v in road.vehicles
                      if v is not ego and v.lane_index[2] == self.slot_lane_index[2]
                      and ego_s < lane.local_coordinates(v.position)[0] < slot_s)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.scenario.get("require_blocked"):
            resets = 1
            while not (info["slot"] in TARGET_SLOTS and self.blocking_cars()):
                if resets >= MAX_SCENARIO_RESETS:
                    raise RuntimeError(f"No blocked episode in {MAX_SCENARIO_RESETS} resets.")
                observation, info = super().reset(options=options)  # Continues the seeded stream.
                resets += 1
            info["scenario_resets"] = resets
            if self.scenario.get("blocker_stays"):
                for _, index in self.blocking_cars():
                    self.highway.road.vehicles[index].enable_lane_change = False
            if self.scenario.get("start_offsets"):
                starts = self.scenario["start_offsets"]
                start = starts[int(self.highway.np_random.integers(len(starts)))]
                if start != "natural" and not self._place_ego(START_OFFSETS[start]):
                    start = "natural"
                if start != "natural":
                    observation, info = self._restart_from_ego_position(info)
                    info["start_reference"] = self.start_reference
                info["start"] = start
        self.previous_potential = self.potential()
        return observation, info

    def _place_ego(self, offset):
        """Move ego to the lane beside the slot's lane, `offset` m from the nearest blocking car. False if unsafe."""
        road, ego = self.highway.road, self.highway.vehicle
        slot_lane = self.slot_lane_index
        neighbours = [(slot_lane[0], slot_lane[1], slot_lane[2] + side) for side in (-1, 1)
                      if 0 <= slot_lane[2] + side < self.env_config["lanes_count"]]
        nearest = min(abs(index[2] - ego.lane_index[2]) for index in neighbours)
        neighbours = [index for index in neighbours if abs(index[2] - ego.lane_index[2]) == nearest]
        lane_index = neighbours[int(self.highway.np_random.integers(len(neighbours)))]
        lane = road.network.get_lane(lane_index)
        self.start_reference = self.blocking_cars()[0][1]  # Index of the car the start is placed against.
        blocker = road.vehicles[self.start_reference]
        s = lane.local_coordinates(blocker.position)[0] + offset
        slot_s = lane.local_coordinates(self.goal_position)[0]
        if s > slot_s - START_BEFORE_SLOT:
            return False
        for car in road.vehicles:
            if car is not ego and car.lane_index[2] == lane_index[2]:
                gap = abs(lane.local_coordinates(car.position)[0] - s) - (car.LENGTH + ego.LENGTH) / 2
                if gap < START_CLEARANCE:
                    return False
        ego.position = lane.position(s, 0)
        ego.heading = lane.heading_at(s)
        ego.target_lane_index = lane_index
        ego.on_state_update()
        return ego.lane_index == lane_index

    def _restart_from_ego_position(self, info):
        """Recompute everything reset derives from ego's start, exactly as the inherited reset does."""
        self.episode_origin = np.array([self.highway.vehicle.position[0], 0.0])
        self.previous_distance = self.distance_to_goal()
        self.previous_shaping_errors = self.shaping_errors()
        observation = self._observation(self.highway.observation_type.observe())
        fresh = self._info(False, False, False)
        return observation, {**info, **fresh}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        spec = self._shaping("potential_progress")
        if spec is None:
            return self._scaled(observation, reward, terminated, truncated, info)
        next_potential = 0.0 if terminated else self.potential()
        shaping = spec["weight"] * (spec["gamma"] * next_potential - self.previous_potential)
        self.previous_potential = next_potential
        self.last_components["potential_progress"] = shaping
        self.episode_totals["potential_progress"] += shaping
        info["reward_components"] = dict(self.last_components)
        info["episode_reward_components"] = dict(self.episode_totals)
        return self._scaled(observation, reward + shaping, terminated, truncated, info)
