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

Chained goals (Step 9; evaluation only)
---------------------------------------
``chain`` (a constructor setting; default none, which changes nothing) turns one episode into a series of
goals: when a goal is reached before the last one, the episode does not end; the next goal is drawn and
everything that depends on the goal is recomputed as at reset. Settings:

- ``goals``: the most goals per chain (K). ``per_goal_seconds``: each goal's own time limit (25 s, the
  training limit; the simulation's own limit is raised to K x that).
- ``mode``: ``"new car"`` draws the next anchor as reset does (a car 30-120 m ahead of ego, not the current
  anchor) with a valid slot; the old anchor may change lanes again. ``"same car"`` keeps the anchor and draws
  a different valid slot of it.
- ``origin``: ``"reset"`` restarts the episode origin at ego (so the goal inputs are what a fresh episode
  from here would show); ``"keep"`` leaves the origin at the first goal's start.
- Feasibility: a candidate goal is used only if ``goal_time_needed`` (time to close the gap along the road at
  the speed limits, plus 2 s per lane change) is at most ``CHAIN_TIME_LIMIT``, the 95th percentile of the same
  measure over goals drawn at reset (computed on seeds 90,000-91,999). So every chained goal is within the
  range of goals the task itself poses. If no candidate qualifies, the chain ends ("no feasible goal"), which
  is recorded and is not a failure of the policy.
- Goal draws come from their own random stream, seeded by the episode seed; the simulator's stream is untouched.
"""
import numpy as np

from baseline_DQN.fast_observation import enable_fast_observation
from baseline_dynamic_waypoint.config import SLOTS
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv

SHAPING_TERMS = ("potential_progress",)
TARGET_SLOTS = ("back", "left", "right")  # FRONT is set aside (EXPERIMENT_NOTES.md, Step 5).
MAX_SCENARIO_RESETS = 1000
START_OFFSETS = {"past": 10.0, "beside": 0.0, "behind": -15.0}  # Ego center minus blocking car center, m.
START_CLEARANCE = 8.0  # Minimum bumper-to-bumper gap to any car in ego's new lane, m.
START_BEFORE_SLOT = 10.0  # Ego must start at least this far before the slot center, m.
CHAIN_MODES = ("new car", "same car")
CHAIN_ORIGINS = ("reset", "keep")
CHAIN_STREAM = 9  # Distinguishes the chain's goal-draw stream from any other stream seeded by the episode seed.
LANE_CHANGE_SECONDS = 2.0  # As the inherited goal check (baseline_waypoint) counts a lane change.
# 95th percentile of goal_time_needed over 2,000 goals drawn at reset (chain_envelope.py, seeds 90,000-91,999;
# median 8.2 s, max 21.1 s; results/chains/envelope.json).
CHAIN_TIME_LIMIT = 15.72


def goal_time_needed(env, anchor, slot):
    """Seconds needed at best to bring ego to `anchor`'s `slot`: the gap along the road closed at the top speed
    (slot ahead) or the lowest speed (slot behind) relative to the anchor's current speed, plus 2 s per lane
    change. Infinite if the speed limits cannot close it."""
    ego = env.highway.vehicle
    offsets = env.goal_config["slot_offsets"]
    along = {"front": offsets["front"], "back": -offsets["back"]}.get(slot, 0.0)
    gap = anchor.position[0] + along - ego.position[0]
    speeds = env.env_config["action"]["target_speeds"]
    relative = (max(speeds) - anchor.speed) if gap >= 0 else (anchor.speed - min(speeds))
    if relative <= 0:
        return float("inf")
    lane = anchor.lane_index[2] + {"left": -1, "right": 1}.get(slot, 0)
    return abs(gap) / relative + LANE_CHANGE_SECONDS * abs(lane - ego.lane_index[2])


class ReachEnv(DynamicGoalHighwayEnv):
    def __init__(self, render_mode=None, reward_scale=1.0, scenario=None, fast_observation=True, chain=None,
                 **settings):
        self.chain = None
        if chain is not None:
            chain = {"goals": 5, "per_goal_seconds": 25.0, "mode": "new car", "origin": "reset", **chain}
            if chain["mode"] not in CHAIN_MODES or chain["origin"] not in CHAIN_ORIGINS or chain["goals"] < 1:
                raise ValueError(f"chain needs mode in {CHAIN_MODES}, origin in {CHAIN_ORIGINS}, goals >= 1: {chain}")
            settings = dict(settings)
            settings["env_config"] = {**settings["env_config"],
                                      "duration": chain["goals"] * chain["per_goal_seconds"] + 1.0}
            self.chain = chain
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
        if self.chain is not None:
            entropy = [CHAIN_STREAM, seed] if seed is not None else [CHAIN_STREAM, int(self.highway.np_random.integers(2**31))]
            self.chain_rng = np.random.default_rng(entropy)
            self.goal_index, self.goal_start_time = 1, float(self.highway.time)
            info["goal_index"] = 1
        return observation, info

    def _chain_options(self):
        """[(car, valid slots, feasible slots)] for the next goal of a chain."""
        ego, road = self.highway.vehicle, self.highway.road
        if self.chain["mode"] == "same car":
            cars = [] if self.anchor_lost() else [self.anchor]
        else:
            low, high = self.goal_config["distance_range"]
            cars = sorted((car for car in road.vehicles if car is not ego and car is not self.anchor and not car.crashed
                           and car.on_road and low <= car.position[0] - ego.position[0] <= high),
                          key=lambda car: car.position[0])
        last = self.env_config["lanes_count"] - 1
        options = []
        for car in cars:
            lane = car.lane_index[2]
            valid = [s for s in SLOTS if (s != "left" or lane > 0) and (s != "right" or lane < last)]
            if self.chain["mode"] == "same car":
                valid = [s for s in valid if s != self.slot]
            options.append((car, valid, [s for s in valid if goal_time_needed(self, car, s) <= CHAIN_TIME_LIMIT]))
        return options

    def _chain_switch(self, info):
        """Draw and install the next goal after a reached one. Returns the new observation, or None when the chain
        cannot continue (info["chain_end"] says why)."""
        options = self._chain_options()
        usable = [(car, feasible) for car, _, feasible in options if feasible]
        info["chain_candidates"] = sum(len(valid) for _, valid, _ in options)
        info["chain_excluded"] = sum(len(valid) - len(feasible) for _, valid, feasible in options)
        if not usable:
            info["chain_end"] = "no car in range" if not options else "no feasible goal"
            return None
        car, feasible = usable[int(self.chain_rng.integers(len(usable)))]
        slot = feasible[int(self.chain_rng.integers(len(feasible)))]
        ego = self.highway.vehicle
        if car is not self.anchor:
            self.anchor.enable_lane_change = True  # Back to ordinary traffic.
        car.enable_lane_change = False
        car.target_lane_index = car.lane_index
        info["next_slot"], info["next_time_needed"] = slot, goal_time_needed(self, car, slot)
        self.anchor, self.slot = car, slot
        self.anchor_id = self.highway.road.vehicles.index(car)
        self.hold_steps = self.hold_gap_steps = 0
        if self.chain["origin"] == "reset":
            self.episode_origin = np.array([ego.position[0], 0.0])
        self._update_goal()
        self.previous_distance = self.distance_to_goal()
        self.previous_shaping_errors = self.shaping_errors()
        self.previous_potential = self.potential()
        self.goal_index += 1
        self.goal_start_time = float(self.highway.time)
        self.status = "Driving"
        info["goal_switched"] = True
        return self._observation(self.highway.observation_type.observe())

    def _chain_step(self, observation, reward, terminated, truncated, info):
        info["goal_index"] = self.goal_index
        if not (terminated or truncated) and \
                self.highway.time - self.goal_start_time >= self.chain["per_goal_seconds"] - 1e-9:
            truncated = True  # This goal's own time limit; the simulation's limit is K goals long.
            self.status = "TIMED OUT"
            info.update(is_timeout=True, termination_reason="TIMED OUT")
        if info["is_success"] and self.goal_index < self.chain["goals"]:
            switched = self._chain_switch(info)
            if switched is not None:
                return switched, reward, False, False, info
        elif info["is_success"]:
            info["chain_end"] = "all goals reached"
        return observation, reward, terminated, truncated, info

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
        if self.chain is not None:
            return self._chain_step(*self._step(action))
        return self._step(action)

    def _step(self, action):
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
