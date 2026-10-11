"""Traffic DFA observers and task guards evaluated from each car's perspective."""
from dataclasses import dataclass, field, replace

import numpy as np

import mainV2 as v2
from task_dfa import TaskDFA


VEHICLE_COLORS = ((76, 224, 167), (91, 188, 255), (255, 192, 91), (192, 148, 255))
OTHER_COLOR = (107, 128, 150)


@dataclass
class ObservedCar:
    uid: int
    label: str
    color: tuple
    previous: v2.Vehicle
    time: float
    dfa: TaskDFA = field(default_factory=TaskDFA)
    target_lane: tuple | None = None
    present: bool = True
    memory: v2.Memory = field(default_factory=v2.Memory)
    context: v2.TaskContext = field(default_factory=v2.TaskContext)
    bindings: dict | None = None
    labels: set = field(default_factory=set)
    options: dict = field(default_factory=dict)

    def refresh_tasks(self, snapshot, config, options_for_agent, sample=True):
        car = next((v for v in snapshot.vehicles if v.uid == self.uid), None)
        if car is None or car.crashed or self.dfa.state[0] == "ended":
            self.options = {}
            return
        # "ego" in the unchanged AP alphabet means this particular car.
        # Every other car, including the controlled ego, remains an obstacle.
        local = replace(snapshot, ego=car)
        lane_changed = (self.memory.previous is not None
                        and self.memory.previous.ego.lane_index != car.lane_index)
        if self.bindings is None or lane_changed or (sample and self.memory.time != snapshot.time):
            if self.bindings is None:
                self.context.rebind(local)
            self.bindings = v2.bind_roles(local, self.context, config)
            # Only lead/intruder roles and immediate side lanes occur in task
            # guards. Reuse V2's evaluator for that projection of the alphabet;
            # gap calculations still check every vehicle in the snapshot.
            roles = dict.fromkeys(v2.ROLES)
            roles.update({name: self.bindings["roles"][name] for name in ("ego", "lead", "intruder")})
            task_bindings = {**self.bindings, "roles": roles,
                             "refs": {**self.bindings["refs"], "passing_lane": None}}
            self.labels = v2.label_highway_state(local, task_bindings, config, self.memory)
        self.options = options_for_agent(local, self.bindings, self.labels,
                                         self.dfa.state[1] != "none")

    def update(self, snapshot, live):
        car = next((v for v in snapshot.vehicles if v.uid == self.uid), None)
        self.present = car is not None
        if car is None or car.crashed:
            self.dfa.dispatch("end")
            return
        dt = snapshot.time - self.time
        if dt <= 0 or self.dfa.state[0] == "ended":
            return
        lane = snapshot.road.network.get_lane(car.lane_index)
        motion = v2.lateral_motion(car, lane)
        target = tuple(getattr(live, "target_lane_index", None) or car.lane_index)
        lateral = self.dfa.state[1]
        moving_sideways = (abs(motion["lateral_speed"]) > .1
                          or abs(motion["heading_error"]) > .02)
        if lateral == "none":
            destination = None
            if target[:2] == car.lane_index[:2] and target != car.lane_index and moving_sideways:
                destination = target
                direction = "left" if target[2] < car.lane_index[2] else "right"
            elif self.previous.lane_index[:2] == car.lane_index[:2] and self.previous.lane_index != car.lane_index:
                destination = car.lane_index
                direction = "left" if car.lane_index[2] < self.previous.lane_index[2] else "right"
            if destination is not None:
                self.target_lane = destination
                self.dfa.dispatch("switch_lane_" + direction)
        if (self.target_lane == car.lane_index and abs(motion["offset"]) <= .2
                and abs(motion["heading_error"]) <= .03):
            self.dfa.dispatch("lane_complete")
            self.target_lane = None

        speed = float(np.linalg.norm(car.velocity))
        acceleration = (speed - np.linalg.norm(self.previous.velocity)) / dt
        # Enter at 0.5 m/s^2 and leave at 0.2 to avoid flickering near zero.
        old_mode = self.dfa.state[0]
        if acceleration < -.5 or (old_mode == "brake" and acceleration < -.2):
            mode = "brake"
        elif acceleration > .5 or (old_mode == "accelerate" and acceleration > .2):
            mode = "accelerate"
        else:
            s = lane.local_coordinates(car.position)[0]
            ahead = [v for v in snapshot.vehicles if v.uid != car.uid
                     and v.lane_index == car.lane_index and lane.local_coordinates(v.position)[0] > s]
            lead = min(ahead, key=lambda v: lane.local_coordinates(v.position)[0], default=None)
            gap = v2.longitudinal_clearance(car, lead, lane) if lead else np.inf
            target_speed = float(getattr(live, "target_speed", speed))
            following = (lead is not None and gap < 1.5 * v2.required_gap(speed, 0.)
                         and speed < target_speed - .5 and self.dfa.state[1] == "none")
            mode = "follow_car" if following else "cruise"
        if mode != self.dfa.state[0]:
            self.dfa.dispatch(mode)
        self.previous, self.time = car, snapshot.time


class TrafficDFAs:
    """Pin three nearest cars at reset; never silently swap their histories."""
    def __init__(self, predicate_config, options_for_agent, ticks_per_sample=6):
        self.cars = []
        self.predicate_config = predicate_config
        self.options_for_agent = options_for_agent
        self.ticks_per_sample = ticks_per_sample
        self.ticks = 0

    def reset(self, snapshot, env):
        self.ticks = 0
        nearest = sorted((v for v in snapshot.vehicles if v.uid != snapshot.ego.uid),
                         key=lambda v: (np.linalg.norm(v.position - snapshot.ego.position), v.uid))[:3]
        self.cars = [ObservedCar(v.uid, label, color, v, snapshot.time)
                     for v, label, color in zip(nearest, ("A", "B", "C"), VEHICLE_COLORS[1:])]
        self.apply_colors(env)
        self.refresh(snapshot)

    def refresh(self, snapshot):
        """Refresh all guard APs on reset or an explicit state refresh."""
        for car in self.cars:
            car.refresh_tasks(snapshot, self.predicate_config, self.options_for_agent)

    def apply_colors(self, env):
        colors = {car.uid: car.color for car in self.cars}
        for vehicle in env.road.vehicles:
            vehicle.color = (VEHICLE_COLORS[0] if vehicle is env.vehicle
                             else colors.get(env._semantic_ids.get(vehicle), OTHER_COLOR))

    def update(self, snapshot, env):
        live = {uid: vehicle for vehicle, uid in env._semantic_ids.items() if vehicle in env.road.vehicles}
        self.ticks += 1
        for i, observed in enumerate(self.cars):
            observed.update(snapshot, live.get(observed.uid))
            # Speed, maneuver and presence guards update every motion tick;
            # stagger the cars' 5 Hz geometric checks between ego label ticks.
            phase = (i * self.ticks_per_sample // max(1, len(self.cars)) + 1) % self.ticks_per_sample
            observed.refresh_tasks(snapshot, self.predicate_config,
                                   self.options_for_agent,
                                   sample=self.ticks % self.ticks_per_sample == phase)

    def cards(self, snapshot, ego_dfa, ego_options=None):
        vehicles = {v.uid: v for v in snapshot.vehicles}
        cards = [{"label": "EGO", "uid": snapshot.ego.uid, "color": VEHICLE_COLORS[0],
                  "dfa": ego_dfa, "vehicle": snapshot.ego, "source": "executed tasks",
                  "options": ego_options or {}}]
        for i in range(3):
            car = self.cars[i] if i < len(self.cars) else None
            cards.append({"label": car.label if car else ("A", "B", "C")[i],
                          "uid": car.uid if car else None, "color": VEHICLE_COLORS[i + 1],
                          "dfa": car.dfa if car else None,
                          "vehicle": vehicles.get(car.uid) if car else None,
                          "source": "observed motion", "options": car.options if car else {}})
        return cards
