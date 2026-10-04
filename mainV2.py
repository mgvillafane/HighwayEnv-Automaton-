"""Interactive traffic facts, without training or a DFA.

Geometry uses lane frames and oriented rectangular footprints. Gap prediction
assumes constant measured velocity for one second (including vehicles entering
the lane); this is a calibrated proxy, never a safety guarantee. Initial gap
hysteresis is false inside [0, 2] metres. See REGISTRY for the fixed alphabet.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 -- registers highway-v0


ROLES = ("lead", "rear", "left_front", "left_rear", "right_front",
         "right_rear", "intruder")
LANES = ("left", "right", "passing_lane")
FRONT_ROLES = ("lead", "left_front", "right_front", "intruder")
REAR_ROLES = ("rear", "left_rear", "right_rear")
# Side neighbours occupy their side lane; pinned passing-lane membership can
# describe progress through an overtake. No Cartesian role/lane product.
OCCUPANCY_GROUNDINGS = (
    ("lead", "passing_lane"), ("rear", "passing_lane"),
    ("left_front", "left"), ("left_rear", "left"),
    ("right_front", "right"), ("right_rear", "right"),
    ("intruder", "left"), ("intruder", "right"),
    ("intruder", "passing_lane"))
REGISTRY = (
    [("present", (r,)) for r in ROLES]
    + [("lane_exists", (l,)) for l in LANES]
    + [("in_lane", grounding) for grounding in OCCUPANCY_GROUNDINGS]
    + [("ahead_of", (r, "ego")) for r in FRONT_ROLES]
    + [("slower_than", (r, "ego")) for r in ROLES]
    + [("closing_on", ("ego", r)) for r in FRONT_ROLES]
    + [("closing_on", (r, "ego")) for r in REAR_ROLES]
    + [("braking", (r,)) for r in ROLES]
    + [(p, ("ego", l)) for p in ("front_gap_adequate", "rear_gap_adequate") for l in LANES]
    + [("moving_into", ("intruder", "ego_lane"))]
    + [("danger_zone", ("ego", "lead"))]
    + [("crashed", ("ego",))])


@dataclass(frozen=True)
class Config:
    speed_difference: float = .5
    closing_threshold: float = .5
    ttc_danger: float = 2.
    acceleration_threshold: float = .5
    motion_offset_threshold: float = .25
    d0: float = 5.
    tau: float = 1.2
    tau_c: float = 1.
    gap_on: float = 2.
    gap_off: float = 0.
    prediction_horizon: float = 1.
    motion_threshold: float = .05
    intruder_distance: float = 60.


@dataclass(frozen=True)
class Vehicle:
    uid: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    length: float
    width: float
    lane_index: tuple
    crashed: bool = False


@dataclass
class Snapshot:
    time: float
    ego: Vehicle
    vehicles: tuple[Vehicle, ...]
    road: object


def extract_snapshot(env):
    h = env.unwrapped
    if not hasattr(h, "_semantic_ids"):
        h._semantic_ids = {}
    def freeze(v):
        if v not in h._semantic_ids:
            h._semantic_ids[v] = len(h._semantic_ids) + 1
        return Vehicle(h._semantic_ids[v], np.array(v.position, dtype=float).copy(),
                       np.array(v.velocity, dtype=float).copy(), float(v.heading),
                       float(v.LENGTH), float(v.WIDTH), tuple(v.lane_index), bool(v.crashed))
    vehicles = tuple(freeze(v) for v in h.road.vehicles)
    ego_id = h._semantic_ids[h.vehicle]
    return Snapshot(float(h.time), next(v for v in vehicles if v.uid == ego_id), vehicles, h.road)


def all_lanes(road):
    return { (a, b, i): lane for a, edges in road.network.graph.items()
             for b, lanes in edges.items() for i, lane in enumerate(lanes) }


def lane_coordinates(vehicle, lane):
    return lane.local_coordinates(vehicle.position)


def vehicle_footprint(vehicle):
    forward = np.array([math.cos(vehicle.heading), math.sin(vehicle.heading)])
    normal = np.array([-forward[1], forward[0]])
    return np.array([vehicle.position + x * vehicle.length / 2 * forward
                     + y * vehicle.width / 2 * normal
                     for x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))])


def extents(vehicle, lane, t=0.):
    points = vehicle_footprint(vehicle) + t * vehicle.velocity
    coords = np.array([lane.local_coordinates(p) for p in points])
    return coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()


def occupies(vehicle, lane, t=0.):
    s0, s1, d0, d1 = extents(vehicle, lane, t)
    width = lane.width_at(np.clip((s0 + s1) / 2, 0, lane.length))
    return s1 >= 0 and s0 <= lane.length and d0 < width / 2 and d1 > -width / 2


def occupied_lanes(vehicle, road):
    return {idx for idx, lane in all_lanes(road).items() if occupies(vehicle, lane)}


def resolve_relative_lanes(snapshot):
    idx = snapshot.ego.lane_index
    lanes = all_lanes(snapshot.road)
    # Highway road topology numbers side lanes from left to right; use only
    # the immediate neighbours on this edge, never world-coordinate signs.
    return {"left": (*idx[:2], idx[2] - 1) if (*idx[:2], idx[2] - 1) in lanes else None,
            "right": (*idx[:2], idx[2] + 1) if (*idx[:2], idx[2] + 1) in lanes else None}


def longitudinal_speed(vehicle, lane):
    s, _ = lane_coordinates(vehicle, lane)
    heading = lane.heading_at(s)
    return float(vehicle.velocity @ np.array([math.cos(heading), math.sin(heading)]))


def longitudinal_clearance(follower, leader, lane):
    return extents(leader, lane)[0] - extents(follower, lane)[1]


def closing_speed(follower, leader, lane):
    return longitudinal_speed(follower, lane) - longitudinal_speed(leader, lane)


def time_to_collision(follower, leader, lane):
    gap = longitudinal_clearance(follower, leader, lane)
    if gap <= 0:
        return 0.
    speed = closing_speed(follower, leader, lane)
    return gap / speed if speed > 1e-6 else math.inf


def nearest(ego, lane, vehicles, ahead):
    if lane is None:
        return None
    s, _ = lane_coordinates(ego, lane)
    candidates = [v for v in vehicles if v.uid != ego.uid and occupies(v, lane)
                  and ((lane_coordinates(v, lane)[0] >= s) if ahead
                       else (lane_coordinates(v, lane)[0] < s))]
    return min(candidates, key=lambda v: (abs(lane_coordinates(v, lane)[0] - s), v.uid), default=None)


def nearest_ahead(ego, lane, vehicles):
    return nearest(ego, lane, vehicles, True)


def nearest_behind(ego, lane, vehicles):
    return nearest(ego, lane, vehicles, False)


def lateral_motion(vehicle, lane):
    s, d = lane_coordinates(vehicle, lane)
    heading = lane.heading_at(s)
    lateral_speed = float(vehicle.velocity @ np.array([-math.sin(heading), math.cos(heading)]))
    error = math.atan2(math.sin(vehicle.heading - heading), math.cos(vehicle.heading - heading))
    return {"offset": float(d), "lateral_speed": lateral_speed, "heading_error": error}


def moving_into(vehicle, lane, config):
    m = lateral_motion(vehicle, lane)
    return (abs(m["lateral_speed"]) > config.motion_threshold
            and m["offset"] * m["lateral_speed"] < 0
            and abs(m["offset"]) > config.motion_offset_threshold)


@dataclass
class TaskContext:
    passing_lane: tuple | None = None

    def rebind(self, snapshot):
        self.passing_lane = resolve_relative_lanes(snapshot)["left"]


def bind_roles(snapshot, task_context, config=Config()):
    refs = resolve_relative_lanes(snapshot)
    refs.update(passing_lane=task_context.passing_lane)
    lanes = all_lanes(snapshot.road)
    roles = dict.fromkeys(ROLES)
    roles["ego"] = snapshot.ego
    # Ego remains an internal reference, outside the alphabet's role list.
    for idx, front, rear in ((snapshot.ego.lane_index, "lead", "rear"),
                              (refs["left"], "left_front", "left_rear"),
                              (refs["right"], "right_front", "right_rear")):
        lane = lanes.get(idx)
        roles[front] = nearest_ahead(snapshot.ego, lane, snapshot.vehicles)
        roles[rear] = nearest_behind(snapshot.ego, lane, snapshot.vehicles)
    current = lanes.get(snapshot.ego.lane_index)
    if current is not None:
        entrants = [v for v in snapshot.vehicles if v.uid != snapshot.ego.uid
                    and np.linalg.norm(v.position - snapshot.ego.position) <= config.intruder_distance
                    and any(occupies(v, lanes[idx]) for idx in (refs["left"], refs["right"])
                            if idx in lanes)
                    and moving_into(v, current, config)]
        # Earliest lateral footprint arrival, then smallest longitudinal
        # clearance magnitude, then episode ID: deterministic conflict priority.
        def priority(v):
            m = lateral_motion(v, current)
            _, _, lo, hi = extents(v, current)
            width = current.width_at(lane_coordinates(v, current)[0])
            arrival = max(0., max(lo - width / 2, -width / 2 - hi)) / abs(m["lateral_speed"])
            return arrival, abs(lane_coordinates(v, current)[0] - lane_coordinates(snapshot.ego, current)[0]), v.uid
        roles["intruder"] = min(entrants, key=priority, default=None)
    return {"roles": roles, "refs": refs, "lanes": lanes}


def estimate_acceleration(vehicle, previous_snapshot, dt, lane=None):
    if previous_snapshot is None or dt <= 0:
        return 0.
    old = next((v for v in previous_snapshot.vehicles if v.uid == vehicle.uid), None)
    if old is None:
        return 0.
    if lane is None:
        lane = all_lanes(previous_snapshot.road).get(vehicle.lane_index)
    return (longitudinal_speed(vehicle, lane) - longitudinal_speed(old, lane)) / dt if lane else 0.


def required_gap(follower_speed, closing_speed, config=Config()):
    return config.d0 + config.tau * max(follower_speed, 0.) + config.tau_c * max(closing_speed, 0.)


def lane_gap_margins(ego, lane, snapshot, config=Config()):
    result = {side: {"margin": math.inf, "clearance": math.inf, "required": 0.,
                     "ttc": math.inf, "vehicle": None, "projection_time": 0.}
              for side in ("front", "rear")}
    if lane is None:
        for data in result.values():
            data["margin"] = -math.inf
        return result
    for v in snapshot.vehicles:
        if v.uid == ego.uid:
            continue
        # For highway-v0's straight lanes, lateral footprint extents translate
        # linearly. Solve the entry/exit interval, avoiding sampling gaps.
        _, _, lo, hi = extents(v, lane)
        vy = lateral_motion(v, lane)["lateral_speed"]
        width = lane.width_at(lane_coordinates(v, lane)[0])
        start, end = 0., config.prediction_horizon
        if abs(vy) < 1e-9:
            if not occupies(v, lane):
                continue
        else:
            times = sorted(((-width / 2 - hi) / vy, (width / 2 - lo) / vy))
            start, end = max(start, times[0]), min(end, times[1])
            if start > end:
                continue
        # Include times at which longitudinal edges cross: a fast vehicle can
        # pass entirely through ego between the two horizon endpoints.
        e0, e1, _, _ = extents(ego, lane)
        v0, v1, _, _ = extents(v, lane)
        relative_speed = longitudinal_speed(v, lane) - longitudinal_speed(ego, lane)
        times_to_check = [start, end]
        if abs(relative_speed) > 1e-9:
            times_to_check += [t for t in ((e1 - v0) / relative_speed,
                                           (e0 - v1) / relative_speed)
                               if start <= t <= end]
        for t in times_to_check:
            e0, e1, _, _ = extents(ego, lane, t)
            v0, v1, _, _ = extents(v, lane, t)
            overlap = v0 <= e1 and v1 >= e0
            sides = ("front", "rear") if overlap else (("front",) if v0 > e1 else ("rear",))
            for side in sides:
                follower, leader = (ego, v) if side == "front" else (v, ego)
                clearance = v0 - e1 if side == "front" else e0 - v1
                need = required_gap(longitudinal_speed(follower, lane), closing_speed(follower, leader, lane), config)
                margin = min(clearance - need, -config.d0) if overlap else clearance - need
                if margin < result[side]["margin"]:
                    result[side] = dict(margin=margin, clearance=clearance, required=need,
                                        ttc=time_to_collision(follower, leader, lane),
                                        vehicle=v.uid, projection_time=t)
    return result


@dataclass
class Memory:
    previous: Snapshot | None = None
    occupants: dict = field(default_factory=dict)
    hysteresis: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    time: float | None = None


def evaluate_predicates(snapshot, bindings, config, memory):
    if memory.time == snapshot.time:
        return memory.values, memory.measurements
    roles, refs, lanes = bindings["roles"], bindings["refs"], bindings["lanes"]
    measurements = {"gaps": {}, "motion": {}, "acceleration": {}, "pairs": {}}
    changed = {r for r, v in roles.items() if memory.occupants.get(r) != (v.uid if v else None)}
    dt = snapshot.time - memory.previous.time if memory.previous else 0.
    for r, v in roles.items():
        if v:
            lane = lanes.get(v.lane_index)
            measurements["acceleration"][r] = (0. if r in changed else
                memory.measurements.get("acceleration", {}).get(r, 0.) if dt == 0 else
                estimate_acceleration(v, memory.previous, dt, lane))
    for l in LANES:
        measurements["gaps"][l] = lane_gap_margins(snapshot.ego, lanes.get(refs[l]), snapshot, config)
    values = {}
    for predicate, args in REGISTRY:
        ap = f"{predicate}({','.join(args)})"
        a = roles.get(args[0])
        value = False
        if predicate == "present":
            value = a is not None
        elif predicate == "lane_exists":
            value = refs[args[0]] in lanes
        elif predicate == "crashed":
            value = a.crashed
        elif predicate == "braking" and a:
            acc = measurements["acceleration"][args[0]]
            value = acc < -config.acceleration_threshold
        elif predicate in ("front_gap_adequate", "rear_gap_adequate"):
            l = args[1]
            side = predicate.split("_")[0]
            margin = measurements["gaps"][l][side]["margin"]
            key = (ap, refs[l])  # history belongs to this physical lane
            old = memory.hysteresis.get(key, False)
            value = True if margin > config.gap_on else False if margin < config.gap_off else old
            memory.hysteresis[key] = value
        elif predicate in ("in_lane", "moving_into") and a:
            lane = lanes.get(snapshot.ego.lane_index if args[1] == "ego_lane" else refs[args[1]])
            if lane:
                m = lateral_motion(a, lane)
                measurements["motion"][','.join(args)] = m
                if predicate == "in_lane":
                    value = occupies(a, lane)
                else:
                    value = moving_into(a, lane, config)
        elif len(args) == 2 and a and roles.get(args[1]):
            b = roles[args[1]]
            lane = lanes.get(snapshot.ego.lane_index)
            if lane:
                gap = longitudinal_clearance(a, b, lane)
                closing = closing_speed(a, b, lane)
                relevant = bool(occupied_lanes(a, snapshot.road) & occupied_lanes(b, snapshot.road))
                follows = lane_coordinates(a, lane)[0] <= lane_coordinates(b, lane)[0]
                ttc = time_to_collision(a, b, lane) if relevant and follows else math.inf
                measurements["pairs"][','.join(args)] = dict(clearance=gap, closing_speed=closing, ttc=ttc)
                if predicate == "ahead_of":
                    value = longitudinal_clearance(b, a, lane) > 0
                elif predicate == "slower_than":
                    value = closing < -config.speed_difference
                elif predicate == "closing_on":
                    # Side-lane approaches use projected longitudinal order/speed;
                    # shared occupancy is required only for collision risk.
                    value = follows and gap > 0 and closing > config.closing_threshold
                elif predicate == "danger_zone":
                    value = relevant and follows and (gap <= 0 or
                        (closing > config.closing_threshold and ttc < config.ttc_danger))
        values[ap] = bool(value)
    memory.occupants = {r: v.uid if v else None for r, v in roles.items()}
    memory.previous, memory.time = snapshot, snapshot.time
    memory.values, memory.measurements = values, measurements
    return values, measurements


def label_highway_state(snapshot, bindings, config, memory):
    values, _ = evaluate_predicates(snapshot, bindings, config, memory)
    return {ap for ap, value in values.items() if value}


class Simulator:
    def __init__(self, seed=7, print_labels=False, config=Config()):
        self.config, self.seed, self.print_labels = config, seed, print_labels
        self.env = gym.make("highway-v0", render_mode="rgb_array", config={
            "lanes_count": 4, "vehicles_count": 20, "controlled_vehicles": 1,
            "vehicles_density": .7, "simulation_frequency": 15, "policy_frequency": 5,
            "duration": 120, "screen_width": 760, "screen_height": 300,
            "offscreen_rendering": True, "action": {"type": "DiscreteMetaAction",
                                                       "longitudinal": True, "lateral": True}})
        self.actions = {name: key for key, name in self.env.unwrapped.action_type.actions.items()}
        self.reset()

    def reset(self, new_seed=False):
        if new_seed:
            self.seed += 1
        self.env.reset(seed=self.seed)
        self.env.unwrapped._semantic_ids = {}
        self.pending, self.done, self.steps = "IDLE", False, 0
        self.memory, self.task = Memory(), TaskContext()
        self.snapshot = extract_snapshot(self.env)
        self.task.rebind(self.snapshot)
        self.labels, self.added, self.removed = set(), set(), set()
        self.refresh()

    def refresh(self):
        self.bindings = bind_roles(self.snapshot, self.task, self.config)
        old = self.labels
        self.labels = label_highway_state(self.snapshot, self.bindings, self.config, self.memory)
        self.added, self.removed = self.labels - old, old - self.labels
        if self.print_labels:
            print("{" + ", ".join(sorted(self.labels)) + "}")

    def available(self):
        return {self.env.unwrapped.action_type.actions[i] for i in self.env.unwrapped.get_available_actions()}

    def queue(self, action):
        if not self.done and action in self.available():
            self.pending = action

    def step(self):
        if self.done:
            return False
        action, self.pending = self.pending, "IDLE"
        _, _, terminated, truncated, _ = self.env.step(self.actions[action])
        self.done, self.steps = bool(terminated or truncated), self.steps + 1
        self.snapshot = extract_snapshot(self.env)
        self.refresh()
        return True

    def rebind(self):
        self.task.rebind(self.snapshot)
        # Re-evaluate at this same time without advancing acceleration history.
        self.memory.hysteresis = {k: v for k, v in self.memory.hysteresis.items()
                                  if "passing_lane" not in k[0]}
        self.memory.time = None
        self.refresh()

    def close(self):
        self.env.close()


class Visualizer:
    def __init__(self, simulator, interval=.6):
        import pygame
        self.pg, self.sim = pygame, simulator
        pygame.init()
        # Render first: offscreen viewer creates surfaces, never another window
        # and never calls handle_events(). Only this loop owns the event queue.
        self.camera_follow = True
        self.camera_anchor = SimpleNamespace(position=simulator.snapshot.ego.position.copy())
        self.frame = self.render_frame(reset_camera=True)
        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        window_size = (max(1000, min(1240, desktop_width - 40)),
                       max(680, min(800, desktop_height - 90)))
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Highway semantic labels")
        pygame.key.set_repeat()  # one-shot commands, ignore repeated KEYDOWN
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 14)
        self.clock = pygame.time.Clock()
        self.play, self.diagnostics, self.scroll = False, False, 0
        self.interval, self.next_step = max(.1, interval), 0
        self.buttons = []
        self.drag_scroll = False
        self.drag_offset = 0
        self.update_layout()

    def update_layout(self):
        width, height = self.screen.get_size()
        self.left_width = min(760, int(width * .62))
        self.panel_x = self.left_width + 30
        self.panel_rect = self.pg.Rect(self.panel_x, 65, width - self.panel_x - 38, height - 110)
        self.row_height = max(22, self.small.get_linesize() + 5)
        self.visible_rows = max(1, self.panel_rect.height // self.row_height)
        self.scroll_up = self.pg.Rect(width - 26, self.panel_rect.top, 18, 22)
        self.scroll_down = self.pg.Rect(width - 26, self.panel_rect.bottom - 22, 18, 22)
        self.scroll_track = self.pg.Rect(width - 25, self.panel_rect.top + 26,
                                        16, self.panel_rect.height - 52)

    def panel_rows(self):
        sim = self.sim
        rows = []
        if self.diagnostics:
            rows += [(f"{r}: #{v.uid}" if v else f"{r}: absent", (200, 220, 240))
                     for r, v in sim.bindings["roles"].items()]
            rows += [(f"{l}: {idx}", (200, 220, 240)) for l, idx in sim.bindings["refs"].items()]
            for section, measurements in sim.memory.measurements.items():
                rows.append((section.upper(), (125, 195, 255)))
                for key, data in measurements.items():
                    rows.append((str(key), (220, 228, 240)))
                    rows.append((str(data), (175, 185, 200)))
            rows.append(("FALSE APs", (125, 195, 255)))
            rows += [(ap, (155, 165, 180)) for ap, value in sim.memory.values.items() if not value]
        else:
            rows += [(('+ ' if ap in sim.added else '  ') + ap,
                      (120, 245, 175) if ap in sim.added else (220, 228, 240)) for ap in sorted(sim.labels)]
            rows.append(("REMOVED at last update (not current)", (255, 190, 120)))
            rows += [("- " + ap, (255, 155, 125)) for ap in sorted(sim.removed)]
        # Wrap to the actual panel width so every expression/measurement is
        # readable, including when the user shrinks the window.
        import textwrap
        columns = max(15, self.panel_rect.width // self.small.size("M")[0])
        return [(line, color) for text, color in rows
                for line in (textwrap.wrap(text, columns) or [""])]

    def scroll_to(self, row):
        maximum = max(0, len(self.panel_rows()) - self.visible_rows)
        self.scroll = max(0, min(int(row), maximum))

    def scrollbar_thumb(self):
        count = len(self.panel_rows())
        maximum = max(0, count - self.visible_rows)
        track = self.scroll_track
        height = min(track.height, max(30, int(track.height * self.visible_rows / max(1, count))))
        offset = int((track.height - height) * self.scroll / maximum) if maximum else 0
        return self.pg.Rect(track.x, track.y + offset, track.width, height)

    def render_frame(self, reset_camera=False):
        """Hold a world view so equal-speed traffic visibly advances on screen."""
        h = self.sim.env.unwrapped
        if h.viewer is None:
            self.sim.env.render()
        viewer = h.viewer
        if reset_camera:
            self.camera_anchor.position = self.sim.snapshot.ego.position.copy()
        if not self.camera_follow:
            # Keep all cars in world motion between discrete camera recentres.
            surface = viewer.sim_surface
            surface.move_display_window_to(self.camera_anchor.position)
            x, _ = surface.vec2pix(self.sim.snapshot.ego.position)
            if x < 80 or x > surface.get_width() - 80:
                self.camera_anchor.position = self.sim.snapshot.ego.position.copy()
        viewer.observer_vehicle = None if self.camera_follow else self.camera_anchor
        return self.sim.env.render()

    def command(self, name):
        pg = self.pg
        if name == "Step":
            self.play = False
            self.sim.step()
        elif name == "Play/Pause":
            self.play = not self.play and not self.sim.done
            self.next_step = pg.time.get_ticks() + int(self.interval * 1000)
        elif name in ("Reset", "New seed"):
            self.play = False
            self.sim.reset(name == "New seed")
            self.scroll = 0
        elif name == "Rebind task":
            self.sim.rebind()
        elif name == "Diagnostics":
            self.diagnostics = not self.diagnostics
            self.scroll = 0
        elif name == "Camera":
            self.camera_follow = not self.camera_follow
            self.camera_anchor.position = self.sim.snapshot.ego.position.copy()
        elif name in ("Faster play", "Slower play"):
            self.interval = min(3., max(.1, self.interval * (.8 if name == "Faster play" else 1.25)))
        else:
            self.sim.queue(name)
        self.frame = self.render_frame(reset_camera=name in ("Reset", "New seed"))

    def text(self, text, x, y, color=(220, 228, 240), small=False):
        self.screen.blit((self.small if small else self.font).render(text, True, color), (x, y))

    def draw(self):
        pg, sim = self.pg, self.sim
        self.screen.fill((19, 25, 36))
        image = pg.surfarray.make_surface(self.frame.swapaxes(0, 1))
        self.screen.blit(pg.transform.smoothscale(image, (self.left_width, 300)), (10, 10))
        state = "ENDED (reset)" if sim.done else "PLAYING" if self.play else "PAUSED"
        for i, line in enumerate([
            f"{state} | t={sim.snapshot.time:.2f}s | steps={sim.steps} | seed={sim.seed}",
            f"Ego speed={np.linalg.norm(sim.snapshot.ego.velocity):.2f} m/s | pending={sim.pending}",
            f"Playback every {self.interval:.2f}s wall | step = 0.20s simulation",
            "IDLE keeps speed/lane targets; it does not stop the car.",
            "Arrows queue | Space step | P play/pause | I IDLE",
            f"R reset | N seed | T rebind | D info | C camera ({'Follow' if self.camera_follow else 'World'})",
            "Gap proxy: constant velocity over 1s; not a safety guarantee."]):
            self.text(line, 14, 325 + i * 23, small=True)
        self.buttons = []
        names = ["Step", "Play/Pause", "Reset", "New seed", "Rebind task", "Diagnostics",
                 "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER", "IDLE", "Faster play", "Slower play", "Camera"]
        available = sim.available()
        button_width = (self.left_width - 22) // 4
        for i, name in enumerate(names):
            rect = pg.Rect(14 + (i % 4) * button_width, 488 + (i // 4) * 44, button_width - 8, 36)
            enabled = (name not in sim.actions or (name in available and not sim.done))
            if name in ("Step", "Play/Pause") and sim.done:
                enabled = False
            pg.draw.rect(self.screen, (48, 69, 92) if enabled else (39, 39, 42), rect, border_radius=5)
            caption = f"Camera: {'Follow' if self.camera_follow else 'World'}" if name == "Camera" else name
            self.text(caption if enabled else caption + " (N/A)", rect.x + 5, rect.y + 11, small=True)
            self.buttons.append((rect, name, enabled))
        self.text("Wheel | PgUp/PgDn | Home/End", self.panel_x, 12, small=True)
        self.text("DIAGNOSTICS" if self.diagnostics else f"TRUE APs ({len(sim.labels)})", self.panel_x, 36)
        rows = self.panel_rows()
        self.scroll_to(self.scroll)
        self.screen.set_clip(self.panel_rect)
        for i, (line, color) in enumerate(rows[self.scroll:self.scroll + self.visible_rows]):
            self.text(line, self.panel_x, self.panel_rect.y + i * self.row_height, color, small=True)
        self.screen.set_clip(None)
        pg.draw.rect(self.screen, (37, 47, 62), self.scroll_track, border_radius=5)
        pg.draw.rect(self.screen, (115, 160, 200), self.scrollbar_thumb(), border_radius=5)
        for rect, caption in ((self.scroll_up, "^"), (self.scroll_down, "v")):
            pg.draw.rect(self.screen, (48, 69, 92), rect, border_radius=3)
            self.text(caption, rect.x + 4, rect.y + 2, small=True)
        self.text(f"Rows {self.scroll + 1}-{min(len(rows), self.scroll + self.visible_rows)} / {len(rows)}",
                  self.panel_x, self.screen.get_height() - 30, small=True)
        pg.display.flip()

    def handle_event(self, event):
        pg = self.pg
        shortcuts = {pg.K_SPACE: "Step", pg.K_p: "Play/Pause", pg.K_LEFT: "LANE_LEFT",
                     pg.K_RIGHT: "LANE_RIGHT", pg.K_UP: "FASTER", pg.K_DOWN: "SLOWER",
                     pg.K_i: "IDLE", pg.K_r: "Reset", pg.K_n: "New seed",
                     pg.K_t: "Rebind task", pg.K_d: "Diagnostics", pg.K_c: "Camera"}
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            return False
        if event.type == pg.VIDEORESIZE:
            self.screen = pg.display.set_mode((max(1000, event.w), max(680, event.h)), pg.RESIZABLE)
            self.update_layout()
            self.scroll_to(self.scroll)
        elif event.type == pg.KEYDOWN:
            destinations = {pg.K_PAGEUP: self.scroll - self.visible_rows,
                            pg.K_PAGEDOWN: self.scroll + self.visible_rows,
                            pg.K_HOME: 0, pg.K_END: len(self.panel_rows())}
            if event.key in destinations:
                self.scroll_to(destinations[event.key])
            elif not getattr(event, "repeat", False) and event.key in shortcuts:
                self.command(shortcuts[event.key])
        elif event.type == pg.MOUSEWHEEL:
            self.scroll_to(self.scroll - event.y * 3)
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button in (4, 5):  # legacy mouse-wheel events
                self.scroll_to(self.scroll + (-3 if event.button == 4 else 3))
            elif event.button == 1:
                if self.scroll_up.collidepoint(event.pos):
                    self.scroll_to(self.scroll - 1)
                elif self.scroll_down.collidepoint(event.pos):
                    self.scroll_to(self.scroll + 1)
                elif self.scroll_track.collidepoint(event.pos):
                    thumb = self.scrollbar_thumb()
                    if thumb.collidepoint(event.pos):
                        self.drag_scroll, self.drag_offset = True, event.pos[1] - thumb.y
                    else:
                        self.scroll_to(self.scroll + (-self.visible_rows if event.pos[1] < thumb.y else self.visible_rows))
                else:
                    for rect, name, enabled in self.buttons:
                        if enabled and rect.collidepoint(event.pos):
                            self.command(name)
                            break
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.drag_scroll = False
        elif event.type == pg.MOUSEMOTION and self.drag_scroll:
            travel = self.scroll_track.height - self.scrollbar_thumb().height
            maximum = max(0, len(self.panel_rows()) - self.visible_rows)
            fraction = (event.pos[1] - self.drag_offset - self.scroll_track.y) / max(1, travel)
            self.scroll_to(round(fraction * maximum))
        return True

    def run(self):
        pg = self.pg
        running = True
        self.draw()  # populate hit targets before accepting the first click
        while running:
            for event in pg.event.get():
                if not self.handle_event(event):
                    running = False
                    break
            now = pg.time.get_ticks()
            if running and self.play and now >= self.next_step:
                self.sim.step()
                self.frame = self.render_frame()
                self.play = not self.sim.done
                self.next_step = now + int(self.interval * 1000)
            self.draw()
            self.clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-labels", action="store_true")
    parser.add_argument("--playback-interval", type=float, default=.6)
    parser.add_argument("--headless-steps", type=int, default=None,
                        help="Run semantic calculations without creating a GUI")
    args = parser.parse_args()
    simulator = None
    try:
        simulator = Simulator(args.seed, args.print_labels)
        if args.headless_steps is not None:
            for _ in range(max(0, args.headless_steps)):
                if not simulator.step():
                    break
        else:
            Visualizer(simulator, args.playback_interval).run()
    finally:
        if simulator:
            simulator.close()
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
