"""Interactive state-derived driving tasks, built on mainV2's traffic semantics.

Run with the existing project environment: .venv/Scripts/python.exe mainV3.py.
Tasks are guarded by the unchanged state alphabet plus physical execution
limits. Selection queues a task; Step or Play executes it. See README_V3.md.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

import mainV2 as v2
from task_dfa import TaskDFA
from live_dfa import TrafficDFAs
from simulator_dashboard import Dashboard
from overtake import OvertakeController


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    symbol: str
    requires: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    lane: str | None = None


# Ground only tasks that make sense in V2's alphabet: immediate left/right
# lanes and the lead role. The pinned passing_lane is not another direction.
LANE_TASKS = tuple(
    TaskSpec(f"switch_lane_{lane}", f"Switch lane {lane}",
             f"switch_lane(ego,{lane})",
             (f"lane_exists({lane})", f"front_gap_adequate(ego,{lane})",
              f"rear_gap_adequate(ego,{lane})"), lane=lane)
    for lane in v2.LANES if lane in ("left", "right"))
TASK_REGISTRY = (
    LANE_TASKS[0],
    TaskSpec("accelerate", "Accelerate", "accelerate(ego)", forbids=(
        "danger_zone(ego,lead)", "moving_into(intruder,ego_lane)")),
    LANE_TASKS[1],
    TaskSpec("follow_car", "Follow car (lead)", "follow_car(ego,lead)",
             ("present(lead)", "ahead_of(lead,ego)")),
    TaskSpec("brake", "Brake", "brake(ego)"),
    TaskSpec("cruise", "Cruise", "cruise(ego)"))
TASKS = {task.name: task for task in TASK_REGISTRY}


@dataclass(frozen=True)
class TaskOption:
    task: TaskSpec
    enabled: bool
    reason: str


@dataclass(frozen=True)
class TaskConfig:
    max_speed: float = 30.
    acceleration: float = 2.
    deceleration: float = 4.
    follow_gap_gain: float = .5
    stopped_speed: float = .1
    lane_center_tolerance: float = .2
    heading_tolerance: float = .03


@dataclass(frozen=True)
class LaneGapConfig:
    """Lane-change spacing, independent of the Follow controller's headway."""
    d0: float = 2.
    tau: float = .3
    tau_c: float = .6
    gap_on: float = .25
    gap_off: float = 0.

    def predicates(self, base):
        return replace(base, d0=self.d0, tau=self.tau, tau_c=self.tau_c,
                       gap_on=self.gap_on, gap_off=self.gap_off)


@dataclass(frozen=True)
class MotionConfig:
    physics_hz: int = 60
    control_hz: int = 30
    labels_hz: int = 5
    traffic_min_speed: float = 18.
    traffic_max_speed: float = 32.


def generate_task_vocabulary(labels, *, done=False, speed=None,
                             config=TaskConfig(), lane_change=False,
                             reachable_lanes=None):
    """Return every task and its guard result without advancing the simulator.

    The symbolic requirements come exclusively from V2's true AP set. Speed,
    reachability, episode end and an ongoing maneuver are execution constraints
    because those facts are not represented in the supplied state vocabulary.
    Unknown/missing positive facts fail closed.
    """
    labels = set(labels)
    options = {}
    for task in TASK_REGISTRY:
        reason = "Available"
        if done:
            reason = "Episode ended; reset to continue"
        elif "crashed(ego)" in labels:
            reason = "Ego has crashed"
        elif task.lane and lane_change:
            reason = "Lane change already in progress"
        elif task.name == "follow_car" and lane_change:
            reason = "Finish the lane change before following"
        else:
            missing = next((ap for ap in task.requires if ap not in labels), None)
            forbidden = next((ap for ap in task.forbids if ap in labels), None)
            if missing:
                if missing.startswith("lane_exists"):
                    reason = f"No lane to the {task.lane}"
                elif missing.startswith("front_gap"):
                    reason = f"Front gap in {task.lane} lane is inadequate"
                elif missing.startswith("rear_gap"):
                    reason = f"Rear gap in {task.lane} lane is inadequate"
                else:
                    reason = "No lead car ahead in ego's lane"
            elif forbidden:
                reason = ("Lead car is in the danger zone" if "danger_zone" in forbidden
                          else "An intruder is moving into ego's lane")
            elif (task.name == "accelerate" and "braking(lead)" in labels
                  and "closing_on(ego,lead)" in labels):
                reason = "Closing on a braking lead car"
            elif (task.lane and reachable_lanes is not None
                  and task.lane not in reachable_lanes):
                reason = f"The {task.lane} lane is not reachable"
            elif task.name == "accelerate" and speed is not None and speed >= config.max_speed - .01:
                reason = "Maximum task speed reached"
            elif task.name == "brake" and speed is not None and speed <= config.stopped_speed:
                reason = "Ego is already stopped"
        options[task.name] = TaskOption(task, reason == "Available", reason)
    return options


class Simulator(v2.Simulator):
    def __init__(self, seed=7, print_labels=False, config=v2.Config(),
                 task_config=TaskConfig(), motion_config=MotionConfig(),
                 lane_gap_config=LaneGapConfig()):
        self.task_config, self.motion_config = task_config, motion_config
        self.lane_gap_config = lane_gap_config
        self.predicate_config = lane_gap_config.predicates(config)
        if (motion_config.physics_hz <= 0 or motion_config.control_hz <= 0
                or motion_config.labels_hz <= 0
                or motion_config.physics_hz % motion_config.control_hz
                or motion_config.control_hz % motion_config.labels_hz):
            raise ValueError("Physics/control/label frequencies must be positive integer multiples")
        super().__init__(seed, print_labels, config)

    @property
    def tick_duration(self):
        return 1. / self.motion_config.control_hz

    @property
    def step_duration(self):
        return 1. / self.motion_config.labels_hz

    @property
    def speed(self):
        return max(0., float(np.linalg.norm(self.snapshot.ego.velocity)))

    def reset(self, new_seed=False):
        self.env.unwrapped.configure({"simulation_frequency": self.motion_config.physics_hz,
                                      "policy_frequency": self.motion_config.control_hz,
                                      # Tasks use full road snapshots, not the unused
                                      # normalized traffic observation DataFrame.
                                      "observation": {"type": "Kinematics", "vehicles_count": 5,
                                                      "normalize": False, "absolute": True}})
        self.dfa = TaskDFA()
        self.traffic_dfas = TrafficDFAs(self.predicate_config, self._options_for_agent,
                                        self.motion_config.control_hz // self.motion_config.labels_hz)
        self.demo = None
        self.overtake = None
        self.motion_ticks = 0
        self.pending_task = None
        self.pending_brake_speed = None
        self.brake_target_speed = 0.
        self.active_task = "cruise"
        self.lane_change_target = None
        self.followed_uid = None
        self.task_message = "Cruise active. Select a task, then Step or Play."
        super().reset(new_seed)
        # Give each IDM driver an individual desired speed as well as an initial
        # velocity. Changing only initial speed makes traffic soon converge.
        rng = np.random.default_rng(self.seed + 104729)
        for car in self.env.unwrapped.road.vehicles:
            if car is not self.env.unwrapped.vehicle:
                car.target_speed = float(rng.uniform(self.motion_config.traffic_min_speed,
                                                     self.motion_config.traffic_max_speed))
                car.speed = float(np.clip(car.target_speed + rng.uniform(-1., 1.),
                                          self.motion_config.traffic_min_speed,
                                          self.motion_config.traffic_max_speed))
        self.memory = v2.Memory()
        self.snapshot = v2.extract_snapshot(self.env)
        self.refresh()
        self.cruise_speed = self.speed
        self.env.unwrapped.vehicle.target_speed = self.cruise_speed
        self.task_options = self.get_task_options()
        self.traffic_dfas.reset(self.snapshot, self.env.unwrapped)

    def refresh(self, refresh_traffic=True):
        self.bindings = v2.bind_roles(self.snapshot, self.task, self.predicate_config)
        old = self.labels
        self.labels = v2.label_highway_state(self.snapshot, self.bindings,
                                             self.predicate_config, self.memory)
        self.added, self.removed = self.labels - old, old - self.labels
        if self.print_labels:
            print("{" + ", ".join(sorted(self.labels)) + "}")
        self.task_options = self.get_task_options()
        if refresh_traffic:
            self.traffic_dfas.refresh(self.snapshot)

    def _options_for_agent(self, snapshot, bindings, labels, lane_change=False):
        """Use identical guards with each vehicle as the local ego reference."""
        reachable = {name for name in ("left", "right")
                     if (lane := bindings["lanes"].get(bindings["refs"][name]))
                     is not None and lane.is_reachable_from(snapshot.ego.position)}
        return generate_task_vocabulary(
            labels, done=self.done, speed=float(np.linalg.norm(snapshot.ego.velocity)),
            config=self.task_config, lane_change=lane_change, reachable_lanes=reachable)

    def get_task_options(self):
        return self._options_for_agent(self.snapshot, self.bindings, self.labels,
                                       self.lane_change_target is not None)

    def available(self):
        """Only guarded task IDs are exposed; raw actions cannot bypass guards."""
        return {name for name, option in self.get_task_options().items() if option.enabled}

    @property
    def task_vocabulary(self):
        return {option.task.symbol for option in self.get_task_options().values() if option.enabled}

    def overtake_availability(self):
        if self.overtake and self.overtake.running:
            return False, "Overtake already in progress"
        option = self.get_task_options()["switch_lane_left"]
        if not option.enabled:
            return False, option.reason
        if self.speed >= self.task_config.max_speed - .01:
            return False, "Maximum task speed reached"
        if not {"present(lead)", "slower_than(lead,ego)"} <= self.labels:
            return False, "A slower lead car is needed to overtake"
        return True, "Lane left, accelerate, lane right, then brake to the starting speed"

    def start_overtake(self):
        if self.memory.time != self.snapshot.time:
            self.refresh()
        enabled, reason = self.overtake_availability()
        if not enabled:
            self.task_message = "Overtake unavailable: " + reason
            return False
        self.demo = None
        self.overtake = OvertakeController(self.snapshot.ego.lane_index,
                                            self.bindings["refs"]["left"],
                                            self.bindings["roles"]["lead"].uid, self.speed)
        self.overtake.request(self)
        return True

    def cancel_overtake(self, reason="Overtake cancelled"):
        if self.overtake and self.overtake.running:
            self.overtake.cancel(reason)
            if self.demo is self.overtake:
                self.demo = None
            self.pending_task = self.pending_brake_speed = None
            self._hold_speed(reason)
            self.env.unwrapped.vehicle.target_speed = self.cruise_speed

    def queue(self, task_name, *, target_speed=None):
        if self.memory.time != self.snapshot.time:
            self.refresh()
        self.task_options = self.get_task_options()
        option = self.task_options.get(task_name)
        if option is None or not option.enabled:
            reason = option.reason if option else "Unknown task"
            self.task_message = f"Rejected: {reason}"
            self.dfa.dispatch("reject")
            return False
        if target_speed is not None and (task_name != "brake" or not np.isfinite(target_speed)
                                         or target_speed < 0 or target_speed > self.speed):
            self.task_message = "Rejected: a brake target must be between zero and the current speed"
            self.dfa.dispatch("reject")
            return False
        self.pending_task = task_name
        self.pending_brake_speed = target_speed
        self.task_message = f"Queued: {option.task.label}"
        return True

    def _hold_speed(self, message):
        self.dfa.dispatch("task_complete")
        if self.overtake:
            self.overtake.task_event("task_complete")
        self.active_task = "cruise"
        self.followed_uid = None
        self.cruise_speed = self.speed
        self.task_message = message

    def _activate(self, task_name, brake_speed=None):
        self.dfa.dispatch(task_name)
        if self.overtake:
            self.overtake.task_started(task_name)
        self.active_task = task_name
        self.followed_uid = None
        self.task_message = f"Active: {TASKS[task_name].label}"
        if task_name == "brake":
            self.brake_target_speed = min(self.speed, brake_speed) if brake_speed is not None else 0.
        if task_name == "cruise" or TASKS[task_name].lane:
            self.cruise_speed = self.speed
            if self.speed <= self.task_config.stopped_speed:
                self.cruise_speed = 0.
        if TASKS[task_name].lane:
            self.lane_change_target = self.bindings["refs"][TASKS[task_name].lane]

    def _speed_target(self):
        car, cfg = self.env.unwrapped.vehicle, self.task_config
        speed = self.speed
        # ControlledVehicle uses KP_A * (target_speed - measured_speed).
        # Setpoints below zero are unnecessary: Brake can approach a full stop.
        lower = max(0., speed - cfg.deceleration / car.KP_A)
        upper = min(cfg.max_speed, speed + cfg.acceleration / car.KP_A)
        if self.active_task == "accelerate":
            return upper
        if self.active_task == "brake":
            return max(self.brake_target_speed, lower)
        if self.active_task == "follow_car":
            uid = self.bindings["roles"]["lead"].uid
            lead = next((v for v in self.snapshot.vehicles if v.uid == uid), None)
            if lead is None:
                return speed
            lane = self.bindings["lanes"][self.snapshot.ego.lane_index]
            self.followed_uid = lead.uid
            lead_speed = max(0., v2.longitudinal_speed(lead, lane))
            gap = v2.longitudinal_clearance(self.snapshot.ego, lead, lane)
            desired_gap = v2.required_gap(speed, speed - lead_speed, self.config)
            target = lead_speed + cfg.follow_gap_gain * (gap - desired_gap)
            if "danger_zone(ego,lead)" in self.labels:
                target = lower
            return float(np.clip(target, lower, max(lower, upper)))
        return self.cruise_speed

    def _finish_lane_change(self):
        target = self.lane_change_target
        if target is None or self.snapshot.ego.lane_index != target:
            return
        lane = self.bindings["lanes"][target]
        motion = v2.lateral_motion(self.snapshot.ego, lane)
        if (abs(motion["offset"]) <= self.task_config.lane_center_tolerance
                and abs(motion["heading_error"]) <= self.task_config.heading_tolerance):
            self.lane_change_target = None
            self.dfa.dispatch("lane_complete")
            if self.overtake:
                self.overtake.task_event("lane_complete")
            if TASKS[self.active_task].lane:
                self.active_task = "cruise"
                self.task_message = "Lane change completed; cruising at the held speed."

    def _validate_active_task(self):
        if (not self.done and self.active_task == "brake"
                and self.speed <= self.brake_target_speed + self.task_config.stopped_speed):
            self._hold_speed(f"Brake target reached: {self.brake_target_speed:.1f} m/s; cruising.")
            self.cruise_speed = self.brake_target_speed
            self.env.unwrapped.vehicle.target_speed = self.cruise_speed
        if not self.done and self.active_task in ("accelerate", "follow_car", "brake"):
            option = self.get_task_options()[self.active_task]
            if not option.enabled:
                self._hold_speed(f"Task ended: {option.reason}; cruising.")
                if self.speed <= self.task_config.stopped_speed:
                    self.cruise_speed = 0.
                self.env.unwrapped.vehicle.target_speed = self.cruise_speed

    def advance(self):
        """One 1/30-second control tick; render it directly during playback."""
        if self.done:
            return False
        ticks_per_step = self.motion_config.control_hz // self.motion_config.labels_hz
        if self.demo and self.motion_ticks % ticks_per_step == 0:
            self.demo.update(self)
        if self.overtake and self.overtake is not self.demo and self.motion_ticks % ticks_per_step == 0:
            self.overtake.update(self)
        if self.pending_task and self.memory.time != self.snapshot.time:
            self.refresh()
        self.task_options = self.get_task_options()
        requested, self.pending_task = self.pending_task, None
        brake_speed, self.pending_brake_speed = self.pending_brake_speed, None
        lane_action = "IDLE"
        if requested:
            option = self.task_options[requested]
            # Revalidate at execution; a queued command is not authorization
            # to use an outdated lane reference or lead binding.
            if option.enabled:
                self._activate(requested, brake_speed)
                if option.task.lane:
                    lane_action = "LANE_" + option.task.lane.upper()
            else:
                self.dfa.dispatch("reject")
                self.task_message = f"Rejected at execution: {option.reason}"
                if self.overtake:
                    self.overtake.rejected(self.task_message)
        self._validate_active_task()
        self.env.unwrapped.vehicle.target_speed = self._speed_target()
        _, _, terminated, truncated, _ = self.env.step(self.actions[lane_action])
        self.motion_ticks += 1
        self.done = bool(terminated or truncated or self.env.unwrapped.time + 1e-9
                         >= self.env.unwrapped.config["duration"])
        self.snapshot = v2.extract_snapshot(self.env)
        self.traffic_dfas.update(self.snapshot, self.env.unwrapped)
        if self.motion_ticks % ticks_per_step == 0 or self.done:
            self.steps += 1
            self.refresh(refresh_traffic=False)
            self._finish_lane_change()
            self._validate_active_task()
        if self.done:
            self.dfa.dispatch("end")
            if self.overtake:
                self.overtake.cancel("Episode ended", ended=True)
            for observed in self.traffic_dfas.cars:
                observed.dfa.dispatch("end")
            self.pending_task = None
            self.pending_brake_speed = None
            self.active_task = None
            self.lane_change_target = None
            self.followed_uid = None
            self.task_message = "Episode ended. Reset to choose tasks again."
        self.task_options = self.get_task_options()
        return True

    def step(self):
        """Advance exactly one label interval (0.2 s), unless the episode ends."""
        if self.done:
            return False
        for _ in range(self.motion_config.control_hz // self.motion_config.labels_hz):
            if not self.advance() or self.done:
                break
        return True


class Visualizer(Dashboard, v2.Visualizer):
    # Keep the established keys while routing them through the task guards.
    SHORTCUT_TASKS = {"LANE_LEFT": "switch_lane_left", "LANE_RIGHT": "switch_lane_right",
                      "FASTER": "accelerate", "SLOWER": "brake", "IDLE": "cruise"}

    def __init__(self, simulator, interval=.2):
        super().__init__(simulator, interval)
        self.font = self.pg.font.SysFont("segoeui", 16)
        self.small = self.pg.font.SysFont("consolas", 12)
        self.title_font = self.pg.font.SysFont("segoeui", 21, bold=True)
        self.node_font = self.pg.font.SysFont("segoeui", 11)
        self.update_layout()
        self.accumulator = 0.
        self.show_overtake_tree = bool(simulator.overtake)
        self.pg.display.set_caption("Highway V3 - state vocabulary and driving tasks")

    def command(self, name):
        name = self.SHORTCUT_TASKS.get(name, name)
        if name in ("Demo overtake", "Demo brake"):
            from task_examples import load_example
            load_example(self.sim, "overtake" if name == "Demo overtake" else "brake")
            self.play = True
            self.accumulator = 0.
            self.scroll = 0
            self.show_overtake_tree = name == "Demo overtake"
            self.frame = self.render_frame(reset_camera=True)
        elif name == "Overtake":
            if self.sim.start_overtake():
                self.show_overtake_tree = True
                self.play = True
                self.accumulator = 0.
        elif name == "Overtake tree":
            self.show_overtake_tree = True
        elif name == "State view":
            self.show_overtake_tree = False
        elif name == "Cancel overtake":
            self.sim.cancel_overtake()
        elif name in TASKS:
            # Manual task selection takes control from an example supervisor.
            self.sim.cancel_overtake("Manual task selected")
            self.sim.demo = None
            self.sim.queue(name)
        elif name in ("Step", "Play/Pause", "Reset", "New seed", "Diagnostics",
                      "Camera", "Faster play", "Slower play"):
            super().command(name)
            if name in ("Step", "Play/Pause", "Reset", "New seed"):
                self.accumulator = 0.
            if name in ("Reset", "New seed"):
                self.show_overtake_tree = False

    def update_playback(self, elapsed):
        """Accumulate wall time; cap catch-up work after dragging the window."""
        if not self.play or self.sim.done:
            self.accumulator = 0.
            return 0
        self.accumulator += min(max(elapsed, 0.), .25) * self.sim.step_duration / self.interval
        ticks = 0
        while self.accumulator + 1e-9 >= self.sim.tick_duration and ticks < 8:
            self.sim.advance()
            self.accumulator -= self.sim.tick_duration
            ticks += 1
            if self.sim.done:
                self.play = False
                self.accumulator = 0.
                break
        if ticks == 8:
            self.accumulator = min(self.accumulator, self.sim.tick_duration)
        if ticks:
            self.frame = self.render_frame()
        return ticks

    def run(self):
        pg = self.pg
        running = True
        previous = pg.time.get_ticks()
        self.draw()
        while running:
            for event in pg.event.get():
                if not self.handle_event(event):
                    running = False
                    break
            now = pg.time.get_ticks()
            if running:
                self.update_playback((now - previous) / 1000.)
                self.draw()
            previous = now
            self.clock.tick(60)

    def handle_event(self, event):
        if event.type == self.pg.KEYDOWN and event.key == self.pg.K_t:
            return True
        if (event.type == self.pg.KEYDOWN and event.key == self.pg.K_f
                and not getattr(event, "repeat", False)):
            self.command("follow_car")
            return True
        return super().handle_event(event)

    def panel_rows(self):
        import textwrap
        columns = max(15, self.panel_rect.width // self.small.size("M")[0])
        rows = [("TASK DFA: " + self.sim.dfa.label, (125, 195, 255))]
        if self.sim.demo:
            rows.append((f"EXAMPLE: {self.sim.demo.kind} / {self.sim.demo.phase}", (255, 215, 140)))
        rows.append(("TASK AVAILABILITY", (125, 195, 255)))
        for option in self.sim.get_task_options().values():
            rows.append((f"{'YES' if option.enabled else 'NO'} {option.task.symbol}",
                         (120, 245, 175) if option.enabled else (255, 190, 120)))
            if not option.enabled:
                rows.append(("  " + option.reason, (190, 195, 205)))
            if self.diagnostics:
                for ap in option.task.requires:
                    rows.append((f"  needs {ap}", (175, 185, 200)))
                for ap in option.task.forbids:
                    rows.append((f"  excludes {ap}", (175, 185, 200)))
        rows.append(("STATE VOCABULARY", (125, 195, 255)))
        if self.diagnostics:
            rows.append(("TRAFFIC: current / target m/s", (125, 195, 255)))
            h = self.sim.env.unwrapped
            for car in h.road.vehicles:
                if car is not h.vehicle:
                    uid = h._semantic_ids.get(car, "?")
                    target = getattr(car, "target_speed", car.speed)
                    rows.append((f"#{uid}: {car.speed:.1f} / {target:.1f}", (200, 220, 240)))
        wrapped = [(line, color) for text, color in rows
                   for line in (textwrap.wrap(text, columns) or [""])]
        return wrapped + super().panel_rows()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-labels", action="store_true")
    parser.add_argument("--playback-interval", type=float, default=.2,
                        help="Wall seconds per 0.2 simulation seconds (default: real time)")
    parser.add_argument("--example", choices=("overtake", "brake"),
                        help="Load an automatic example using the guarded tasks")
    parser.add_argument("--task", choices=tuple(TASKS), help="Queue an initial driving task")
    parser.add_argument("--headless-steps", type=int, default=None)
    args = parser.parse_args()
    simulator = None
    try:
        simulator = Simulator(args.seed, args.print_labels)
        if args.example:
            from task_examples import load_example
            load_example(simulator, args.example)
        if args.task and not simulator.queue(args.task):
            print(simulator.task_message)
        if args.headless_steps is not None:
            for _ in range(max(0, args.headless_steps)):
                if not simulator.step():
                    break
            print(f"steps={simulator.steps} speed={simulator.speed:.2f} m/s | {simulator.task_message}")
            if simulator.demo:
                print("\n".join(simulator.demo.trace))
                print(f"example={simulator.demo.phase} | DFA={simulator.dfa.label}")
        else:
            view = Visualizer(simulator, args.playback_interval)
            view.play = bool(args.example)
            view.run()
    finally:
        if simulator:
            simulator.close()
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
