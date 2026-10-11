"""Ordered composition of four guarded task DFAs for one overtake."""
from dataclasses import dataclass, field

import mainV2 as v2
from task_dfa import TaskDFA


@dataclass
class OvertakeStage:
    task: str
    label: str
    dfa: TaskDFA = field(default_factory=TaskDFA)
    status: str = "planned"
    reason: str = ""


@dataclass
class OvertakeController:
    original_lane: tuple
    left_lane: tuple
    lead_uid: int
    previous_speed: float
    kind: str = field(default="overtake", init=False)
    phase: str = "change_left"
    completed: bool = False
    stopped: bool = False
    index: int = 0
    trace: list[str] = field(default_factory=list)
    stages: list[OvertakeStage] = field(default_factory=list)

    def __post_init__(self):
        self.stages = [OvertakeStage(task, label) for task, label in (
            ("switch_lane_left", "Lane left"), ("accelerate", "Accelerate"),
            ("switch_lane_right", "Lane right"), ("brake", f"Brake to {self.previous_speed:.1f} m/s"))]

    @property
    def running(self):
        return not self.completed and not self.stopped

    @property
    def current(self):
        return self.stages[self.index]

    def request(self, sim):
        target = self.previous_speed if self.current.task == "brake" else None
        if sim.queue(self.current.task, target_speed=target):
            self.current.status, self.current.reason = "queued", ""
            self.trace.append(f"t={sim.snapshot.time:.2f}s: {self.current.task} -> {self.phase}")
        else:
            self.current.status, self.current.reason = "waiting", sim.task_message

    def task_started(self, task):
        if not self.running:
            return
        if task != self.current.task or self.current.status != "queued":
            self.cancel("Manual task selected")
            return
        self.current.status, self.current.reason = "active", ""
        self.current.dfa.dispatch(task)

    def task_event(self, event):
        if self.running and self.current.status == "active":
            self.current.dfa.dispatch(event)

    def rejected(self, reason):
        if self.running:
            self.current.status, self.current.reason = "waiting", reason

    def cancel(self, reason, ended=False):
        if not self.running:
            return
        self.stopped = True
        self.phase = "ended" if ended else "cancelled"
        for stage in self.stages:
            if stage.status != "complete":
                stage.status, stage.reason = self.phase, reason
                stage.dfa.dispatch("end")

    def finish_stage(self):
        self.current.dfa.dispatch("lane_complete" if self.current.task.startswith("switch_lane") else "task_complete")
        self.current.status, self.current.reason = "complete", ""

    def next_stage(self, sim):
        self.finish_stage()
        self.index += 1
        self.phase = ("change_left", "passing", "change_right", "restoring_speed")[self.index]
        self.request(sim)

    def finish(self, sim):
        self.finish_stage()
        self.completed, self.phase = True, "complete"
        sim._hold_speed("Overtake complete; cruising at the restored speed.")
        if abs(sim.speed - self.previous_speed) <= sim.task_config.stopped_speed:
            sim.cruise_speed = self.previous_speed
        sim.env.unwrapped.vehicle.target_speed = sim.cruise_speed
        self.trace.append(f"t={sim.snapshot.time:.2f}s: complete -> cruise {sim.cruise_speed:.1f} m/s")

    def update(self, sim):
        if not self.running:
            return
        if sim.done:
            self.cancel("Episode ended", ended=True)
            return
        # A request is a queued task until the simulator actually executes it.
        if self.current.status == "queued":
            return
        if self.index == 3 and sim.speed <= self.previous_speed + sim.task_config.stopped_speed:
            self.finish(sim)
            return
        if self.current.status in ("planned", "waiting"):
            self.request(sim)
            return
        if self.index == 0:
            if sim.lane_change_target is None and sim.snapshot.ego.lane_index == self.left_lane:
                self.next_stage(sim)
        elif self.index == 1:
            lead = next((v for v in sim.snapshot.vehicles if v.uid == self.lead_uid), None)
            if lead is None:
                self.cancel("The car being overtaken is no longer present")
                sim._hold_speed("Overtake cancelled: tracked lead disappeared.")
                return
            lane = sim.bindings["lanes"][self.original_lane]
            passed = v2.longitudinal_clearance(lead, sim.snapshot.ego, lane) > sim.config.d0
            option = sim.get_task_options()["switch_lane_right"]
            if passed and option.enabled:
                self.next_stage(sim)
            else:
                self.current.reason = option.reason if passed else f"Passing car #{self.lead_uid}"
        elif self.index == 2:
            if sim.lane_change_target is None and sim.snapshot.ego.lane_index == self.original_lane:
                self.finish_stage()
                self.index, self.phase = 3, "restoring_speed"
                if sim.speed <= self.previous_speed + sim.task_config.stopped_speed:
                    self.finish(sim)
                else:
                    self.request(sim)
