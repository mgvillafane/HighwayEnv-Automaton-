"""Small deterministic scenarios that issue only V3's guarded tasks."""
from dataclasses import dataclass, field

import numpy as np
from highway_env.vehicle.kinematics import Vehicle

import mainV2 as v2


@dataclass
class ExampleController:
    """Stopped-lead braking example; overtaking uses the composite controller."""
    kind: str
    original_lane: tuple
    lead_uid: int
    phase: str = "approach"
    completed: bool = False
    trace: list[str] = field(default_factory=list)

    def select(self, sim, task, phase):
        if sim.queue(task):
            self.phase = phase
            self.trace.append(f"t={sim.snapshot.time:.2f}s: {task} -> {phase}")
            return True
        return False

    def update(self, sim):
        if self.completed or sim.done:
            return
        labels, available = sim.labels, sim.available()
        if self.phase == "approach":
            slower_lead = {"present(lead)", "slower_than(lead,ego)", "closing_on(ego,lead)"} <= labels
            if not slower_lead:
                return
            if "brake" in available:
                self.select(sim, "brake", "braking")
        elif self.phase == "braking" and sim.speed <= sim.task_config.stopped_speed:
            self.select(sim, "cruise", "complete")
            self.completed = True


def load_example(sim, kind):
    if kind not in ("overtake", "brake"):
        raise ValueError(f"Unknown example: {kind}")
    sim.reset()
    h, ego = sim.env.unwrapped, sim.env.unwrapped.vehicle
    origin = ("0", "1", 1)
    ego.position = np.array([100., 4.])
    ego.heading = 0.
    ego.speed = 24. if kind == "overtake" else 18.
    ego.lane_index = ego.target_lane_index = origin
    ego.lane = h.road.network.get_lane(origin)
    ego.target_speed = ego.speed
    lead = Vehicle(h.road, [160., 4.], speed=16. if kind == "overtake" else 0.)
    # Constant-speed traffic makes these examples reproducible and easy to
    # inspect. Normal seeded traffic uses the existing IDM behavior instead.
    if kind == "brake":
        traffic = [Vehicle(h.road, [100., 0.], speed=18.),
                   Vehicle(h.road, [100., 8.], speed=18.)]
    else:
        traffic = [Vehicle(h.road, [450., 0.], speed=28.),
                   Vehicle(h.road, [250., 8.], speed=20.)]
    h.road.vehicles = [ego, lead, *traffic]
    h._semantic_ids = {}
    sim.memory = v2.Memory()
    sim.snapshot = v2.extract_snapshot(sim.env)
    sim.task.rebind(sim.snapshot)
    sim.refresh()
    sim.traffic_dfas.reset(sim.snapshot, h)
    sim._hold_speed(f"{kind.title()} example ready. Press Play or Step.")
    sim.env.unwrapped.vehicle.target_speed = sim.cruise_speed
    if kind == "overtake":
        if not sim.start_overtake():
            raise RuntimeError(sim.task_message)
        sim.demo = sim.overtake
    else:
        sim.demo = ExampleController(kind, origin, h._semantic_ids[lead])
    return sim.demo
