"""Focused full-state semantic and control regression checks; no desktop needed."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import unittest
from unittest.mock import patch
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
import mainV2 as m


def vehicle(uid, x, y=4., vx=20., vy=0., heading=0.):
    return m.Vehicle(uid, np.array([x, y]), np.array([vx, vy]), heading,
                     5., 2., ("a", "b", int(round(y / 4))))


class Semantics(unittest.TestCase):
    def setUp(self):
        network = RoadNetwork()
        for y in (0., 4., 8.):
            network.add_lane("a", "b", StraightLane([0, y], [1000, y]))
        self.road = Road(network=network)
        self.ego = vehicle(1, 100.)
        self.lane = network.get_lane(("a", "b", 1))

    def snapshot(self, others=(), time=0., ego=None):
        ego = ego or self.ego
        return m.Snapshot(time, ego, (ego, *others), self.road)

    def label(self, snapshot, memory=None, task=None):
        task = task or m.TaskContext()
        if task.origin_lane is None:
            task.rebind(snapshot)
        memory = memory or m.Memory()
        bindings = m.bind_roles(snapshot, task)
        return m.label_highway_state(snapshot, bindings, m.Config(), memory), memory

    def test_empty_and_missing_lanes(self):
        labels, _ = self.label(self.snapshot())
        self.assertIn("front_gap_adequate(ego,left)", labels)
        self.assertIn("rear_gap_adequate(ego,left)", labels)
        labels, _ = self.label(self.snapshot(ego=vehicle(1, 100, 0)))
        self.assertNotIn("lane_exists(left)", labels)
        self.assertNotIn("front_gap_adequate(ego,left)", labels)
        self.assertNotIn("rear_gap_adequate(ego,left)", labels)

    def test_fast_rear_and_alongside(self):
        for other in (vehicle(2, 80, vx=40), vehicle(2, 100)):
            gaps = m.lane_gap_margins(self.ego, self.lane, self.snapshot([other]))
            self.assertLess(gaps["rear"]["margin"], 0)
        gaps = m.lane_gap_margins(self.ego, self.lane, self.snapshot([vehicle(2, 100)]))
        self.assertLess(gaps["front"]["margin"], 0)

    def test_projected_entry_and_crossing(self):
        entering = vehicle(2, 100, 8, vy=-4)
        gaps = m.lane_gap_margins(self.ego, self.lane, self.snapshot([entering]))
        self.assertLess(gaps["rear"]["margin"], 0)
        bindings = m.bind_roles(self.snapshot([entering]), m.TaskContext())
        self.assertEqual(bindings["roles"]["intruder"].uid, 2)
        # Ego passes a stopped vehicle inside the horizon, with both endpoints separated.
        fast = vehicle(1, 100, vx=200)
        gaps = m.lane_gap_margins(fast, self.lane, self.snapshot([vehicle(2, 200, vx=0)], ego=fast))
        self.assertLess(gaps["rear"]["margin"], 0)

    def test_boundary_is_not_settled(self):
        crossing = vehicle(1, 100, 2, vy=-1, heading=-.1)
        self.assertEqual(len(m.occupied_lanes(crossing, self.road)), 2)
        labels, _ = self.label(self.snapshot(ego=crossing))
        self.assertIn("in_lane(ego,current)", labels)
        self.assertNotIn("lane_settled(ego,current)", labels)
        self.assertTrue(m.moving_into(crossing, self.road.network.get_lane(("a", "b", 0)), m.Config()))
        labels, _ = self.label(self.snapshot())
        self.assertIn("lane_settled(ego,current)", labels)

    def test_role_history_and_simulation_time(self):
        task, memory = m.TaskContext(), m.Memory()
        s = self.snapshot([vehicle(2, 130)])
        task.rebind(s)
        self.label(s, memory, task)
        s = self.snapshot([vehicle(2, 132, vx=18)], .2)
        labels, _ = self.label(s, memory, task)
        self.assertIn("braking(lead)", labels)
        self.assertAlmostEqual(memory.measurements["acceleration"]["lead"], -10.)
        previous = memory.previous
        for _ in range(10):
            self.assertEqual(self.label(s, memory, task)[0], labels)
            self.assertIs(memory.previous, previous)
        s = self.snapshot([vehicle(3, 130, vx=3), vehicle(2, 140, vx=18)], .4)
        labels, _ = self.label(s, memory, task)
        self.assertNotIn("braking(lead)", labels)
        self.assertEqual(memory.measurements["acceleration"]["lead"], 0.)
        self.assertEqual(task.target_id, 2)

    def test_ttc_and_hysteresis(self):
        self.assertEqual(m.time_to_collision(self.ego, vehicle(2, 120, vx=30), self.lane), float("inf"))
        self.assertEqual(m.time_to_collision(self.ego, vehicle(2, 100), self.lane), 0.)
        task, memory = m.TaskContext(), m.Memory()
        # speed 20 -> required 29, bumper gap = x - 105
        s = self.snapshot([vehicle(2, 135)])  # margin 1: initially false
        task.rebind(s)
        self.assertNotIn("front_gap_adequate(ego,current)", self.label(s, memory, task)[0])
        self.assertIn("front_gap_adequate(ego,current)", self.label(self.snapshot([vehicle(2, 138)], .2), memory, task)[0])
        self.assertIn("front_gap_adequate(ego,current)", self.label(self.snapshot([vehicle(2, 135)], .4), memory, task)[0])
        self.assertNotIn("front_gap_adequate(ego,current)", self.label(self.snapshot([vehicle(2, 133)], .6), memory, task)[0])


class Controls(unittest.TestCase):
    def test_steps_reset_rebind_and_termination(self):
        sim = m.Simulator()
        try:
            initial = sim.snapshot
            with patch.object(sim.env, "step", wraps=sim.env.step) as step:
                sim.queue("SLOWER")
                self.assertTrue(sim.step())
                self.assertEqual(step.call_count, 1)
                self.assertEqual(sim.pending, "IDLE")
                self.assertAlmostEqual(sim.snapshot.time - initial.time, .2)
                acceleration = sim.memory.measurements["acceleration"]["ego"]
                sim.rebind()
                self.assertEqual(sim.memory.measurements["acceleration"]["ego"], acceleration)
                sim.done = True
                self.assertFalse(sim.step())
                self.assertEqual(step.call_count, 1)
            sim.queue("FASTER")
            sim.reset()
            self.assertEqual(sim.pending, "IDLE")
            self.assertEqual(sim.steps, 0)
            self.assertEqual(sim.snapshot.time, 0)
            self.assertFalse(sim.done)
            self.assertTrue(all(v == 0 for v in sim.memory.measurements["acceleration"].values()))
            sim.env.unwrapped.config["duration"] = .2
            sim.step()
            self.assertTrue(sim.done)
            self.assertFalse(sim.step())
        finally:
            sim.close()

    def test_window_events_and_paused_redraw(self):
        import pygame as pg
        sim = m.Simulator()
        try:
            view = m.Visualizer(sim)
            # Dummy SDL suppresses renderer drawing; enable offscreen surfaces.
            sim.env.unwrapped.viewer.enabled = True
            view.frame = sim.env.render()
            self.assertGreater(view.frame.max(), 0)
            labels, positions = sim.labels.copy(), [v.position.copy() for v in sim.snapshot.vehicles]
            with patch.object(sim.env, "step", wraps=sim.env.step) as step:
                for _ in range(3):
                    view.draw()
                self.assertEqual(step.call_count, 0)
                self.assertEqual(sim.labels, labels)
                for old, v in zip(positions, sim.snapshot.vehicles):
                    np.testing.assert_array_equal(old, v.position)
                pg.event.clear()
                pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE))
                pg.event.post(pg.event.Event(pg.QUIT))
                view.run()
                self.assertEqual(step.call_count, 1)
            self.assertTrue(sim.env.unwrapped.viewer.offscreen)
        finally:
            sim.close()
            pg.quit()
        self.assertFalse(pg.get_init())


if __name__ == "__main__":
    unittest.main()
