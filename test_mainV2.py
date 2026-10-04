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
        if task.passing_lane is None:
            task.rebind(snapshot)
        memory = memory or m.Memory()
        bindings = m.bind_roles(snapshot, task)
        return m.label_highway_state(snapshot, bindings, m.Config(), memory), memory

    def test_bounded_registry(self):
        self.assertEqual(m.ROLES, ("lead", "rear", "left_front", "left_rear",
                                  "right_front", "right_rear", "intruder"))
        self.assertEqual(m.LANES, ("left", "right", "passing_lane"))
        self.assertEqual(len(m.REGISTRY), 53)
        self.assertEqual(len(set(m.REGISTRY)), len(m.REGISTRY))
        for predicate, args in m.REGISTRY:
            self.assertNotIn("current", args)
            self.assertNotEqual(predicate, "gap_critical")
            if predicate in ("ahead_of", "slower_than"):
                self.assertIn(args[0], m.ROLES)
                self.assertEqual(args[1], "ego")
            elif predicate == "closing_on":
                self.assertIn(args, tuple(("ego", r) for r in m.FRONT_ROLES)
                              + tuple((r, "ego") for r in m.REAR_ROLES))
            elif predicate == "moving_into":
                self.assertEqual(args[0], "intruder")
                self.assertEqual(args[1], "ego_lane")
            elif predicate in ("present", "braking", "in_lane"):
                self.assertIn(args[0], m.ROLES)
        self.assertLess(len(m.OCCUPANCY_GROUNDINGS), len(m.ROLES) * len(m.LANES))

    def test_directional_relationships_and_intruder(self):
        labels, _ = self.label(self.snapshot([vehicle(2, 130, vx=15), vehicle(3, 80, vx=15)]))
        self.assertIn("ahead_of(lead,ego)", labels)
        self.assertIn("slower_than(lead,ego)", labels)
        self.assertIn("closing_on(ego,lead)", labels)
        self.assertIn("slower_than(rear,ego)", labels)
        self.assertNotIn("ahead_of(rear,ego)", labels)
        self.assertNotIn("closing_on(ego,rear)", labels)
        labels, _ = self.label(self.snapshot([vehicle(2, 110, 8, vy=-4)]))
        self.assertIn("moving_into(intruder,ego_lane)", labels)
        self.assertNotIn("moving_into(intruder,right)", labels)

    def test_danger_zone_and_rear_closing(self):
        for other, dangerous in ((vehicle(2, 114, vx=15), True),
                                 (vehicle(2, 115, vx=15), False),
                                 (vehicle(2, 110, vx=25), False),
                                 (vehicle(2, 103, vx=20), True)):
            labels, _ = self.label(self.snapshot([other]))
            self.assertEqual("danger_zone(ego,lead)" in labels, dangerous)
        self.assertNotIn("danger_zone(ego,lead)", self.label(self.snapshot())[0])
        labels, _ = self.label(self.snapshot([vehicle(2, 80, vx=30),
                                             vehicle(3, 80, 0, vx=30),
                                             vehicle(4, 80, 8, vx=30)]))
        for role in m.REAR_ROLES:
            self.assertIn(f"closing_on({role},ego)", labels)
        labels, _ = self.label(self.snapshot([vehicle(2, 130, 0, vx=15)]))
        self.assertIn("closing_on(ego,left_front)", labels)
        self.assertNotIn("danger_zone(ego,lead)", labels)

    def test_intruder_requires_adjacent_entry(self):
        for other in (vehicle(2, 110, 8), vehicle(2, 110, 8, vy=1),
                      vehicle(2, 110, 12, vy=-1), vehicle(2, 110, 4.5, vy=-1)):
            labels, _ = self.label(self.snapshot([other]))
            self.assertNotIn("present(intruder)", labels)
        labels, _ = self.label(self.snapshot([vehicle(2, 110, 8, vy=-1)]))
        self.assertIn("present(intruder)", labels)
        self.assertIn("moving_into(intruder,ego_lane)", labels)
        # Lane changes immediately change the physical cut-in destination.
        labels, _ = self.label(self.snapshot([vehicle(2, 110, 8, vy=-1)],
                                            ego=vehicle(1, 100, 8)))
        self.assertNotIn("present(intruder)", labels)

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

    def test_boundary_occupancy_and_entry(self):
        crossing = vehicle(2, 130, 2, vy=1, heading=.1)
        self.assertEqual(len(m.occupied_lanes(crossing, self.road)), 2)
        labels, _ = self.label(self.snapshot([crossing]))
        self.assertIn("in_lane(left_front,left)", labels)
        self.assertTrue(m.moving_into(crossing, self.lane, m.Config()))
        labels, _ = self.label(self.snapshot([vehicle(2, 130, 0)]))
        self.assertIn("in_lane(left_front,left)", labels)

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
        self.assertEqual(task.passing_lane, ("a", "b", 0))

    def test_ttc_and_hysteresis(self):
        self.assertEqual(m.time_to_collision(self.ego, vehicle(2, 120, vx=30), self.lane), float("inf"))
        self.assertEqual(m.time_to_collision(self.ego, vehicle(2, 100), self.lane), 0.)
        task, memory = m.TaskContext(), m.Memory()
        # speed 20 -> required 29, bumper gap = x - 105
        s = self.snapshot([vehicle(2, 135, 0)])  # margin 1: initially false
        task.rebind(s)
        self.assertNotIn("front_gap_adequate(ego,passing_lane)", self.label(s, memory, task)[0])
        self.assertIn("front_gap_adequate(ego,passing_lane)", self.label(self.snapshot([vehicle(2, 138, 0)], .2), memory, task)[0])
        self.assertIn("front_gap_adequate(ego,passing_lane)", self.label(self.snapshot([vehicle(2, 135, 0)], .4), memory, task)[0])
        self.assertNotIn("front_gap_adequate(ego,passing_lane)", self.label(self.snapshot([vehicle(2, 133, 0)], .6), memory, task)[0])


class Controls(unittest.TestCase):
    def test_traffic_moves_and_world_camera(self):
        import pygame as pg
        sim = m.Simulator()
        try:
            view = m.Visualizer(sim)
            self.assertTrue(view.camera_follow)
            view.command("Camera")  # opt into World view for the motion check
            viewer = sim.env.unwrapped.viewer
            viewer.enabled = True
            view.frame = view.render_frame()
            anchor = viewer.window_position().copy()
            before = sim.snapshot
            surface = viewer.sim_surface
            old_pixels = [surface.vec2pix(v.position) for v in before.vehicles[1:]]
            view.command("Step")
            np.testing.assert_array_equal(viewer.window_position(), anchor)
            for old, new, pixels in zip(before.vehicles[1:], sim.snapshot.vehicles[1:], old_pixels):
                self.assertGreater(np.linalg.norm(new.position - old.position), 1.)
                self.assertGreater(surface.vec2pix(new.position)[0], pixels[0])
            view.command("Camera")
            self.assertTrue(view.camera_follow)
            np.testing.assert_array_equal(viewer.window_position(), sim.snapshot.ego.position)
            view.command("Camera")
            self.assertFalse(view.camera_follow)
            # Recenter the world view before ego disappears, without stepping.
            view.camera_anchor.position -= np.array([1000., 0.])
            view.frame = view.render_frame()
            np.testing.assert_array_equal(viewer.window_position(), sim.snapshot.ego.position)
            self.assertEqual(sim.steps, 1)
            view.command("Reset")
            np.testing.assert_array_equal(viewer.window_position(), sim.snapshot.ego.position)
            self.assertEqual(sim.steps, 0)
        finally:
            sim.close()
            pg.quit()

    def test_ap_panel_scroll_and_resize_without_stepping(self):
        import pygame as pg
        sim = m.Simulator()
        try:
            view = m.Visualizer(sim)
            # Exercise overflow with the entire registry, beyond real traffic's
            # usually smaller true set, so the last AP must remain reachable.
            sim.labels = {f"{name}({','.join(args)})" for name, args in m.REGISTRY}
            sim.added, sim.removed = set(), set()
            view.draw()
            self.assertGreater(len(view.panel_rows()), view.visible_rows)
            snapshot, labels = sim.snapshot, sim.labels.copy()
            with patch.object(sim.env, "step", wraps=sim.env.step) as step:
                view.handle_event(pg.event.Event(pg.MOUSEWHEEL, y=-2))
                self.assertEqual(view.scroll, 6)
                view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_PAGEDOWN))
                self.assertGreater(view.scroll, 6)
                view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_END))
                maximum = len(view.panel_rows()) - view.visible_rows
                self.assertEqual(view.scroll, maximum)
                with patch.object(view, "text", wraps=view.text) as draw_text:
                    view.draw()
                    rendered = [call.args[0] for call in draw_text.call_args_list]
                    self.assertIn("  " + sorted(labels)[-1], rendered)
                view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=view.scroll_up.center))
                self.assertEqual(view.scroll, maximum - 1)
                view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_HOME))
                self.assertEqual(view.scroll, 0)
                thumb = view.scrollbar_thumb()
                view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=thumb.center))
                view.handle_event(pg.event.Event(pg.MOUSEMOTION, pos=(thumb.centerx, view.scroll_track.bottom)))
                view.handle_event(pg.event.Event(pg.MOUSEBUTTONUP, button=1))
                self.assertEqual(view.scroll, maximum)
                self.assertFalse(view.drag_scroll)
                view.handle_event(pg.event.Event(pg.VIDEORESIZE, w=1000, h=680))
                self.assertEqual(view.screen.get_size(), (1000, 680))
                self.assertLess(view.panel_rect.bottom, view.screen.get_height())
                view.draw()
                self.assertTrue(all(view.small.size(text)[0] <= view.panel_rect.width
                                    for text, _ in view.panel_rows()))
                self.assertEqual(step.call_count, 0)
                self.assertIs(sim.snapshot, snapshot)
                self.assertEqual(sim.labels, labels)
        finally:
            sim.close()
            pg.quit()

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
