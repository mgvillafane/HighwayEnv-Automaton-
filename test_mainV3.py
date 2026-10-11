"""Task guards and real simulator/UI execution checks, using dummy SDL."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import unittest
from unittest.mock import patch

import numpy as np
from highway_env.vehicle.kinematics import Vehicle

import mainV2 as v2
import mainV3 as m


class Vocabulary(unittest.TestCase):
    def test_groundings_use_existing_state_alphabet(self):
        alphabet = {f"{p}({','.join(args)})" for p, args in v2.REGISTRY}
        self.assertEqual(len(m.TASK_REGISTRY), 6)
        self.assertEqual(len(m.TASKS), 6)
        for task in m.TASK_REGISTRY:
            self.assertTrue(set(task.requires + task.forbids) <= alphabet)
        self.assertEqual({task.lane for task in m.LANE_TASKS}, {"left", "right"})
        self.assertEqual(m.TASKS["follow_car"].symbol, "follow_car(ego,lead)")

    def test_no_lanes_or_lead_fail_closed(self):
        options = m.generate_task_vocabulary(set())
        self.assertFalse(options["switch_lane_left"].enabled)
        self.assertFalse(options["switch_lane_right"].enabled)
        self.assertFalse(options["follow_car"].enabled)
        self.assertIn("No lane", options["switch_lane_right"].reason)
        self.assertTrue(options["accelerate"].enabled)
        self.assertTrue(options["brake"].enabled)
        self.assertTrue(options["cruise"].enabled)

    def test_lane_needs_both_gaps(self):
        for side in ("left", "right"):
            facts = {f"lane_exists({side})", f"front_gap_adequate(ego,{side})",
                     f"rear_gap_adequate(ego,{side})"}
            name = f"switch_lane_{side}"
            self.assertTrue(m.generate_task_vocabulary(facts)[name].enabled)
            for fact in facts:
                self.assertFalse(m.generate_task_vocabulary(facts - {fact})[name].enabled)

    def test_hazards_limits_and_terminal_guards(self):
        for facts in ({"danger_zone(ego,lead)"}, {"moving_into(intruder,ego_lane)"},
                      {"braking(lead)", "closing_on(ego,lead)"}):
            options = m.generate_task_vocabulary(facts)
            self.assertFalse(options["accelerate"].enabled)
            self.assertTrue(options["brake"].enabled)
        self.assertFalse(m.generate_task_vocabulary(set(), speed=30)["accelerate"].enabled)
        self.assertFalse(m.generate_task_vocabulary(set(), speed=0)["brake"].enabled)
        for kwargs in ({"done": True}, {}):
            labels = set() if kwargs else {"crashed(ego)"}
            self.assertFalse(any(o.enabled for o in m.generate_task_vocabulary(labels, **kwargs).values()))


class Execution(unittest.TestCase):
    def setUp(self):
        self.sim = m.Simulator()
        self.prepare()

    def tearDown(self):
        self.sim.close()
        import pygame
        pygame.quit()

    def prepare(self, lane=1, speed=20.):
        sim = self.sim
        h, car = sim.env.unwrapped, sim.env.unwrapped.vehicle
        index = ("0", "1", lane)
        car.position = np.array([100., lane * 4.])
        car.heading, car.speed = 0., speed
        car.lane_index = car.target_lane_index = index
        car.lane = h.road.network.get_lane(index)
        car.target_speed = speed
        car.crashed = False
        h.road.vehicles = [car]
        sim.pending_task = None
        sim.lane_change_target = None
        self.refresh()
        sim._hold_speed("Cruise active")

    def refresh(self):
        self.sim.memory = v2.Memory()
        self.sim.snapshot = v2.extract_snapshot(self.sim.env)
        self.sim.task.rebind(self.sim.snapshot)
        self.sim.refresh()

    def add_car(self, x, lane=1, speed=15.):
        road = self.sim.env.unwrapped.road
        car = Vehicle(road, [x, lane * 4.], heading=0., speed=speed)
        road.vehicles.append(car)
        self.refresh()
        return car

    def test_boundaries_and_blocked_lanes_use_actual_state(self):
        for lane, absent in ((0, "left"), (3, "right")):
            self.prepare(lane=lane)
            self.assertNotIn(f"switch_lane_{absent}", self.sim.available())
            target = self.sim.env.unwrapped.vehicle.target_lane_index
            self.assertFalse(self.sim.queue(f"switch_lane_{absent}"))
            self.sim.step()
            self.assertEqual(self.sim.env.unwrapped.vehicle.target_lane_index, target)
        self.prepare()
        self.add_car(100, lane=0)
        self.assertFalse(self.sim.queue("switch_lane_left"))
        self.assertIn("gap", self.sim.task_message)
        self.prepare()
        self.add_car(60, lane=2, speed=40.)
        self.assertFalse(self.sim.queue("switch_lane_right"))
        self.assertIn("Rear gap", self.sim.task_message)

    def test_queued_task_is_revalidated_and_raw_actions_are_rejected(self):
        self.assertTrue(self.sim.queue("switch_lane_left"))
        self.add_car(100, lane=0)
        self.assertFalse(self.sim.queue("LANE_RIGHT"))
        self.sim.step()
        self.assertEqual(self.sim.env.unwrapped.vehicle.target_lane_index, ("0", "1", 1))
        self.assertIn("Rejected at execution", self.sim.task_message)

    def test_closer_lane_gaps_allow_front_and_rear_but_still_reject_tight_gaps(self):
        front = self.add_car(115, lane=0, speed=20.)
        rear = self.add_car(85, lane=2, speed=20.)
        self.assertTrue({"switch_lane_left", "switch_lane_right"} <= self.sim.available())
        previous_config = m.LaneGapConfig(d0=3., tau=.6, gap_on=.5).predicates(v2.Config())
        for side, direction in (("front", "left"), ("rear", "right")):
            lane = self.sim.bindings["lanes"][self.sim.bindings["refs"][direction]]
            old = v2.lane_gap_margins(self.sim.snapshot.ego, lane, self.sim.snapshot, previous_config)
            self.assertLess(old[side]["margin"], 0.)  # 10 m bumper gap failed the previous V3 rule.
            self.assertIn(f"{side}_gap_adequate(ego,{direction})", self.sim.labels)
            self.assertGreater(self.sim.memory.measurements["gaps"][direction][side]["margin"], 0.)
        # Follow retains its original headway, independently of lane spacing.
        self.assertEqual(v2.required_gap(20., 0., self.sim.config), 29.)
        front.position[0], rear.position[0] = 110., 90.
        self.refresh()
        options = self.sim.get_task_options()
        self.assertFalse(options["switch_lane_left"].enabled)
        self.assertIn("Front gap", options["switch_lane_left"].reason)
        self.assertFalse(options["switch_lane_right"].enabled)
        self.assertIn("Rear gap", options["switch_lane_right"].reason)

    def test_cruise_captures_actual_speed_and_does_not_step_when_queued(self):
        self.prepare(speed=23.7)
        self.sim.env.unwrapped.vehicle.target_speed = 30.
        before = self.sim.snapshot
        self.assertTrue(self.sim.queue("cruise"))
        self.assertIs(self.sim.snapshot, before)
        self.assertEqual(self.sim.steps, 0)
        self.sim.step()
        self.assertAlmostEqual(self.sim.cruise_speed, 23.7)
        for _ in range(4):
            self.sim.step()
        self.assertAlmostEqual(self.sim.speed, 23.7, places=5)

    def test_accelerate_persists_and_brake_can_stop_below_discrete_minimum(self):
        self.sim.queue("accelerate")
        self.sim.step()
        first = self.sim.speed
        self.sim.step()
        self.assertGreater(first, 20.)
        self.assertGreater(self.sim.speed, first)
        self.assertEqual(self.sim.active_task, "accelerate")
        self.sim.queue("brake")
        for _ in range(50):
            self.sim.step()
        self.assertLess(self.sim.speed, self.sim.task_config.stopped_speed)
        self.assertAlmostEqual(self.sim.env.unwrapped.vehicle.target_speed, 0.)
        self.assertNotIn("brake", self.sim.available())
        self.assertIsNone(self.sim.pending_task)

    def test_lane_change_is_issued_once_and_completes_physically(self):
        self.assertTrue(self.sim.queue("switch_lane_left"))
        with patch.object(self.sim.env.unwrapped.action_type, "act",
                          wraps=self.sim.env.unwrapped.action_type.act) as act:
            self.sim.step()
            self.assertIsNotNone(self.sim.lane_change_target)
            self.assertFalse(self.sim.queue("switch_lane_left"))
            self.assertNotIn("switch_lane_right", self.sim.available())
            for _ in range(35):
                self.sim.step()
                if self.sim.lane_change_target is None:
                    break
            calls = [call.args[0] for call in act.call_args_list]
            self.assertEqual(calls.count(self.sim.actions["LANE_LEFT"]), 1)
        self.assertIsNone(self.sim.lane_change_target)
        self.assertEqual(self.sim.snapshot.ego.lane_index, ("0", "1", 0))
        self.assertEqual(self.sim.active_task, "cruise")
        # Steering changes the longitudinal projection slightly during settling.
        self.assertAlmostEqual(self.sim.speed, 20., delta=.05)

    def test_follow_requires_lead_and_reacts_to_its_speed_and_distance(self):
        self.add_car(145, lane=0)
        self.assertFalse(self.sim.queue("follow_car"))
        lead = self.add_car(150, speed=15.)
        self.assertTrue(self.sim.queue("follow_car"))
        for _ in range(40):
            self.sim.step()
        self.assertEqual(self.sim.active_task, "follow_car")
        self.assertEqual(self.sim.followed_uid, self.sim.bindings["roles"]["lead"].uid)
        self.assertLess(abs(self.sim.speed - 15.), 2.)
        self.assertFalse(self.sim.done)
        lane = self.sim.bindings["lanes"][self.sim.snapshot.ego.lane_index]
        gap = v2.longitudinal_clearance(self.sim.snapshot.ego, self.sim.bindings["roles"]["lead"], lane)
        self.assertGreater(gap, 5.)
        self.sim.env.unwrapped.road.vehicles.remove(lead)
        self.refresh()
        speed = self.sim.speed
        self.sim.step()
        self.assertEqual(self.sim.active_task, "cruise")
        self.assertIsNone(self.sim.followed_uid)
        self.assertAlmostEqual(self.sim.speed, speed)

    def test_follow_can_slow_for_stopped_lead(self):
        self.add_car(145, speed=0.)
        self.assertTrue(self.sim.queue("follow_car"))
        before = self.sim.speed
        self.sim.step()
        self.assertLess(self.sim.speed, before)

    def test_reset_truncation_and_rebind(self):
        self.sim.queue("accelerate")
        self.sim.step()
        self.sim.rebind()
        self.assertEqual(self.sim.active_task, "accelerate")
        self.sim.queue("brake")
        self.sim.reset()
        self.assertEqual(self.sim.active_task, "cruise")
        self.assertIsNone(self.sim.pending_task)
        self.assertIsNone(self.sim.lane_change_target)
        self.sim.env.unwrapped.config["duration"] = .2
        self.sim.step()
        self.assertTrue(self.sim.done)
        self.assertIsNone(self.sim.active_task)
        self.assertFalse(self.sim.available())
        self.assertFalse(self.sim.step())
        self.assertFalse(self.sim.queue("cruise"))

    def test_ui_mouse_and_keys_use_same_guards_and_preserve_paused_state(self):
        import pygame as pg
        self.prepare(lane=0)
        view = m.Visualizer(self.sim)
        self.sim.env.unwrapped.viewer.enabled = True
        view.frame = view.render_frame()
        view.draw()
        self.assertGreater(view.frame.max(), 0)
        with patch.object(self.sim.env, "step", wraps=self.sim.env.step) as step:
            left = next(button for button in view.buttons if button[1] == "switch_lane_left")
            self.assertFalse(left[2])
            view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=left[0].center))
            self.assertIsNone(self.sim.pending_task)
            view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_LEFT))
            self.assertIsNone(self.sim.pending_task)
            self.assertIn("No lane", self.sim.task_message)
            view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_f))
            self.assertIn("No lead", self.sim.task_message)
            accel = next(button for button in view.buttons if button[1] == "accelerate")
            view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=accel[0].center))
            self.assertEqual(self.sim.pending_task, "accelerate")
            view.handle_event(pg.event.Event(pg.MOUSEWHEEL, y=-2))
            view.handle_event(pg.event.Event(pg.VIDEORESIZE, w=1000, h=680))
            view.draw()
            self.assertTrue(all(rect.bottom <= view.screen.get_height() for rect, _, _ in view.buttons))
            self.assertTrue(all(view.small.size(text)[0] <= view.panel_rect.width
                                for text, _ in view.panel_rows()))
            self.assertEqual(step.call_count, 0)
            view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE))
            self.assertEqual(step.call_count, self.sim.motion_config.control_hz // self.sim.motion_config.labels_hz)
            self.assertAlmostEqual(self.sim.snapshot.time, .2)
            self.assertEqual(self.sim.active_task, "accelerate")

    def test_small_motion_ticks_and_realtime_playback(self):
        import pygame as pg
        view = m.Visualizer(self.sim)
        before = self.sim.snapshot.ego.position.copy()
        view.play = True
        self.assertEqual(view.update_playback(.1), 3)
        self.assertAlmostEqual(self.sim.snapshot.time, .1)
        self.assertEqual(self.sim.steps, 0)
        self.assertGreater(self.sim.snapshot.ego.position[0], before[0])
        self.assertAlmostEqual(self.sim.memory.time, 0.)
        self.assertEqual(view.update_playback(.1), 3)
        self.assertAlmostEqual(self.sim.snapshot.time, .2)
        self.assertAlmostEqual(self.sim.memory.time, .2)
        self.assertEqual(self.sim.steps, 1)
        view.command("Play/Pause")
        snapshot = self.sim.snapshot
        self.assertEqual(view.update_playback(1.), 0)
        self.assertIs(self.sim.snapshot, snapshot)
        view.command("Play/Pause")
        self.assertLessEqual(view.update_playback(10.), 8)
        view.command("Reset")
        self.assertFalse(view.play)
        self.assertEqual(view.accumulator, 0.)
        self.assertEqual(self.sim.snapshot.time, 0.)

    def test_traffic_has_distinct_seeded_current_and_target_speeds(self):
        def speeds():
            h = self.sim.env.unwrapped
            return np.array([(v.speed, v.target_speed) for v in h.road.vehicles if v is not h.vehicle])
        self.sim.reset()
        self.assertTrue(self.sim.env.observation_space.contains(self.sim.env.unwrapped.observation_type.observe()))
        first = speeds()
        self.assertGreater(np.ptp(first[:, 0]), 5.)
        self.assertGreater(np.ptp(first[:, 1]), 5.)
        self.assertTrue(np.all(first >= self.sim.motion_config.traffic_min_speed))
        self.assertTrue(np.all(first <= self.sim.motion_config.traffic_max_speed))
        self.sim.reset()
        np.testing.assert_array_equal(speeds(), first)
        self.sim.reset(new_seed=True)
        self.assertFalse(np.array_equal(speeds(), first))

    def test_dfa_tracks_parallel_lane_and_speed_tasks(self):
        self.assertEqual(self.sim.dfa.state, ("cruise", "none"))
        self.sim.queue("switch_lane_left")
        self.assertEqual(self.sim.dfa.state, ("cruise", "none"))
        self.sim.step()
        self.assertEqual(self.sim.dfa.state, ("cruise", "left"))
        self.sim.queue("accelerate")
        self.sim.step()
        self.assertEqual(self.sim.dfa.state, ("accelerate", "left"))
        for _ in range(20):
            self.sim.step()
            if self.sim.lane_change_target is None:
                break
        self.assertEqual(self.sim.dfa.state, ("accelerate", "none"))

    def test_examples_complete_without_collision_and_use_guarded_tasks(self):
        from task_examples import load_example
        for kind in ("overtake", "brake"):
            demo = load_example(self.sim, kind)
            for _ in range(80):
                self.sim.step()
                if demo.completed:
                    break
            self.assertTrue(demo.completed, demo.trace)
            self.assertFalse(self.sim.done)
            self.assertFalse(self.sim.snapshot.ego.crashed)
            self.assertEqual(self.sim.dfa.state, ("cruise", "none"))
            self.assertEqual(self.sim.snapshot.ego.lane_index, demo.original_lane)
            if kind == "brake":
                self.assertLessEqual(self.sim.speed, self.sim.task_config.stopped_speed)
                self.assertEqual(self.sim.env.unwrapped.vehicle.target_speed, 0.)
                lead = next(v for v in self.sim.snapshot.vehicles if v.uid == demo.lead_uid)
                lane = self.sim.bindings["lanes"][demo.original_lane]
                self.assertGreater(v2.longitudinal_clearance(self.sim.snapshot.ego, lead, lane), 5.)
            else:
                # Closer return gaps can let the right-lane request replace
                # Accelerate before it completes at the speed limit.
                self.assertEqual([event for _, event, _ in self.sim.dfa.history if event != "task_complete"],
                                 ["switch_lane_left", "lane_complete", "accelerate",
                                  "switch_lane_right", "lane_complete", "brake"])
                self.assertAlmostEqual(self.sim.speed, demo.previous_speed, delta=.1)
                self.assertAlmostEqual(self.sim.env.unwrapped.vehicle.target_speed, demo.previous_speed)
                lead = next(v for v in self.sim.snapshot.vehicles if v.uid == demo.lead_uid)
                lane = self.sim.bindings["lanes"][demo.original_lane]
                self.assertGreater(v2.longitudinal_clearance(lead, self.sim.snapshot.ego, lane), self.sim.config.d0)

    def test_overtake_button_queues_current_scene_and_tree_does_not_advance_time(self):
        import pygame as pg
        self.add_car(160, speed=12.)
        view = m.Visualizer(self.sim)
        view.draw()
        button, _, enabled = next(b for b in view.buttons if b[1] == "Overtake")
        self.assertTrue(enabled)
        snapshot, vehicles = self.sim.snapshot, list(self.sim.env.unwrapped.road.vehicles)
        view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center))
        self.assertIs(self.sim.snapshot, snapshot)
        self.assertEqual(self.sim.env.unwrapped.road.vehicles, vehicles)
        self.assertTrue(view.play)
        self.assertTrue(view.show_overtake_tree)
        self.assertEqual(self.sim.pending_task, "switch_lane_left")
        self.assertEqual(self.sim.overtake.previous_speed, 20.)
        self.assertEqual([s.status for s in self.sim.overtake.stages], ["queued", "planned", "planned", "planned"])
        self.assertTrue(all(not s.dfa.history for s in self.sim.overtake.stages))
        view.command("Play/Pause")
        for width, height in ((1000, 680), (1240, 800), (1500, 900)):
            view.handle_event(pg.event.Event(pg.VIDEORESIZE, w=width, h=height))
            view.draw()
            self.assertEqual(len(view.overtake_tree_nodes), 4)
            self.assertTrue(all(view.inspector_rect.contains(r) for r in view.overtake_tree_nodes))
            self.assertTrue(all(not a.colliderect(b) for i, a in enumerate(view.overtake_tree_nodes)
                                for b in view.overtake_tree_nodes[i + 1:]))
            self.assertEqual(len(view.dfa_nodes), 4)
        self.assertIs(self.sim.snapshot, snapshot)
        view.command("State view")
        view.draw()
        self.assertFalse(view.show_overtake_tree)
        self.assertTrue(any(name == "Overtake tree" for _, name, _ in view.buttons))
        view.command("Overtake tree")
        view.draw()
        self.assertEqual(len(view.overtake_tree_nodes), 4)
        cancel = next(rect for rect, name, _ in view.buttons if name == "Cancel overtake")
        view.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=cancel.center))
        self.assertEqual(self.sim.overtake.phase, "cancelled")
        self.assertIsNone(self.sim.pending_task)
        self.assertIs(self.sim.snapshot, snapshot)

    def test_overtake_restores_the_captured_speed_and_completes_each_child_dfa(self):
        from task_dfa import TASK_EVENTS
        for initial_speed in (20., 27.):
            self.sim.reset()
            self.prepare(speed=initial_speed)
            self.add_car(160, speed=initial_speed - 8.)
            self.assertTrue(self.sim.start_overtake())
            controller = self.sim.overtake
            restoring = []
            for _ in range(120):
                self.sim.step()
                if controller.phase == "restoring_speed":
                    restoring.append(self.sim.speed)
                if controller.completed or self.sim.done:
                    break
            self.assertTrue(controller.completed, controller.trace)
            self.assertFalse(self.sim.done)
            self.assertEqual(self.sim.snapshot.ego.lane_index, controller.original_lane)
            self.assertEqual(controller.previous_speed, initial_speed)
            self.assertEqual([s.status for s in controller.stages], ["complete"] * 4)
            self.assertTrue(restoring)
            self.assertTrue(all(speed >= initial_speed - .05 for speed in restoring))
            self.assertGreater(restoring[0], restoring[-1])
            self.assertAlmostEqual(self.sim.speed, initial_speed, delta=.1)
            self.assertAlmostEqual(self.sim.cruise_speed, initial_speed)
            self.assertEqual([event for _, event, _ in self.sim.dfa.history if event in TASK_EVENTS],
                             ["switch_lane_left", "accelerate", "switch_lane_right", "brake"])
            for stage in controller.stages:
                self.assertEqual(stage.dfa.state, ("cruise", "none"))
                self.assertEqual([event for _, event, _ in stage.dfa.history if event in TASK_EVENTS], [stage.task])
            for _ in range(5):
                self.sim.step()
            self.assertAlmostEqual(self.sim.speed, initial_speed, delta=.05)

    def test_overtake_rejects_missing_lead_missing_left_lane_and_blocked_left_lane(self):
        self.assertFalse(self.sim.start_overtake())
        self.assertIsNone(self.sim.overtake)
        self.prepare(lane=0)
        self.add_car(160, lane=0)
        self.assertFalse(self.sim.start_overtake())
        self.assertIn("No lane", self.sim.task_message)
        self.prepare()
        self.add_car(160)
        self.add_car(100, lane=0)
        self.assertFalse(self.sim.start_overtake())
        self.assertIn("gap", self.sim.task_message)
        self.assertIsNone(self.sim.pending_task)

    def test_overtake_revalidates_and_retries_a_newly_blocked_lane(self):
        self.add_car(160)
        self.assertTrue(self.sim.start_overtake())
        controller = self.sim.overtake
        blocker = self.add_car(100, lane=0)
        self.sim.advance()
        self.assertEqual(controller.current.status, "waiting")
        self.assertIn("gap", controller.current.reason)
        self.assertFalse(controller.current.dfa.history)
        self.assertIsNone(self.sim.lane_change_target)
        self.sim.env.unwrapped.road.vehicles.remove(blocker)
        self.refresh()
        for _ in range(6):
            self.sim.advance()
        self.assertEqual(controller.current.status, "active")
        self.assertEqual(controller.current.dfa.state, ("cruise", "left"))
        self.assertIsNotNone(self.sim.lane_change_target)

    def test_overtake_manual_cancel_reset_and_episode_end(self):
        from task_examples import load_example
        load_example(self.sim, "overtake")
        controller = self.sim.overtake
        self.sim.advance()
        view = m.Visualizer(self.sim)
        view.command("brake")
        self.assertFalse(controller.running)
        self.assertEqual(controller.phase, "cancelled")
        self.assertIsNone(self.sim.demo)
        self.sim.advance()
        self.assertEqual(self.sim.active_task, "brake")
        self.assertEqual(self.sim.brake_target_speed, 0.)
        view.command("Reset")
        self.assertIsNone(self.sim.overtake)
        self.assertFalse(view.show_overtake_tree)
        load_example(self.sim, "overtake")
        self.sim.env.unwrapped.config["duration"] = self.sim.tick_duration
        self.sim.advance()
        self.assertTrue(self.sim.done)
        self.assertEqual(self.sim.overtake.phase, "ended")
        self.assertFalse(self.sim.overtake.running)
        self.assertIsNone(self.sim.pending_task)
        self.assertIsNone(self.sim.pending_brake_speed)

    def test_overtake_waits_for_the_return_lane_before_starting_final_brake(self):
        self.add_car(160, speed=15.)
        self.assertTrue(self.sim.start_overtake())
        controller = self.sim.overtake
        for _ in range(20):
            self.sim.step()
            if controller.index == 1:
                break
        self.assertEqual(controller.index, 1)
        lane = self.sim.bindings["lanes"][controller.original_lane]
        # Catch the first measured pass before the supervisor's next update,
        # then put traffic alongside in the return lane.
        for _ in range(400):
            self.sim.advance()
            lead = next(v for v in self.sim.snapshot.vehicles if v.uid == controller.lead_uid)
            if v2.longitudinal_clearance(lead, self.sim.snapshot.ego, lane) > self.sim.config.d0:
                break
        self.assertGreater(v2.longitudinal_clearance(lead, self.sim.snapshot.ego, lane), self.sim.config.d0)
        self.assertEqual(controller.index, 1)
        blocker = self.add_car(self.sim.snapshot.ego.position[0], lane=1, speed=self.sim.speed)
        self.sim.step()
        self.assertEqual(controller.index, 1)
        self.assertNotIn("switch_lane_right", self.sim.available())
        self.assertEqual(controller.stages[2].status, "planned")
        self.assertEqual(controller.stages[3].status, "planned")
        self.sim.env.unwrapped.road.vehicles.remove(blocker)
        self.refresh()
        for _ in range(100):
            self.sim.step()
            if controller.completed:
                break
        self.assertTrue(controller.completed, controller.trace)
        self.assertAlmostEqual(self.sim.speed, controller.previous_speed, delta=.1)

    def test_brake_to_positive_speed_keeps_moving_and_manual_brake_still_stops(self):
        self.prepare(speed=27.)
        for task, target in (("brake", -1.), ("brake", 28.), ("brake", np.nan), ("cruise", 20.)):
            self.assertFalse(self.sim.queue(task, target_speed=target))
        self.assertTrue(self.sim.queue("brake", target_speed=20.))
        for _ in range(50):
            self.sim.step()
            if self.sim.active_task == "cruise":
                break
        self.assertAlmostEqual(self.sim.speed, 20., delta=.1)
        self.assertAlmostEqual(self.sim.cruise_speed, 20.)
        self.assertIn("brake", self.sim.available())
        self.assertTrue(self.sim.queue("brake"))
        self.sim.step()
        self.assertEqual(self.sim.brake_target_speed, 0.)

    def test_live_dfas_keep_three_distinct_vehicle_identities_and_colors(self):
        from live_dfa import VEHICLE_COLORS
        from highway_env.vehicle.graphics import VehicleGraphics
        self.sim.reset()
        tracked = self.sim.traffic_dfas.cars
        uids = [car.uid for car in tracked]
        self.assertEqual(len(set(uids)), 3)
        self.assertNotIn(self.sim.snapshot.ego.uid, uids)
        self.assertEqual([car.label for car in tracked], ["A", "B", "C"])
        h = self.sim.env.unwrapped
        actual = {h._semantic_ids[v]: VehicleGraphics.get_color(v) for v in h.road.vehicles}
        self.assertEqual(actual[self.sim.snapshot.ego.uid], VEHICLE_COLORS[0])
        for car in tracked:
            self.assertEqual(actual[car.uid], car.color)
        for _ in range(10):
            self.sim.advance()
        self.assertEqual([car.uid for car in self.sim.traffic_dfas.cars], uids)
        missing = next(v for v in h.road.vehicles if h._semantic_ids[v] == uids[0])
        h.road.vehicles.remove(missing)
        self.sim.advance()
        cards = self.sim.traffic_dfas.cards(self.sim.snapshot, self.sim.dfa)
        self.assertEqual(cards[1]["uid"], uids[0])
        self.assertIsNone(cards[1]["vehicle"])
        self.assertEqual(cards[1]["dfa"].state, ("ended", "none"))
        self.assertFalse(cards[1]["options"])

    def test_each_live_graph_uses_its_own_lanes_lead_speed_and_gap_guards(self):
        stopped = self.add_car(100, lane=0, speed=0.)
        lead = self.add_car(125, lane=1, speed=20.)
        fast = self.add_car(100, lane=3, speed=30.)
        h = self.sim.env.unwrapped
        self.sim.traffic_dfas.reset(self.sim.snapshot, h)
        observed = {car.uid: car for car in self.sim.traffic_dfas.cars}
        def allowed(live):
            return {name for name, option in observed[h._semantic_ids[live]].options.items() if option.enabled}
        self.assertNotIn("switch_lane_left", allowed(stopped))
        self.assertNotIn("brake", allowed(stopped))
        self.assertNotIn("switch_lane_right", allowed(stopped))  # Ego alongside in its right lane.
        self.assertNotIn("switch_lane_right", allowed(fast))
        self.assertNotIn("accelerate", allowed(fast))
        self.assertNotIn("follow_car", allowed(lead))
        self.assertIn("follow_car", self.sim.available())
        self.assertIn("switch_lane_right", self.sim.available())

        # New stopped traffic is a lead hazard for the observed car, while ego
        # still sees its original, equal-speed lead. Sample the actual next state.
        self.add_car(142, lane=1, speed=0.)
        self.sim.step()
        self.assertIn("follow_car", allowed(lead))
        self.assertNotIn("accelerate", allowed(lead))
        self.assertIn("accelerate", self.sim.available())
        self.assertIn("danger_zone(ego,lead)", observed[h._semantic_ids[lead]].labels)
        view = m.Visualizer(self.sim)
        view.draw()
        cards = self.sim.traffic_dfas.cards(self.sim.snapshot, self.sim.dfa, self.sim.get_task_options())
        from task_dfa import transition
        for card, nodes, edges in zip(cards, view.dfa_nodes, view.dfa_edges):
            expected = {name for name, option in card["options"].items() if option.enabled}
            self.assertEqual(set(nodes), expected)
            self.assertEqual(edges, {name: transition(card["dfa"].state, name) for name in expected})

    def test_traffic_graph_guards_react_to_maneuver_crash_and_episode_end(self):
        from task_examples import load_example
        load_example(self.sim, "brake")
        observed = self.sim.traffic_dfas.cars[0]
        h = self.sim.env.unwrapped
        car = next(v for v in h.road.vehicles if h._semantic_ids[v] == observed.uid)
        car.target_lane_index = (*car.lane_index[:2], car.lane_index[2] + 1)
        car.heading = .05
        self.sim.advance()
        for name in ("switch_lane_left", "switch_lane_right", "follow_car"):
            self.assertFalse(observed.options[name].enabled)
        car.crashed = True
        self.sim.advance()
        self.assertFalse(observed.options)
        view = m.Visualizer(self.sim)
        view.draw()
        self.assertFalse(view.dfa_nodes[1])
        self.assertFalse(view.dfa_edges[1])
        h.config["duration"] = self.sim.snapshot.time + self.sim.tick_duration
        self.sim.advance()
        view.draw()
        self.assertTrue(self.sim.done)
        self.assertTrue(all(not nodes for nodes in view.dfa_nodes))

    def test_traffic_guard_projection_matches_full_state_evaluation(self):
        self.sim.reset()
        memories = {car.uid: v2.Memory() for car in self.sim.traffic_dfas.cars}
        guard_aps = {ap for task in m.TASK_REGISTRY for ap in task.requires + task.forbids}
        guard_aps.update({"crashed(ego)", "braking(lead)", "closing_on(ego,lead)"})
        for tick in range(25):
            for car in self.sim.traffic_dfas.cars:
                reference = memories[car.uid]
                if reference.time != car.memory.time:
                    snapshot = car.memory.previous
                    bindings = v2.bind_roles(snapshot, car.context, self.sim.predicate_config)
                    full = v2.label_highway_state(snapshot, bindings, self.sim.predicate_config, reference)
                    self.assertEqual(car.labels & guard_aps, full & guard_aps)
            if tick < 24:
                self.sim.advance()
        # Traffic geometry is spread between ego's label ticks, at the same
        # frequency, instead of evaluating all four cars in a single frame.
        self.assertEqual(len({car.memory.time for car in self.sim.traffic_dfas.cars}), 3)
        self.assertTrue(all(0. <= self.sim.snapshot.time - car.memory.time < self.sim.step_duration
                            for car in self.sim.traffic_dfas.cars))

    def test_observed_dfas_respond_to_motion_and_complete_lane_changes(self):
        from task_examples import load_example
        load_example(self.sim, "brake")
        observed = self.sim.traffic_dfas.cars[0]
        h = self.sim.env.unwrapped
        car = next(v for v in h.road.vehicles if h._semantic_ids[v] == observed.uid)
        car.speed += 2.
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("accelerate", "none"))
        car.speed -= 3.
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("brake", "none"))
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("cruise", "none"))
        original = car.lane_index
        car.target_lane_index = (*original[:2], original[2] + 1)
        car.heading = .05
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("cruise", "right"))
        car.position = np.array([190., 4.])
        car.heading = 0.
        car.lane_index = car.target_lane_index
        car.lane = h.road.network.get_lane(car.lane_index)
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("cruise", "none"))
        self.assertIn("lane_complete", [event for _, event, _ in observed.dfa.history])

    def test_observed_follow_state_uses_same_lane_lead(self):
        from task_examples import load_example
        load_example(self.sim, "brake")
        observed = self.sim.traffic_dfas.cars[0]
        h = self.sim.env.unwrapped
        car = next(v for v in h.road.vehicles if h._semantic_ids[v] == observed.uid)
        car.target_speed = 30.
        ahead = Vehicle(h.road, car.position + [30., 0.], speed=18.)
        h.road.vehicles.append(ahead)
        self.sim.advance()
        self.assertEqual(observed.dfa.state, ("follow_car", "none"))

    def test_dashboard_groups_controls_draws_four_graphs_and_removes_rebind(self):
        import pygame as pg
        from task_examples import load_example
        load_example(self.sim, "brake")
        view = m.Visualizer(self.sim)
        self.sim.env.unwrapped.viewer.enabled = True
        view.frame = view.render_frame()
        view.draw()
        self.assertEqual(len(view.dfa_nodes), 4)
        self.assertEqual(len(view.road_badges), 4)
        with patch.object(self.sim, "rebind", wraps=self.sim.rebind) as rebind:
            view.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_t))
            view.command("Rebind task")
            self.assertEqual(rebind.call_count, 0)
        self.assertFalse(any("rebind" in name.lower() for _, name, _ in view.buttons))
        snapshots = self.sim.snapshot
        histories = [list(c.dfa.history) for c in self.sim.traffic_dfas.cars]
        for width, height in ((1000, 680), (1240, 800), (1500, 900)):
            view.handle_event(pg.event.Event(pg.VIDEORESIZE, w=width, h=height))
            view.draw()
            sections = [view.road_rect, view.sim_control_rect, view.task_control_rect,
                        view.demo_rect, view.inspector_rect, *view.dfa_rects]
            for i, rect in enumerate(sections):
                self.assertTrue(view.screen.get_rect().contains(rect))
                self.assertFalse(any(rect.colliderect(other) for other in sections[i + 1:]))
            cards = self.sim.traffic_dfas.cards(self.sim.snapshot, self.sim.dfa, self.sim.get_task_options())
            for rect, nodes, card in zip(view.dfa_rects, view.dfa_nodes, cards):
                expected = {name for name, option in card["options"].items() if option.enabled}
                self.assertEqual(set(nodes), expected)
                self.assertTrue(all(rect.contains(node) for node in nodes.values()))
            self.assertTrue(view.sim_control_rect.contains(next(r for r, n, _ in view.buttons if n == "Step")))
            self.assertTrue(view.task_control_rect.contains(next(r for r, n, _ in view.buttons if n == "brake")))
            self.assertTrue(view.demo_rect.contains(next(r for r, n, _ in view.buttons if n == "Demo brake")))
        self.assertIs(self.sim.snapshot, snapshots)
        self.assertEqual(histories, [c.dfa.history for c in self.sim.traffic_dfas.cars])
        view.command("Demo overtake")
        self.assertTrue(view.play)
        self.assertEqual(len(self.sim.traffic_dfas.cars), 3)
        self.assertTrue(all(car.dfa.state == ("cruise", "none") for car in self.sim.traffic_dfas.cars))


class Automaton(unittest.TestCase):
    def test_transition_function_is_total_and_closed(self):
        from task_dfa import ALPHABET, STATES, transition
        self.assertEqual(len(STATES), 11)
        for state in STATES:
            for event in ALPHABET:
                self.assertIn(transition(state, event), STATES)

    def test_rejected_requests_and_terminal_state(self):
        from task_dfa import TaskDFA
        dfa = TaskDFA()
        dfa.dispatch("switch_lane_left")
        self.assertEqual(dfa.state, ("cruise", "left"))
        dfa.dispatch("reject")
        self.assertEqual(dfa.state, ("cruise", "left"))
        dfa.dispatch("brake")
        self.assertEqual(dfa.state, ("brake", "left"))
        dfa.dispatch("lane_complete")
        self.assertEqual(dfa.state, ("brake", "none"))
        dfa.dispatch("end")
        dfa.dispatch("accelerate")
        self.assertEqual(dfa.state, ("ended", "none"))
        dfa.dispatch("reset")
        self.assertEqual(dfa.state, ("cruise", "none"))


if __name__ == "__main__":
    unittest.main()
