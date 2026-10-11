# Coding-agent prompt

Extend the existing highway-env project by creating `mainV3.py` based on
`mainV2.py`. Inspect and reuse the existing code and Pygame interface. Preserve
`mainV2.py`, `main.py` and the existing state predicate registry and semantics.

Create a task vocabulary whose availability is derived from the current true
state predicates. There are six unique tasks (Accelerate was listed twice):

- Switch lane left: `switch_lane(ego,left)`.
- Accelerate: `accelerate(ego)`.
- Switch lane right: `switch_lane(ego,right)`.
- Follow car: `follow_car(ego,lead)`; follow only the current lead in ego's lane.
- Brake: `brake(ego)`.
- Cruise: `cruise(ego)`; hold the measured velocity when selected.

Represent task schemas and their preconditions explicitly. Ground lane-change
tasks only over `left` and `right` from the existing LANES vocabulary, and Follow
only over the `lead` role. The pinned `passing_lane` is a state reference, not
another lane-change direction. Keep task availability separate from the state
alphabet and expose a function that takes true APs and returns enabled tasks
and reasons for disabled tasks.

Use these guards:

- Disable every task if `crashed(ego)` or the episode has ended.
- Lane changes require `lane_exists(direction)`,
  `front_gap_adequate(ego,direction)` and
  `rear_gap_adequate(ego,direction)`. A missing lane must be rejected through
  both mouse and keyboard controls. Check physical lane reachability too.
- Disable additional lane changes while a maneuver is in progress; finish
  using measured lane position and heading rather than elapsed time alone.
- Follow requires `present(lead)` and `ahead_of(lead,ego)`. Never substitute a
  side, rear or intruder role. Recompute the current lead and control speed
  from its measured speed and bumper gap on every policy step. If the lead
  disappears, return to Cruise at the current speed.
- Accelerate is unavailable in `danger_zone(ego,lead)`, while
  `moving_into(intruder,ego_lane)`, when closing on a braking lead, or at the
  configured maximum speed.
- Brake is unavailable when already stopped. It must be able to slow below
  the environment's minimum discrete target speed and approach a full stop.
- Cruise captures actual speed, rather than assuming the low-level IDLE action
  holds the current measured speed.

The supplied AP registry does not encode ego speed, maneuver progress or
episode termination. Keep these as explicit runtime checks; do not invent
new state predicates or claim the AP set contains facts that it does not.

Integrate all six tasks into the existing window with clear task labels,
disabled buttons, visible reasons, and active/queued task status. Use arrows
for lane changes/acceleration/braking, F for Follow and I for Cruise. Keep
Step, Play/Pause, Reset, New seed, diagnostics, camera,
playback speed, scrolling and resizing working.

Selecting a task queues it without stepping. Step/Play revalidates the queued
task before starting it. Longitudinal tasks persist until replaced or their
guards fail; lane commands execute once and track physical completion. Avoid
multiple lateral commands from one selection. Reset clears all task/controller
state. Stop cleanly on termination or truncation. Document the fallback and
any traffic-model limitations without claiming collision avoidance guarantees.

Use the existing dependencies and virtual environment. Add focused tests for
lane boundaries, missing lanes, front/rear gap rejection, absent/side-only
lead cars, Follow speed response and lead disappearance, acceleration,
braking to a stop, Cruise holding actual speed, queued-command revalidation,
single lane-change execution, task lifecycle, reset/end states and consistent
mouse/keyboard guards. Check that paused drawing, task selection and resizing
do not advance time. Run existing V2 regressions, new V3 tests and a headless
V3 smoke check. Add concise run instructions and report the results.

Follow-up requirements: use smooth real-time playback with small physics/control
ticks and a separate symbolic-label cadence. Keep Step equal to 0.2 simulation
seconds. Give every normal traffic car seeded individual initial and desired
speeds. Implement a deterministic task automaton that retains both longitudinal
mode and lateral maneuver state, rejects unavailable requests, and handles
completion/end/reset. Provide its full transition table and diagram. Add runnable
overtaking and braking examples that use the same guarded tasks as manual input,
with UI buttons and console traces. Verify example completion without collisions,
motion timing, traffic reproducibility and totality of the DFA transition function.

Interface requirements: remove the manual Rebind lane control and T shortcut.
Separate Simulator controls, ego Task controls and Demos into distinct panels.
Show four live DFA views for ego and three nearest traffic cars, using
stable vehicle IDs and consistent green/blue/amber/purple colors on graphs and
road badges. Show each car's current state and only its enabled task transitions;
hide unavailable task nodes and arrows. Evaluate the same symbolic and runtime
guards independently from each car's perspective, including its own adjacent
lanes, lead, hazards, front/rear gaps, speed and maneuver state. Keep per-car
predicate histories and sample geometric facts on the existing label cadence;
do not recalculate geometry every redraw. Preserve the full internal 11-state
automaton and static DOT export. Ego's graph
reflects executed task events; traffic graphs must be labelled as observers of
measured movement, without inventing task commands from their IDM controller.
Pin those identities until reset/demo load and handle offscreen/missing cars.
Preserve scrolling, resizing, task guards and the smooth playback clock.

Allow closer lane changes using a separate configurable `LaneGapConfig`:
2 m base distance, 0.3 s follower headway, 0.6 s closing-speed allowance,
0.25 m switch-on margin and 0 m switch-off margin for both front and rear.
Retain the one-second gap prediction, overlap and fast-approach rejection,
and keep the Follow controller's original headway independent of these values.
Test per-car availability, disappearing disabled branches, stopped/max-speed
cars, lane boundaries, maneuver/crash/end handling, and closer front/rear gaps.

Add an **Overtake** button in Task controls that acts on the current scene and
shows a live tree of four dependent task DFAs: Lane left, Accelerate, Lane right,
Brake to the speed captured when the maneuver started. Keep the six atomic
task symbols unchanged. Give each child its own TaskDFA and visible lifecycle;
activate them in order through the guarded queue, wait for physical lane
completion and track the original lead's identity through the pass. Revalidate
the right lane's front/rear gaps before returning. Give the final Brake a
positive speed target and end in Cruise at that target, within 0.1 m/s.
Normal Brake still targets zero. Display the saved/current speed, planned,
queued, active, completed and waiting stages, plus guard reasons. Preserve the
four colored per-car DFA views. Provide State/Tree navigation and cancellation;
manual tasks cancel the sequence, reset clears it, and episode end stops it.
Use this same controller for the reproducible Overtake demo. Test the button
without resetting traffic or advancing time, exact task order, multiple saved
speeds, brake target behavior, lane revalidation, cancellation/end/reset and
paused/resized tree rendering. Update the example trace and documentation.
