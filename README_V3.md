# State-derived driving tasks

`mainV3.py` adds a guarded task layer and task controls to the existing Pygame
interface. It imports `mainV2.py` for the simulator setup, the unchanged 53 state
predicates, geometry, role binding, camera and scrolling. Keep both
files together. `mainV2.py` and `main.py` are preserved.

Run from the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe mainV3.py --seed 7
```

The repeated Accelerate entry is one task, giving six unique tasks. The fixed
task schemas are grounded only over immediate left/right lanes and the lead
role. `generate_task_vocabulary(labels)` evaluates their guards from the set of
true state predicates and returns each task's availability and reason.
`simulator.task_vocabulary` is the set of currently executable task symbols.
Task availability is separate from the state registry; no task predicates are
added to V2's state alphabet.

| Task | State requirements | Execution |
| --- | --- | --- |
| Switch lane left | `lane_exists(left)`, `front_gap_adequate(ego,left)`, `rear_gap_adequate(ego,left)` | Issue one left command; maintain speed until centered in that lane |
| Accelerate | No `danger_zone(ego,lead)` or `moving_into(intruder,ego_lane)`; cannot be closing on a braking lead | Increase speed toward 30 m/s |
| Switch lane right | `lane_exists(right)`, `front_gap_adequate(ego,right)`, `rear_gap_adequate(ego,right)` | Issue one right command; maintain speed until centered in that lane |
| Follow car (lead) | `present(lead)` and `ahead_of(lead,ego)` | Recompute speed using current lead speed and bumper gap each policy step |
| Brake | Ego is moving | Decelerate toward a full stop, including below V2's discrete speed targets |
| Cruise | No additional symbolic requirement | Hold the measured speed captured when the task starts |

Every task is disabled after a crash or episode end. Runtime checks additionally
enforce reachable lanes, the speed limit and stopped threshold. A second lane
change and Follow are disabled while a lane change is underway. Ego speed,
physical maneuver completion, reachability and episode termination are not
encoded by the supplied AP alphabet, so these remain explicit execution checks.
Task execution parameters are in `TaskConfig`. `LaneGapConfig` controls V3's
front/rear lane-change gap predicates independently of the Follow controller.
Its required bumper spacing is `2 + 0.3 * follower_speed + 0.6 * closing_speed`
metres, with negative speeds/closing speeds clamped to zero. Gaps turn adequate
above a 0.25 m margin and inadequate below zero. For equal speeds of 25 m/s,
a previously inadequate lane becomes available just above 9.75 m bumper
spacing, compared with 18.5 m under the previous V3 defaults. Both front and
rear gaps must pass the existing one-second prediction, including overlap and approaching
traffic checks. V2's `Config` retains the Follow controller's original headway.

Selection queues a task without advancing time. Step or Play revalidates it and
starts execution; a new valid selection replaces the previous queued selection.
The active task persists across subsequent steps. Accelerate ends at the speed
limit or when its guards fail. Brake ends at the stopped threshold. Follow binds
only to the current `lead` role, updating the bound vehicle as the role changes,
and returns to Cruise if there is no lead ahead. Lane changes finish based on
measured lane index, lateral offset and heading; new speed tasks can be selected
during a lane change. Reset clears pending tasks, maneuver state and follow
targets, and initializes Cruise at the reset speed. The pinned passing-lane
reference is initialized on reset; the interface has no manual rebind control.

The window starts paused with three separate control areas:

- **Simulator controls**: Play/Pause, Step, Reset, New seed and playback speed.
- **Task controls**: the six guarded ego tasks, active/queued status and an **Overtake** button.
- **Demos**: Overtake demo and Brake demo, with the example's current phase.

**Overtake** captures the current speed and runs an ordered tree of four task
DFAs in the current scene: **Lane left → Accelerate → Lane right → Brake to
the captured speed**. It needs a slower lead, an available left lane change
and room below the task speed limit. Each later task checks its own guards
before execution. The right maneuver waits until the original lead has been
passed and both return-lane gaps are adequate. Brake ends within 0.1 m/s of
the saved speed, then Cruise holds that speed. Normal Brake targets a full stop.

The right panel switches to the live **Overtake / DFA tree**, showing the four
child DFAs, queued/running/completed status and the saved/current speed.
Future children are marked planned; their execution waits for earlier children.
**State** returns to the facts panel; **Tree** reopens the maneuver view.
**Cancel overtake** holds the current speed, while an accepted lateral maneuver
finishes. Selecting a manual task cancels the sequence. Pause and Step work
throughout. Reset clears the tree and saved speed. **Overtake demo** loads
the reproducible scenario and runs the same four-task controller.

Camera control is in the header. The scrollable State & availability panel
shows task availability, reasons and true state predicates. Details toggles
diagnostics and task requirements. Disabled tasks are dimmed; hovering a task
shows its availability reason. The active task has a green outline.

Four live DFA graphs remain visible across the bottom: **EGO** in green,
**A** in blue, **B** in amber and **C** in purple. Colors match the vehicle
bodies and letter badges on the road. Each graph shows its current DFA state
and only the task transitions currently available to that car, with its speed.
Unavailable tasks disappear entirely from the graph. Each car evaluates the
same state vocabulary and task guards from its own perspective: its adjacent
lanes, lead, front/rear gaps, hazards, speed and maneuver progress. A stopped
car has no Brake branch; a car in the rightmost lane has no Lane right branch.
A crashed, removed or ended car has no available task branches.

Ego's DFA records executed task commands. The other three are explicitly
labelled **observed motion**: they infer task modes from the existing traffic
controller's measured movement, rather than claiming it executes V3 tasks.
Acceleration and braking use measured speed change with hysteresis. Follow
requires a nearby same-lane lead, reduced speed relative to the driver's
desired speed and no lateral maneuver. Lane transitions require measured
sideways motion or a lane crossing and finish when centered in the destination.

The three nearest traffic cars are selected on reset or demo load. Their
vehicle IDs, colors and histories stay pinned until the next reset. An offscreen
car keeps its graph and is marked offscreen; a removed car is marked unavailable.
Untracked traffic is muted gray. Rendering and resizing do not advance any DFA.
Repeated graph labels are cached to keep redraws inexpensive. V3 consumes full
road snapshots for its predicates; the separate, unused Gym observation is
left unnormalized to avoid redundant work during motion ticks.
Traffic graphs evaluate the task-relevant predicates with the same V2
evaluator. Their 5 Hz geometric checks are staggered between motion ticks;
speed and maneuver guards update every tick. Drawing uses cached results.

| Key | Task/control |
| --- | --- |
| Left / Right | Queue Switch lane left / right |
| Up / Down | Queue Accelerate / Brake |
| F | Queue Follow car (lead) |
| I | Queue Cruise |
| Space / P | Single step / Play or pause |
| R / N | Reset / New seed |
| D / C | Diagnostics / Camera |
| Escape | Close |

Keyboard and mouse use the same guards. Existing playback speed controls,
window resizing and AP panel scrolling remain available.

Playback now runs at real time by default, using 60 Hz physics and 30 Hz motion
and control updates. The window redraws at up to 60 FPS; traffic facts update
at 5 Hz. Step still advances exactly 0.2 simulation seconds using six small
control ticks. Playback uses accumulated wall time with bounded catch-up after
window stalls; there is no longer a 0.6-second jump between motion updates.
`--playback-interval 0.4` gives half-speed playback and `0.1` gives double speed.

Each normal traffic vehicle has a seeded initial speed and an individual
desired speed between 18 and 32 m/s. Existing IDM behavior still handles its
acceleration, following and lane changes. Current and target speeds are listed
in Diagnostics. Reset repeats the same seeded traffic; New seed changes it.

The live graphs show the currently enabled outgoing DFA transitions. The complete
11-state automaton, its transition table, Mermaid view and runnable overtaking
and braking examples are documented in [TASK_DFA.md](TASK_DFA.md).
`task_dfa.py` implements the DFA and `TASK_DFA.dot` exports its complete graph.
Click **Overtake demo** or **Brake demo** to load a controlled scenario and
automatically play its task sequence. Manual task selection takes control.

For a command line smoke check and the regression suite:

```powershell
.\.venv\Scripts\python.exe mainV3.py --headless-steps 20 --task brake
.\.venv\Scripts\python.exe mainV3.py --example overtake
.\.venv\Scripts\python.exe mainV3.py --example brake --headless-steps 40
.\.venv\Scripts\python.exe -m unittest test_mainV2 test_mainV3 -v
```

Following is a bounded proportional speed/gap controller using V2's required
gap formula. It updates once per motion tick (1/30 s); guard facts update every
0.2 s and are refreshed when a task request is queued or executed. Acceleration and
deceleration commands are limited through the speed controller's setpoint.
Lane-gap predicates are the existing one-second constant-velocity proxy.
These controls do not guarantee collision avoidance. If Accelerate or Follow
becomes unavailable, Cruise captures current speed; it does not automatically
perform an evasive maneuver. An accepted lane change is not automatically
aborted if traffic changes during the maneuver.

The tests exercise real highway-env dynamics and offscreen Pygame controls
under dummy SDL. Physical desktop input and appearance require a manual run.

The reusable coding-agent prompt is in `PROMPT_MAINV3.md`.
