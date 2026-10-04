# Interactive semantic-label visualizer

`mainV2.py` manually controls `highway-v0` and labels full simulator state.
`main.py` is preserved. No training or DFA is involved.

## Install and run (PowerShell)

From the workspace root:

```powershell
cd highway_project
# Only create the environment if .venv does not already exist:
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe mainV2.py --seed 7
```

Required packages are already in `requirements.txt`: highway-env, gymnasium,
pygame, numpy. Checks passed using Python 3.13, highway-env 1.12.1,
Gymnasium 1.3.0, Pygame 2.6.1, and NumPy 2.5.3. The existing Pygame installation
is reused; no additional GUI package is required.

Optional commands:

```powershell
.\.venv\Scripts\python.exe mainV2.py --print-labels --playback-interval 0.6
.\.venv\Scripts\python.exe mainV2.py --headless-steps 20 --print-labels
.\.venv\Scripts\python.exe -m unittest test_mainV2 -v
```

## Controls

The window starts paused, showing the initial frame and labels. Click the
buttons or use these shortcuts:

| Key | Effect |
| --- | --- |
| Space | Pause playback and execute exactly one step |
| P | Play/pause |
| Left / Right | Queue LANE_LEFT / LANE_RIGHT |
| Up / Down | Queue FASTER / SLOWER |
| I | Queue IDLE |
| R / N | Reset current seed / reset with incremented seed |
| T | Rebind the pinned passing lane |
| D | Toggle diagnostics |
| C | Switch World/Follow camera |
| Escape | Close |

Driving commands replace the pending command and execute once at the next
step, then revert to IDLE. Key repeat is disabled. IDLE maintains existing
speed and lane targets; it does not stop the vehicle. Unavailable action
buttons are marked N/A. Faster/Slower play buttons change wall-clock playback
interval independently of simulation time. The AP/diagnostics panel scrolls
with the mouse wheel, Page Up/Down, Home/End, the scrollbar arrows, or its
draggable thumb. End jumps to the last rows. The window is resizable, starts
within the desktop height, and adjusts the visible rows and text wrapping to
fit. Scrolling and resizing never advance the simulation.

Current labels are sorted. Green `+` labels became true at the last semantic
update. Removed labels appear separately and are not members of the current
label set. Diagnostics show episode-local role IDs, lane references, gap
clearances, required gaps, margins, TTC, projected times, acceleration, lateral
measurements, and false predicate values.

The environment advances three 15 Hz physics frames per 5 Hz policy step,
approximately 0.2 simulated seconds. Playback initially steps every 0.6 wall
seconds. Termination/truncation freezes stepping and retains the final frame.
Reset clears labels, history, pending actions, and playback. Repaint does not
advance predicate memory. Offscreen rendering prevents Highway Env from opening
another window or consuming this application's events.

Other vehicles use Highway Env's default autonomous controllers and their
original speeds, advancing on every simulation step. The default Follow camera
tracks ego and displays relative motion, so equal-speed cars can look stationary.
Use C or the Camera button to switch to a fixed World view, which recentres
when ego approaches the horizontal edge. All traffic remains frozen while paused.

## Grounded vocabulary and assumptions

The fixed `REGISTRY` in `mainV2.py` documents every supported grounding:

- Presence and braking for seven roles: lead, rear, left_front, left_rear,
  right_front, right_rear, intruder. Ego is an internal reference.
- Lane existence and ego front/rear gaps for left, right, passing_lane.
- Occupancy predicates for nine explicit combinations: lead/rear
  in passing_lane, side neighbours in their respective side lane, and intruder
  in left, right, passing_lane.
- Ahead-of for front roles and intruder; slower-than for all seven roles.
- Closing-on from ego to front roles/intruder and from rear roles to ego.
  Side-lane closing describes longitudinal catching even without shared occupancy.
- moving_into(intruder,ego_lane), danger_zone(ego,lead), crashed(ego).

This is a bounded registry of 53 APs. ego_lane is an internal destination token
used only for cut-in motion, outside the named lane set. Lead/rear are the nearest
vehicles ahead/behind in ego's current lane; side roles are the nearest vehicles
in those relative positions. Neighbours use occupied footprints, lane-frame
centre ordering, distance, and episode ID to break ties. The passing lane stays
pinned until reset or T. Adjacent lanes use Highway Env's left-to-right topology
numbering on the current road edge. No lane is invented at a road boundary.
Intruders must occupy an immediate adjacent lane and physically move toward
ego's current lane within 60 m. Priority is earliest lateral footprint arrival,
then nearest longitudinal centre, then episode ID.
Missing roles and lanes make their relevant predicates false.

Vehicles use oriented rectangular footprints. Occupancy can include two lanes.
Moving_into requires measured lateral speed
over 0.05 m/s progressing toward the lane centre and offset above 0.25 m.
Actions and target lanes alone never imply motion or completion.

Longitudinal gaps are projected bumper clearances. Gap requirements are
`5 + 1.2 * max(follower_speed, 0) + max(closing_speed, 0)` metres. Rear gaps use
the rear vehicle as follower. Existing empty lanes have infinite margins;
missing lanes have negative infinite margins. Every other vehicle is considered,
including alongside vehicles and entering footprints. A one-second constant
velocity horizon solves lateral occupancy intervals and checks longitudinal
edge crossings and interval endpoints, taking minimum margins. This prediction
is specific to highway-v0's straight lanes and ignores future steering,
acceleration and uncertainty. Projected overlap makes gaps inadequate.

Adequate gap turns on above +2 m, off below 0 m, and retains its value between.
An initial value in the band is false. Hysteresis belongs to the physical lane;
passing-lane rebinding clears passing-lane history. Acceleration is measured from consecutive
snapshots using simulation elapsed time, initially zero. A role occupant change
resets that role's acceleration to zero for its first sample. Speed/closing and
acceleration thresholds are 0.5 m/s and 0.5 m/s². TTC is zero for overlap and
infinite for separated non-closing vehicles. Closing_on requires a positive bumper gap, longitudinal following order,
and closing speed above the threshold. danger_zone(ego,lead) requires shared
occupancy and either overlapping longitudinal footprints or closing speed above
0.5 m/s with TTC strictly below Config.ttc_danger (default 2 seconds). Absent,
separated non-closing, and side-only vehicles do not trigger danger_zone. These are traffic-fact proxies,
not safety guarantees.
All calibration thresholds reside in `Config`.

## Verification

Automated tests cover the bounded registry and relationship directions,
traffic movement and World/Follow camera behaviour, empty/nonexistent lanes, fast rear approaches,
alongside vehicles, projected entry and overlap, boundary crossing occupancy, TTC, hysteresis, occupant changes, simulation-time acceleration,
paused redraws, AP scrolling and window resizing, step counts, reset/rebind,
truncation, event ownership and cleanup.
Pygame checks run under dummy SDL with offscreen drawing explicitly enabled.
Actual desktop appearance and physical keyboard/mouse interaction were not
verified in this run.
