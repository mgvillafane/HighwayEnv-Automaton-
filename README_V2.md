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
| T | Rebind origin lane, passing lane, and task target |
| D | Toggle diagnostics |
| Escape | Close |

Driving commands replace the pending command and execute once at the next
step, then revert to IDLE. Key repeat is disabled. IDLE maintains existing
speed and lane targets; it does not stop the vehicle. Unavailable action
buttons are marked N/A. Faster/Slower play buttons change wall-clock playback
interval independently of simulation time. Mouse wheel scrolls the right panel.

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

## Grounded vocabulary and assumptions

The fixed `REGISTRY` in `mainV2.py` documents every supported grounding:

- Presence and measured acceleration predicates for all nine roles.
- Lane existence and ego front/rear gap predicates for all five lane references.
- Occupancy and settled predicates for ego and task_target in all five lanes,
  plus intruder in current.
- Both ego/other-role orderings for ahead_of, slower_than, closing_on, and
  gap_critical (including both task_target orderings).
- Moving_into for ego in all five references and intruder in current; crashed ego.

Dynamic neighbours use occupied footprints, lane-frame centre ordering, distance,
and episode ID to break ties. Task references stay pinned until reset or T.
Adjacent lanes use Highway Env's left-to-right topology numbering on the current
road edge. No adjacent lane is invented at a road boundary. Intruder candidates
must physically move toward the current lane within 60 m. Priority is earliest
lateral footprint arrival, then nearest longitudinal centre, then episode ID.
Missing roles and lanes make their relevant predicates false.

Vehicles use oriented rectangular footprints. Occupancy can include two lanes;
settled requires lateral offset <= 0.25 m, lateral speed <= 0.2 m/s, heading
error <= 0.05 rad, and occupancy. Moving_into requires measured lateral speed
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
task rebinding clears task-lane history. Acceleration is measured from consecutive
snapshots using simulation elapsed time, initially zero. A role occupant change
resets that role's acceleration to zero for its first sample. Speed/closing and
acceleration thresholds are 0.5 m/s and 0.5 m/s². TTC is zero for overlap and
infinite for separated non-closing vehicles. Critical TTC is below two seconds
only when the ordered pair shares occupied lane geometry and follows in
longitudinal centre order. These are traffic-fact proxies, not safety guarantees.
All calibration thresholds reside in `Config`.

## Verification

Eight automated tests cover empty/nonexistent lanes, fast rear approaches,
alongside vehicles, projected entry and overlap, boundary crossing versus
settling, TTC, hysteresis, occupant changes, simulation-time acceleration,
paused redraws, step counts, reset/rebind, truncation, event ownership and cleanup.
Pygame checks run under dummy SDL with offscreen drawing explicitly enabled.
Actual desktop appearance and physical keyboard/mouse interaction were not
verified in this run.
