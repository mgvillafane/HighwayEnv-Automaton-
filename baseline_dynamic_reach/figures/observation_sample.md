Real observation at reset, evaluation seed 10,000: BACK slot, initial distance 53.7 m. 53 inputs.

| Input | Refers to | Value | Physical | Meaning and scale |
|---|---|---:|---:|---|
| `vehicles[0].presence` | ego | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[0].x` | ego | +0.000 | +0.0 m | ego progress since reset, / 200 m |
| `vehicles[0].y` | ego | +0.000 | +0.00 m | ego lateral position, / 12 m |
| `vehicles[0].vx` | ego | +0.667 | +20.00 m/s | ego forward speed, / 30 m/s |
| `vehicles[0].vy` | ego | +0.000 | +0.00 m/s | ego lateral speed, / 10 m/s |
| `vehicles[1].presence` | car 1 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[1].x` | car 1 | +0.175 | +35.00 m | x relative to ego, / 200 m |
| `vehicles[1].y` | car 1 | +0.000 | +0.00 m | y relative to ego, / 12 m |
| `vehicles[1].vx` | car 1 | -0.267 | -8.00 m/s | vx relative to ego, / 30 m/s |
| `vehicles[1].vy` | car 1 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `vehicles[2].presence` | car 2 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[2].x` | car 2 | +0.341 | +68.14 m | x relative to ego, / 200 m |
| `vehicles[2].y` | car 2 | +0.667 | +8.00 m | y relative to ego, / 12 m |
| `vehicles[2].vx` | car 2 | -0.254 | -7.63 m/s | vx relative to ego, / 30 m/s |
| `vehicles[2].vy` | car 2 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `vehicles[3].presence` | car 3 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[3].x` | car 3 | +0.499 | +99.89 m | x relative to ego, / 200 m |
| `vehicles[3].y` | car 3 | +0.667 | +8.00 m | y relative to ego, / 12 m |
| `vehicles[3].vx` | car 3 | -0.159 | -4.76 m/s | vx relative to ego, / 30 m/s |
| `vehicles[3].vy` | car 3 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `vehicles[4].presence` | car 4 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[4].x` | car 4 | +0.657 | +131.49 m | x relative to ego, / 200 m |
| `vehicles[4].y` | car 4 | +0.000 | +0.00 m | y relative to ego, / 12 m |
| `vehicles[4].vx` | car 4 | -0.188 | -5.63 m/s | vx relative to ego, / 30 m/s |
| `vehicles[4].vy` | car 4 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `vehicles[5].presence` | car 5 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[5].x` | car 5 | +0.815 | +162.94 m | x relative to ego, / 200 m |
| `vehicles[5].y` | car 5 | +0.000 | +0.00 m | y relative to ego, / 12 m |
| `vehicles[5].vx` | car 5 | -0.144 | -4.32 m/s | vx relative to ego, / 30 m/s |
| `vehicles[5].vy` | car 5 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `vehicles[6].presence` | car 6 | +1.000 | 1 | 1 if this row holds a car |
| `vehicles[6].x` | car 6 | +0.994 | +198.74 m | x relative to ego, / 200 m |
| `vehicles[6].y` | car 6 | +0.000 | +0.00 m | y relative to ego, / 12 m |
| `vehicles[6].vx` | car 6 | -0.223 | -6.69 m/s | vx relative to ego, / 30 m/s |
| `vehicles[6].vy` | car 6 | +0.000 | +0.00 m/s | vy relative to ego, / 10 m/s |
| `achieved_goal.x` | ego | +0.000 | +0.00 m | x position in the episode frame (origin: ego's start x, road y = 0), / 200 m |
| `achieved_goal.y` | ego | +0.000 | +0.00 m | y position in the episode frame (origin: ego's start x, road y = 0), / 12 m |
| `desired_goal.x` | slot | +0.266 | +53.14 m | x position in the episode frame (origin: ego's start x, road y = 0), / 200 m |
| `desired_goal.y` | slot | +0.667 | +8.00 m | y position in the episode frame (origin: ego's start x, road y = 0), / 12 m |
| `goal_delta.x` | slot minus ego | +0.266 | +53.14 m | x difference, / 200 m |
| `goal_delta.y` | slot minus ego | +0.667 | +8.00 m | y difference, / 12 m |
| `anchor.x` | anchor minus ego | +0.341 | +68.14 m | normalized like the Kinematics rows, clipped to [-1, 1] |
| `anchor.y` | anchor minus ego | +0.667 | +8.00 m | normalized like the Kinematics rows, clipped to [-1, 1] |
| `anchor.vx` | anchor minus ego | -0.254 | -7.63 m/s | normalized like the Kinematics rows, clipped to [-1, 1] |
| `anchor.vy` | anchor minus ego | +0.000 | +0.00 m/s | normalized like the Kinematics rows, clipped to [-1, 1] |
| `slot.front` | target slot | 0 |  | one-hot: which slot is the goal |
| `slot.back` | target slot | 1 |  | one-hot: which slot is the goal |
| `slot.left` | target slot | 0 |  | one-hot: which slot is the goal |
| `slot.right` | target slot | 0 |  | one-hot: which slot is the goal |
| `hold_progress` | ego | 0.000 |  | decisions held in the box / required; always 0 before success in reach |
| `slot_error.along` | local error | +1.000 | +10.00 m | slot minus ego along the slot lane, / 10 m, clipped to [-1, 1] (saturates beyond 10 m) |
| `slot_error.across` | local error | +1.000 | +4.00 m | slot lane center minus ego, across, / 4 m, clipped to [-1, 1] (saturates beyond 4 m) |
| `slot_error.speed` | local error | -0.763 | -7.63 m/s | anchor speed minus ego speed, / 10 m/s, clipped to [-1, 1] (saturates beyond 10 m/s) |
