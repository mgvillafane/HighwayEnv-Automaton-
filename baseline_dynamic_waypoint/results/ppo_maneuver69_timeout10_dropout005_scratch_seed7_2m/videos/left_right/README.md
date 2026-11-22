# LEFT and RIGHT: final 2M route-progress model

Final evaluation: LEFT 33/33 success; RIGHT 21/22 success (one timeout).

The first two evaluation seeds for each goal are included, plus the single RIGHT timeout.
Every replay is checked against its evaluation outcome, return, step count, and duration.
All simulation ticks are recorded at 10 fps, with a two-second final-frame hold.

[LEFT examples](left_examples.mp4) | [RIGHT examples](right_examples.mp4) | [Combined video](left_and_right.mp4)

| Goal | Seed | Initial / goal lane | Outcome | Seconds | Video |
| --- | ---: | --- | --- | ---: | --- |
| LEFT | 10004 | 1 / 1 | GOAL REACHED | 4.7 | [Watch](episode_01_seed10004_goal_reached.mp4) |
| LEFT | 10011 | 2 / 0 | GOAL REACHED | 5.9 | [Watch](episode_02_seed10011_goal_reached.mp4) |
| RIGHT | 10000 | 0 / 1 | GOAL REACHED | 4.6 | [Watch](episode_03_seed10000_goal_reached.mp4) |
| RIGHT | 10001 | 1 / 2 | GOAL REACHED | 4.4 | [Watch](episode_04_seed10001_goal_reached.mp4) |
| RIGHT | 10080 | 0 / 1 | TIMED OUT | 30.0 | [Watch](episode_05_seed10080_timed_out.mp4) |
