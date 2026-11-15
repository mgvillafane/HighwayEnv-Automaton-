# RIGHT and FRONT videos at 1,450,000 saved training steps

Training was stopped at the user's request. These recordings use the latest saved
checkpoint with direct deterministic PPO actions. Each video includes every
0.1-second simulation tick and a two-second hold of the final frame.

The first two evaluation seeds for each goal were used regardless of outcome.

[RIGHT examples](right_examples.mp4) | [FRONT examples](front_examples.mp4) | [Combined video](right_and_front.mp4)

| Goal | Seed | Outcome | Simulated seconds | Video |
| --- | ---: | --- | ---: | --- |
| RIGHT | 10000 | TIMED OUT | 30.0 | [Watch](episode_01_seed10000_timed_out.mp4) |
| RIGHT | 10001 | TIMED OUT | 30.0 | [Watch](episode_02_seed10001_timed_out.mp4) |
| FRONT | 10002 | TIMED OUT | 30.0 | [Watch](episode_03_seed10002_timed_out.mp4) |
| FRONT | 10003 | COLLISION | 5.5 | [Watch](episode_04_seed10003_collision.mp4) |
