# Committed-lane trained PPO episodes

[Watch all goals](all_goals.mp4)

First held-out seed per goal, independent of outcome. Every internal physics frame is recorded at 10 fps. Each lane-change action keeps its target until completion or a terminal event.

| Goal | Seed | Outcome | PPO decisions | Physics ticks | Video |
| --- | --- | --- | ---: | ---: | --- |
| BACK | 10009 | GOAL REACHED | 14 | 50 | [Watch](back/episode.mp4) |
| FRONT | 10002 | TIMED OUT | 262 | 300 | [Watch](front/episode.mp4) |
| LEFT | 10004 | GOAL REACHED | 67 | 67 | [Watch](left/episode.mp4) |
| RIGHT | 10000 | TIMED OUT | 300 | 300 | [Watch](right/episode.mp4) |
