# dqn_quickstart_gamma095_seed2_200k_fast

Final DQN model, gamma 0.95, 200,000 training steps.

Full saved evaluation: **13/50 collisions (26%)**.

[Watch all eight episodes](all_episodes.mp4) | [Watch collision replays](collisions_only.mp4)

Collision replays show the last six seconds at half speed and hold the terminal frame for two seconds.

These examples deliberately include four crashes. Use the full 50-episode evaluation above for the collision rate.

Every recording reproduced its saved evaluation outcome, reward, duration and policy-step count.

| Episode | Seed | Slot | Outcome | Simulated seconds | Video | Collision replay |
| --- | --- | --- | --- | ---: | --- | --- |
| 1 | 10000 | - | COMPLETED | 30.0 | [Watch](episode_01_seed10000_completed.mp4) |  |
| 2 | 10001 | - | COMPLETED | 30.0 | [Watch](episode_02_seed10001_completed.mp4) |  |
| 3 | 10002 | - | COMPLETED | 30.0 | [Watch](episode_03_seed10002_completed.mp4) |  |
| 4 | 10003 | - | COLLISION | 10.0 | [Watch](episode_04_seed10003_collision.mp4) | [Half speed](collision_04_seed10003_slow.mp4) |
| 5 | 10004 | - | COMPLETED | 30.0 | [Watch](episode_05_seed10004_completed.mp4) |  |
| 6 | 10005 | - | COLLISION | 15.0 | [Watch](episode_06_seed10005_collision.mp4) | [Half speed](collision_06_seed10005_slow.mp4) |
| 7 | 10022 | - | COLLISION | 21.0 | [Watch](episode_07_seed10022_collision.mp4) | [Half speed](collision_07_seed10022_slow.mp4) |
| 8 | 10049 | - | COLLISION | 9.0 | [Watch](episode_08_seed10049_collision.mp4) | [Half speed](collision_08_seed10049_slow.mp4) |

![Collision final frames](collision_contact_sheet.jpg)

[All evaluation collisions and seeds](collisions.csv) | [Recording metadata](episodes.json)
