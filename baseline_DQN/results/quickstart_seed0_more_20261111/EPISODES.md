# Twelve fresh DQN episodes

Model: `dqn_quickstart_seed0.zip`. Deterministic actions, original saved task settings, consecutive traffic seeds 30000-30011. Every outcome is retained.

6/12 collisions; 6/12 completed the 30-second time limit. This short sample is for viewing, not a precise collision-rate estimate.

[Watch all 12 episodes](all_12_episodes.mp4)

| Episode | Traffic seed | Outcome | Simulated seconds | Video |
| --- | --- | --- | ---: | --- |
| 1 | 30000 | Collision | 28 | [Watch](episode_01.mp4) |
| 2 | 30001 | Completed | 30 | [Watch](episode_02.mp4) |
| 3 | 30002 | Collision | 18 | [Watch](episode_03.mp4) |
| 4 | 30003 | Collision | 23 | [Watch](episode_04.mp4) |
| 5 | 30004 | Completed | 30 | [Watch](episode_05.mp4) |
| 6 | 30005 | Completed | 30 | [Watch](episode_06.mp4) |
| 7 | 30006 | Collision | 30 | [Watch](episode_07.mp4) |
| 8 | 30007 | Collision | 15 | [Watch](episode_08.mp4) |
| 9 | 30008 | Completed | 30 | [Watch](episode_09.mp4) |
| 10 | 30009 | Completed | 30 | [Watch](episode_10.mp4) |
| 11 | 30010 | Completed | 30 | [Watch](episode_11.mp4) |
| 12 | 30011 | Collision | 8 | [Watch](episode_12.mp4) |

The reel labels each episode and preserves the original 5 Hz simulator frames. Blue is ego; amber is surrounding traffic; magenta is an initial reference car. Episode reward sums positive native driving rewards and can remain high even when the episode eventually crashes.

[Collision diagnosis and fresh 50-episode results](../quickstart_seed0_diagnostics_20261111/DIAGNOSIS.md)
