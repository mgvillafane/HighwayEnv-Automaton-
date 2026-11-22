# New experiment episode videos

Videos use the final trained checkpoints and the original saved evaluation settings.

Each run has eight full episodes, including four selected collisions. Collision reels replay the last six seconds at half speed. Collision rates below use all 50 saved evaluation episodes, with paired seeds 10000-10049 within each task.

DQN completion means surviving the 30-second episode. PPO success means reaching and holding the moving slot. The two algorithms have different tasks, so their rates are not a controlled algorithm comparison.

| Experiment | Training steps | Evaluation collisions | All episodes | Collision replays | Episode list |
| --- | ---: | ---: | --- | --- | --- |
| dqn_quickstart_gamma09_seed2_200k | 200,000 | 5/50 (10%) | [Watch](baseline_DQN/results/dqn_quickstart_gamma09_seed2_200k/videos/all_episodes.mp4) | [Watch](baseline_DQN/results/dqn_quickstart_gamma09_seed2_200k/videos/collisions_only.mp4) | [Details](baseline_DQN/results/dqn_quickstart_gamma09_seed2_200k/videos/EPISODES.md) |
| dqn_quickstart_gamma095_seed2_200k_fast | 200,000 | 13/50 (26%) | [Watch](baseline_DQN/results/dqn_quickstart_gamma095_seed2_200k_fast/videos/all_episodes.mp4) | [Watch](baseline_DQN/results/dqn_quickstart_gamma095_seed2_200k_fast/videos/collisions_only.mp4) | [Details](baseline_DQN/results/dqn_quickstart_gamma095_seed2_200k_fast/videos/EPISODES.md) |
| ppo_front20m_current_seed7_1m | 1,000,448 | 27/50 (54%) | [Watch](baseline_dynamic_waypoint/results/ppo_front20m_current_seed7_1m/videos/all_episodes.mp4) | [Watch](baseline_dynamic_waypoint/results/ppo_front20m_current_seed7_1m/videos/collisions_only.mp4) | [Details](baseline_dynamic_waypoint/results/ppo_front20m_current_seed7_1m/videos/EPISODES.md) |
| ppo_front20m_gamma099_seed7_1m | 1,000,448 | 21/50 (42%) | [Watch](baseline_dynamic_waypoint/results/ppo_front20m_gamma099_seed7_1m/videos/all_episodes.mp4) | [Watch](baseline_dynamic_waypoint/results/ppo_front20m_gamma099_seed7_1m/videos/collisions_only.mp4) | [Details](baseline_dynamic_waypoint/results/ppo_front20m_gamma099_seed7_1m/videos/EPISODES.md) |
