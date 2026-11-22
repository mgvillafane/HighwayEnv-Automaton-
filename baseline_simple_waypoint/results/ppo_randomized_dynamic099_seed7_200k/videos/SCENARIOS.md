# Trained PPO episode videos

[Watch learned successes](successful_examples.mp4) | [Watch first scenario for every goal](all_goals.mp4)

These videos use the saved PPO policy after 200,000 additional transitions. No scripted controller is used.

The first held-out seed for each goal is shown regardless of outcome. Two additional examples show the first successful BACK and LEFT cases. Success rates come from all 100 evaluation episodes.

| Selection | Goal | Seed | Initial ego / traffic lane | Traffic speed | Outcome | Video |
| --- | --- | --- | --- | ---: | --- | --- |
| First seed | BACK | 10009 | 2 / 0 | 16.16 m/s | TIMED OUT | [Watch](back/scenario.mp4) |
| First seed | FRONT | 10002 | 2 / 1 | 14.02 m/s | TIMED OUT | [Watch](front/scenario.mp4) |
| First seed | LEFT | 10004 | 1 / 2 | 14.18 m/s | TIMED OUT | [Watch](left/scenario.mp4) |
| First seed | RIGHT | 10000 | 0 / 0 | 14.44 m/s | TIMED OUT | [Watch](right/scenario.mp4) |
| First success | BACK | 10016 | 2 / 1 | 13.29 m/s | GOAL REACHED | [Watch](success_back/scenario.mp4) |
| First success | LEFT | 10011 | 2 / 1 | 17.65 m/s | GOAL REACHED | [Watch](success_left/scenario.mp4) |
