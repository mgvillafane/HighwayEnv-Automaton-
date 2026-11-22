# Legacy 57-input observation, timeout -10, dropout 0.1

Restored the committed-lane observation used by the earlier BACK/LEFT checkpoint. Its learned weights were copied exactly. Adding actor/critic dropout 0.1 reset the optimizer; deterministic evaluation disables dropout and reproduces the source checkpoint's original outcomes.

Completed exactly **200,000 additional PPO decisions** in 4.72 minutes. All four goals were enabled. Timeout costs -10 once; other baseline reward weights and goal conditions are retained.

| Goal | Evaluation cases | Success before | Success after | Collisions after |
| --- | ---: | ---: | ---: | ---: |
| FRONT | 18 | 0.0% | 0.0% | 33.3% |
| BACK | 27 | 81.5% | 59.3% | 0.0% |
| LEFT | 33 | 100.0% | 69.7% | 15.2% |
| RIGHT | 22 | 0.0% | 0.0% | 0.0% |

Overall success: 55.0% -> 39.0%. Collisions: 7.0% -> 11.0%. Final timeouts: 50.0%.

Both evaluations use the same 100 held-out seeds (10000-10099). This experiment changes both observation layout and initialization relative to the recent 54-input runs; comparisons across those runs do not isolate the observation's effect.

[TensorBoard](http://127.0.0.1:6014/live) | [All-goal video](videos/all_goals.mp4) | [Model](../../models/ppo_legacy57_timeout10_dropout01_seed7_200k.zip) | [Configuration](run.json) | [Verification](verification.json) | [Training curves](training_curves.png)

Videos use the first evaluation seed for each goal; outcomes, rewards, lengths and frame counts were verified.
