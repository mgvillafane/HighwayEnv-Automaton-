# DQN collision diagnosis and more episodes

The default saved policy (`dqn_quickstart_seed0.zip`) crashed in 22/50 fresh episodes (44%) on traffic seeds 30000-30049. It completed 28/50 (56%). Evaluation used deterministic actions and the saved task configuration, with no training or policy changes.

The earlier evaluation had 19/50 collisions (38%) for this model. Across the five independently trained quickstart models, rates were 38%, 38%, 36%, 20%, and 54%, averaging 37.2%. These rates concern whole episodes lasting up to 30 seconds. The same original 50 traffic seeds were shared across models, so their results are correlated.

## What the new traces show

- All 22/22 collision partners were included in the policy observation immediately before impact; all 22/22 were ahead in the same lane at that decision. Missing rear traffic does not explain these particular crashes.
- At the final decision, the gap divided by the closing speed was 0.01-0.79 seconds, assuming the relative speed stayed constant. Every case had less than one second of remaining clearance by this approximation, while decisions occur once per second.
- Fifteen final actions were `SLOWER`, six were `LANE_LEFT`, and one was `LANE_RIGHT`. Braking is often too late or interrupted by renewed acceleration. A `SLOWER` action lowers a speed target by one discrete level; it is not an emergency-brake command. The installed simulator has ten target speeds spaced between 10 and 30 m/s, so a step lowers the target by about 2.22 m/s.
- In 3/22 cases the final action attempted to steer beyond the outermost lane. The simulator clips the lane target, so the command leaves the vehicle following the same lane at its previous speed target. Several earlier decisions repeat these ineffective commands.

Concrete examples from the recordings:

- [Episode 1, seed 30000](../quickstart_seed0_more_20261111/episode_01.mp4): at 23-27 seconds the agent repeatedly chooses `LANE_LEFT` while already in the leftmost lane. Its gap shrinks from 27.17 m to 4.35 m while it keeps roughly 25.6 m/s behind a car traveling around 19.9 m/s. It crashes at 28 seconds.
- [Episode 8, seed 30007](../quickstart_seed0_more_20261111/episode_08.mp4): braking begins at 12 seconds with a 20.72 m gap and about 10.32 m/s closing speed. Three successive `SLOWER` decisions cannot remove the closing speed in time; it crashes at 15 seconds.
- [Episode 12, seed 30011](../quickstart_seed0_more_20261111/episode_12.mp4): it accelerates again at 6 seconds with only 10.63 m clearance to a slower lead car, then tries to change lanes at 7 seconds with only 1.36 m clearance. It crashes at 8 seconds.

## Why the training recipe can learn this behavior

The model had 20,000 training transitions. Its discount is gamma=0.8 at one decision per second: a reward ten seconds away receives only 0.8^10 = 10.7% of its immediate weight. The native reward favors speed and right lanes; it has no explicit safe-headway or time-to-collision term. Normalization maps the collision-weighted reward to a small terminal reward rather than applying a large negative terminal cost. Ending the episode still forfeits future driving rewards.

These settings are plausible contributors to the observed short-sighted choices. The traces establish late or ineffective reactions; they do not establish that changing one hyperparameter alone will fix them. The policy sees ego plus only four nearby cars and excludes most rear traffic, which can limit general safety but was not the immediate issue in these 22 crashes. There is no action mask that stops outward lane commands at road boundaries.

The plotted training return also includes exploratory actions and measures reward, not collision avoidance. In the new recordings, episode 1 crashes with return 23.43, while episode 6 survives with return 23.22; a high episode return alone does not establish safe driving.

## Recordings and data

- [Watch all 12 labeled episodes](../quickstart_seed0_more_20261111/all_12_episodes.mp4). These are consecutive traffic seeds 30000-30011, retained without outcome selection: six crashes and six completed episodes.
- [Individual episode index](../quickstart_seed0_more_20261111/EPISODES.md).
- [Fresh 50-episode summary](evaluation.json), [episode rows](evaluation_episodes.csv), [all decisions](decision_traces.json), and [last six decisions for every crash](crash_traces.json).
- All 12 recordings match the unrendered evaluation exactly in outcome, duration and reward.

Record the episodes again with a new output name:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --headless --record --episodes 12 --seed 30000 --output-name my_more_episodes
```

Run the standard paired DQN-versus-random evaluation on the same 50 fresh traffic seeds with a new output name:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/evaluate.py --episodes 50 --seed 30000 --output-name my_fresh_evaluation
```

Source evidence: saved model configuration; `baseline_DQN/evaluate.py` for deterministic playback; installed `highway_env/envs/highway_env.py` for reward and termination; installed `highway_env/vehicle/controller.py` for speed targets and lane clipping. Original five-seed results are in `../quickstart_comparison/REPORT.md`.
