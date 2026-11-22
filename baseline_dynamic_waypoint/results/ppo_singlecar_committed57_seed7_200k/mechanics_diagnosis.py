"""Read-only checks of lane coordinates, sampling, encoder and steering symmetry."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baseline_dynamic_waypoint.single_car import make_single_car_env

HERE = Path(__file__).resolve().parent
metadata = json.loads((ROOT / 'baseline_dynamic_waypoint/models/ppo_singlecar_committed57_seed7_200k.json').read_text())
settings = metadata['environment']


def configured(seed, y, speed):
    env = make_single_car_env(**settings)
    env.reset(seed=seed, options={'slot': 'front'})
    ego, anchor = env.highway.vehicle, env.anchor
    ego.position[1] = y
    ego.heading = 0.0
    ego.speed = ego.target_speed = speed
    ego.speed_index = ego.speed_to_index(speed)
    ego.on_state_update()
    ego.target_lane_index = ego.lane_index
    anchor.position[1] = 4.0
    anchor.heading = 0.0
    anchor.speed = anchor.target_speed = 15.0
    anchor.on_state_update()
    anchor.target_lane_index = anchor.lane_index
    env._update_goal()
    env.previous_distance = env.distance_to_goal()
    env.previous_shaping_errors = env.shaping_errors()
    return env


pairs = []
for speed in (10, 12.5, 15, 17.5, 20, 22.5, 25):
    right, left = configured(10002, 0, speed), configured(10002, 8, speed)
    tracks = [[], []]
    for env, track in zip((right, left), tracks):
        env.physics_step_callback = lambda o, r, t, tr, i, e=env, x=track: x.append({
            'position': e.highway.vehicle.position.copy().tolist(), 'heading': e.highway.vehicle.heading,
            'reward': r, 'progress': i['reward_components']['progress']})
    _, rr, rt, rtr, ri = right.step(2)
    _, lr, lt, ltr, li = left.step(0)
    a, b = tracks
    pairs.append({'speed': speed, 'right_ticks': len(a), 'left_ticks': len(b),
        'right_complete': ri['lane_change_completed'], 'left_complete': li['lane_change_completed'],
        'right_reward': rr, 'left_reward': lr,
        'max_x_difference': max(abs(x['position'][0] - y['position'][0]) for x, y in zip(a, b)),
        'max_mirrored_y_difference': max(abs(x['position'][1] - (8-y['position'][1])) for x, y in zip(a, b)),
        'max_reward_difference': max(abs(x['reward'] - y['reward']) for x, y in zip(a, b)),
        'termination_equal': (rt, rtr) == (lt, ltr)})
    right.close()
    left.close()

env = make_single_car_env(**settings)
counts = Counter()
directions = {s: Counter() for s in ('front', 'back', 'left', 'right')}
ego_lanes = Counter()
traffic_lanes = Counter()
y_errors = []
for seed in range(20000, 22000):
    obs, info = env.reset(seed=seed)
    counts[env.slot] += 1
    ego_lanes[int(env.highway.vehicle.lane_index[2])] += 1
    traffic_lanes[int(env.anchor.lane_index[2])] += 1
    delta = env.goal_lane - env.highway.vehicle.lane_index[2]
    directions[env.slot]['right' if delta > 0 else 'left' if delta < 0 else 'same'] += 1
    dy = env.goal_position[1] - env.highway.vehicle.position[1]
    y_errors.append(abs(float(obs['goal_delta'][1]) - dy/12))
    y_errors.append(abs(float(obs['slot_error'][1]) - np.clip(dy/4, -1, 1)))
env.close()

slow_settings = deepcopy(settings)
slow_settings['env_config']['fast_observation'] = False
slow, fast = make_single_car_env(**slow_settings), make_single_car_env(**settings)
encoder_states = 0
for seed in range(30000, 30030):
    a, _ = slow.reset(seed=seed)
    b, _ = fast.reset(seed=seed)
    for _ in range(20):
        assert all(np.array_equal(a[k], b[k]) for k in a)
        encoder_states += 1
        action = int(np.random.default_rng(seed + encoder_states).integers(5))
        a, ar, at, atr, ai = slow.step(action)
        b, br, bt, btr, bi = fast.step(action)
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert (ar, at, atr, ai['internal_steps']) == (br, bt, btr, bi['internal_steps'])
        if at or atr:
            break
slow.close()
fast.close()

result = {'mirrored_controller_pairs': pairs, 'sampling_seeds': [20000, 21999],
    'goal_counts': counts, 'initial_required_direction': directions,
    'ego_lane_counts': ego_lanes, 'traffic_lane_counts': traffic_lanes,
    'maximum_y_feature_formula_error': max(y_errors),
    'fast_native_exact_equal_states': encoder_states,
    'notes': ['No model or reward modifications.',
              'slot_error lateral clips two-lane errors to the same magnitude as one-lane errors, but signed goal_delta preserves full difference.',
              'Absolute achieved/desired y have nonnegative values; relative goal_delta and slot_error are signed.',
              'Controller mirror setup: ego y=0 RIGHT versus y=8 LEFT, traffic y=4, identical x and velocities, FRONT goal.']}
(HERE / 'mechanics_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
