"""Verify observation reflection against reflected simulator states."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baseline_dynamic_waypoint.single_car import make_single_car_env

HERE = Path(__file__).resolve().parent
meta = json.loads((ROOT / 'baseline_dynamic_waypoint/models/ppo_singlecar_committed57_seed7_200k.json').read_text())


def reflect(obs):
    out = {k: v.copy() for k, v in obs.items()}
    for key in ('achieved_goal', 'desired_goal'):
        out[key][1] = 8/12 - out[key][1]
    for key in ('goal_delta', 'slot_error'):
        out[key][1] *= -1
    out['anchor'][[1, 3]] *= -1
    out['vehicles'][0, 2] = 8/12 - out['vehicles'][0, 2]
    out['vehicles'][1:, 2] *= -1
    out['vehicles'][:, 4] *= -1
    out['requested_lane'] = out['requested_lane'][::-1].copy()
    out['slot'][[2, 3]] = out['slot'][[3, 2]]
    return out


worst = {}
states = 0
for seed in range(10000, 10040):
    a, b = [make_single_car_env(**meta['environment']) for _ in range(2)]
    oa, _ = a.reset(seed=seed)
    b.reset(seed=seed)
    for car in b.highway.road.vehicles:
        car.position[1] = 8-car.position[1]
        car.heading = -car.heading
        old_target = car.target_lane_index
        car.on_state_update()
        car.target_lane_index = (*old_target[:2], 2-old_target[2])
    b.slot = {'left': 'right', 'right': 'left'}.get(a.slot, a.slot)
    b._update_goal()
    b.previous_distance = b.distance_to_goal()
    b.previous_shaping_errors = b.shaping_errors()
    ob = b._observation(b.highway.observation_type.observe())
    rng = np.random.default_rng(seed)
    for _ in range(10):
        reflected = reflect(oa)
        for key in oa:
            delta = float(np.max(np.abs(reflected[key]-ob[key])))
            worst[key] = max(worst.get(key, 0), delta)
            assert delta < 2e-6, (seed, key, delta)
        states += 1
        action = int(rng.integers(5))
        mirrored = {0: 2, 2: 0}.get(action, action)
        oa, ar, at, atr, ai = a.step(action)
        ob, br, bt, btr, bi = b.step(mirrored)
        assert abs(ar-br) < 1e-10, (seed, ar, br)
        assert (at, atr, ai['internal_steps']) == (bt, btr, bi['internal_steps'])
        if at or atr:
            break
    a.close()
    b.close()
result = {'mirrored_states': states, 'maximum_error_by_key': worst,
          'reward_and_termination_symmetry': True}
(HERE / 'mechanics_reflection.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
