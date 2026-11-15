"""Transfer PPO input weights by field semantics; new inputs start at zero."""
import numpy as np
import torch
from stable_baselines3 import PPO


def transfer_observation_weights(source, target):
    old_keys = list(source.policy.features_extractor.extractors)
    new_keys = list(target.policy.features_extractor.extractors)
    old_columns, new_columns = {}, {}
    for keys, space, offsets in ((old_keys, source.observation_space, old_columns),
                                 (new_keys, target.observation_space, new_columns)):
        start = 0
        for key in keys:
            size = int(np.prod(space[key].shape))
            offsets[key] = slice(start, start + size)
            start += size
    old_state = source.policy.state_dict()
    new_state = target.policy.state_dict()
    lane_goal_transfer = (source.observation_space["desired_goal"].shape == (2,)
                          and target.observation_space["desired_goal"].shape == (3,)
                          and "lane_difference" in new_columns)
    column_pairs = []
    dropped = []
    for key in old_keys:
        old_shape = source.observation_space[key].shape
        new_shape = target.observation_space[key].shape if key in new_columns else None
        if lane_goal_transfer and (key not in new_columns or key == "desired_goal"):
            dropped.append(key)
            continue
        if (key == "vehicles" and lane_goal_transfer and new_shape is not None
                and len(new_shape) == 2 and old_shape == (new_shape[0], 5) and new_shape[1] == 6):
            for row in range(old_shape[0]):
                column_pairs.extend((old_columns[key].start + row * 5 + col,
                                     new_columns[key].start + row * 6 + col) for col in range(5))
        elif old_shape == new_shape:
            column_pairs.extend(zip(range(old_columns[key].start, old_columns[key].stop),
                                    range(new_columns[key].start, new_columns[key].stop)))
        else:
            raise ValueError(f"Incompatible source observation: {key}.")
    expanded = []
    for name, destination in new_state.items():
        original = old_state[name]
        if name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
            destination.zero_()
            for old_column, new_column in column_pairs:
                destination[:, new_column].copy_(original[:, old_column])
            expanded.append(name)
        elif destination.shape == original.shape:
            destination.copy_(original)
        else:
            raise ValueError(f"Cannot transfer parameter {name}: {original.shape} -> {destination.shape}.")
    target.policy.load_state_dict(new_state)
    return {"kind": "observation_schema_weight_transfer" if lane_goal_transfer else "expanded_observation_weight_transfer", "expanded_layers": expanded,
            "old_features": source.policy.features_extractor.features_dim,
            "new_features": target.policy.features_extractor.features_dim,
            "new_input_connections": "zero initialized", "optimizer_reset": True,
            "dropped_or_redefined_inputs": dropped,
            "policy_outputs_preserved": not dropped}
