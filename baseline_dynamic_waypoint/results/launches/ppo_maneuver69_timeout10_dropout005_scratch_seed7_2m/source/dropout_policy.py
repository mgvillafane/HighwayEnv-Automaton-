"""Dictionary-input PPO policy with dropout in both actor and critic MLPs."""
import math

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from torch import nn


class DropoutMultiInputPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, dropout_rate=0.1, **kwargs):
        if not math.isfinite(dropout_rate) or not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be finite and in [0, 1).")
        self.dropout_rate = float(dropout_rate)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        for network in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
            # SB3 alternates Linear and activation modules. Nest dropout with
            # each activation so all learned Linear parameter names stay stable.
            for index in range(1, len(network), 2):
                network[index] = nn.Sequential(network[index], nn.Dropout(self.dropout_rate))

    def _get_constructor_parameters(self):
        parameters = super()._get_constructor_parameters()
        parameters["dropout_rate"] = self.dropout_rate
        return parameters
