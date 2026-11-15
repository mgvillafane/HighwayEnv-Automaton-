"""The reach environment: the inherited DynamicGoalHighwayEnv plus optional reward terms.

Every term here is off unless an arm gives it a weight. With none of them weighted, ReachEnv
behaves exactly like DynamicGoalHighwayEnv (tested), so arms trained before a term existed
are unaffected by its addition.

Terms
-----
``potential_progress``: potential-based shaping toward the slot (Ng, Harada & Russell 1999).
    Configured as ``{"weight": w, "gamma": g}`` in ``reward_weights``. With the potential
    ``Phi(s) = -distance to the slot center (m)``, each decision pays

        F = w * (g * Phi(s') - Phi(s)),   with Phi(s') = 0 when the episode terminates.

    When ``g`` equals PPO's discount (train.py enforces it), the discounted sum of F over any
    terminated episode is exactly ``-w * Phi(s_0) = w * initial distance``: a constant that does
    not depend on what the policy does, so the best policy under the shaped reward is the best
    policy under the unshaped one. The term only changes *when* credit arrives. On a time-limit
    or anchor-lost truncation the episode is not over for the value function, which bootstraps,
    so Phi(s') keeps its real value there.

    The inherited ``progress`` term pays ``w * (d - d')``; it omits g, so its discounted sum
    depends on the path and it can change the best policy.
"""
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv

SHAPING_TERMS = ("potential_progress",)


class ReachEnv(DynamicGoalHighwayEnv):
    def _shaping(self, name):
        spec = self.reward_weights.get(name)
        return spec if isinstance(spec, dict) and spec.get("weight", 0.0) else None

    def potential(self):
        return -self.distance_to_goal()

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.previous_potential = self.potential()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        spec = self._shaping("potential_progress")
        if spec is None:
            return observation, reward, terminated, truncated, info
        next_potential = 0.0 if terminated else self.potential()
        shaping = spec["weight"] * (spec["gamma"] * next_potential - self.previous_potential)
        self.previous_potential = next_potential
        self.last_components["potential_progress"] = shaping
        self.episode_totals["potential_progress"] += shaping
        info["reward_components"] = dict(self.last_components)
        info["episode_reward_components"] = dict(self.episode_totals)
        return observation, reward + shaping, terminated, truncated, info
