"""NumPy implementation of the simulator's standard kinematics observation.

The native configuration, object selection, normalization arithmetic and RNG
calls are preserved. Extended feature schemas keep the native implementation.
This changes only array assembly, not the driving task or its observations.
"""
from types import MethodType

import numpy as np

from highway_env.envs.common.observation import KinematicObservation
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import RoadObject


class FastKinematicObservation(KinematicObservation):
    """Avoid DataFrames and unused vehicle fields for the five basic features."""

    def _record(self, obj, origin=None):
        # Unknown vehicle/object implementations retain their own to_dict behavior.
        if isinstance(obj, Vehicle) and type(obj).to_dict is Vehicle.to_dict:
            velocity = obj.velocity
            record = np.array([1.0, *obj.position, *velocity], dtype=np.float64)
        elif isinstance(obj, RoadObject) and type(obj).to_dict is RoadObject.to_dict:
            record = np.array([1.0, *obj.position, 0.0, 0.0], dtype=np.float64)
        else:
            values = obj.to_dict() if origin is None else obj.to_dict(
                origin, observe_intentions=self.observe_intentions)
            return np.array([values[name] for name in self.FEATURES], dtype=np.float64)
        if origin is not None:
            record[1:] -= self._record(origin)[1:]
        return record

    def observe(self):
        if (self.vehicles_count < 2 or len(set(self.features)) != len(self.features)
                or not set(self.features).issubset(self.FEATURES)):
            return super().observe()
        if not self.env.road:
            return np.zeros(self.space().shape)

        ego = self.observer_vehicle
        close = self.env.road.close_objects_to(
            ego, self.env.PERCEPTION_DISTANCE, count=self.vehicles_count - 1,
            see_behind=self.see_behind, sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles)
        origin = None if self.absolute else ego
        records = [self._record(ego)]
        records.extend(self._record(obj, origin)
                       for obj in close[-self.vehicles_count + 1:])
        columns = [self.FEATURES.index(name) for name in self.features]
        values = np.asarray(records, dtype=np.float64)[:, columns]

        if self.normalize:
            if not self.features_range:
                lanes = self.env.road.network.all_side_lanes(ego.lane_index)
                self.features_range = {
                    "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                    "y": [-AbstractLane.DEFAULT_WIDTH * len(lanes),
                          AbstractLane.DEFAULT_WIDTH * len(lanes)],
                    "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                    "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                }
            for name, limits in self.features_range.items():
                if name in self.features:
                    column = self.features.index(name)
                    # Keep utils.lmap's operation order exactly, in float64.
                    values[:, column] = -1 + (values[:, column] - limits[0]) * 2 / (limits[1] - limits[0])
                    if self.clip:
                        np.clip(values[:, column], -1, 1, out=values[:, column])

        observation = np.zeros(self.space().shape, dtype=np.float64)
        observation[:len(values)] = values
        if self.order == "shuffled":
            self.env.np_random.shuffle(observation[1:])
        return observation.astype(self.space().dtype)


def _upgrade_observation(highway):
    if type(highway.observation_type) is KinematicObservation:
        # The subclass adds no instance state; keep the native options and observer.
        highway.observation_type.__class__ = FastKinematicObservation


def _define_fast_spaces(highway):
    highway._dqn_native_define_spaces()
    _upgrade_observation(highway)


def enable_fast_observation(env):
    """Enable equivalent observation assembly on this environment only.

    highway-env recreates observation_type during every reset. Hook the instance's
    define_spaces method so subsequent resets retain the optimization without
    changing native configuration or patching installed packages. Return env.
    """
    highway = env.unwrapped
    if not isinstance(highway.observation_type, KinematicObservation):
        raise ValueError("Fast observations require a native Kinematics observation.")
    if not hasattr(highway, "_dqn_native_define_spaces"):
        highway._dqn_native_define_spaces = highway.define_spaces
        highway.define_spaces = MethodType(_define_fast_spaces, highway)
    _upgrade_observation(highway)
    return env
