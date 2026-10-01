import io
import logging
import math
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "20,40")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
output_sink = io.StringIO()
with redirect_stdout(output_sink), redirect_stderr(output_sink):
    import gymnasium as gym
    import highway_env


ENV_CONFIG = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "vehicles_count": 10,
    "controlled_vehicles": 1,
    "vehicles_density": 0.7,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 120,
}

FRONT_DISTANCE = 15.0
REAR_DISTANCE = 15.0
SPEED_EPSILON = 0.5
ACCELERATION_EPSILON = 0.2
STEP_DELAY = 0.5


def _lane_number(vehicle):
    lane_index = vehicle.lane_index
    return int(lane_index[-1] if isinstance(lane_index, tuple) else lane_index)


def _adjacent_lane_numbers(env, ego):
    lane_index = ego.lane_index
    if not isinstance(lane_index, tuple):
        return {}

    ego_lane = env.road.network.get_lane(lane_index)
    longitudinal, _ = ego_lane.local_coordinates(ego.position)
    ego_center = ego_lane.position(longitudinal, 0)
    heading = ego_lane.heading_at(longitudinal)
    left_normal = (math.sin(heading), -math.cos(heading))
    ego_lane_number = _lane_number(ego)
    adjacent_lanes = {}

    for lane_number in (ego_lane_number - 1, ego_lane_number + 1):
        try:
            candidate = env.road.network.get_lane((lane_index[0], lane_index[1], lane_number))
        except (IndexError, KeyError, TypeError, ValueError):
            continue

        candidate_center = candidate.position(longitudinal, 0)
        lateral_offset = (
            (candidate_center[0] - ego_center[0]) * left_normal[0]
            + (candidate_center[1] - ego_center[1]) * left_normal[1]
        )
        if lateral_offset > 0:
            adjacent_lanes["Left"] = lane_number
        elif lateral_offset < 0:
            adjacent_lanes["Right"] = lane_number

    return adjacent_lanes


def _vehicle_acceleration(vehicle, previous_speeds, step_duration):
    previous_speed = previous_speeds.get(id(vehicle))
    if previous_speed is not None:
        return (vehicle.speed - previous_speed) / step_duration

    for attribute in ("acceleration", "acc"):
        value = getattr(vehicle, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)

    action = getattr(vehicle, "action", None)
    if isinstance(action, dict):
        value = action.get("acceleration")
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def get_alphabet(env):
    """Return the currently true traffic symbols for the ego vehicle."""
    highway = env.unwrapped
    ego = highway.vehicle
    vehicles = highway.road.vehicles
    previous_speeds = getattr(highway, "_alphabet_previous_speeds", {})
    step_duration = 1.0 / highway.config["policy_frequency"]
    active_symbols = set()

    ego_lane = _lane_number(ego)
    adjacent_lanes = _adjacent_lane_numbers(highway, ego)
    ego_x = ego.position[0]
    ego_speed = ego.speed
    lane_vehicles = {-1: [], 0: [], 1: []}
    lane_numbers = {
        -1: adjacent_lanes.get("Left"),
        0: ego_lane,
        1: adjacent_lanes.get("Right"),
    }

    for vehicle in vehicles:
        if vehicle is ego:
            continue
        vehicle_lane_index = vehicle.lane_index
        if not isinstance(vehicle_lane_index, tuple) or vehicle_lane_index[:2] != ego.lane_index[:2]:
            continue
        vehicle_lane_number = _lane_number(vehicle)
        lane_offset = next(
            (offset for offset, lane_number in lane_numbers.items() if lane_number == vehicle_lane_number),
            None,
        )
        if lane_offset is not None:
            delta_x = vehicle.position[0] - ego_x
            acceleration = _vehicle_acceleration(vehicle, previous_speeds, step_duration)
            lane_vehicles[lane_offset].append((delta_x, vehicle, acceleration))

    def nearest_front(offset):
        candidates = [entry for entry in lane_vehicles[offset] if entry[0] > 0]
        return min(candidates, key=lambda entry: entry[0]) if candidates else None

    def nearest_rear(offset):
        candidates = [entry for entry in lane_vehicles[offset] if entry[0] < 0]
        return max(candidates, key=lambda entry: entry[0]) if candidates else None

    front = nearest_front(0)
    if front is None or front[0] >= FRONT_DISTANCE:
        active_symbols.add("FrontAvailable")
    if front is not None:
        _, front_vehicle, front_acceleration = front
        if 0 < front[0] < FRONT_DISTANCE:
            active_symbols.add("FrontClose")
        if ego_speed - front_vehicle.speed > SPEED_EPSILON:
            active_symbols.add("FrontClosing")
        if front_acceleration < -ACCELERATION_EPSILON:
            active_symbols.add("FrontBraking")
        if front_acceleration > ACCELERATION_EPSILON:
            active_symbols.add("FrontAccelerating")

    for side, offset in (("Left", -1), ("Right", 1)):
        if side not in adjacent_lanes:
            continue

        side_front = nearest_front(offset)
        side_rear = nearest_rear(offset)
        front_close = side_front is not None and 0 < side_front[0] < FRONT_DISTANCE
        rear_close = side_rear is not None and -REAR_DISTANCE < side_rear[0] < 0

        if not front_close and not rear_close:
            active_symbols.add(f"{side}LaneAvailable")

        if side_rear is not None:
            _, rear_vehicle, rear_acceleration = side_rear
            if rear_close:
                active_symbols.add(f"{side}RearClose")
            if rear_vehicle.speed - ego_speed > SPEED_EPSILON:
                active_symbols.add(f"{side}RearClosing")
            if rear_acceleration < -ACCELERATION_EPSILON:
                active_symbols.add(f"{side}RearBraking")
            if rear_acceleration > ACCELERATION_EPSILON:
                active_symbols.add(f"{side}RearAccelerating")

        if side_front is not None:
            _, front_vehicle, front_acceleration = side_front
            if front_close:
                active_symbols.add(f"{side}FrontClose")
            if ego_speed - front_vehicle.speed > SPEED_EPSILON:
                active_symbols.add(f"{side}FrontClosing")
            if front_acceleration < -ACCELERATION_EPSILON:
                active_symbols.add(f"{side}FrontBraking")
            if front_acceleration > ACCELERATION_EPSILON:
                active_symbols.add(f"{side}FrontAccelerating")

    highway._alphabet_previous_speeds = {id(vehicle): vehicle.speed for vehicle in vehicles}
    return active_symbols


def _step_environment(env, episode_state):
    if episode_state["done"]:
        env.reset()
        env.unwrapped._alphabet_previous_speeds = {}
        episode_state["done"] = False

    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    env.render()
    episode_state["done"] = terminated or truncated
    return get_alphabet(env)


def main():
    env = None
    root = None
    episode_state = {"done": False}
    auto_running = False
    auto_after_id = None

    try:
        with redirect_stdout(output_sink), redirect_stderr(output_sink):
            env = gym.make("highway-v0", config=ENV_CONFIG, render_mode="human")
            env.reset()
            env.render()

        import tkinter as tk
        from tkinter import scrolledtext, ttk

        root = tk.Tk()
        root.title("Highway Alphabet")
        root.resizable(True, True)
        root.minsize(340, 320)

        panel_width = 360
        highway_width = int(env.unwrapped.config.get("screen_width", 600))
        screen_width = root.winfo_screenwidth()
        panel_x = min(20 + highway_width + 16, max(0, screen_width - panel_width))
        root.geometry(f"{panel_width}x380+{panel_x}+40")

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))

        content = ttk.Frame(root, padding=16)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Traffic alphabet", style="Title.TLabel").pack(anchor="w")

        status = tk.StringVar(value="Ready")
        ttk.Label(content, textvariable=status, style="Status.TLabel").pack(anchor="w", pady=(4, 10))
        ttk.Label(content, text="CURRENT SYMBOLS").pack(anchor="w", pady=(0, 5))

        alphabet_view = scrolledtext.ScrolledText(
            content,
            height=12,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            padx=9,
            pady=8,
        )
        alphabet_view.pack(fill="both", expand=True)

        controls = ttk.Frame(content)
        controls.pack(fill="x", pady=(12, 0))

        def show_alphabet(symbols):
            alphabet_view.configure(state="normal")
            alphabet_view.delete("1.0", "end")
            alphabet_view.insert("1.0", f"{{{', '.join(sorted(symbols))}}}")
            alphabet_view.configure(state="disabled")

        def update_controls():
            step_button.configure(state="disabled" if auto_running else "normal")
            auto_button.configure(state="disabled" if auto_running else "normal")
            stop_button.configure(state="normal" if auto_running else "disabled")
            status.set("Auto stepping" if auto_running else "Paused")

        def stop_auto():
            nonlocal auto_running, auto_after_id
            auto_running = False
            if auto_after_id is not None:
                root.after_cancel(auto_after_id)
                auto_after_id = None
            if root.winfo_exists():
                update_controls()

        def step_once():
            try:
                show_alphabet(_step_environment(env, episode_state))
                return True
            except Exception as error:
                stop_auto()
                status.set(f"Stopped: {error}")
                return False

        def manual_step():
            if not auto_running and step_once():
                status.set("Paused")

        def automatic_step():
            nonlocal auto_after_id
            auto_after_id = None
            if auto_running and step_once() and auto_running:
                auto_after_id = root.after(int(STEP_DELAY * 1000), automatic_step)

        def start_auto():
            nonlocal auto_running
            if not auto_running:
                auto_running = True
                update_controls()
                automatic_step()

        step_button = ttk.Button(controls, text="Step", command=manual_step)
        auto_button = ttk.Button(controls, text="Auto", command=start_auto)
        stop_button = ttk.Button(controls, text="Stop", command=stop_auto, state="disabled")
        step_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        auto_button.pack(side="left", fill="x", expand=True, padx=5)
        stop_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        show_alphabet(get_alphabet(env))

        def close_windows():
            stop_auto()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close_windows)
        root.mainloop()
    finally:
        if root is not None:
            try:
                if root.winfo_exists():
                    root.destroy()
            except Exception:
                pass
        if env is not None:
            with redirect_stdout(output_sink), redirect_stderr(output_sink):
                env.close()


if __name__ == "__main__":
    main()