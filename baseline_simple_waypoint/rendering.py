"""A clear two-car view with the same moving goal box as the original task."""
import pygame

from baseline_dynamic_waypoint.rendering import DynamicGoalViewer


class FollowingViewer(DynamicGoalViewer):
    def __init__(self, env, task):
        super().__init__(env, task)
        self.enabled = True  # SDL's dummy driver can still draw offscreen surfaces.

    def _draw_hud(self):
        task, surface = self.task, self.sim_surface
        color = (255, 95, 95) if self.env.vehicle.crashed else (100, 240, 155)
        hold = task.hold_steps / task.env_config["policy_frequency"]
        hold_required = task.goal_config["hold_seconds"]
        speed_tolerance = task.goal_config["speed_tolerance"]
        mode = getattr(task, "demo_label", "Moving goal task")
        along, across = task.slot_error()
        lines = [
            (f"{mode}  |  {task.slot.upper()}  |  {task.status}  |  Time {self.env.time:.1f}/{task.env_config['duration']:.0f} s", color),
            (f"Ego {self.env.vehicle.speed:.1f} m/s  |  Traffic {task.anchor.speed:.1f} m/s  |  "
             f"Slot error {along:+.1f} m along / {across:+.1f} m across  |  Hold {hold:.1f}/{hold_required:.1f} s", (230, 235, 245)),
            (f"BLUE: your car    AMBER / MAGENTA: traffic    GREEN: follow here    Match speed within {speed_tolerance:g} m/s", (175, 210, 235)),
        ]
        pygame.draw.rect(surface, (20, 27, 40), (0, 0, surface.get_width(), 85))
        for index, (text, text_color) in enumerate(lines):
            surface.blit(self.font.render(text, True, text_color), (14, 7 + index * 25))
