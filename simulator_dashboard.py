"""Responsive dashboard with each car's currently enabled DFA transitions."""
from functools import lru_cache

import numpy as np

from task_dfa import transition


BG = (12, 18, 29)
CARD = (21, 30, 44)
BORDER = (43, 57, 75)
TEXT = (225, 234, 245)
MUTED = (146, 163, 184)
ACCENT = (76, 224, 167)


@lru_cache(maxsize=2048)
def _render_text(font, text, color):
    return font.render(text, True, color)


@lru_cache(maxsize=2048)
def _fit_text(font, text, max_width):
    if max_width is not None and font.size(text)[0] > max_width:
        while text and font.size(text + "…")[0] > max_width:
            text = text[:-1]
        return text + "…" if text else ""
    return text


class Dashboard:
    def update_layout(self):
        pg = self.pg
        width, height = self.screen.get_size()
        margin, gap = 14, 12
        full_width = width - 2 * margin
        band_height = max(224, min(260, int(height * .31)))
        self.dfa_top = height - margin - band_height
        card_width = (full_width - 3 * gap) // 4
        self.dfa_rects = [pg.Rect(margin + i * (card_width + gap), self.dfa_top + 27,
                                  card_width, band_height - 27) for i in range(4)]
        top_height = self.dfa_top - gap - 60
        inspector_width = max(290, min(380, int(full_width * .31)))
        self.left_width = full_width - inspector_width - gap
        self.inspector_rect = pg.Rect(margin + self.left_width + gap, 60, inspector_width, top_height)
        self.panel_x = self.inspector_rect.x + 12
        self.panel_rect = pg.Rect(self.panel_x, 100, inspector_width - 44, top_height - 67)
        self.row_height = max(18, self.small.get_linesize() + 3)
        self.visible_rows = max(1, self.panel_rect.height // self.row_height)
        self.scroll_up = pg.Rect(self.inspector_rect.right - 23, self.panel_rect.top, 14, 18)
        self.scroll_down = pg.Rect(self.scroll_up.x, self.panel_rect.bottom - 18, 14, 18)
        self.scroll_track = pg.Rect(self.scroll_up.x, self.panel_rect.top + 23,
                                    14, self.panel_rect.height - 46)
        road_height = top_height - 142 - 62 - 2 * gap
        self.road_rect = pg.Rect(margin, 60, self.left_width, road_height)
        self.road_image_rect = self.road_rect.inflate(-16, -40)
        self.road_image_rect.y = self.road_rect.y + 32
        controls_y = self.road_rect.bottom + gap
        sim_width = int(self.left_width * .43)
        self.sim_control_rect = pg.Rect(margin, controls_y, sim_width, 142)
        self.task_control_rect = pg.Rect(margin + sim_width + gap, controls_y,
                                         self.left_width - sim_width - gap, 142)
        self.demo_rect = pg.Rect(margin, controls_y + 142 + gap, self.left_width, 62)

    def _label(self, text, x, y, color=TEXT, font=None, max_width=None):
        font = font or self.small
        text = _fit_text(font, text, max_width)
        self.screen.blit(_render_text(font, text, color), (int(x), int(y)))

    def _card(self, rect, title, note=None):
        self.pg.draw.rect(self.screen, CARD, rect, border_radius=10)
        self.pg.draw.rect(self.screen, BORDER, rect, width=1, border_radius=10)
        self._label(title, rect.x + 12, rect.y + 8, TEXT, self.font)
        if note:
            self._label(note, rect.right - self.small.size(note)[0] - 12, rect.y + 12, MUTED)

    def _button(self, rect, name, caption, enabled=True, active=False, color=ACCENT):
        pg = self.pg
        hover = enabled and rect.collidepoint(pg.mouse.get_pos())
        fill = (44, 62, 82) if hover else (32, 46, 64) if enabled else (26, 33, 44)
        pg.draw.rect(self.screen, fill, rect, border_radius=6)
        if active:
            pg.draw.rect(self.screen, color, rect, width=2, border_radius=6)
        font = self.small
        self._label(caption, rect.x + 8, rect.y + (rect.height - font.get_linesize()) // 2,
                    TEXT if enabled else (102, 116, 135), max_width=rect.width - 16)
        self.buttons.append((rect, name, enabled))

    def _draw_controls(self):
        pg, sim = self.pg, self.sim
        self._card(self.sim_control_rect, "Simulator controls")
        self._card(self.task_control_rect, "Task controls")
        enabled, _ = sim.overtake_availability()
        self._button(pg.Rect(self.task_control_rect.right - 100, self.task_control_rect.y + 8, 88, 23),
                     "Overtake", "Overtake", enabled, bool(sim.overtake and sim.overtake.running))
        sim_names = [("Play/Pause", "Pause [P]" if self.play else "Play [P]"),
                     ("Step", "Step [Space]"), ("Reset", "Reset [R]"), ("New seed", "New seed [N]"),
                     ("Slower play", "Speed −"), ("Faster play", "Speed +")]
        rect = self.sim_control_rect
        button_width = (rect.width - 30) // 2
        for i, (name, label) in enumerate(sim_names):
            button = pg.Rect(rect.x + 10 + i % 2 * (button_width + 10), rect.y + 37 + i // 2 * 28,
                             button_width, 23)
            self._button(button, name, label, not (sim.done and name in ("Step", "Play/Pause")))
        self._label(f"{sim.step_duration / self.interval:.2f}x · step {sim.step_duration:.1f}s",
                    rect.x + 12, rect.bottom - 18, MUTED)
        rect = self.task_control_rect
        options = sim.get_task_options()
        definitions = [("switch_lane_left", "← Lane left"), ("accelerate", "↑ Accelerate"),
                       ("switch_lane_right", "→ Lane right"), ("follow_car", "F Follow lead"),
                       ("brake", "↓ Brake"), ("cruise", "I Cruise")]
        button_width = (rect.width - 36) // 3
        for i, (name, caption) in enumerate(definitions):
            button = pg.Rect(rect.x + 10 + i % 3 * (button_width + 8), rect.y + 37 + i // 3 * 32,
                             button_width, 27)
            self._button(button, name, caption, options[name].enabled, name == sim.active_task)
        queued = options[sim.pending_task].task.label if sim.pending_task else "None"
        active = options[sim.active_task].task.label if sim.active_task else "None"
        self._label("Active: " + active, rect.x + 12, rect.y + 103, ACCENT, max_width=rect.width - 24)
        self._label("Queued: " + queued + " · Step / Play", rect.x + 12, rect.bottom - 19, MUTED,
                    max_width=rect.width - 24)
        self._card(self.demo_rect, "Demos")
        width = min(135, (self.demo_rect.width - 96) // 3)
        x, y = self.demo_rect.x + 80, self.demo_rect.y + 11
        for name, label in (("Demo overtake", "Overtake demo"), ("Demo brake", "Brake demo")):
            self._button(pg.Rect(x, y, width, 25), name, label,
                         active=bool(sim.demo and sim.demo.kind == ("overtake" if "overtake" in name else "brake")))
            x += width + 8
        demo_status = f"{sim.demo.kind}: {sim.demo.phase}" if sim.demo else "Manual control · choose a demo to start"
        self._label(demo_status, self.demo_rect.x + 12, self.demo_rect.y + 41, MUTED,
                    max_width=self.demo_rect.width - 24)

    def _draw_road(self, cards):
        pg = self.pg
        self._card(self.road_rect, "Highway", "Colors match the live DFA cards")
        viewer = self.sim.env.unwrapped.viewer
        crop_top, crop_bottom = 0, self.frame.shape[0]
        if viewer:
            centers = []
            for lane in self.sim.bindings["lanes"].values():
                s = lane.local_coordinates(self.sim.snapshot.ego.position)[0]
                centers.append(viewer.sim_surface.vec2pix(lane.position(s, 0.))[1])
            if centers:
                ideal = self.frame.shape[1] * self.road_image_rect.height / self.road_image_rect.width
                crop_height = min(self.frame.shape[0], int(max(ideal, max(centers) - min(centers) + 44)))
                middle = (min(centers) + max(centers)) / 2
                crop_top = int(np.clip(middle - crop_height / 2, 0, self.frame.shape[0] - crop_height))
                crop_bottom = crop_top + crop_height
        image = pg.surfarray.make_surface(self.frame[crop_top:crop_bottom].swapaxes(0, 1))
        self.screen.blit(pg.transform.smoothscale(image, self.road_image_rect.size), self.road_image_rect)
        self.road_badges = []
        if viewer is None:
            return
        image_rect = self.road_image_rect
        old_clip = self.screen.get_clip()
        self.screen.set_clip(image_rect)
        for card in cards:
            vehicle = card["vehicle"]
            if vehicle is None:
                continue
            px, py = viewer.sim_surface.vec2pix(vehicle.position)
            x = image_rect.x + px * image_rect.width / self.frame.shape[1]
            y = image_rect.y + (py - crop_top) * image_rect.height / (crop_bottom - crop_top)
            if not image_rect.collidepoint(x, y):
                continue
            label = card["label"]
            badge = pg.Rect(int(x - 13), int(y - 23), max(26, self.small.size(label)[0] + 10), 17)
            pg.draw.line(self.screen, card["color"], (x, y - 3), (x, badge.bottom), 1)
            pg.draw.rect(self.screen, card["color"], badge, border_radius=4)
            self._label(label, badge.x + 5, badge.y + 1, BG)
            self.road_badges.append((card["uid"], card["color"], badge))
        self.screen.set_clip(old_clip)

    def _arrow(self, start, end, color, width=1):
        pg = self.pg
        vector = pg.Vector2(end) - pg.Vector2(start)
        if vector.length() < 2:
            return
        direction = vector.normalize()
        pg.draw.line(self.screen, color, start, end, width)
        normal = pg.Vector2(-direction.y, direction.x)
        tip = pg.Vector2(end)
        pg.draw.polygon(self.screen, color,
                        [tip, tip - direction * 5 + normal * 3, tip - direction * 5 - normal * 3])

    def _draw_dfa(self, rect, card):
        pg = self.pg
        color, dfa, vehicle = card["color"], card["dfa"], card["vehicle"]
        self._card(rect, "")
        pg.draw.rect(self.screen, color, pg.Rect(rect.x, rect.y + 12, 3, rect.height - 24), border_radius=2)
        name = card["label"] + (f" · #{card['uid']}" if card["uid"] is not None else "")
        self._label(name, rect.x + 12, rect.y + 9, color, self.font)
        speed = f"{np.linalg.norm(vehicle.velocity):.1f} m/s" if vehicle else "No car"
        self._label(speed, rect.right - self.small.size(speed)[0] - 10, rect.y + 13, MUTED)
        source = card["source"]
        if vehicle and card["uid"] not in {uid for uid, _, _ in self.road_badges}:
            source += " · offscreen"
        self._label(source, rect.x + 12, rect.y + 31, MUTED, max_width=rect.width - 24)
        # This is the guarded outgoing view of the DFA. Only enabled task
        # transitions get boxes; the current state is a separate root caption.
        alive = dfa is not None and vehicle is not None and dfa.state[0] != "ended"
        order = ("switch_lane_left", "switch_lane_right", "accelerate", "brake", "follow_car", "cruise")
        events = [name for name in order if alive and name in card["options"]
                  and card["options"][name].enabled]
        root = pg.Rect(rect.x + 12, rect.y + 54, rect.width - 24, 23)
        pg.draw.rect(self.screen, (31, 47, 63), root, border_radius=5)
        pg.draw.rect(self.screen, color if alive else BORDER, root, 1, border_radius=5)
        state_label = dfa.label.replace("follow_car", "follow") if dfa and vehicle else "Vehicle unavailable"
        self._label("Now: " + state_label, root.x + 7, root.y + 3,
                    color if alive else MUTED, max_width=root.width - 14)
        nodes, edges = {}, {}
        node_width = (rect.width - 44) // 2
        for i, event in enumerate(events):
            nodes[event] = pg.Rect(rect.x + 12 + i % 2 * (node_width + 20),
                                    rect.y + 91 + i // 2 * 28, node_width, 22)
            edges[event] = transition(dfa.state, event)
        self.dfa_nodes.append(nodes)
        self.dfa_edges.append(edges)
        if nodes:
            trunk = rect.x + 12 + node_width + 10
            pg.draw.line(self.screen, color, (root.centerx, root.bottom),
                         (trunk, max(node.centery for node in nodes.values())), 1)
            for i, node in enumerate(nodes.values()):
                end = (node.right + 2, node.centery) if i % 2 == 0 else (node.left - 2, node.centery)
                self._arrow((trunk, node.centery), end, color)
        else:
            message = "No tasks available" if vehicle else "Vehicle unavailable"
            self._label(message, rect.x + 12, rect.y + 100, MUTED, max_width=rect.width - 24)
        captions = {"switch_lane_left": "Lane left", "switch_lane_right": "Lane right",
                    "accelerate": "Accelerate", "brake": "Brake", "follow_car": "Follow lead", "cruise": "Cruise"}
        for event, node in nodes.items():
            active = event == dfa.state[0]
            fill = color if active else (29, 40, 56)
            pg.draw.rect(self.screen, fill, node, border_radius=5)
            pg.draw.rect(self.screen, color, node, width=1, border_radius=5)
            caption = captions[event]
            self._label(caption, node.centerx - self.node_font.size(caption)[0] / 2,
                        node.centery - self.node_font.get_linesize() / 2,
                        BG if active else TEXT, self.node_font)
        self._label(f"{len(events)} available tasks", rect.x + 12, rect.bottom - 18, MUTED,
                    max_width=rect.width - 24)

    def _draw_overtake_tree(self):
        pg, rect, controller = self.pg, self.inspector_rect, self.sim.overtake
        self._card(rect, "Overtake / DFA tree")
        self._button(pg.Rect(rect.right - 75, rect.y + 7, 63, 24), "State view", "State")
        self.overtake_tree_nodes = []
        if controller is None:
            self._label("Select Overtake to start the sequence", rect.x + 12, rect.y + 45,
                        MUTED, max_width=rect.width - 24)
            return
        self._label(f"Saved: {controller.previous_speed:.1f}  |  Now: {self.sim.speed:.1f} m/s",
                    rect.x + 12, rect.y + 41, TEXT, max_width=rect.width - 24)
        root = pg.Rect(rect.x + 12, rect.y + 64, rect.width - 24, 28)
        pg.draw.rect(self.screen, (32, 46, 64), root, border_radius=6)
        pg.draw.rect(self.screen, ACCENT if controller.running or controller.completed else MUTED,
                     root, 1, border_radius=6)
        self._label("Overtake: " + controller.phase.replace("_", " "), root.x + 8, root.y + 6,
                    ACCENT, max_width=root.width - 16)
        leaf_height = min(52, (rect.height - 184) // 4)
        trunk = rect.x + 23
        captions = ("Left", "Accel", "Right", "Brake")
        for i, stage in enumerate(controller.stages):
            leaf = pg.Rect(rect.x + 37, root.bottom + 11 + i * (leaf_height + 7),
                           rect.width - 49, leaf_height)
            self.overtake_tree_nodes.append(leaf)
            color = ((255, 192, 91) if stage.status in ("active", "queued", "waiting")
                     else ACCENT if stage.status == "complete"
                     else (255, 145, 145) if stage.status in ("cancelled", "ended") else MUTED)
            pg.draw.line(self.screen, BORDER, (trunk, root.bottom), (trunk, leaf.centery))
            self._arrow((trunk, leaf.centery), (leaf.left - 2, leaf.centery), color)
            pg.draw.rect(self.screen, (26, 37, 52), leaf, border_radius=6)
            pg.draw.rect(self.screen, color, leaf, 1, border_radius=6)
            status = stage.status.upper()
            self._label(f"{i + 1}. {stage.label}", leaf.x + 7, leaf.y + 4, TEXT,
                        max_width=leaf.width - self.node_font.size(status)[0] - 20)
            self._label(status, leaf.right - self.node_font.size(status)[0] - 7,
                        leaf.y + 5, color, self.node_font)
            node_width = (leaf.width - 42) // 3
            active = 2 if stage.status == "complete" else 1 if stage.status == "active" else 0
            middle = captions[i]
            if stage.status == "active" and stage.dfa.state == ("cruise", "none"):
                middle = "Cruise"
            for j, caption in enumerate(("Ready", middle, "Done")):
                node = pg.Rect(leaf.x + 7 + j * (node_width + 14), leaf.bottom - 21, node_width, 16)
                if j:
                    self._arrow((node.left - 12, node.centery), (node.left - 2, node.centery), BORDER)
                selected = j == active and stage.status not in ("cancelled", "ended")
                pg.draw.rect(self.screen, color if selected else CARD, node, border_radius=3)
                self._label(caption, node.centerx - self.node_font.size(caption)[0] / 2,
                            node.y, BG if selected else MUTED, self.node_font)
        reason = controller.current.reason or ("Starting speed restored" if controller.completed
                                                else "Each step waits for its task guards")
        self._label(reason, rect.x + 12, rect.bottom - 46, MUTED, max_width=rect.width - 24)
        self._button(pg.Rect(rect.x + 12, rect.bottom - 30, 122, 23),
                     "Cancel overtake", "Cancel overtake", controller.running)
        self._label(f"{sum(s.status == 'complete' for s in controller.stages)}/4 complete",
                    rect.x + 144, rect.bottom - 25, MUTED)

    def _draw_inspector(self):
        pg = self.pg
        self.overtake_tree_nodes = []
        if getattr(self, "show_overtake_tree", False):
            self._draw_overtake_tree()
            return
        self._card(self.inspector_rect, "State & availability")
        if self.sim.overtake:
            self._button(pg.Rect(self.inspector_rect.right - 144, self.inspector_rect.y + 7, 50, 24),
                         "Overtake tree", "Tree")
        rect = pg.Rect(self.inspector_rect.right - 86, self.inspector_rect.y + 7, 76, 24)
        self._button(rect, "Diagnostics", "Facts" if self.diagnostics else "Details", active=self.diagnostics)
        rows = self.panel_rows()
        self.scroll_to(self.scroll)
        self.screen.set_clip(self.panel_rect)
        for i, (line, color) in enumerate(rows[self.scroll:self.scroll + self.visible_rows]):
            self._label(line, self.panel_x, self.panel_rect.y + i * self.row_height, color)
        self.screen.set_clip(None)
        pg.draw.rect(self.screen, (30, 43, 60), self.scroll_track, border_radius=5)
        pg.draw.rect(self.screen, (88, 116, 145), self.scrollbar_thumb(), border_radius=5)
        for rect, caption in ((self.scroll_up, "↑"), (self.scroll_down, "↓")):
            pg.draw.rect(self.screen, (41, 58, 78), rect, border_radius=3)
            self._label(caption, rect.x + 3, rect.y, MUTED)
        self._label("Wheel / PgUp / PgDn · " + str(len(rows)) + " rows",
                    self.panel_x, self.inspector_rect.bottom - 21, MUTED)

    def draw(self):
        pg, sim = self.pg, self.sim
        self.screen.fill(BG)
        self.buttons, self.dfa_nodes, self.dfa_edges = [], [], []
        self._label("HIGHWAY / TASK LAB", 14, 12, TEXT, self.title_font)
        status = "ENDED" if sim.done else "PLAYING" if self.play else "PAUSED"
        self._label(f"{status}  ·  {sim.snapshot.time:.2f}s  ·  Ego {sim.speed:.1f} m/s  ·  Seed {sim.seed}",
                    300, 20, ACCENT if self.play else MUTED)
        cam = pg.Rect(self.screen.get_width() - 139, 14, 125, 27)
        self._button(cam, "Camera", "Camera: " + ("Follow" if self.camera_follow else "World"))
        cards = sim.traffic_dfas.cards(sim.snapshot, sim.dfa, sim.get_task_options())
        self._draw_road(cards)
        self._draw_controls()
        self._draw_inspector()
        self._label("LIVE TASK DFAs", 14, self.dfa_top + 2, TEXT, self.font)
        self._label("Current state → available tasks · Guards evaluated for each car", 158,
                    self.dfa_top + 6, MUTED)
        for rect, card in zip(self.dfa_rects, cards):
            self._draw_dfa(rect, card)
        hover = pg.mouse.get_pos()
        for rect, name, _ in self.buttons:
            if (name in sim.get_task_options() or name == "Overtake") and rect.collidepoint(hover):
                reason = sim.overtake_availability()[1] if name == "Overtake" else sim.get_task_options()[name].reason
                pg.draw.rect(self.screen, CARD, pg.Rect(self.task_control_rect.x + 10,
                              self.task_control_rect.bottom - 21, self.task_control_rect.width - 20, 17))
                self._label(reason, self.task_control_rect.x + 12, self.task_control_rect.bottom - 19,
                            (255, 192, 91), max_width=self.task_control_rect.width - 24)
        pg.display.flip()
