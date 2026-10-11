"""Finite task controller over guard-validated requests and lifecycle events."""
from dataclasses import dataclass, field


TASK_EVENTS = ("switch_lane_left", "accelerate", "switch_lane_right",
               "follow_car", "brake", "cruise")
ALPHABET = TASK_EVENTS + ("lane_complete", "task_complete", "reject", "tick", "end", "reset")
# Following is allowed only with no lateral maneuver. Other longitudinal modes
# may run while completing a previously accepted left/right lane change.
STATES = tuple((mode, lateral) for lateral in ("none", "left", "right")
               for mode in ("cruise", "accelerate", "follow_car", "brake")
               if mode != "follow_car" or lateral == "none") + (("ended", "none"),)
INITIAL = ("cruise", "none")


def transition(state, event):
    """Total deterministic transition function over STATES x ALPHABET.

    A request failing the simulator's traffic guards emits 'reject', not a
    task event. Invalid structural combinations also self-loop defensively.
    """
    if state not in STATES or event not in ALPHABET:
        raise ValueError(f"Unknown DFA state/event: {state!r}, {event!r}")
    mode, lateral = state
    if event == "reset":
        return INITIAL
    if event == "end" or mode == "ended":
        return ("ended", "none")
    if event in ("tick", "reject"):
        return state
    if event.startswith("switch_lane_"):
        return ("cruise", event.removeprefix("switch_lane_")) if lateral == "none" else state
    if event == "lane_complete":
        return (mode, "none")
    if event == "task_complete":
        return ("cruise", lateral)
    if event == "follow_car" and lateral != "none":
        return state
    return (event, lateral)


@dataclass
class TaskDFA:
    state: tuple[str, str] = INITIAL
    last_event: str = "reset"
    history: list = field(default_factory=list)

    @property
    def label(self):
        mode, lateral = self.state
        return mode if lateral == "none" else f"{mode} + lane {lateral}"

    def dispatch(self, event):
        old = self.state
        self.state = transition(old, event)
        self.last_event = event
        if old != self.state:
            self.history.append((old, event, self.state))
        return self.state


def to_dot():
    """Export the complete DFA; grouped edge labels include self-loops."""
    def key(state):
        return "_".join(state)
    lines = ["digraph Tasks {", "  rankdir=LR;", "  node [shape=circle];",
             '  start [shape=point];', f'  start -> {key(INITIAL)};']
    for state in STATES:
        shape = "doublecircle" if state == INITIAL else "circle"
        lines.append(f'  {key(state)} [shape={shape}, label="{state[0]}\\n{state[1]}"];')
        edges = {}
        for event in ALPHABET:
            edges.setdefault(transition(state, event), []).append(event)
        for dest, events in edges.items():
            lines.append(f'  {key(state)} -> {key(dest)} [label="{", ".join(events)}"];')
    return "\n".join(lines + ["}", ""])
