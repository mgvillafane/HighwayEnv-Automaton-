"""View the four newest dynamic-waypoint PPO and DQN runs in live TensorBoard."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PPO = ROOT / "baseline_dynamic_waypoint/results"
DEFAULT_DQN = ROOT / "baseline_DQN/results"


def latest_runs(roots, count):
    """Select event-bearing run folders by saved creation time, not update time."""
    candidates = []
    for family, root in roots.items():
        for metadata_path in root.rglob("run.json"):
            directory = metadata_path.parent
            if not any(directory.glob("events.out.tfevents.*")):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
                continue
            relative = directory.relative_to(root)
            name = relative.as_posix() if relative.parts else directory.name
            candidates.append((created, f"{family}/{name}", directory))
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    return {name: path for _, name, path in candidates[:count]}


class LiveSettings:
    """Enable TensorBoard's normal chart refresh for this browser, then show it."""

    def __init__(self, app, chart_reload_seconds):
        self.app = app
        milliseconds = round(chart_reload_seconds * 1000)
        # TensorBoard's browser refresh is separate from --reload_interval.
        # Merge the settings so existing display preferences are preserved.
        self.page = ("""<!doctype html><html><head><meta charset="utf-8">
<title>Live training</title></head><body>Opening live training charts...
<script>
let settings = {};
try { settings = JSON.parse(localStorage.getItem('_tb_global_settings') || '{}'); }
catch (_) {}
settings.autoReload = true;
settings.autoReloadPeriodInMs = RELOAD_MILLISECONDS;
localStorage.setItem('_tb_global_settings', JSON.stringify(settings));
location.replace('/#timeseries');
</script></body></html>""".replace("RELOAD_MILLISECONDS", str(milliseconds))).encode("utf-8")

    def __call__(self, environ, start_response):
        if environ.get("PATH_INFO", "").rstrip("/") != "/live":
            return self.app(environ, start_response)
        start_response("200 OK", [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(self.page))),
            ("Cache-Control", "no-store"),
        ])
        return [self.page]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-logdir", type=Path, default=DEFAULT_PPO,
                        help="PPO results root or individual run folder; scanned recursively.")
    parser.add_argument("--dqn-logdir", type=Path, default=DEFAULT_DQN,
                        help="DQN results root or individual run folder; scanned recursively.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--latest", type=int, default=4,
                           help="Show this many newest runs by run.json creation time (default: 4).")
    selection.add_argument("--all-runs", action="store_true",
                           help="Show every event-bearing run under the two result roots.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--reload-seconds", type=float, default=5.0,
                        help="Seconds between TensorBoard event-file reloads.")
    parser.add_argument("--chart-reload-seconds", type=float, default=30.0,
                        help="Browser chart refresh interval, at least 30 seconds.")
    parser.add_argument("--check", action="store_true",
                        help="Validate TensorBoard configuration without starting a server.")
    args = parser.parse_args()
    if args.latest < 1:
        parser.error("latest must be positive.")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535.")
    if not 0 < args.reload_seconds < float("inf"):
        parser.error("reload-seconds must be positive and finite.")
    if not 30 <= args.chart_reload_seconds < float("inf"):
        parser.error("chart-reload-seconds must be finite and at least 30.")
    roots = {"PPO": args.ppo_logdir.resolve(), "DQN": args.dqn_logdir.resolve()}
    runs = roots if args.all_runs else latest_runs(roots, args.latest)
    if not runs:
        parser.error("No training runs with valid run.json creation times and TensorBoard events were found.")
    if any("," in str(path) for path in runs.values()):
        parser.error("TensorBoard's logdir_spec cannot represent paths containing commas.")

    try:
        from tensorboard import program
    except ImportError:
        parser.error("Run with baseline_waypoint/.venv/Scripts/python.exe, which has TensorBoard installed.")

    def server_factory(wsgi_app, flags):
        app = LiveSettings(wsgi_app, args.chart_reload_seconds)
        return program.create_port_scanning_werkzeug_server(app, flags)

    tensorboard = program.TensorBoard(server_class=server_factory)
    tensorboard.configure(argv=[
        "tensorboard", "--host", args.host, "--port", str(args.port),
        "--logdir_spec", ",".join(f"{name}:{path}" for name, path in runs.items()),
        "--reload_interval", str(args.reload_seconds), "--load_fast=false",
    ])
    details = {"url": f"http://{args.host}:{args.port}/live",
               "event_reload_seconds": args.reload_seconds,
               "chart_reload_seconds": args.chart_reload_seconds,
               "selection": "all" if args.all_runs else f"latest {args.latest}",
               "recursive": args.all_runs,
               "runs": {name: str(path) for name, path in runs.items()}}
    print(json.dumps(details, indent=2), flush=True)
    if args.check:
        return 0
    return tensorboard.main()


if __name__ == "__main__":
    raise SystemExit(main())
