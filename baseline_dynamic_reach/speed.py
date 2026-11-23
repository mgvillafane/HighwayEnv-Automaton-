"""Step 8-0: how fast does training run on this machine, and which speed-ups leave it bit-identical?

    python -B baseline_dynamic_reach/speed.py replay B3 --seed 7          # a PC-trained run: redo its first 61,440 transitions, compare with its saved log
    python -B baseline_dynamic_reach/speed.py variants P6 --seed 7        # worker processes + TensorBoard vs in-process environments vs no TensorBoard
    python -B baseline_dynamic_reach/speed.py profile P4 --seed 7         # where one run's time goes (cProfile, in-process environments)
    python -B baseline_dynamic_reach/speed.py concurrency P6 --vec dummy --runs 8 16 24

A speed change is allowed only if training stays bit-identical: the same training episodes in every logged
column and the same final weights. Anything that changes what is computed (more environments, larger
batches, the GPU) is an independent variable, not a speed-up, and is not tried here.

Every run is built by train.build_model, the code train.py itself uses, and runs in its own process.
Scratch runs go to smoke/speed/ (not part of the chain); reports to results/speed_check/.
"""
import argparse
import cProfile
import csv
import hashlib
import io
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dynamic_reach.config import ARMS, RESULTS_DIR, SMOKE_DIR, arm_config, run_name

SCRATCH = SMOKE_DIR / "speed"
REPORT_DIR = RESULTS_DIR / "speed_check"
REPLAY_TRANSITIONS = 61_440  # 80 rollouts of 768 transitions: Step 7a's reproduction check.
PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve())


def weights_sha(model):
    state = model.policy.state_dict()
    return hashlib.sha256(b"".join(state[key].cpu().numpy().tobytes() for key in sorted(state))).hexdigest()


def prefix(arm, seed, transitions, vec, tensorboard, out_dir, profile=False):
    """Train `arm` for `transitions` exactly as train.py would; write result.json into out_dir."""
    from baseline_dynamic_reach.train import build_model, ignore_console_interrupts
    ignore_console_interrupts()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = arm_config(arm)
    started = time.perf_counter()
    model, vec_env, metrics, _ = build_model(config, seed, out_dir, vec=vec, tensorboard=tensorboard)
    built = time.perf_counter()
    profiler = cProfile.Profile() if profile else None
    try:
        if profiler:
            profiler.enable()
        model.learn(total_timesteps=transitions, callback=metrics)
        if profiler:
            profiler.disable()
    finally:
        learned = time.perf_counter()
        if metrics.handle is not None and not metrics.handle.closed:
            metrics.handle.close()
        vec_env.close()
    result = {"arm": arm, "seed": seed, "vec": vec, "tensorboard": tensorboard, "transitions": model.num_timesteps,
              "setup_seconds": built - started, "learn_seconds": learned - built,
              "transitions_per_second": model.num_timesteps / (learned - built), "weights_sha256": weights_sha(model)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if profiler:
        profiler.dump_stats(str(out_dir / "profile.pstats"))
    return result


def spawn(arm, seed, transitions, vec, tensorboard, out_dir, profile=False):
    command = [PYTHON, "-B", SCRIPT, "prefix", arm, "--seed", str(seed), "--transitions", str(transitions),
               "--vec", vec, "--out", str(out_dir)] + ([] if tensorboard else ["--no-tensorboard"]) + (["--profile"] if profile else [])
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def finish(process, out_dir):
    _, err = process.communicate()
    if process.returncode:
        raise RuntimeError(f"{out_dir} failed:\n{err[-3000:]}")
    return json.loads((Path(out_dir) / "result.json").read_text(encoding="utf-8"))


def episodes(path, last=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    if last is not None:
        body = [row for row in body if int(row[0]) <= last]
    return header, body


def write(name, lines):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def replay(arm, seed):
    """Re-run a saved run's first 61,440 transitions with today's code and compare the training log."""
    saved = RESULTS_DIR / run_name(arm, seed) / "training_episodes.csv"
    sidecar = json.loads((RESULTS_DIR.parent / "models" / f"{run_name(arm, seed)}.json").read_text(encoding="utf-8"))
    out = SCRATCH / f"replay_{arm}_seed{seed}"
    result = finish(spawn(arm, seed, REPLAY_TRANSITIONS, "subproc", True, out), out)
    header, new = episodes(out / "training_episodes.csv")
    saved_header, old = episodes(saved, last=result["transitions"])
    same = header == saved_header and new == old
    differing = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
    lines = [f"# Replay of {run_name(arm, seed)}: first {result['transitions']:,} transitions", "",
             f"- Saved run: trained on {sidecar['platform']['system']} {sidecar['platform']['release']}, "
             f"Python {sidecar['platform']['python']}, {sidecar['created_utc'][:10]}.",
             f"- Replay: today's code, worker processes, TensorBoard on (as train.py by default); "
             f"{result['learn_seconds']:.1f} s ({result['transitions_per_second']:.0f} transitions/s).",
             f"- Training episodes: {len(new)} in the replay, {len(old)} in the saved log; columns "
             f"{'identical' if header == saved_header else 'DIFFERENT'}.",
             f"- **{'Identical in every column of every episode.' if same else f'DIFFERENT: first differing episode {differing[:1] or len(new)}.'}**"]
    write(f"replay_{arm}_seed{seed}.md", lines)
    return same


def variants(arm, seed, transitions):
    """The same prefix under each candidate speed-up, one process each; all must match the reference."""
    setups = {"worker processes + TensorBoard (train.py until now)": ("subproc", True),
              "worker processes, no TensorBoard": ("subproc", False),
              "in-process environments + TensorBoard": ("dummy", True),
              "in-process environments, no TensorBoard": ("dummy", False)}
    results, logs = {}, {}
    for label, (vec, tensorboard) in setups.items():  # One at a time, so the timings do not compete.
        out = SCRATCH / f"variant_{arm}_seed{seed}_{vec}_{'tb' if tensorboard else 'notb'}"
        results[label] = finish(spawn(arm, seed, transitions, vec, tensorboard, out), out)
        logs[label] = episodes(out / "training_episodes.csv")
    reference = next(iter(setups))
    lines = [f"# Speed-up candidates: {arm}, seed {seed}, {transitions:,} transitions, one run alone", "",
             "| Setup | Transitions/s | Setup time (s) | Training log vs reference | Final weights vs reference |",
             "|---|---:|---:|---|---|"]
    for label, r in results.items():
        log_same = logs[label] == logs[reference]
        weight_same = r["weights_sha256"] == results[reference]["weights_sha256"]
        lines.append(f"| {label} | {r['transitions_per_second']:.0f} | {r['setup_seconds']:.1f} | "
                     f"{'identical' if log_same else '**different**'} ({len(logs[label][1])} episodes) | "
                     f"{'identical' if weight_same else '**different**'} |")
    write(f"variants_{arm}_seed{seed}.md", lines)
    return results


CATEGORIES = (("highway_env", "simulator (highway-env)"), ("fast_observation", "observation encoder"),
              ("baseline_dynamic_reach", "reach environment and logging"),
              ("baseline_dynamic_waypoint", "dynamic-goal environment"),
              ("stable_baselines3", "SB3 (rollout bookkeeping, buffers)"), ("torch", "torch (networks, updates)"),
              ("gymnasium", "gymnasium wrappers"), ("numpy", "numpy"), ("tensorboard", "TensorBoard"),
              ("csv", "CSV logging"))


def profile(arm, seed, transitions):
    out = SCRATCH / f"profile_{arm}_seed{seed}"
    result = finish(spawn(arm, seed, transitions, "dummy", True, out, profile=True), out)
    stats = pstats.Stats(str(out / "profile.pstats"))
    total = sum(entry[2] for entry in stats.stats.values())  # Own time of every function.
    shares = {label: 0.0 for _, label in CATEGORIES}
    shares["other (Python built-ins, C extensions)"] = 0.0
    for (filename, _, _), entry in stats.stats.items():
        path = filename.replace("\\", "/")
        label = next((label for key, label in CATEGORIES if f"/{key}" in path or path.startswith(key)), None)
        shares[label or "other (Python built-ins, C extensions)"] += entry[2]
    buffer = io.StringIO()
    pstats.Stats(str(out / "profile.pstats"), stream=buffer).sort_stats("cumulative").print_stats(35)
    top = [line for line in buffer.getvalue().splitlines() if line.strip()][-36:]
    lines = [f"# Where the time goes: {arm}, seed {seed}, {result['transitions']:,} transitions, in-process environments", "",
             f"{result['transitions_per_second']:.0f} transitions/s under the profiler (it slows Python code, so "
             "shares matter more than the speed). Own time by package:", "",
             "| Part | Share of time |", "|---|---:|"]
    lines += [f"| {label} | {100 * value / total:.1f}% |" for label, value in sorted(shares.items(), key=lambda kv: -kv[1]) if value]
    lines += ["", "Top functions by cumulative time:", "", "```", *top, "```"]
    write(f"profile_{arm}_seed{seed}.md", lines)


def concurrency(arm, levels, vec, transitions, seed=7):
    """k copies of the same run at once (the same seed, so a trained init_policy exists for every copy)."""
    rows = []
    for level in levels:
        outs = [SCRATCH / f"concurrency_{arm}_{vec}_{level}" / f"copy{i}" for i in range(level)]
        started = time.perf_counter()
        processes = [spawn(arm, seed, transitions, vec, False, out) for out in outs]
        results = [finish(process, out) for process, out in zip(processes, outs)]
        wall = time.perf_counter() - started
        per_run = [r["transitions_per_second"] for r in results]
        total = sum(per_run)
        rows.append({"runs": level, "per_run_min": min(per_run), "per_run_max": max(per_run), "total": total,
                     "wall": wall, "runs_per_hour_2m": 3600 * total / 2_000_000})
        print(rows[-1], flush=True)
    lines = [f"# Throughput with several runs at once: {arm}, {'in-process environments' if vec == 'dummy' else 'worker processes'}, "
             f"no TensorBoard, {transitions:,} transitions each", "",
             "Total = sum over runs of each run's own training speed while all were running.", "",
             "| Runs at once | Per run (transitions/s) | Total (transitions/s) | 2M runs per hour | Wall time (s) |",
             "|---:|---|---:|---:|---:|"]
    lines += [f"| {r['runs']} | {r['per_run_min']:.0f}–{r['per_run_max']:.0f} | {r['total']:.0f} | {r['runs_per_hour_2m']:.1f} | "
              f"{r['wall']:.0f} |" for r in rows]
    write(f"concurrency_{arm}_{vec}.md", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prefix", help="(internal) one training prefix in this process")
    one.add_argument("arm", choices=list(ARMS))
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--transitions", type=int, required=True)
    one.add_argument("--vec", choices=("subproc", "dummy"), required=True)
    one.add_argument("--no-tensorboard", action="store_true")
    one.add_argument("--profile", action="store_true")
    one.add_argument("--out", required=True)
    for name in ("replay", "variants", "profile"):
        sub = commands.add_parser(name)
        sub.add_argument("arm", choices=list(ARMS))
        sub.add_argument("--seed", type=int, default=7)
        sub.add_argument("--transitions", type=int, default=REPLAY_TRANSITIONS)
    many = commands.add_parser("concurrency")
    many.add_argument("arm", choices=list(ARMS))
    many.add_argument("--runs", type=int, nargs="+", required=True)
    many.add_argument("--vec", choices=("subproc", "dummy"), required=True)
    many.add_argument("--transitions", type=int, default=30_720)
    args = parser.parse_args()
    if args.command == "prefix":
        prefix(args.arm, args.seed, args.transitions, args.vec, not args.no_tensorboard, args.out, args.profile)
    elif args.command == "replay":
        sys.exit(0 if replay(args.arm, args.seed) else 1)
    elif args.command == "variants":
        variants(args.arm, args.seed, args.transitions)
    elif args.command == "profile":
        profile(args.arm, args.seed, args.transitions)
    else:
        concurrency(args.arm, args.runs, args.vec, args.transitions)


if __name__ == "__main__":
    main()
