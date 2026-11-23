"""Step 8 launcher: train arms x seeds with a limit on parallel processes, then evaluate them.

    python -B baseline_dynamic_reach/grid_run.py --stage 8a
    python -B baseline_dynamic_reach/grid_run.py --stage 8b --width n64
    python -B baseline_dynamic_reach/grid_run.py --stage 8c --order E --width n64
    python -B baseline_dynamic_reach/grid_run.py --stage 8c2 --order N O --width n64   # time cost
    python -B baseline_dynamic_reach/grid_run.py --stage 8d --cells E_c1w01 E_c05w0 --width n64
    python -B baseline_dynamic_reach/grid_run.py --stage 8e --cells N_c05w0t1 N_c1w0t4 O_c1w0t4  # 256 + round 4
    python -B baseline_dynamic_reach/grid_run.py --arms E1 --seeds 7 --runs 4        # any list of arms
    python -B baseline_dynamic_reach/grid_run.py --arms N3_c05w0t1 O3_c1w0t4_n256 --seeds 7 10 --final-test s60000 --evaluate-only

A run starts when the run it continues from (its init_policy, same seed) has been trained. Each run is
`train.py --no-tensorboard --checkpoint-every 200000 --skip-evaluation`, at most --runs at once, with
evaluations (one process each) alongside, at most --eval-jobs at once. Step 8-0 measured the throughput
on this PC (EXPERIMENT_NOTES.md); TensorBoard off and the --vec choice leave training bit-identical.

Evaluation, deterministic PPO only (random and scripted are measured once per episode set):
- every run's final model on the full task, on the two selection sets (seeds 10,000-10,199 and 40,000-40,199);
- for rounds 3 and 4, also the two checkpoints 400k and 200k before the end (the score averages all three);
- runs trained on practice episodes, also on the 200 natural-start passing episodes (H1's task).
Outputs: results/<run>/eval8/<set>_<snapshot>/; references in results/grid8/references/. Finished work is
skipped, so the launcher can be re-run after an interruption. Logs: results/logs/<run>.log.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dynamic_reach.config import (ARMS, MODELS_DIR, P6_REWARD, RESULTS_DIR, REWARDS, SMOKE_DIR, STEP8_ARMS,
                                           TIME_REWARDS,
                                           STEP8_CHECKPOINT_EVERY, STEP8_CONFIRM_SEEDS, STEP8_SCREEN_SEEDS,
                                           arm_config, run_name, step8_name)

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
GRID_DIR = RESULTS_DIR / "grid8"
SELECTION_SETS = {"s10000": (10_000, 200), "s40000": (40_000, 200)}
# Fresh test sets, each used once: s50000 for the 8d final test; s60000 for 8f (the 8d result helped pick 8e's recipes).
TEST_SETS = {"s50000": (50_000, 500), "s60000": (60_000, 500)}
FINAL_TEST = {"s50000": TEST_SETS["s50000"]}  # --final-test [set] picks one.
PASSING_SET = {"pass10000": (10_000, 200)}  # H1's task: blocked episodes, natural start, blocker held.
FULL_TASK_ARM = step8_name("N", 3, P6_REWARD)  # Any full-task arm: references do not depend on the reward.
PASSING_ARM = "H1"
P6_SEEDS = (7, 8, 9)
SCORED_BEFORE_END = (400_000, 200_000)  # For a 2M round these are the 1.6M and 1.8M checkpoints, as before 8e.
# Step 8-0 on this PC: with few runs, worker processes are faster per run (12 runs: 9,150 transitions/s in
# total); with many, in-process environments give the most in total (32 runs: 11,550/s). Training is identical.
SUBPROC_UP_TO = 12
SMOKE = False  # --smoke: tiny runs in smoke/ (outside the chain), to test the launcher end to end.  # Checkpoints nearest 1.6M and 1.8M of a 2M round, besides the final model.


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def stage_arms(args):
    width = "" if args.width in (None, "n64") else args.width
    if args.arms:
        return args.arms
    if args.stage == "8a":
        return [step8_name("E", r, P6_REWARD if r > 1 else None, w) for w in ("", "n256") for r in (1, 2, 3)]
    if args.stage == "8b":
        return [step8_name(o, r, P6_REWARD if r > 1 else None, width) for o in ("N", "H", "O") for r in (1, 2, 3)
                if not (o == "O" and r == 1)]
    if args.stage == "8c":
        orders = args.order or []
        return [step8_name(o, r, reward, width) for o in orders for reward in REWARDS if reward != P6_REWARD
                for r in (2, 3)]
    if args.stage == "8c2":  # Time cost (after 8c): rounds 2-3 of each order, the time-cost rewards only.
        return [step8_name(o, r, reward, width) for o in (args.order or []) for reward in TIME_REWARDS for r in (2, 3)]
    if args.stage == "8d":
        arms = []
        for cell in args.cells:
            order, reward = cell.split("_")
            arms += [step8_name(order, r, reward if r > 1 else None, width) for r in (1, 2, 3)]
        return list(dict.fromkeys(arms))
    if args.stage == "8e":  # Width x longer training: every cell at 256 (rounds 1-4), then round 4 at 64.
        arms = []
        for w in ("n256", ""):
            for cell in args.cells:
                order, reward = cell.split("_")
                arms += [step8_name(order, r, reward if r > 1 else None, w) for r in (1, 2, 3, 4)]
        return list(dict.fromkeys(arms))  # Runs already trained (64-wide rounds 1-3, E1_n256 on 7/10/13) are skipped.
    raise SystemExit("Give --stage or --arms.")


def trained(run):
    sidecar = MODELS_DIR / f"{run}.json"
    return sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get("status") == "trained"


def train_command(arm, seed, vec):
    command = [PYTHON, "-B", str(HERE / "train.py"), "--arm", arm, "--seed", str(seed), "--vec", vec,
               "--no-tensorboard", "--checkpoint-every", str(STEP8_CHECKPOINT_EVERY), "--skip-evaluation"]
    if SMOKE:
        return command + ["--smoke"]
    parent = ARMS[arm]["parent"]
    if ARMS[arm].get("iv") != "round" and parent is not None and not trained(run_name(parent, seed)):
        command.append("--parent-in-progress")  # Re-checked with ivcheck on the saved sidecars afterwards.
    return command


def snapshots(run):
    """[(label, model path)] to evaluate: the final model, plus the two checkpoints before it (400k and 200k
    earlier: ~1.6M and ~1.8M of a 2M round, ~2.6M and ~2.8M of Step 8e's 3M round 4) for a round 3 or 4."""
    found = [("final", MODELS_DIR / f"{run}.zip")]
    if ARMS[run.rsplit("_seed", 1)[0]].get("round") in (3, 4):
        sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
        total = sidecar["requested_timesteps"]
        points = sorted((int(path.stem.split("_")[-2]), path) for path in (RESULTS_DIR / run / "checkpoints").glob("*.zip"))
        for before in SCORED_BEFORE_END:
            steps, path = min(points, key=lambda point: abs(point[0] - (total - before)))
            found.append((str(steps), path))
    return found


def evaluation_jobs(run, final_test=False):
    arm = run.rsplit("_seed", 1)[0]
    if final_test and ARMS[arm].get("round") not in (3, 4, None):  # None: P6, an arm from before Step 8.
        return []  # The final test scores finished pipelines (their final models) only.
    jobs = []
    sets = FINAL_TEST if final_test else SELECTION_SETS
    practice = arm_config(arm)["environment"].get("scenario")
    # Selection and final-test sets are always the full task. A run trained on practice episodes would
    # otherwise be measured on its own practice task (fixed 2026-11-18; see EXPERIMENT_NOTES.md, Step 8a).
    full_task = FULL_TASK_ARM if practice else None
    for label, path in snapshots(run):
        if final_test and label != "final":
            continue
        for name, (first, count) in sets.items():
            jobs.append((run, path, name, label, first, count, full_task))
    if practice and not final_test:
        for name, (first, count) in PASSING_SET.items():
            jobs.append((run, MODELS_DIR / f"{run}.zip", name, "final", first, count, PASSING_ARM))
    return jobs


def evaluation_output(run, set_name, label):
    return RESULTS_DIR / run / "eval8" / f"{set_name}_{label}"


def evaluate_command(job):
    run, path, set_name, label, first, count, environment_of = job
    output = evaluation_output(run, set_name, label)
    command = [PYTHON, "-B", str(HERE / "evaluate.py"), "--run", run, "--seed", str(first), "--episodes", str(count),
               "--no-references", "--output-name", str(output.relative_to(RESULTS_DIR)),
               "--models-dir", str(MODELS_DIR), "--results-dir", str(RESULTS_DIR)]
    if label != "final":
        command += ["--checkpoint", str(path)]
    if environment_of:
        command += ["--environment-of", environment_of]
    return command, output


def reference_jobs(final_test=False):
    jobs = []
    sets = FINAL_TEST if final_test else {**SELECTION_SETS, **PASSING_SET}
    for name, (first, count) in sets.items():
        arm = PASSING_ARM if name.startswith("pass") else FULL_TASK_ARM
        output = GRID_DIR / "references" / name
        command = [PYTHON, "-B", str(HERE / "evaluate.py"), "--references", arm, "--seed", str(first),
                   "--episodes", str(count), "--output-name", str(output.relative_to(RESULTS_DIR)),
                   "--results-dir", str(RESULTS_DIR)]
        jobs.append((f"references {name}", command, output))
    return jobs


def relabel_own_task_evaluations(runs):
    """Before the 2026-11-18 fix, selection-set evaluations of practice-trained runs used their own practice task.
    Those folders are kept, renamed own_task_<set>_<snapshot>, so the full-task evaluation can be made under the
    proper name. Returns the renamed folders."""
    moved = []
    for run in runs:
        for folder in sorted((RESULTS_DIR / run / "eval8").glob("s*_*")):
            summary = folder / "evaluation.json"
            if not summary.exists():
                continue
            header = json.loads(summary.read_text(encoding="utf-8"))
            if header.get("environment", {}).get("scenario"):
                target = folder.with_name(f"own_task_{folder.name}")
                folder.rename(target)
                moved.append(target)
    return moved


def run_process(label, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        handle.flush()
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return label, code, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("8a", "8b", "8c", "8c2", "8d", "8e"))
    parser.add_argument("--arms", nargs="+", choices=list(STEP8_ARMS))
    parser.add_argument("--width", choices=("n64", "n256"), default="n64")
    parser.add_argument("--order", nargs="+", choices=("N", "H", "E", "O"), help="8c: the order(s) that get the grid.")
    parser.add_argument("--cells", nargs="+", help="8d: finalist cells as <order>_<reward>, e.g. E_c1w01.")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--runs", type=int, default=32, help="Training runs at once.")
    parser.add_argument("--eval-jobs", type=int, default=4, help="Evaluation processes at once.")
    parser.add_argument("--vec", choices=("auto", "dummy", "subproc"), default="auto",
                        help="auto: worker processes while at most 12 runs can train, else in-process environments.")
    parser.add_argument("--final-test", nargs="?", const="s50000", choices=tuple(TEST_SETS),
                        help="Evaluate the final models on a fresh test set only (default s50000, the 8d final test).")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny runs and evaluations under smoke/ (tests the launcher).")
    args = parser.parse_args()
    global SMOKE, MODELS_DIR, RESULTS_DIR, GRID_DIR, SELECTION_SETS, PASSING_SET, FINAL_TEST, STEP8_CHECKPOINT_EVERY
    if args.final_test:
        FINAL_TEST = {args.final_test: TEST_SETS[args.final_test]}
    if args.smoke:
        SMOKE, MODELS_DIR, RESULTS_DIR = True, SMOKE_DIR / "models", SMOKE_DIR / "results"
        GRID_DIR, STEP8_CHECKPOINT_EVERY = RESULTS_DIR / "grid8", 768
        SELECTION_SETS = {name: (first, 3) for name, (first, _) in SELECTION_SETS.items()}
        PASSING_SET = {name: (first, 3) for name, (first, _) in PASSING_SET.items()}
        FINAL_TEST = {name: (first, 3) for name, (first, _) in FINAL_TEST.items()}
    from baseline_dynamic_reach.train import ignore_console_interrupts
    ignore_console_interrupts()
    seeds = args.seeds or {"8d": STEP8_CONFIRM_SEEDS, "8e": STEP8_SCREEN_SEEDS + STEP8_CONFIRM_SEEDS}.get(
        args.stage, STEP8_SCREEN_SEEDS)
    arms = stage_arms(args)
    runs = [(arm, seed) for arm in arms for seed in seeds]
    log(f"{len(runs)} runs: {', '.join(arms)} x seeds {', '.join(map(str, seeds))}; "
        f"up to {args.runs} runs and {args.eval_jobs} evaluations at once")

    for folder in relabel_own_task_evaluations([run_name(a, s) for a, s in runs]):
        log(f"renamed practice-task evaluation to {folder.relative_to(RESULTS_DIR)}")
    pending_train = [] if args.evaluate_only else [(a, s) for a, s in runs if not trained(run_name(a, s))]
    pending_eval = []
    failed = set()
    for job in reference_jobs(args.final_test):
        if not (job[2] / "evaluation.json").exists():
            pending_eval.append(("ref", job))
    # The 8d final test also scores P6 itself (trained on the Mac, P4 -> P5 -> P6, 1M per round) on the same episodes.
    p6_runs = [run_name("P6", seed) for seed in P6_SEEDS] if args.final_test == "s50000" else []
    for run in [run_name(arm, seed) for arm, seed in runs] + p6_runs:
        if trained(run):
            pending_eval += [("run", job) for job in evaluation_jobs(run, args.final_test)
                             if not (evaluation_output(job[0], job[2], job[3]) / "evaluation.json").exists()]
    running = {}
    with ThreadPoolExecutor(max_workers=args.runs + args.eval_jobs) as pool:
        while pending_train or pending_eval or running:
            busy = sum(1 for kind in running.values() if kind == "train")
            ready = [(a, s) for a, s in pending_train if not (ARMS[a].get("iv") == "round"
                                                             and not trained(run_name(ARMS[a]["parent"], s)))]
            vec = args.vec if args.vec != "auto" else ("subproc" if busy + len(ready) <= SUBPROC_UP_TO else "dummy")
            for arm, seed in list(pending_train):
                if busy >= args.runs:
                    break
                init = ARMS[arm]["parent"] if ARMS[arm].get("iv") == "round" else None
                if init and run_name(init, seed) in failed:
                    pending_train.remove((arm, seed))
                    failed.add(run_name(arm, seed))
                    log(f"SKIP {run_name(arm, seed)}: its previous round failed")
                    continue
                if init and not trained(run_name(init, seed)):
                    continue
                pending_train.remove((arm, seed))
                run = run_name(arm, seed)
                log(f"train {run} ({vec})")
                running[pool.submit(run_process, run, train_command(arm, seed, vec), RESULTS_DIR / "logs" / f"{run}.log")] = "train"
                busy += 1
            while pending_eval and sum(1 for kind in running.values() if kind == "eval") < args.eval_jobs:
                kind, job = pending_eval.pop(0)
                if kind == "ref":
                    label, command, output = job
                    log_path = GRID_DIR / "logs" / f"{label.replace(' ', '_')}.log"
                else:
                    command, output = evaluate_command(job)
                    label = f"evaluate {job[0]} {job[2]} {job[3]}"
                    log_path = RESULTS_DIR / "logs" / f"{job[0]}.log"
                running[pool.submit(run_process, label, command, log_path)] = "eval"
            if not running:
                if pending_train:
                    log(f"Nothing can start: {len(pending_train)} runs wait for failed or missing previous rounds.")
                break
            # Re-check every minute too: a waiting run's previous round may be trained by another launcher.
            done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            for future in done:
                kind = running.pop(future)
                label, code, seconds = future.result()
                log(f"{'done' if code == 0 else 'FAILED'} ({seconds / 60:.1f} min): {label}")
                if kind == "train":
                    if code == 0 and trained(label):
                        pending_eval += [("run", job) for job in evaluation_jobs(label, args.final_test)]
                    else:
                        failed.add(label)
    log(f"Finished. Failed: {sorted(failed) or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
