"""Step 7a: is the training pipeline doing what we think? Read-only checks of PPO's learning signal.

    python -B baseline_dynamic_reach/diagnose_updates.py curves                 # noise vs. change in the training curves; PPO's logged update sizes
    python -B baseline_dynamic_reach/diagnose_updates.py run B3m_seed7 B4_seed7 # gradient shares and one update, for trained runs
    python -B baseline_dynamic_reach/diagnose_updates.py start B5 --seed 7      # the same for an arm's starting point, before training
    python -B baseline_dynamic_reach/diagnose_updates.py checkpoints B5_seed7 B5_seed8   # deterministic learning curves

`run` and `start` collect one rollout with the run's own settings, then
- compute, per minibatch, the policy and value gradient norms before clipping and the factor by which
  SB3's single gradient-norm clip (max_grad_norm over both networks) scales them;
- apply one real PPO update (all epochs) in memory, once with the configured clip and once without,
  and measure how far the policy moved (KL between the action distributions before and after, on the
  rollout's states). Nothing is saved to the model files.

`checkpoints` evaluates every saved checkpoint (train.py --checkpoint-every) of the given runs on the fixed
200 evaluation episodes with deterministic actions, in parallel processes: a learning curve without the
sampling noise of the training curves.

Results go to results/pipeline_check/ (one .json per target, and a .md table per command).
"""
import argparse
import copy
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baseline_dynamic_reach.config import ARMS, MODELS_DIR, RESULTS_DIR, SEEDS, arm_config, run_name

OUTPUT_DIR = RESULTS_DIR / "pipeline_check"
ROLLOUT_SEED = 123  # Same rollout seed for every target, so targets see comparable episodes.


def binomial_ratio(rows, bin_size=25_000, after=200_000):
    """Spread of per-bin success rates divided by the spread expected from counting alone."""
    rows = [r for r in rows if int(r["timesteps"]) > after]
    bins = {}
    for r in rows:
        bins.setdefault(int(r["timesteps"]) // bin_size, []).append(int(r["success"]))
    rates = [statistics.mean(v) for v in bins.values() if len(v) >= 20]
    counts = [len(v) for v in bins.values() if len(v) >= 20]
    p = statistics.mean(int(r["success"]) for r in rows)
    expected = (p * (1 - p) / statistics.mean(counts)) ** 0.5
    return statistics.mean(counts), statistics.stdev(rates), expected


def curves():
    lines = ["# Training curves: counting noise and PPO's update sizes", "",
             "Episodes per 25k-transition bin after 200k; spread of the bin success rates against pure binomial noise "
             "(1.0 = the curve's wobble is all counting noise). PPO diagnostics are medians after 200k from progress.csv.", "",
             "| Run | Episodes per bin | Observed spread | Counting noise | Ratio | BACK/LEFT/RIGHT ratio | approx KL | clip fraction | explained variance | entropy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            run = run_name(arm, seed)
            episodes, progress = RESULTS_DIR / run / "training_episodes.csv", RESULTS_DIR / run / "progress.csv"
            if not episodes.exists() or not progress.exists():
                continue
            rows = list(csv.DictReader(episodes.open(encoding="utf-8")))
            n, observed, expected = binomial_ratio(rows)
            _, blr_observed, blr_expected = binomial_ratio([r for r in rows if r["slot"] != "front"])
            logged = [r for r in csv.DictReader(progress.open(encoding="utf-8"))
                      if r.get("train/approx_kl") and float(r["time/total_timesteps"]) > 200_000]
            med = lambda key, sign=1: statistics.median(sign * float(r[key]) for r in logged)
            ratio = lambda a, b: f"{a / b:.2f}" if b > 0 else "–"  # B1 never succeeds: no binomial spread to compare.
            lines.append(f"| {run} | {n:.0f} | {observed:.3f} | {expected:.3f} | {ratio(observed, expected)} | "
                         f"{ratio(blr_observed, blr_expected)} | {med('train/approx_kl'):.4f} | {med('train/clip_fraction'):.3f} | "
                         f"{med('train/explained_variance'):.2f} | {med('train/entropy_loss', -1):.2f} |")
    write(lines, "curves.md")


def probe(model, vec):
    """Gradient norms per minibatch and the KL of one update with and without the clip (in memory)."""
    import torch
    total, callback = model._setup_learn(model.n_steps * vec.num_envs, None, reset_num_timesteps=True)
    callback.on_training_start(locals(), globals())
    model.collect_rollouts(vec, callback, model.rollout_buffer, model.n_steps)
    model._update_current_progress_remaining(0, total)
    policy_params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    value_params = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    # Read the states before the first get(): it flattens the buffer's arrays in place.
    buffer = model.rollout_buffer.observations
    observations = {key: torch.as_tensor(value.reshape(-1, *value.shape[2:])) for key, value in buffer.items()}
    with torch.no_grad():
        before = model.policy.get_distribution(observations).distribution.probs
    norms = []
    for data in model.rollout_buffer.get(model.batch_size):
        values, log_prob, entropy = model.policy.evaluate_actions(data.observations, data.actions.long().flatten())
        advantages = (data.advantages - data.advantages.mean()) / (data.advantages.std() + 1e-8)
        ratio = torch.exp(log_prob - data.old_log_prob)
        clip = model.clip_range(1.0)
        policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
        value_loss = torch.nn.functional.mse_loss(data.returns, values.flatten())
        model.policy.zero_grad()
        (policy_loss - model.ent_coef * entropy.mean() + model.vf_coef * value_loss).backward()
        g_policy = float(torch.sqrt(sum((p.grad ** 2).sum() for p in policy_params)))
        g_value = float(torch.sqrt(sum((p.grad ** 2).sum() for p in value_params)))
        norm = (g_policy ** 2 + g_value ** 2) ** 0.5
        norms.append({"policy": g_policy, "value": g_value, "value_share": g_value ** 2 / norm ** 2,
                      "clip_factor": min(1.0, model.max_grad_norm / (norm + 1e-6)), "value_loss": float(value_loss),
                      "returns_sd": float(data.returns.std())})
    saved_policy, saved_optimizer = copy.deepcopy(model.policy.state_dict()), copy.deepcopy(model.policy.optimizer.state_dict())
    configured, updates = model.max_grad_norm, {}
    for label, max_norm in (("configured", configured), ("no_clip", 1e9)):
        model.policy.load_state_dict(saved_policy)
        model.policy.optimizer.load_state_dict(saved_optimizer)
        model.max_grad_norm = max_norm
        model.train()
        with torch.no_grad():
            after = model.policy.get_distribution(observations).distribution.probs
        kl = float((before * (torch.log(before + 1e-12) - torch.log(after + 1e-12))).sum(-1).mean())
        updates[label] = {"kl": kl, "mean_abs_probability_change": float((after - before).abs().mean())}
    model.max_grad_norm = configured
    model.policy.load_state_dict(saved_policy)
    summary = {key: statistics.median(n[key] for n in norms) for key in norms[0]}
    return {"minibatch_median": summary, "update": updates, "minibatches": norms}


def make_vec(environment, num_envs):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from baseline_dynamic_reach.environment import ReachEnv
    vec = DummyVecEnv([lambda: ReachEnv(**deepcopy(environment)) for _ in range(num_envs)])
    vec.seed(ROLLOUT_SEED)
    return vec


def table(results):
    lines = ["| Target | Policy gradient | Value gradient | Value share | Clip factor | Value loss | KL, configured clip | KL, no clip |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        m, u = r["minibatch_median"], r["update"]
        lines.append(f"| {name} | {m['policy']:.3f} | {m['value']:.3f} | {m['value_share']:.1%} | {m['clip_factor']:.3f} | "
                     f"{m['value_loss']:.3g} | {u['configured']['kl']:.4f} | {u['no_clip']['kl']:.4f} |")
    return lines


def run_targets(runs):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    results = {}
    for run in runs:
        sidecar = json.loads((MODELS_DIR / f"{run}.json").read_text(encoding="utf-8"))
        vec = make_vec(sidecar["environment"], sidecar["num_envs"])
        results[run] = probe(PPO.load(MODELS_DIR / f"{run}.zip", env=vec, device="cpu"), vec)
        vec.close()
        save(run, results[run])
    write(["# Update probe, trained runs", "", "Medians over the minibatches of one rollout (seed 123); "
           "KL: one full PPO update in memory. Clip factor 1 = the gradient is not clipped.", "", *table(results)],
          "updates_" + "_".join(runs) + ".md")


def start_target(arm, seed):
    import torch
    from stable_baselines3 import PPO
    from baseline_dynamic_reach.train import load_initial_weights
    torch.set_num_threads(1)
    config = arm_config(arm)
    vec = make_vec(config["environment"], config["num_envs"])
    model = PPO("MultiInputPolicy", vec, seed=seed, verbose=0, **config["ppo_config"])
    note = "fresh weights"
    if config.get("init_policy"):
        path, ratio = load_initial_weights(model, config, seed)
        note = f"{path.name}, value output x {ratio:g}"
    name = f"{arm} start (seed {seed}; {note})"
    result = probe(model, vec)
    vec.close()
    save(f"{arm}_start_seed{seed}", result)
    write([f"# Update probe, starting point of {arm}", "", "As for trained runs, but with the arm's settings and its "
           "initial weights, before any training (fresh Adam state, as in train.py).", "", *table({name: result})],
          f"start_{arm}_seed{seed}.md")


def evaluate_checkpoint(job):
    """Outcome counts of one checkpoint on the fixed evaluation episodes (deterministic actions)."""
    import torch
    from stable_baselines3 import PPO
    from baseline_dynamic_reach.config import EVAL_EPISODES, EVAL_SEED
    from baseline_dynamic_reach.environment import TARGET_SLOTS, ReachEnv
    torch.set_num_threads(1)
    run, path, environment = job
    model = PPO.load(path, device="cpu")
    env = ReachEnv(**deepcopy(environment))
    counts = {"success": 0, "collision": 0, "timeout": 0, "blr_success": 0, "blr_collision": 0, "blr_timeout": 0}
    try:
        for seed in range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(int(model.predict(obs, deterministic=True)[0]))
                done = terminated or truncated
            outcome = "success" if info["is_success"] else "collision" if info["is_collision"] else "timeout"
            counts[outcome] += 1
            if info["slot"] in TARGET_SLOTS:
                counts[f"blr_{outcome}"] += 1
    finally:
        env.close()
    steps = int(Path(path).stem.split("_")[-2])
    return {"run": run, "timesteps": steps, **counts}


def checkpoints(runs, processes):
    from multiprocessing import Pool
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jobs = []
    for run in runs:
        sidecar = MODELS_DIR / f"{run}.json"  # Written when training ends; the arm defines the same settings.
        environment = (json.loads(sidecar.read_text(encoding="utf-8"))["environment"] if sidecar.exists()
                       else arm_config(run.rsplit("_seed", 1)[0])["environment"])
        jobs += [(run, str(path), environment) for path in sorted((RESULTS_DIR / run / "checkpoints").glob("*.zip"))]
    with Pool(processes) as pool:
        rows = sorted(pool.map(evaluate_checkpoint, jobs), key=lambda r: (r["run"], r["timesteps"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "checkpoints_" + "_".join(runs)
    with (OUTPUT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for run in runs:
        mine = [r for r in rows if r["run"] == run]
        x = [r["timesteps"] / 1e6 for r in mine]
        axes[0].plot(x, [r["blr_success"] for r in mine], marker="o", ms=3, label=run)
        axes[1].plot(x, [r["blr_collision"] for r in mine], marker="o", ms=3, label=run)
    for ax, title in zip(axes, ("BACK/LEFT/RIGHT reached (of 138)", "BACK/LEFT/RIGHT crashes (of 138)")):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("training transitions (millions)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Deterministic policy at each checkpoint, fixed 200 evaluation episodes", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
    lines = [f"# Checkpoint evaluations: {', '.join(runs)}", "", f"![curves]({name}.png)", "",
             "| Run | Transitions | Reached | Crash | Timeout | BACK/LEFT/RIGHT reached / crash / timeout |", "|---|---:|---:|---:|---:|---|"]
    lines += [f"| {r['run']} | {r['timesteps']:,} | {r['success']} | {r['collision']} | {r['timeout']} | "
              f"{r['blr_success']}/{r['blr_collision']}/{r['blr_timeout']} |" for r in rows]
    write(lines, f"{name}.md")


def save(name, result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"updates_{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def write(lines, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("curves")
    runs = commands.add_parser("run")
    runs.add_argument("runs", nargs="+")
    start = commands.add_parser("start")
    start.add_argument("arm", choices=list(ARMS))
    start.add_argument("--seed", type=int, default=7)
    points = commands.add_parser("checkpoints")
    points.add_argument("runs", nargs="+")
    points.add_argument("--processes", type=int, default=9)
    args = parser.parse_args()
    if args.command == "curves":
        curves()
    elif args.command == "run":
        run_targets(args.runs)
    elif args.command == "checkpoints":
        checkpoints(args.runs, args.processes)
    else:
        start_target(args.arm, args.seed)


if __name__ == "__main__":
    main()
