"""Verify strict goal selection and the live TensorBoard data for this run."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from baseline_dynamic_waypoint.single_car import make_single_car_env
from baseline_waypoint.plot_results import plot_training

RUN = Path(__file__).resolve().parent
URL = "http://127.0.0.1:6011"


def rows(name):
    with (RUN / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    metadata = json.loads((RUN / "run.json").read_text())
    assert metadata["status"] == "trained"
    assert metadata["initialization"] is None
    assert metadata["actual_timesteps"] == metadata["total_lifetime_timesteps"] == 200_000
    reference_dir = RUN.parent / "ppo_singlecar_front_right_only_scratch_seed7_200k"
    reference = json.loads((reference_dir / "run.json").read_text())
    assert metadata["environment"]["env_config"] == reference["environment"]["env_config"]
    assert metadata["environment"]["reward_weights"] == reference["environment"]["reward_weights"]
    assert metadata["ppo_config"] == reference["ppo_config"]
    for key in ("seed", "requested_timesteps", "num_envs", "reward_profile", "evaluation_first_seed"):
        assert metadata[key] == reference[key]
    current_goal, reference_goal = dict(metadata["environment"]["goal_config"]), dict(reference["environment"]["goal_config"])
    for goal in (current_goal, reference_goal):
        goal.pop("enabled_slots")
        goal.pop("sampled_slots")
    assert current_goal == reference_goal
    assert set(metadata["enabled_slots"]) == set(metadata["training_slots"]) == set(metadata["evaluation_slots"]) == {"back", "left"}
    training, scenarios = rows("training_episodes.csv"), rows("sampled_scenarios.csv")
    before, after = rows("evaluation_before_episodes.csv"), rows("evaluation_episodes.csv")
    assert len(training) == len(scenarios)
    assert len(before) == len(after) == 100
    for data in (training, scenarios, before, after):
        assert set(row["slot"] for row in data) == {"back", "left"}
    for previous, current in zip(before, after):
        for key in previous:
            if key.startswith("initial_") or key in ("seed", "slot", "anchor_initial_gap", "goal_lane"):
                assert previous[key] == current[key]
    env = make_single_car_env(**metadata["environment"])
    try:
        for slot in ("front", "right"):
            try:
                env.reset(seed=7, options={"slot": slot})
            except ValueError as error:
                assert "disabled" in str(error)
            else:
                raise AssertionError(f"Disabled goal accepted: {slot}")
    finally:
        env.close()
    model = PPO.load(RUN.parents[1] / "models" / f"{RUN.name}.zip", device="cpu")
    assert model.num_timesteps == 200_000
    assert model.policy.features_extractor.features_dim == 54
    events = EventAccumulator(str(RUN)).Reload()
    tags = events.Tags()["scalars"]
    assert not any(tag.startswith(("episodes/front/", "episodes/right/", "evaluation/front/", "evaluation/right/")) for tag in tags)
    wanted = [f"episodes/{slot}/{metric}_mean_100" for slot in ("back", "left") for metric in ("success", "reward", "collision", "timeout")]
    wanted += [f"evaluation/{slot}/{metric}_rate" for slot in ("back", "left") for metric in ("success", "collision", "timeout")]
    for tag in wanted:
        last_step = events.Scalars(tag)[-1].step
        assert 0 < last_step <= 200_000
        if tag.startswith("evaluation/"):
            assert last_step == 200_000
    assert events.Scalars("setup/enabled_front")[0].value == events.Scalars("setup/enabled_right")[0].value == 0
    assert events.Scalars("setup/enabled_back")[0].value == events.Scalars("setup/enabled_left")[0].value == 1
    with urllib.request.urlopen(URL + "/data/plugin/scalars/tags", timeout=5) as response:
        live_tags = json.load(response)
    assert len(live_tags) == 2
    live_run = next(name for name in live_tags if RUN.name in name)
    assert any(reference_dir.name in name for name in live_tags)
    live_metrics = {}
    for tag in wanted:
        assert tag in live_tags[live_run]
        query = urllib.parse.urlencode({"run": live_run, "tag": tag})
        with urllib.request.urlopen(URL + "/data/plugin/scalars/scalars?" + query, timeout=5) as response:
            points = json.load(response)
        assert points[-1][1] == events.Scalars(tag)[-1].step
        live_metrics[tag] = {"points": len(points), "last_step": points[-1][1], "last_value": points[-1][2]}
    (RUN / "tensorboard_metrics.json").write_text(json.dumps(live_metrics, indent=2) + "\n")
    plot_training(RUN)
    verification = {"from_scratch": True, "training_decisions": 200_000, "enabled_goals": ["back", "left"],
                    "disabled_goals": ["front", "right"], "explicit_disabled_resets_rejected": True,
                    "training_counts": dict(Counter(row["slot"] for row in scenarios)),
                    "evaluation_counts": dict(Counter(row["slot"] for row in after)),
                    "before_after_initial_scenarios_identical": True, "tensorboard_url": URL + "/live",
                    "live_tensorboard_run": live_run, "live_goal_metrics_verified_at_step": 200_000,
                    "front_right_training_and_evaluation_rows": 0,
                    "non_goal_settings_match_front_right_run": True}
    (RUN / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    result = json.loads((RUN / "evaluation.json").read_text())
    reference_result = json.loads((reference_dir / "evaluation.json").read_text())
    (RUN / "comparison.json").write_text(json.dumps({"left_back": result["ppo"], "front_right": reference_result["ppo"],
        "same_ppo_environment_reward_and_seed": True,
        "difference": "Enabled goals are LEFT/BACK instead of RIGHT/FRONT; each evaluation samples its enabled pair."}, indent=2) + "\n")
    lines = ["# BACK/LEFT only: training and evaluation", "", "Fresh PPO weights and optimizer; 54-input observation; exactly 200,000 training decisions.", "", "FRONT and RIGHT are disabled in the run's environment, including explicit reset requests. Training and evaluation both sample only BACK and LEFT.", "", f"Training time: {metadata['training_seconds'] / 60:.2f} minutes; physics ticks: {metadata['actual_physics_steps']:,}.", "", "| Goal | Training episodes | Training successes | Evaluation episodes | Evaluation success | Evaluation collisions |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for slot in ("back", "left"):
        train = [row for row in training if row["slot"] == slot]
        stats = result["ppo_by_slot"][slot]
        lines.append(f"| {slot.upper()} | {len(train)} | {sum(int(row['success']) for row in train)} | {stats['episodes']} | {stats['success_rate']:.1%} | {stats['collision_rate']:.1%} |")
    lines += ["", f"Evaluation success: {result['ppo']['success_rate']:.1%}; collisions: {result['ppo']['collision_rate']:.1%}; timeouts: {result['ppo']['timeout_rate']:.1%}. Evaluation uses deterministic actions on 100 held-out seeds (10000-10099). Training actions are sampled.", "", "## TensorBoard", "", "[Open the dashboard](" + URL + "/live). Only this run is displayed. Event reload: 5 seconds; browser refresh: 30 seconds.", "", "Training charts: `episodes/back/success_mean_100`, `episodes/left/success_mean_100`, the corresponding `reward_mean_100` and `collision_mean_100` tags, and `sampling/back/episodes` / `sampling/left/episodes`.", "", "Held-out charts: `evaluation/back/success_rate` and `evaluation/left/success_rate`, with before/after points at steps 0 and 200,000. All goal charts were verified through the running server's scalar API.", "", "The server uses an isolated Python environment with TensorBoard 2.21.0 and grpcio 1.76.0. The training environment's grpcio build was blocked by Windows Application Control; no system policy was changed.", "", "Restart the dashboard from the repository root:", "", "```powershell", ".\\.tensorboard-venv\\Scripts\\python.exe -u -B monitor_training.py --port 6011 --ppo-logdir baseline_dynamic_waypoint/results/" + RUN.name + " --latest 1", "```", "", "Repeat training with a fresh run name:", "", "```powershell", ".\\baseline_waypoint\\.venv\\Scripts\\python.exe -u -B -m baseline_dynamic_waypoint.train_single_car --committed-lanes --observation-schema lane_goal_v2 --enabled-slots left back --timesteps 200000 --num-envs 4 --rollout-steps 250 --seed 7 --reward-profile baseline --run-name YOUR_NEW_RUN --eval-episodes 100 --eval-seed 10000 --quiet", "```", "", "[Run settings](run.json) | [Evaluation](evaluation.json) | [Verification](verification.json) | [TensorBoard values](tensorboard_metrics.json) | [Training curves](training_curves.png)", ""]
    lines = [line.replace("Only this run is displayed.", "The LEFT/BACK and prior FRONT/RIGHT runs are displayed together.")
             .replace("--latest 1", "--latest 2").replace("--ppo-logdir baseline_dynamic_waypoint/results/" + RUN.name,
                                                                 "--ppo-logdir baseline_dynamic_waypoint/results") for line in lines]
    lines += ["", f"Reference FRONT/RIGHT run: success {reference_result['ppo']['success_rate']:.1%}, collisions {reference_result['ppo']['collision_rate']:.1%}. The same training settings and seed were used; goal assignments and therefore scenarios differ.", "", "[Configuration comparison and results](comparison.json)", ""]
    (RUN / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
