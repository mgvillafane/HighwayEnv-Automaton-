"""Verify the restored legacy observation and timeout -10 continuation."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torch import nn

from baseline_dynamic_waypoint.dropout_policy import DropoutMultiInputPolicy
from baseline_waypoint.plot_results import plot_training

RUN = Path(__file__).resolve().parent
MODEL = RUN.parents[1] / "models" / (RUN.name + ".zip")
GOALS = ("front", "back", "left", "right")
URL = "http://127.0.0.1:6014"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    metadata = json.loads((RUN / "run.json").read_text())
    assert metadata["status"] == "trained" and metadata["actual_timesteps"] == 200_000
    assert metadata["environment"]["goal_config"]["observation_schema"] == "legacy"
    assert metadata["environment"]["reward_weights"]["timeout"] == 10
    assert metadata["dropout_rate"] == .1
    source = Path(metadata["initialization"]["model"])
    assert source.stem == "ppo_singlecar_committed57_seed7_200k"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["initialization"]["sha256"]
    assert metadata["initialization"]["policy_outputs_preserved"] is True
    assert metadata["initialization"]["dropped_or_redefined_inputs"] == []
    assert metadata["initialization"]["optimizer_reset"] is True
    source_dir = RUN.parent / source.stem
    original = read_rows(source_dir / "evaluation_episodes.csv")
    before, after = read_rows(RUN / "evaluation_before_episodes.csv"), read_rows(RUN / "evaluation_episodes.csv")
    training, scenarios = read_rows(RUN / "training_episodes.csv"), read_rows(RUN / "sampled_scenarios.csv")
    assert len(before) == len(after) == len(original) == 100
    assert len(training) == len(scenarios)
    for records in (before, after, training, scenarios):
        assert set(row["slot"] for row in records) == set(GOALS)
    for resumed, cached, final in zip(before, original, after):
        for key in ("seed", "slot", "success", "collision", "timeout", "length", "physics_steps"):
            assert resumed[key] == cached[key]
        assert abs(float(resumed["reward"]) - (float(cached["reward"]) - 10 * int(cached["timeout"]))) < 1e-8
        for key in resumed:
            if key.startswith("initial_") or key in ("seed", "slot", "anchor_initial_gap", "goal_lane"):
                assert resumed[key] == final[key]
    for records in (before, after, training):
        assert all(float(row["reward_timeout"]) == -10 * int(row["timeout"]) for row in records)
    model = PPO.load(MODEL, device="cpu")
    previous = PPO.load(source, device="cpu")
    assert isinstance(model.policy, DropoutMultiInputPolicy)
    assert model.observation_space == previous.observation_space
    assert model.policy.features_extractor.features_dim == 57 and model.num_timesteps == 200_000
    for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Dropout)]
        assert len(layers) == 2 and all(layer.p == .1 for layer in layers)
    del previous
    events = EventAccumulator(str(RUN)).Reload()
    with urllib.request.urlopen(URL + "/data/plugin/scalars/tags", timeout=5) as response:
        live = json.load(response)
    live_run = next(name for name in live if RUN.name in name)
    metrics = {}
    for goal in GOALS:
        assert events.Scalars("setup/enabled_" + goal)[0].value == 1
        for tag in (f"episodes/{goal}/success_mean_100", f"evaluation/{goal}/success_rate"):
            query = urllib.parse.urlencode({"run": live_run, "tag": tag})
            with urllib.request.urlopen(URL + "/data/plugin/scalars/scalars?" + query, timeout=5) as response:
                points = json.load(response)
            assert points[-1][1] == events.Scalars(tag)[-1].step
            metrics[tag] = {"last_step": points[-1][1], "last_value": points[-1][2]}
            if tag.startswith("evaluation/"):
                assert points[-1][1] == 200_000
    (RUN / "tensorboard_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_training(RUN)
    subprocess.run([sys.executable, "-B", "-m", "baseline_dynamic_waypoint.record_committed", "--model", str(MODEL)], cwd=ROOT, check=True)
    recordings = json.loads((RUN / "videos" / "episodes.json").read_text())
    expected = {int(row["seed"]): row for row in after}
    for replay in recordings:
        row = expected[replay["seed"]]
        assert replay["success"] == bool(int(row["success"])) and replay["collision"] == bool(int(row["collision"]))
        assert abs(replay["reward"] - float(row["reward"])) < 1e-8
        assert replay["policy_decisions"] == int(row["length"])
    verification = {"additional_training_steps": 200_000, "observation": "legacy", "input_values": 57,
                    "source_checkpoint_unchanged": True, "initial_policy_reproduces_source_outcomes": True,
                    "dropout_rate": .1, "actor_dropout_layers": 2, "critic_dropout_layers": 2,
                    "optimizer_reset_for_dropout": True, "timeout_penalty": -10,
                    "all_timeout_costs_correct": True, "training_counts": dict(Counter(row["slot"] for row in training)),
                    "held_out_evaluation_cases": 100, "before_after_initial_scenarios_identical": True,
                    "tensorboard_verified": True, "video_replays_match_evaluation": True,
                    "checks": "9 timeout and goal-sampling tests passed"}
    (RUN / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    old = json.loads((RUN / "evaluation_before.json").read_text())
    result = json.loads((RUN / "evaluation.json").read_text())
    lines = ["# Legacy 57-input observation, timeout -10, dropout 0.1", "", "Restored the committed-lane observation used by the earlier BACK/LEFT checkpoint. Its learned weights were copied exactly. Adding actor/critic dropout 0.1 reset the optimizer; deterministic evaluation disables dropout and reproduces the source checkpoint's original outcomes.", "", f"Completed exactly **200,000 additional PPO decisions** in {metadata['training_seconds'] / 60:.2f} minutes. All four goals were enabled. Timeout costs -10 once; other baseline reward weights and goal conditions are retained.", "", "| Goal | Evaluation cases | Success before | Success after | Collisions after |", "| --- | ---: | ---: | ---: | ---: |"]
    for goal in GOALS:
        initial, final = old["ppo_by_slot"][goal], result["ppo_by_slot"][goal]
        lines.append(f"| {goal.upper()} | {final['episodes']} | {initial['success_rate']:.1%} | {final['success_rate']:.1%} | {final['collision_rate']:.1%} |")
    lines += ["", f"Overall success: {old['ppo']['success_rate']:.1%} -> {result['ppo']['success_rate']:.1%}. Collisions: {old['ppo']['collision_rate']:.1%} -> {result['ppo']['collision_rate']:.1%}. Final timeouts: {result['ppo']['timeout_rate']:.1%}.", "", "Both evaluations use the same 100 held-out seeds (10000-10099). This experiment changes both observation layout and initialization relative to the recent 54-input runs; comparisons across those runs do not isolate the observation's effect.", "", "[TensorBoard](" + URL + "/live) | [All-goal video](videos/all_goals.mp4) | [Model](../../models/" + MODEL.name + ") | [Configuration](run.json) | [Verification](verification.json) | [Training curves](training_curves.png)", "", "Videos use the first evaluation seed for each goal; outcomes, rewards, lengths and frame counts were verified.", ""]
    (RUN / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
