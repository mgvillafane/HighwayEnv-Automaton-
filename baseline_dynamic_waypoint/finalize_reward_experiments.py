"""Validate completed reward experiments, report sampling, and replay a success."""
import csv
import hashlib
import json
import os
from pathlib import Path

import imageio_ffmpeg
import torch

from baseline_dynamic_waypoint.evaluate import load_experiment
from baseline_dynamic_waypoint.record_committed import record_episode


ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT / "results" / "reward_guidance_comparison_seed7"


def main():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    audit = json.loads((COMPARISON / "stochastic_comparison.json").read_text(encoding="utf-8"))
    for name in ("source", "approach"):
        with (COMPARISON / f"stochastic_{name}_episodes.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 100
        assert [int(row["seed"]) for row in rows] == list(range(10000, 10100))
        stats = audit["models"][name]["summary"]
        for flag in ("success", "collision", "timeout"):
            assert sum(int(row[flag]) for row in rows) == stats[flag]
        path = Path(audit["models"][name]["model"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["models"][name]["sha256"]
    with (COMPARISON / "stochastic_approach_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # First successful FRONT case. Selection is disclosed, not an average-case claim.
    chosen = next(row for row in rows if row["slot"] == "front" and int(row["success"]))
    seed = int(chosen["seed"])
    model_path = Path(audit["models"]["approach"]["model"])
    video_dir = model_path.parent.parent / "results" / model_path.stem / "videos" / "front_sampled_success"
    if not (video_dir / "episode.json").exists():
        model, env, _ = load_experiment(model_path, "rgb_array")
        env.demo_label = "PPO sampled / selected success"
        try:
            episode = record_episode(model, env, model_path, seed, video_dir,
                                     deterministic=False, sampling_seed=seed + 1_000_000)
        finally:
            env.close()
    else:
        episode = json.loads((video_dir / "episode.json").read_text(encoding="utf-8"))
    assert episode["success"] and not episode["collision"]
    assert episode["deterministic"] is False
    assert episode["policy_decisions"] == int(chosen["policy_decisions"])
    assert episode["physics_steps"] == int(chosen["physics_steps"])
    assert abs(episode["reward"] - float(chosen["reward"])) < 1e-8
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video_dir / "episode.mp4"))
    assert frames == episode["physics_steps"] + 21
    starts = [decision["action"] for decision in episode["decisions"] if decision["started"]]
    assert 0 in starts and 2 in starts
    (video_dir / "SELECTION.md").write_text(
        "# Sampled PPO FRONT success\n\n"
        f"First successful FRONT episode in the seeded stochastic evaluation: seed {seed}. "
        "This is a selected successful case, not a representative success-rate estimate.\n\n"
        f"Direct saved PPO, deterministic=False, torch seed={seed + 1_000_000}; no reflection, "
        "action transformation, or scripted steering. All internal physics ticks are recorded. "
        f"Outcome: success in {episode['seconds']:.1f} s. Initiated lane actions: {starts}.\n",
        encoding="utf-8")

    readme = COMPARISON / "README.md"
    text = readme.read_text(encoding="utf-8")
    opening = "Two PPO runs each trained for **200,000 additional policy decisions**, initialized from the same saved actor and optimizer."
    replacement = ("Four experiments each trained for **200,000 policy decisions**, totaling **800,000 new decisions**. "
                   "The first two form a matched reward comparison initialized from the same saved actor and optimizer. "
                   "The third starts fresh, and the fourth continues that fresh agent for a 400,000-decision lifetime.")
    text = text.replace(opening, replacement)
    marker = "\n## Final conclusion and sampled-action check\n"
    if marker in text:
        text = text.split(marker)[0]
    source = audit["models"]["source"]["summary"]
    approach = audit["models"]["approach"]["summary"]
    text += marker
    text += "\nThe tested rewards have **not produced a reliable FRONT/RIGHT lane-changing solution**. "
    text += "No weaker model has replaced the previous checkpoint, and reward profiles remain explicit opt-ins.\n\n"
    text += "- In deterministic evaluation, the first guided continuation improves BACK performance but still has zero FRONT/RIGHT successes.\n"
    text += "- The fresh guided model's RIGHT successes (4/22 fixed and 13/46 fresh) all start in the goal lane. They do not demonstrate learned rightward lane changes.\n"
    text += "- The stronger follow-up fails all deterministic goals (0/100 fixed, 0/200 fresh), so it must not be promoted as a successful fix.\n\n"
    text += ("A separate seeded check uses **direct PPO categorical sampling**, with `deterministic=False`, "
             "on seeds 10000–10099. This is the behavior mode used during PPO training; no observations or actions are transformed. "
             "Torch is seeded with episode seed + 1,000,000 at each reset.\n\n")
    text += "| Sampled policy | Overall success | Collisions | FRONT successes | RIGHT successes |\n"
    text += "| --- | ---: | ---: | ---: | ---: |\n"
    for label, stats in (("Original", source), ("Stronger follow-up", approach)):
        text += (f"| {label} | {stats['success']}/100 | {stats['collision']}/100 | "
                 f"{stats['by_slot']['front']['success']}/18 | {stats['by_slot']['right']['success']}/22 |\n")
    text += ("\nThis shows some FRONT behavior under sampling, but no RIGHT-goal recovery in this check, "
             "and lower overall success with more collisions. Sampling does not turn the result into a reliable fix. "
             "One seeded trajectory per scenario is a finite check, not a multi-seed robustness claim.\n\n")
    text += (f"[Selected sampled FRONT overtake and merge](../{model_path.stem}/videos/front_sampled_success/episode.mp4) "
             f"uses seed {seed}, succeeds in {episode['seconds']:.1f} s, and initiates one LEFT and one RIGHT maneuver. "
             "The clip is explicitly success-selected. [Selection and reproduction details]"
             f"(../{model_path.stem}/videos/front_sampled_success/SELECTION.md).\n\n")
    text += ("[Sampled evaluation data](stochastic_comparison.json) · "
             "[FRONT deterministic comparison videos](videos/front_comparison.mp4) · "
             "[RIGHT deterministic comparison videos](videos/right_comparison.mp4) · "
             "[Live TensorBoard for all four runs](http://127.0.0.1:6010/live).\n\n"
             "Validation: eight new reward checks and thirteen existing single-car/committed-lane checks passed. "
             "Each model was checked on the same 100 fixed and 200 additional scenarios; source hashes, saved profiles, "
             "parent lineage, and non-reward environment settings were verified. Full-physics recordings match saved evaluations.\n")
    readme.write_text(text, encoding="utf-8")
    verification = {"stochastic_rows_verified": 200, "models_unchanged": True,
                    "selected_video_seed": seed, "selected_video_physics_steps": episode["physics_steps"],
                    "selected_video_frames": frames, "selected_video_lane_actions": starts,
                    "reflection_or_action_transforms": False}
    (COMPARISON / "final_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
