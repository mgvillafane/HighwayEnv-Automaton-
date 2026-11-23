"""Which code produced a model or an evaluation: recorded in every Step 8 sidecar and evaluation.

The git commit alone is not enough when the working tree has uncommitted changes, so a SHA-256 over the
source files that training and evaluation import is recorded too. Two runs with the same source hash ran
the same code, committed or not.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("baseline_dynamic_reach", "baseline_dynamic_waypoint", "baseline_waypoint", "baseline_DQN")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@lru_cache(maxsize=1)
def code_provenance():
    files = sorted(path for folder in SOURCE_DIRS for path in (REPO / folder).rglob("*.py")
                   if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_head": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_files_modified": None if status is None else bool(status),
            "source_sha256": digest.hexdigest(), "source_files": len(files)}
