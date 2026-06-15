"""
Wiener Linien Fleet Spotter — Dataset-Size Helpers
====================================================
Maps a point in time to the state of data/annotated/ in effect then,
reconstructed from its git history. Shared by scripts/compare_runs.py
(training-run comparison) and app/app.py (run selector grouping).
"""

import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "valid", "test")


def dataset_commits():
    """Commits that touched data/annotated/, as (sha, datetime) sorted oldest first."""
    out = subprocess.run(
        ["git", "log", "--format=%H %aI", "--", "data/annotated"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    commits = [(sha, datetime.fromisoformat(date)) for sha, date in (line.split(" ", 1) for line in out)]
    return sorted(commits, key=lambda c: c[1])


def count_images(commit, split):
    """Number of files under data/annotated/<split>/images/ at the given commit."""
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, f"data/annotated/{split}/images"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return len(out)


def dataset_size_at(run_time, commits):
    """Image counts per split for the dataset state in effect at `run_time`."""
    chosen = commits[0][0]
    for sha, date in commits:
        if date <= run_time:
            chosen = sha
        else:
            break
    return {split: count_images(chosen, split) for split in SPLITS}
