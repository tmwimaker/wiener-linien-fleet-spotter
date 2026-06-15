"""
Wiener Linien Fleet Spotter — Training Run Comparison
=======================================================
Builds a table and chart comparing all training runs in model/runs/,
including the dataset size each run was trained on (reconstructed from the
git history of data/annotated/). Useful for showing the project's
progression — from a small dataset with weak results to a larger dataset
with better metrics.

Usage:
    python scripts/compare_runs.py
"""

import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "model" / "runs"

# Runs that completed fewer epochs than this are treated as aborted/interrupted
# restarts rather than real results, and are left out of the comparison.
MIN_EPOCHS = 3


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
    return {split: count_images(chosen, split) for split in ("train", "val", "test")}


def load_run(run_dir, commits):
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        return None
    last = df.iloc[-1]
    if int(last["epoch"]) < MIN_EPOCHS:
        return None

    args = {}
    if args_yaml.exists():
        with open(args_yaml) as f:
            args = yaml.safe_load(f)

    timestamped = args_yaml if args_yaml.exists() else run_dir
    run_time = datetime.fromtimestamp(timestamped.stat().st_mtime).astimezone()
    sizes = dataset_size_at(run_time, commits)

    return {
        "run": run_dir.name,
        "date": run_time.strftime("%Y-%m-%d %H:%M"),
        "model": args.get("model", "?"),
        "epochs": f"{int(last['epoch'])}/{args.get('epochs', '?')}",
        "train_imgs": sizes["train"],
        "val_imgs": sizes["val"],
        "test_imgs": sizes["test"],
        "total_imgs": sum(sizes.values()),
        "precision": round(float(last["metrics/precision(B)"]), 3),
        "recall": round(float(last["metrics/recall(B)"]), 3),
        "mAP50": round(float(last["metrics/mAP50(B)"]), 3),
        "mAP50-95": round(float(last["metrics/mAP50-95(B)"]), 3),
        "_sort_time": run_time,
    }


def main():
    commits = dataset_commits()

    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        row = load_run(run_dir, commits)
        if row:
            rows.append(row)

    if not rows:
        print("No runs with results.csv found in model/runs/.")
        return

    rows.sort(key=lambda r: r["_sort_time"])
    for r in rows:
        del r["_sort_time"]

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_csv = RUNS_DIR / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table: {out_csv}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax1.bar(x, df["total_imgs"], color="#A8DADC", label="Datensatzgröße (Bilder gesamt)")
    ax1.set_ylabel("Bilder im Datensatz")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["run"], rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, df["mAP50"], color="#E63946", marker="o", label="mAP50")
    ax2.plot(x, df["mAP50-95"], color="#1D3557", marker="o", label="mAP50-95")
    ax2.set_ylabel("mAP")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Trainings-Fortschritt: Datensatzgröße vs. Modellgüte")
    plt.tight_layout()

    out_png = RUNS_DIR / "comparison.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved chart: {out_png}")


if __name__ == "__main__":
    main()
