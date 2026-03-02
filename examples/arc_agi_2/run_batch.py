#!/usr/bin/env python3
"""
Batch runner for ARC-AGI 2 tasks.
Iterates over task JSON files and spawns per-task evolution runs.

Usage:
    # Run all tasks in a directory:
    python run_batch.py --data_dir data/training/

    # Resume interrupted batch (skips tasks with existing results):
    python run_batch.py --data_dir data/training/ --resume

    # Limit number of tasks:
    python run_batch.py --data_dir data/training/ --max_tasks 10
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime


def find_best_score(results_dir: str) -> float:
    """Find the best score achieved for a task by reading its evolution DB results."""
    metrics_pattern = os.path.join(results_dir, "**", "metrics.json")
    best_score = 0.0
    for metrics_path in glob.glob(metrics_pattern, recursive=True):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            score = metrics.get("combined_score", 0.0)
            best_score = max(best_score, score)
        except (json.JSONDecodeError, OSError):
            continue
    return best_score


def run_batch(args):
    """Run evolution on all tasks in the data directory."""
    data_dir = args.data_dir
    results_base = args.results_dir
    os.makedirs(results_base, exist_ok=True)

    # Find all task JSON files
    task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not task_files:
        print(f"No .json files found in {data_dir}")
        sys.exit(1)

    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    print(f"Found {len(task_files)} tasks")

    summary = []
    for i, task_path in enumerate(task_files):
        task_id = os.path.splitext(os.path.basename(task_path))[0]
        task_results_dir = os.path.join(results_base, task_id)

        # Skip if resuming and results exist
        if args.resume and os.path.exists(task_results_dir):
            best_score = find_best_score(task_results_dir)
            print(f"[{i+1}/{len(task_files)}] Skipping {task_id} (existing results, best={best_score:.2f})")
            summary.append({
                "task_id": task_id,
                "status": "skipped",
                "best_score": best_score,
            })
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(task_files)}] Running task: {task_id}")
        print(f"{'='*60}")

        env = os.environ.copy()
        env["ARC_TASK_PATH"] = os.path.abspath(task_path)

        cmd = [
            sys.executable, "run_evo.py",
            "--task_path", os.path.abspath(task_path),
            "--results_dir", task_results_dir,
            "--generations", str(args.generations),
        ]

        try:
            result = subprocess.run(
                cmd,
                env=env,
                timeout=args.timeout,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            status = "success" if result.returncode == 0 else f"failed (rc={result.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
            print(f"Task {task_id} timed out after {args.timeout}s")
        except Exception as e:
            status = f"error: {e}"
            print(f"Task {task_id} error: {e}")

        best_score = find_best_score(task_results_dir)
        summary.append({
            "task_id": task_id,
            "status": status,
            "best_score": best_score,
        })

        print(f"Task {task_id}: {status}, best_score={best_score:.2f}")

    # Write summary
    summary_path = os.path.join(results_base, "summary.json")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(task_files),
        "num_solved": sum(1 for s in summary if s["best_score"] == 1.0),
        "avg_score": sum(s["best_score"] for s in summary) / len(summary) if summary else 0.0,
        "tasks": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Batch complete: {summary_data['num_solved']}/{len(task_files)} tasks solved")
    print(f"Average score: {summary_data['avg_score']:.3f}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for ARC-AGI 2 tasks")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing task JSON files")
    parser.add_argument("--results_dir", type=str, default="results_batch", help="Base results directory")
    parser.add_argument("--generations", type=int, default=30, help="Generations per task")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of tasks to run")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per task in seconds (default: 1h)")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with existing results")
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
