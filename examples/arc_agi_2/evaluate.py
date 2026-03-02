#!/usr/bin/env python3
"""
Standalone evaluator for ARC-AGI 2 tasks.
No external dependencies on evolve_anything package.

Task data is passed via the ARC_TASK_PATH environment variable,
which points to a JSON file with the task definition.
"""

import os
import json
import signal
import argparse
import importlib.util
import numpy as np
from typing import Optional, Dict, Any


# =============================================================================
# Generic utilities - same pattern as circle_packing/evaluate.py
# =============================================================================


def load_python_module(program_path: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
):
    """Save metrics.json and correct.json to the results directory."""
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=2)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# =============================================================================
# Timeout handling
# =============================================================================

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


# =============================================================================
# ARC-AGI evaluation logic
# =============================================================================


def load_task(task_path: str) -> dict:
    """Load an ARC-AGI task from a JSON file."""
    with open(task_path, "r") as f:
        return json.load(f)


def evaluate_on_examples(module, examples: list, timeout_sec: int = 5) -> dict:
    """Run the program's transform on a list of input/output examples.

    Returns:
        dict with keys: num_correct, num_total, errors (list of str)
    """
    num_correct = 0
    num_total = len(examples)
    errors = []

    for i, example in enumerate(examples):
        input_grid = example["input"]
        expected_output = np.array(example["output"], dtype=int)

        try:
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)

            try:
                predicted = module.run_transform(input_grid)
                predicted = np.array(predicted, dtype=int)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            if np.array_equal(predicted, expected_output):
                num_correct += 1
            else:
                errors.append(
                    f"Example {i}: shape {predicted.shape} vs {expected_output.shape}, "
                    f"mismatches={int(np.sum(predicted != expected_output)) if predicted.shape == expected_output.shape else 'N/A'}"
                )

        except TimeoutError:
            errors.append(f"Example {i}: timed out after {timeout_sec}s")
        except Exception as e:
            errors.append(f"Example {i}: {type(e).__name__}: {e}")

    return {
        "num_correct": num_correct,
        "num_total": num_total,
        "errors": errors,
    }


def evaluate(program_path: str, results_dir: str):
    """Run the ARC-AGI evaluation."""
    print(f"Evaluating: {program_path}")
    print(f"Results dir: {results_dir}")

    # Get task path from environment
    task_path = os.environ.get("ARC_TASK_PATH")
    if not task_path:
        error = "ARC_TASK_PATH environment variable not set"
        print(f"ERROR: {error}")
        save_results(results_dir, {"combined_score": 0.0, "error": error}, correct=False, error=error)
        return

    try:
        task = load_task(task_path)
        train_examples = task.get("train", [])

        if not train_examples:
            raise ValueError("Task has no training examples")

        # Load the evolved program
        module = load_python_module(program_path)
        if not hasattr(module, "run_transform"):
            raise AttributeError("Program must define 'run_transform' function")

        # Evaluate on training examples
        result = evaluate_on_examples(module, train_examples)
        score = result["num_correct"] / result["num_total"]

        # Also evaluate on test examples if available (for reporting only)
        test_examples = task.get("test", [])
        test_result = None
        if test_examples:
            test_result = evaluate_on_examples(module, test_examples)

        metrics = {
            "combined_score": score,
            "public": {
                "train_correct": result["num_correct"],
                "train_total": result["num_total"],
                "train_score": score,
            },
            "private": {
                "train_errors": result["errors"],
            },
        }

        if test_result:
            test_score = test_result["num_correct"] / test_result["num_total"]
            metrics["public"]["test_correct"] = test_result["num_correct"]
            metrics["public"]["test_total"] = test_result["num_total"]
            metrics["public"]["test_score"] = test_score
            metrics["private"]["test_errors"] = test_result["errors"]

        is_correct = score == 1.0
        error_msg = "; ".join(result["errors"]) if result["errors"] else None
        save_results(results_dir, metrics, is_correct, error_msg)

        if is_correct:
            print(f"SUCCESS: All {result['num_total']} training examples correct")
        else:
            print(f"PARTIAL: {result['num_correct']}/{result['num_total']} training examples correct")
            for err in result["errors"]:
                print(f"  - {err}")

    except Exception as e:
        print(f"ERROR: {e}")
        metrics = {"combined_score": 0.0, "error": str(e)}
        save_results(results_dir, metrics, correct=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI 2 evaluator")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args.program_path, args.results_dir)
