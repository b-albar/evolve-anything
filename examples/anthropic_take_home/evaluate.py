#!/usr/bin/env python3
import os
import json
import argparse
import importlib.util
import sys
from typing import Dict, Any, Optional


def load_python_module(program_path: str):
    """Dynamically load a Python module from a file path."""
    module_name = os.path.basename(program_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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


def evaluate(program_path: str, results_dir: str):
    """Run the performance takehome evaluation."""
    print(f"Evaluating: {program_path}")
    print(f"Results dir: {results_dir}")

    try:
        # Load the program
        module = load_python_module(program_path)

        try:
            from problem import (
                Machine,
                build_mem_image,
                reference_kernel2,
                Tree,
                Input,
                N_CORES,
            )
        except ImportError:
            # Fallback if running from a different directory
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from problem import (
                Machine,
                build_mem_image,
                reference_kernel2,
                Tree,
                Input,
                N_CORES,
            )

        # Run correctness check
        # We use a similar logic to do_kernel_test in submission_tests.py but adapted
        # to use the KernelBuilder from the loaded module

        # Test parameters for correctness
        forest_height = 10
        rounds = 16
        batch_size = 256

        # Setup problem (deterministic seed for correctness)
        import random

        # Ensure we don't mess up global state if the module changes it,
        # but here we just need a consistent test case.
        # Note: submission_tests.py says "Note the random generator is not seeded here" for do_kernel_test
        # but CorrectnessTests usually relies on some consistency or multiple runs.
        # We will use a fixed seed for the data generation to be consistent.
        random.seed(42)

        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)

        # Build kernel using the user's module
        kb = module.KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

        # Run simulation
        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Validate against reference
        for ref_mem in reference_kernel2(mem):
            pass

        inp_values_p = ref_mem[6]
        output_correct = (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        )

        if not output_correct:
            raise RuntimeError(
                "Correctness test failed: Output values do not match reference."
            )

        # If correct, we can get the cycle count from this run or run a separate performance run.
        # The prompt asks to "minimize the cycle count".
        # The takehome uses the same parameters for test_kernel_cycles: 10, 16, 256.
        # So we can just use the cycles from this run!
        cycles = machine.cycle

        # Compute metrics
        metrics = {
            "combined_score": float(cycles),
            "public": {
                "cycles": float(cycles),
            },
            "private": {
                "baseline_speedup": 147734.0 / float(cycles) if cycles > 0 else 0.0
            },
        }

        save_results(results_dir, metrics, correct=True)
        print(f"SUCCESS: Cycles = {cycles}")

    except Exception as e:
        print(f"ERROR: {e}")
        # Use a very high score for failures since we are minimizing
        metrics = {"combined_score": 1e9, "error": str(e)}
        save_results(results_dir, metrics, correct=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anthropic Performance Evaluator")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args.program_path, args.results_dir)
