#!/usr/bin/env python3
import os
import sys
import argparse
import importlib.util
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


def run_tests(program_path: str):
    """Run a quick correctness test (validation)."""
    print(f"Validating: {program_path}")

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
            # Fallback for when running from a different directory
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from problem import (
                Machine,
                build_mem_image,
                reference_kernel2,
                Tree,
                Input,
                N_CORES,
            )

        # Test parameters for correctness
        forest_height = 10
        rounds = 16
        batch_size = 256
        BASELINE = 147734

        # Single trial for validation
        random.seed(42)

        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)

        kb = module.KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

        value_trace = {}
        machine = Machine(
            mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace
        )
        machine.enable_pause = True  # Enable pause to match reference yields
        machine.enable_debug = False

        for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
            machine.run()

            inp_values_p = ref_mem[6]
            output_correct = (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            )

            if not output_correct:
                raise RuntimeError(
                    f"Correctness test failed on round {i}: Output values do not match reference."
                )

            # Optional: check indices too if desired, matching perf_takehome
            inp_indices_p = ref_mem[5]
            indices_correct = (
                machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
                == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            )

            if not indices_correct:
                raise RuntimeError(
                    f"Correctness test failed on round {i}: Input indices do not match reference."
                )

        print(f"CYCLES: {machine.cycle}")
        print(f"Speedup over baseline: {BASELINE / machine.cycle}")
        print("Validation SUCCESS.")
        sys.exit(0)

    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        # Also print the exception message to stdout for the summary
        print(f"Validation ERROR: {sys.exc_info()[1]}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quick validation tests")
    parser.add_argument("program_path", type=str, help="Path to the program to test")
    args = parser.parse_args()
    run_tests(args.program_path)
