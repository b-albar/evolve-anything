#!/usr/bin/env python3
"""
Run evolution for a single ARC-AGI 2 task.

Usage:
    # Via environment variable (used by run_batch.py):
    ARC_TASK_PATH=data/task_001.json python run_evo.py

    # Via CLI argument:
    python run_evo.py --task_path data/task_001.json

    # With custom settings:
    python run_evo.py --task_path data/task_001.json --results_dir my_results --generations 50
"""

import os
import sys
import json
import argparse

import dotenv
from evolve_anything.core import EvolutionRunner, EvolutionConfig
from evolve_anything.database import DatabaseConfig
from evolve_anything.launch import JobConfig

dotenv.load_dotenv()

# ARC color names for LLM context
ARC_COLORS = {
    0: "black (background)",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}


def format_grid(grid: list) -> str:
    """Format a 2D grid as a readable string."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_task_sys_msg(task: dict) -> str:
    """Build the system message for the LLM from an ARC task."""
    train_examples = task.get("train", [])

    # Format training examples
    examples_str = ""
    for i, example in enumerate(train_examples):
        input_grid = example["input"]
        output_grid = example["output"]
        examples_str += f"\n### Training Example {i + 1}\n"
        examples_str += f"Input ({len(input_grid)}x{len(input_grid[0])}):\n"
        examples_str += f"```\n{format_grid(input_grid)}\n```\n"
        examples_str += f"Output ({len(output_grid)}x{len(output_grid[0])}):\n"
        examples_str += f"```\n{format_grid(output_grid)}\n```\n"

    # Color legend
    color_legend = "\n".join(f"  {k}: {v}" for k, v in ARC_COLORS.items())

    sys_msg = f"""You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.

## Task Description

You must discover the transformation rule that maps each input grid to its output grid.
Grids are 2D arrays of integers (0-9), where each integer represents a color:

{color_legend}

## Training Examples
{examples_str}

## Instructions

1. Carefully analyze ALL training examples to find the pattern
2. The transformation must work for ALL training examples, not just one
3. Common patterns include: rotations, reflections, translations, color substitutions, filling regions, extending patterns, counting, symmetry operations
4. Your `transform` function receives a numpy ndarray and must return a numpy ndarray
5. The output grid may have a different shape than the input
6. Focus on finding the EXACT rule, not an approximation

Write a `transform(input_grid: np.ndarray) -> np.ndarray` function that implements the discovered pattern."""

    return sys_msg


def main():
    parser = argparse.ArgumentParser(description="Run evolution for a single ARC-AGI 2 task")
    parser.add_argument("--task_path", type=str, default=None, help="Path to task JSON (overrides ARC_TASK_PATH env)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    args = parser.parse_args()

    # Resolve task path
    task_path = args.task_path or os.environ.get("ARC_TASK_PATH")
    if not task_path:
        print("Error: provide --task_path or set ARC_TASK_PATH environment variable")
        sys.exit(1)

    # Set env var so evaluate.py can read it
    os.environ["ARC_TASK_PATH"] = task_path

    # Load task
    with open(task_path, "r") as f:
        task = json.load(f)

    task_id = os.path.splitext(os.path.basename(task_path))[0]
    results_dir = args.results_dir or f"results_{task_id}"

    # Build configs
    job_config = JobConfig(
        eval_program_path="evaluate.py",
        use_sandbox=False,
        packages=["numpy"],
    )

    db_config = DatabaseConfig(
        db_path=os.path.join(results_dir, "evolution_db.sqlite"),
        num_islands=2,
        archive_size=30,
        elite_selection_ratio=0.3,
        num_archive_inspirations=3,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )

    task_sys_msg = build_task_sys_msg(task)

    evo_config = EvolutionConfig(
        task_sys_msg=task_sys_msg,
        patch_types=["diff", "full"],
        patch_type_probs=[0.6, 0.4],
        minimize=False,
        num_generations=args.generations,
        max_parallel_jobs=3,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=[
            "google/gemini-3-flash-preview",
            "openai/gpt-5-mini",
        ],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 1.0],
            reasoning_efforts=["auto", "low", "medium", "high"],
            max_tokens=16384,
        ),
        meta_llm_models=["openai/gpt-5-nano"],
        meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
        embedding_model="openai/text-embedding-3-small",
        code_embed_sim_threshold=0.995,
        novelty_llm_models=["openai/gpt-5-nano"],
        novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
        init_program_path="initial.py",
        results_dir=results_dir,
        meta_rec_interval=10,
    )

    print(f"Starting evolution for task: {task_id}")
    print(f"Training examples: {len(task.get('train', []))}")
    print(f"Results directory: {results_dir}")

    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
