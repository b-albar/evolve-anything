#!/usr/bin/env python3

import sys
import os

# Add the project root to sys.path to allow importing evolve_anything
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import dotenv
from evolve_anything.core import EvolutionRunner, EvolutionConfig
from evolve_anything.database import DatabaseConfig
from evolve_anything.launch import JobConfig

dotenv.load_dotenv()

job_config = JobConfig(
    eval_program_path="evaluate.py",
    use_sandbox=False,
    packages=[],
)

parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

search_task_sys_msg = """You are an expert performance engineer specializing in VLIW SIMD architecture optimization.
Your goal is to minimize the cycle count of a kernel performing tree operations defined in `KernelBuilder.build_kernel`.

The kernel operates on a simulator defined in `problem.py`.
The instructions supported are `alu`, `valu` (vector ALU), `load`, `store`, `flow`.

# Architecture Details
SLOT_LIMITS per cycle:
- alu: 12
- valu: 6
- load: 2
- store: 2
- flow: 1

VLEN (Vector Length) = 8
Scratch Memory: 1536 integers (serves as registers/cache).
Memory: Flat 32-bit integer array.

# Instructions
Effects of instructions take effect at the end of the cycle.
All arguments are scratch memory addresses unless specified otherwise.

## ALU (Scalar)
Op types: +, -, *, //, cdiv (ceil div), ^, &, |, <<, >>, %, <, ==
Format: `(op, dest, a1, a2)`
- dest = a1 op a2

## VALU (Vector)
Operates on VLEN (8) elements.
- `("vbroadcast", dest, src)`: Broadcasts scalar at `src` to `dest[0..7]`.
- `("multiply_add", dest, a, b, c)`: `dest[i] = (a[i] * b[i]) + c[i]`.
- `(op, dest, a1, a2)`: standard ALU op applied element-wise.

## LOAD
- `("load", dest, addr)`: `dest = mem[addr]`.
- `("load_offset", dest, addr, offset)`: `dest[offset] = mem[addr[offset]]`.
- `("vload", dest, addr)`: `dest[0..7] = mem[addr..addr+7]`. `addr` is scalar base address.
- `("const", dest, val)`: `dest = val` (immediate value).

## STORE
- `("store", addr, src)`: `mem[addr] = src`.
- `("vstore", addr, src)`: `mem[addr..addr+7] = src[0..7]`. `addr` is scalar base address.

## FLOW
- `("select", dest, cond, a, b)`: `dest = a if cond != 0 else b`.
- `("vselect", dest, cond, a, b)`: Element-wise select.
- `("add_imm", dest, a, imm)`: `dest = a + imm`.
- `("halt",)`: Stop execution.
- `("jump", addr)`: `pc = addr`. (Absolute jump, immediate address).
- `("cond_jump", cond, addr)`: `if cond != 0: pc = addr`.
- `("cond_jump_rel", cond, offset)`: `if cond != 0: pc += offset`.

Guidance:
1. Vectorization is key. Use `vload`, `vstore`, `valu`.
2. Instruction Level Parallelism (ILP): Reorder instructions to fill all execution slots in a cycle.
3. Optimize the hash function implementation using vector ops.
4. Unroll loops if beneficial.
5. Use the scratchpad memory effectively to cache values.
6. **BE CAREFUL WITH MEMORY ACCESS**: Ensure you are not reading/writing out of bounds. `vload`/`vstore` access 8 contiguous elements.

CRITICAL: Distinguish between ENGINE names and OPERATION names.
- Valid ENGINE names: "alu", "valu", "load", "store", "flow".
- `vload` is an OPERATION for the "load" ENGINE.
- `vstore` is an OPERATION for the "store" ENGINE.

Use `body.append(("load", ("vload", ...)))` NOT `body.append(("vload", ("vload", ...)))`.
Use `body.append(("store", ("vstore", ...)))` NOT `body.append(("vstore", ("vstore", ...)))`.

The target function to optimize is `KernelBuilder.build_kernel`.
Do NOT modify the behavior such that it fails correctness tests.
The system validates correctness against a reference implementation.
"""

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.6, 0.4],
    minimize=True,  # Minimize cycles
    num_generations=50,
    max_parallel_jobs=2,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        # "google/gemini-3-flash-preview",
        # "moonshotai/kimi-k2.5",
        # "qwen/qwen3-coder",
        # "z-ai/glm-4.7",
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "mistralai/devstral-medium",
        "x-ai/grok-4.1-fast",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.7],
        max_tokens=4096,
    ),
    meta_llm_models=["z-ai/glm-4.7"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=2048),
    # novelties
    novelty_llm_models=["z-ai/glm-4.7"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=2048),
    init_program_path="initial.py",
    results_dir="results_anthropic",
    meta_rec_interval=5,
    validation_command=f"python {os.path.abspath('run_tests.py')} {{filename}}",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
