#!/usr/bin/env python3

import dotenv
from evolve_anything.core import EvolutionRunner, EvolutionConfig
from evolve_anything.database import DatabaseConfig
from evolve_anything.launch import JobConfig

dotenv.load_dotenv()

job_config = JobConfig(
    eval_program_path="evaluate.py",
    use_sandbox=False,
    packages=["numpy"],
)

parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,  # chance to migrate program to random island
    island_elitism=True,  # Island elite is protected from migration
    **parent_config,
)

search_task_sys_msg = """You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Make sure that all circles are disjoint and lie inside the unit square.

Be creative and try to find a new solution better than the best known result."""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    minimize=False,
    num_generations=40,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "mistralai/devstral-medium",
        "x-ai/grok-4.1-fast",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_llm_models=["openai/gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="openai/text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["openai/gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    enable_web_research=True,
    web_research_llm_models=["google/gemini-3-flash-preview"],
    web_research_llm_kwargs=dict(temperatures=[0.5], max_tokens=16384),
    web_research_interval=1,
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_cpack_web",
    meta_rec_interval=10,
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
