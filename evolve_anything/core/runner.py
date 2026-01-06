import shutil
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import yaml
from rich.logging import RichHandler
from rich.console import Console
from typing import List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from evolve_anything.launch import JobScheduler, JobConfig, JobHandle
from evolve_anything.database import ProgramDatabase, DatabaseConfig, Program
from evolve_anything.llm import (
    LLMClient,
    extract_between,
    EmbeddingClient,
    BanditBase,
    AsymmetricUCB,
)
from evolve_anything.edit import (
    apply_diff_patch,
    apply_full_patch,
    summarize_diff,
    redact_immutable,
)
from evolve_anything.core.sampler import PromptSampler
from evolve_anything.core.summarizer import MetaSummarizer
from evolve_anything.core.novelty_judge import NoveltyJudge
from evolve_anything.core.patch_metadata import PatchMetadata
from evolve_anything.utils.format_config import (
    DEFAULT_FORMAT,
    LANGUAGE_EXTENSIONS,
    get_comment_char,
)

from line_profiler import profile

FOLDER_PREFIX = "gen"
LOG_SEPARATOR = "=" * 80


def _fmt_score(x) -> str:
    """Format a score value for display."""
    return f"{x:.4f}" if isinstance(x, (float, int)) else "None"


@dataclass
class EvolutionConfig:
    """Configuration for the evolution runner.

    Attributes:
        task_sys_msg: System message describing the optimization task for the LLM.
        patch_types: Types of patches to use ("diff", "full", "cross").
        patch_type_probs: Probability weights for each patch type.
        num_generations: Total number of generations to evolve.
        max_parallel_jobs: Maximum number of concurrent evaluation jobs.
        max_patch_resamples: Max retries with different parent/inspirations per generation.
        max_patch_attempts: Max LLM query retries per patch generation.
        job_type: Execution backend ("local" or "slurm").
        language: Programming language of the code being evolved.
        minimize: If True, minimize combined_score; if False (default), maximize.
        llm_models: List of LLM model names for code generation (OpenRouter format).
        llm_dynamic_selection: Bandit algorithm for dynamic model selection.
        llm_dynamic_selection_kwargs: Additional kwargs for the bandit algorithm.
        llm_kwargs: Additional kwargs passed to LLM queries.
        meta_rec_interval: Programs between meta-analysis updates (None to disable).
        meta_llm_models: LLM models for meta-analysis (defaults to llm_models).
        meta_llm_kwargs: Additional kwargs for meta LLM queries.
        meta_max_recommendations: Max recommendations to generate per meta update.
        embedding_model: Model for code embeddings (None to disable).
        init_program_path: Path to initial program file (None to generate).
        results_dir: Directory for results (auto-generated if None).
        max_novelty_attempts: Max retries when novelty check fails.
        code_embed_sim_threshold: Similarity threshold for novelty rejection (0-1).
        novelty_llm_models: LLM models for novelty assessment.
        novelty_llm_kwargs: Additional kwargs for novelty LLM queries.
        use_text_feedback: Whether to include text feedback in prompts.
    """

    task_sys_msg: Optional[str] = None
    patch_types: List[str] = field(default_factory=lambda: ["diff"])
    patch_type_probs: List[float] = field(default_factory=lambda: [1.0])
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    max_patch_attempts: int = 5
    job_type: str = "local"
    language: str = "python"
    minimize: bool = False  # If True, minimize score; if False, maximize
    llm_models: List[str] = field(
        default_factory=lambda: ["google/gemini-2.5-flash-lite"]
    )
    llm_dynamic_selection: Optional[Union[str, BanditBase]] = None
    llm_dynamic_selection_kwargs: dict = field(default_factory=lambda: {})
    llm_kwargs: dict = field(default_factory=lambda: {})
    meta_rec_interval: Optional[int] = None
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    embedding_model: Optional[str] = None
    init_program_path: Optional[str] = "initial.py"
    results_dir: Optional[Path] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 1.0
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False
    enable_web_research: bool = True
    web_research_llm_models: Optional[List[str]] = None
    web_research_llm_kwargs: dict = field(default_factory=lambda: {})
    web_research_interval: int = 1
    web_research_allow_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "arxiv.org",
            "wikipedia.org",
            "scholar.google.com",
            "github.com",
            "medium.com",
            "towardsdatascience.com",
            "stackoverflow.com",
            "paperswithcode.com",
        ]
    )


@dataclass
class RunningJob:
    """Represents a running job in the queue."""

    job_id: JobHandle
    exec_fname: str
    results_dir: str
    start_time: float
    generation: int
    parent_id: Optional[str]
    archive_insp_ids: List[str]
    top_k_insp_ids: List[str]
    code_diff: Optional[str]
    meta_patch_data: Optional[PatchMetadata]
    code_embedding: List[float] = field(default_factory=list)


# Set up logging
logger = logging.getLogger(__name__)


class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.verbose = verbose

        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"results_{timestamp}")
        else:
            self.results_dir = Path(evo_config.results_dir)

        if self.verbose:
            # Create log file path in results directory
            log_filename = f"{self.results_dir}/evolution_run.log"
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Set up logging with both console and file handlers
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),  # Console output (clean)
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),  # File output (detailed)
                ],
            )

            # Also log the initial setup information
            logger.info(LOG_SEPARATOR)
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Log file: {log_filename}")
            logger.info(LOG_SEPARATOR)

        # Check if we are resuming a run
        resuming_run = False
        db_path = Path(f"{self.results_dir}/{db_config.db_path}")
        if self.evo_config.results_dir is not None and db_path.exists():
            resuming_run = True

        # Initialize LLM selection strategy
        if evo_config.llm_dynamic_selection is None:
            self.llm_selection = None
        elif isinstance(evo_config.llm_dynamic_selection, BanditBase):
            self.llm_selection = evo_config.llm_dynamic_selection
        elif (evo_config.llm_dynamic_selection.lower() == "ucb") or (
            evo_config.llm_dynamic_selection.lower() == "ucb1"
        ):
            self.llm_selection = AsymmetricUCB(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        else:
            raise ValueError("Invalid llm_dynamic_selection")

        # Initialize database and scheduler
        db_config.db_path = str(db_path)
        db_config.minimize = evo_config.minimize  # Sync minimize setting
        embedding_model_to_use = evo_config.embedding_model or "text-embedding-3-small"
        self.db = ProgramDatabase(
            config=db_config, embedding_model=embedding_model_to_use
        )
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,  # type: ignore
            verbose=verbose,
        )

        self.llm = LLMClient(
            model_names=evo_config.llm_models,
            model_selection=self.llm_selection,
            **evo_config.llm_kwargs,
            verbose=verbose,
        )

        # Optional clients - initialized only if models are specified
        self.embedding = (
            EmbeddingClient(model_name=evo_config.embedding_model, verbose=verbose)
            if evo_config.embedding_model
            else None
        )
        self.meta_llm = (
            LLMClient(
                model_names=evo_config.meta_llm_models,
                **evo_config.meta_llm_kwargs,
                verbose=verbose,
            )
            if evo_config.meta_llm_models
            else None
        )
        self.novelty_llm = (
            LLMClient(
                model_names=evo_config.novelty_llm_models,
                **evo_config.novelty_llm_kwargs,
                verbose=verbose,
            )
            if evo_config.novelty_llm_models
            else None
        )
        self.web_research_llm = (
            LLMClient(
                model_names=evo_config.web_research_llm_models,
                **evo_config.web_research_llm_kwargs,
                verbose=verbose,
            )
            if evo_config.web_research_llm_models
            else None
        )

        # Initialize PromptSampler for handling LLM code prompts
        self.prompt_sampler = PromptSampler(
            task_sys_msg=evo_config.task_sys_msg,
            language=evo_config.language,
            patch_types=evo_config.patch_types,
            patch_type_probs=evo_config.patch_type_probs,
            use_text_feedback=evo_config.use_text_feedback,
            minimize=evo_config.minimize,
        )

        # Initialize MetaSummarizer for meta-recommendations
        self.meta_summarizer = MetaSummarizer(
            meta_llm_client=self.meta_llm,
            language=evo_config.language,
            use_text_feedback=evo_config.use_text_feedback,
            max_recommendations=evo_config.meta_max_recommendations,
            results_dir=str(self.results_dir),
            enable_web_research=evo_config.enable_web_research,
            web_research_llm_client=self.web_research_llm,
            web_research_interval=evo_config.web_research_interval,
            web_research_allow_list=evo_config.web_research_allow_list,
        )

        # Initialize NoveltyJudge for novelty assessment
        self.novelty_judge = NoveltyJudge(
            novelty_llm_client=self.novelty_llm,
            language=evo_config.language,
            similarity_threshold=evo_config.code_embed_sim_threshold,
            max_novelty_attempts=evo_config.max_novelty_attempts,
        )

        # Initialize rich console for formatted output
        self.console = Console()

        # Set language extension
        if self.evo_config.language not in LANGUAGE_EXTENSIONS:
            raise ValueError(f"Language {self.evo_config.language} not supported")
        self.lang_ext = LANGUAGE_EXTENSIONS[self.evo_config.language]

        # Queue for managing parallel jobs
        self.running_jobs: List[RunningJob] = []
        self.best_program_id: Optional[str] = None
        self.next_generation_to_submit = 0

        # Thread pool for parallel patch generation
        # Use num_islands as the number of workers (one per island)
        num_islands = getattr(db_config, "num_islands", 1)
        self._num_parallel_generations = max(1, num_islands)
        self._generation_executor = ThreadPoolExecutor(
            max_workers=self._num_parallel_generations,
            thread_name_prefix="patch_gen",
        )
        self._generation_lock = threading.Lock()
        self._pending_generations: List[Future] = []

        if resuming_run:
            self.completed_generations = self.db.last_iteration + 1
            self.next_generation_to_submit = self.completed_generations
            logger.info(LOG_SEPARATOR)
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info(LOG_SEPARATOR)
            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {self.completed_generations} "
                "previously completed generations."
            )
            logger.info(LOG_SEPARATOR)
            self._update_best_solution()
            # Restore meta memory state when resuming
            self.meta_summarizer.restore(verbose=self.verbose)
        else:
            self.completed_generations = 0

        # Save experiment configuration to a YAML file
        self._save_experiment_config(evo_config, job_config, db_config)

    def _save_experiment_config(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
    ) -> None:
        """Save experiment configuration to a YAML file."""
        config_data = {
            "evolution_config": asdict(evo_config),
            "job_config": asdict(job_config),
            "database_config": asdict(db_config),
            "timestamp": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
        }

        config_path = Path(self.results_dir) / "experiment_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Experiment configuration saved to {config_path}")

    def run(self):
        """Run evolution with parallel job queue.

        This method now parallelizes patch generation across islands by:
        1. Submitting patch generation tasks to a thread pool
        2. Collecting results and submitting evaluation jobs
        3. Processing completed evaluation jobs
        """
        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        max_parallel_gens = self._num_parallel_generations
        logger.info(
            f"Starting evolution with {max_jobs} parallel eval jobs, "
            f"{max_parallel_gens} parallel patch generations (num_islands), "
            f"target: {target_gens} generations"
        )

        try:
            # First, run generation 0 sequentially to populate the database
            if self.completed_generations == 0 and target_gens > 0:
                logger.info(
                    "Running generation 0 sequentially to initialize database..."
                )
                self._run_generation_0()
                self.completed_generations = 1
                self.next_generation_to_submit = 1
                logger.info(f"Completed generation 0, total: 1/{target_gens}")

            # Now start parallel execution for remaining generations
            if self.completed_generations < target_gens:
                logger.info(
                    "Starting parallel patch generation and execution "
                    "for remaining generations..."
                )

                # Main loop: monitor jobs and submit new ones
                while (
                    self.completed_generations < target_gens
                    or len(self.running_jobs) > 0
                    or len(self._pending_generations) > 0
                ):
                    # 1. Collect completed patch generations and submit jobs
                    self._collect_completed_generations()

                    # 2. Check for completed evaluation jobs
                    completed_jobs = self._check_completed_jobs()

                    # 3. Process completed jobs
                    if completed_jobs:
                        for job in completed_jobs:
                            self._process_completed_job(job)

                        # Update completed generations count
                        self._update_completed_generations()

                        if self.verbose:
                            logger.info(
                                f"Processed {len(completed_jobs)} jobs. "
                                f"Total completed generations: "
                                f"{self.completed_generations}/{target_gens}"
                            )

                    # 4. Check if we've completed all generations
                    if self.completed_generations >= target_gens:
                        logger.info("All generations completed, exiting...")
                        break

                    # 5. Submit new patch generation tasks to fill the queue
                    # Limited to num_islands parallel generations
                    # Batch-sample from DB first (sequential, main thread)
                    # then submit all LLM tasks at once (parallel)
                    tasks_to_submit = []
                    while (
                        len(self._pending_generations) + len(tasks_to_submit)
                        < max_parallel_gens
                        and self.next_generation_to_submit < target_gens
                    ):
                        # Sample and prepare job context in main thread
                        job_context = self._prepare_generation_context()
                        if job_context:
                            tasks_to_submit.append(job_context)

                    # Now submit all LLM tasks to thread pool at once
                    for job_context in tasks_to_submit:
                        future = self._generation_executor.submit(
                            self._run_patch_in_thread, job_context
                        )
                        self._pending_generations.append(future)

                    if tasks_to_submit and self.verbose:
                        logger.info(
                            f"Submitted {len(tasks_to_submit)} parallel patch generation tasks"
                        )

                    # Wait a bit before checking again
                    time.sleep(0.5)  # Reduced from 2s for faster response

                # All jobs are now handled by the main loop above

        finally:
            # Ensure executor is properly shutdown
            self._generation_executor.shutdown(wait=True)

        # Perform final meta summary for any remaining unprocessed programs
        best_program = self.db.get_best_program()
        self.meta_summarizer.perform_final_summary(str(self.results_dir), best_program)

        # Save final meta memory state
        self.meta_summarizer.save()

        self.db.print_summary()
        logger.info(f"Evolution completed! {self.completed_generations} generations")
        logger.info(LOG_SEPARATOR)
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Evolution run ended at {end_time}")
        logger.info(LOG_SEPARATOR)

    @profile
    def generate_initial_program(self):
        """Generate initial program with LLM, with retries."""
        llm_kwargs = self.llm.get_kwargs()

        sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()
        msg_history = []
        total_input_tokens = 0
        total_output_tokens = 0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                llm_kwargs=llm_kwargs,
                msg_history=msg_history,
            )
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: LLM response content was None."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    if response and response.new_msg_history:
                        msg_history = response.new_msg_history
                    continue
                else:
                    break

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            code_start = DEFAULT_FORMAT.code_start_tag.format(
                language=self.evo_config.language
            )
            initial_code = extract_between(
                response.content,
                code_start,
                DEFAULT_FORMAT.code_end_tag,
                False,
            )

            if initial_code:
                patch_name = extract_between(
                    response.content,
                    DEFAULT_FORMAT.name_start_tag,
                    DEFAULT_FORMAT.name_end_tag,
                    False,
                )
                patch_description = extract_between(
                    response.content,
                    DEFAULT_FORMAT.description_start_tag,
                    DEFAULT_FORMAT.description_end_tag,
                    False,
                )
                comment_char = get_comment_char(self.evo_config.language)

                initial_code = (
                    f"{comment_char} {DEFAULT_FORMAT.evolve_block_start}\n"
                    f"{initial_code}\n"
                    f"{comment_char} {DEFAULT_FORMAT.evolve_block_end}\n"
                )

                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "SUCCESS."
                    )
                return (
                    initial_code,
                    patch_name,
                    patch_description,
                    total_input_tokens,
                    total_output_tokens,
                )
            else:  # code extraction failed
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: Could not extract code from response."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        f"Please make sure to enclose the code in "
                        f"`{code_start}`...`{DEFAULT_FORMAT.code_end_tag}` tags."
                    )
                    msg_history = response.new_msg_history
                else:  # last attempt
                    break

        raise ValueError(
            "LLM failed to generate a valid initial program after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    def _parse_results(self, results: Optional[dict]) -> dict:
        """Parse job results into standardized format.

        Args:
            results: Raw results dict from job execution.

        Returns:
            Dict with correct, combined_score, public_metrics, private_metrics,
            text_feedback, stdout_log, stderr_log.
        """
        if not results:
            return {
                "correct": False,
                "combined_score": 0.0,
                "public_metrics": {},
                "private_metrics": {},
                "text_feedback": "",
                "stdout_log": "",
                "stderr_log": "",
            }

        correct_dict = results.get("correct", {})
        metrics = results.get("metrics", {})

        return {
            "correct": correct_dict.get("correct", False),
            "combined_score": metrics.get("combined_score", 0.0),
            "public_metrics": metrics.get("public", {}),
            "private_metrics": metrics.get("private", {}),
            "text_feedback": metrics.get("text_feedback", ""),
            "stdout_log": results.get("stdout_log", ""),
            "stderr_log": results.get("stderr_log", ""),
        }

    def _add_program_to_db(
        self,
        exec_fname: str,
        results: Optional[dict],
        rtime: float,
        generation: int,
        code_embedding: List[float],
        parent_id: Optional[str] = None,
        archive_insp_ids: Optional[List[str]] = None,
        top_k_insp_ids: Optional[List[str]] = None,
        code_diff: Optional[str] = None,
        meta_patch_data: Optional[PatchMetadata] = None,
        set_baseline: bool = False,
    ) -> Program:
        """Create and add a program to the database.

        This shared helper handles:
        - Reading evaluated code
        - Parsing results
        - Creating Program object
        - Adding to database
        - Meta memory tracking
        - LLM selection update

        Args:
            exec_fname: Path to the executed code file.
            results: Job execution results dict.
            rtime: Runtime in seconds.
            generation: Generation number.
            code_embedding: Code embedding vector.
            parent_id: Parent program ID (None for gen 0).
            archive_insp_ids: Archive inspiration IDs.
            top_k_insp_ids: Top-k inspiration IDs.
            code_diff: Code diff from parent.
            meta_patch_data: Patch metadata.
            set_baseline: If True, set LLM selection baseline score (gen 0).

        Returns:
            The created Program object.
        """
        # Read the evaluated code
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for {exec_fname}. Error: {e}")
            evaluated_code = ""

        # Parse results
        parsed = self._parse_results(results)

        # Build metadata
        metadata = {
            "compute_time": rtime,
            **(meta_patch_data.to_dict() if meta_patch_data else {}),
            "stdout_log": parsed["stdout_log"],
            "stderr_log": parsed["stderr_log"],
        }

        # Create program object
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=parent_id,
            generation=generation,
            archive_inspiration_ids=archive_insp_ids or [],
            top_k_inspiration_ids=top_k_insp_ids or [],
            code_diff=code_diff,
            embedding=code_embedding,
            correct=parsed["correct"],
            combined_score=parsed["combined_score"],
            public_metrics=parsed["public_metrics"],
            private_metrics=parsed["private_metrics"],
            text_feedback=parsed["text_feedback"],
            metadata=metadata,
        )

        self.db.add(db_program, verbose=True)

        # Add to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs = self.meta_summarizer.update_meta_memory(best_program)
            if updated_recs:
                self.meta_summarizer.write_meta_output(str(self.results_dir))

        # Handle LLM selection
        if set_baseline and self.llm_selection is not None:
            # Generation 0: set baseline score
            self.llm_selection.set_baseline_score(
                db_program.combined_score if parsed["correct"] else 0.0,
            )
        elif self.llm_selection is not None:
            # Other generations: update model selection
            if "model_name" not in db_program.metadata:
                logger.warning(
                    "No model_name found in program metadata, "
                    "unable to update model selection algorithm."
                )
            else:
                parent = (
                    self.db.get(db_program.parent_id) if db_program.parent_id else None
                )
                baseline = parent.combined_score if parent else None
                reward = db_program.combined_score if parsed["correct"] else None
                model_name = db_program.metadata["model_name"]
                result = self.llm_selection.update(
                    arm=model_name,
                    reward=reward,
                    baseline=baseline,
                )
                if result and self.verbose:
                    normalized_score, baseline = result
                    logger.debug(
                        f"==> UPDATED LLM SELECTION: model: "
                        f"{model_name.split('/')[-1][-25:]}..., "
                        f"score: {_fmt_score(normalized_score)}, "
                        f"raw score: {_fmt_score(reward)}, "
                        f"baseline: {_fmt_score(baseline)}"
                    )
                    self.llm_selection.print_summary()

        self.db.save()
        self._update_best_solution()
        self.meta_summarizer.save()

        return db_program

    @profile
    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        initial_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        Path(initial_dir).mkdir(parents=True, exist_ok=True)
        exec_fname = f"{initial_dir}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        patch_name = "initial_program"
        patch_description = "Initial program from file."
        patch_type = "init"

        total_input_tokens = 0
        total_output_tokens = 0

        if self.evo_config.init_program_path:
            if self.verbose:
                logger.info(
                    f"Copying initial program from {self.evo_config.init_program_path}"
                )
            shutil.copy(self.evo_config.init_program_path, exec_fname)
        else:
            if self.verbose:
                logger.info(
                    "`init_program_path` not provided, "
                    "generating initial program with LLM..."
                )
            (
                initial_code,
                patch_name,
                patch_description,
                total_input_tokens,
                total_output_tokens,
            ) = self.generate_initial_program()
            with open(exec_fname, "w", encoding="utf-8") as f:
                f.write(initial_code)

            if self.verbose:
                logger.info(f"Initial program generated and saved to {exec_fname}")

        # Run the evaluation synchronously
        results, rtime = self.scheduler.run(
            exec_fname=exec_fname,
            results_dir=results_dir,
        )

        code_embedding = self.get_code_embedding(exec_fname)

        # Create patch metadata for generation 0
        meta_patch_data = PatchMetadata(
            patch_type=patch_type,
            patch_name=patch_name,
            patch_description=patch_description,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        # Use shared helper to add program to database
        self._add_program_to_db(
            exec_fname=exec_fname,
            results=results,
            rtime=rtime,
            generation=0,
            code_embedding=code_embedding,
            meta_patch_data=meta_patch_data,
            set_baseline=True,
        )

    def _prepare_generation_context(self) -> Optional[dict]:
        """Prepare context for a patch generation task (main thread).

        This method samples from the database in the main thread (required for
        SQLite thread safety) and prepares all data needed for LLM work.
        The actual thread pool submission happens in the caller.

        Returns:
            Job context dict, or None if no more generations to submit.
        """
        with self._generation_lock:
            current_gen = self.next_generation_to_submit
            if current_gen >= self.evo_config.num_generations:
                return None
            self.next_generation_to_submit += 1

        exec_fname = (
            f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        )
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get current meta-recommendations for this job (main thread - safe)
        meta_recs, meta_summary, meta_scratch, _ = self.meta_summarizer.get_current()

        # Sample parent and inspiration programs (main thread - DB access)
        (
            parent_program,
            archive_programs,
            top_k_programs,
        ) = self.db.sample(
            target_generation=current_gen,
            novelty_attempt=1,
            max_novelty_attempts=self.evo_config.max_novelty_attempts,
            resample_attempt=1,
            max_resample_attempts=self.evo_config.max_patch_resamples,
        )
        archive_insp_ids = [p.id for p in archive_programs]
        top_k_insp_ids = [p.id for p in top_k_programs]
        parent_id = parent_program.id

        if self.verbose:
            logger.info(f"Prepared generation {current_gen}: parent={parent_id[:8]}...")

        # Return job context (all data needed by worker thread)
        return {
            "current_gen": current_gen,
            "exec_fname": exec_fname,
            "results_dir": results_dir,
            "parent_program": parent_program,
            "archive_programs": archive_programs,
            "top_k_programs": top_k_programs,
            "parent_id": parent_id,
            "archive_insp_ids": archive_insp_ids,
            "top_k_insp_ids": top_k_insp_ids,
            "meta_recs": meta_recs,
            "meta_summary": meta_summary,
            "meta_scratch": meta_scratch,
        }

    def _run_patch_in_thread(self, job_context: dict) -> dict:
        """Run patch generation in thread pool (LLM calls only).

        This method only handles the LLM work - no database access.
        Database-dependent operations are handled in the main thread.

        Args:
            job_context: Dict with parent_program, inspirations, and metadata.

        Returns:
            Dict with patch results ready for post-processing.
        """
        current_gen = job_context["current_gen"]
        parent_program = job_context["parent_program"]
        archive_programs = job_context["archive_programs"]
        top_k_programs = job_context["top_k_programs"]

        # Run patch (LLM calls - thread safe)
        code_diff, meta_patch_data, num_applied_attempt = self.run_patch(
            parent_program,
            archive_programs,
            top_k_programs,
            current_gen,
            novelty_attempt=1,
            resample_attempt=1,
        )

        # Add meta-recommendations/summary/scratchpad to meta_patch_data
        meta_recs = job_context["meta_recs"]
        meta_summary = job_context["meta_summary"]
        meta_scratch = job_context["meta_scratch"]

        if meta_recs is not None and meta_patch_data is not None:
            meta_patch_data.meta_recommendations = meta_recs
            meta_patch_data.meta_summary = meta_summary
            meta_patch_data.meta_scratch_pad = meta_scratch

        # Return results for post-processing in main thread
        return {
            "exec_fname": job_context["exec_fname"],
            "results_dir": job_context["results_dir"],
            "generation": current_gen,
            "parent_id": job_context["parent_id"],
            "archive_insp_ids": job_context["archive_insp_ids"],
            "top_k_insp_ids": job_context["top_k_insp_ids"],
            "code_diff": code_diff,
            "meta_patch_data": meta_patch_data,
            "num_applied_attempt": num_applied_attempt,
        }

    def _collect_completed_generations(self):
        """Collect completed patch generation tasks and submit evaluation jobs.

        This method checks for completed futures from the thread pool and
        submits the corresponding evaluation jobs.
        """
        still_pending = []

        for future in self._pending_generations:
            if future.done():
                try:
                    job_data = future.result()

                    # Get code embedding in main thread (safe for any API calls)
                    code_embedding = self.get_code_embedding(job_data["exec_fname"])

                    # Submit the evaluation job
                    job_id = self.scheduler.submit_async(
                        job_data["exec_fname"], job_data["results_dir"]
                    )

                    # Create RunningJob and add to queue
                    running_job = RunningJob(
                        job_id=job_id,
                        exec_fname=job_data["exec_fname"],
                        results_dir=job_data["results_dir"],
                        start_time=time.time(),
                        generation=job_data["generation"],
                        parent_id=job_data["parent_id"],
                        archive_insp_ids=job_data["archive_insp_ids"],
                        top_k_insp_ids=job_data["top_k_insp_ids"],
                        code_diff=job_data["code_diff"],
                        meta_patch_data=job_data["meta_patch_data"],
                        code_embedding=code_embedding,
                    )
                    self.running_jobs.append(running_job)

                    if self.verbose:
                        logger.info(
                            f"Submitted evaluation job for generation "
                            f"{job_data['generation']}, "
                            f"queue size: {len(self.running_jobs)}"
                        )
                except Exception as e:
                    logger.error(f"Error in patch generation task: {e}")
            else:
                still_pending.append(future)

        self._pending_generations = still_pending

    def _update_completed_generations(self):
        """
        Update the count of completed generations from the database.
        A generation `g` is considered complete if all generations from 0..g
        have at least one program in the database. This ensures the count
        advances sequentially without gaps.
        """
        last_gen = self.db.last_iteration
        if last_gen == -1:
            self.completed_generations = 0
            return

        # Check for contiguous generations from 0 up to last_gen
        completed_up_to = 0
        for i in range(last_gen + 1):
            if self.db.get_programs_by_generation(i):
                completed_up_to = i + 1
            else:
                # Found a gap, so contiguous sequence is broken
                self.completed_generations = completed_up_to
                return

        self.completed_generations = completed_up_to

    def _check_completed_jobs(self) -> List[RunningJob]:
        """Check for completed jobs and return them."""
        completed = []
        still_running = []

        for job in self.running_jobs:
            is_running = self.scheduler.check_job_status(job)
            if not is_running:
                # Job completed
                if self.verbose:
                    logger.info(f"Job {job.job_id} completed!")
                completed.append(job)
            else:
                # Job still running
                still_running.append(job)

        self.running_jobs = still_running
        return completed

    def _process_completed_job(self, job: RunningJob):
        """Process a completed job and add results to database."""
        end_time = time.time()
        rtime = end_time - job.start_time

        # Get job results
        results = self.scheduler.get_job_results(job.job_id, job.results_dir)

        # Use shared helper to add program to database
        self._add_program_to_db(
            exec_fname=job.exec_fname,
            results=results,
            rtime=rtime,
            generation=job.generation,
            code_embedding=job.code_embedding,
            parent_id=job.parent_id,
            archive_insp_ids=job.archive_insp_ids,
            top_k_insp_ids=job.top_k_insp_ids,
            code_diff=job.code_diff,
            meta_patch_data=job.meta_patch_data,
        )

    def _update_best_solution(self):
        """Checks and updates the best program."""
        best_programs = self.db.get_top_programs(n=1, correct_only=True)
        if not best_programs:
            if self.verbose:
                logger.debug(
                    "No correct programs found yet, cannot determine best solution."
                )
            return

        best_program = best_programs[0]

        if best_program.id == self.best_program_id:
            return  # No change

        self.best_program_id = best_program.id

        source_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{best_program.generation}"
        best_dir = Path(self.results_dir) / "best"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_dir, best_dir)

        if self.verbose:
            logger.info(
                f"New best program found: gen {best_program.generation}, "
                f"id {best_program.id[:6]}... "
                f"Copied to {best_dir}"
            )

    def run_patch(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
    ) -> tuple[Optional[str], dict, int]:
        """Run patch generation for a specific generation."""
        max_patch_attempts = self.evo_config.max_patch_attempts
        if self.verbose:
            logger.info(
                f"Edit Cycle {generation} -> {generation + 1}, "
                f"Max Patch Attempts: {max_patch_attempts}"
            )
        # Get current meta recommendations
        meta_recs, _, _, _ = self.meta_summarizer.get_current()
        # Construct edit / code change message
        patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            meta_recommendations=meta_recs,
        )

        if patch_type in ["full", "cross"]:
            apply_patch = apply_full_patch
        elif patch_type == "diff":
            apply_patch = apply_diff_patch
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
            # apply_patch = apply_paper_patch
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        total_input_tokens = 0
        total_output_tokens = 0
        msg_history = []
        llm_kwargs = self.llm.get_kwargs()
        if self.llm_selection is not None:
            model_name = llm_kwargs["model_name"]
            self.llm_selection.update_submitted(model_name)
        code_diff = None  # Initialize code_diff
        num_applied_attempt = 0  # Initialize num_applied_attempt
        error_attempt = (
            "Max attempts reached without successful patch."  # Default error
        )
        patch_name = None
        patch_description = None
        output_path_attempt = None
        patch_txt_attempt = None
        patch_path = None
        diff_summary = {}

        for patch_attempt in range(max_patch_attempts):
            response = self.llm.query(
                msg=patch_msg,
                system_msg=patch_sys,
                msg_history=msg_history,
                llm_kwargs=llm_kwargs,
            )
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: LLM response content was None."
                    )
                # Prepare for next attempt or exit
                error_attempt = "LLM response content was None."
                num_applied_attempt = 0
                patch_txt_attempt = None
                if patch_attempt < max_patch_attempts - 1:
                    patch_msg = (
                        "The previous attempt to get an edit was not "
                        "successful because the LLM response was empty. "
                        "Try again."
                    )
                    if response:
                        msg_history = response.new_msg_history
                    continue
                else:  # Last attempt
                    break

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            patch_name = extract_between(
                response.content,
                DEFAULT_FORMAT.name_start_tag,
                DEFAULT_FORMAT.name_end_tag,
                False,
            )
            patch_description = extract_between(
                response.content,
                DEFAULT_FORMAT.description_start_tag,
                DEFAULT_FORMAT.description_end_tag,
                False,
            )

            # Apply the code patch (diff/full rewrite)
            (
                _,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{generation}",
                language=self.evo_config.language,
                verbose=False,
            )

            if error_attempt is None and num_applied_attempt > 0:
                if patch_path:  # Ensure patch_path is not None
                    diff_summary = summarize_diff(
                        str(patch_path)
                    )  # Convert Path to str
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} SUCCESS. "
                        f"Output: {output_path_attempt}, "
                        f"Patches Applied: {num_applied_attempt}."
                    )

                code_diff = patch_txt_attempt
                break  # Break from patch attempts
            else:
                error_str = (
                    str(error_attempt) if error_attempt else "No changes applied."
                )
                patch_msg = (
                    "The previous edit was not successful."
                    + " This was the error message: \n\n"
                    + error_str
                    + "\n\n Try again."
                )
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: '{error_str}', "
                        f"Patches Applied: {num_applied_attempt}."
                    )
                msg_history = response.new_msg_history
                code_diff = None
                if patch_attempt == max_patch_attempts - 1:  # Last attempt failed
                    # error_attempt is already set from apply_patch or default
                    pass

        # Only consider the diff summary for the original source file
        original_filename = f"original.{self.lang_ext}"
        if original_filename in diff_summary:
            diff_summary = diff_summary[original_filename]

        meta_edit_data = PatchMetadata(
            patch_type=patch_type,
            patch_name=patch_name,
            patch_description=patch_description,
            num_applied=num_applied_attempt,
            error_attempt=error_attempt,
            novelty_attempt=novelty_attempt,
            resample_attempt=resample_attempt,
            patch_attempt=patch_attempt + 1,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            llm_kwargs=llm_kwargs,
            llm_result=response.to_dict() if response else None,
            diff_summary=diff_summary if isinstance(diff_summary, dict) else {},
        )
        if self.verbose and num_applied_attempt > 0:
            meta_edit_data.print_table(
                generation=generation,
                num_generations=self.evo_config.num_generations,
                max_novelty_attempts=self.evo_config.max_novelty_attempts,
                max_patch_resamples=self.evo_config.max_patch_resamples,
                max_patch_attempts=self.evo_config.max_patch_attempts,
                console=self.console,
            )
        return code_diff, meta_edit_data, num_applied_attempt

    def get_code_embedding(self, exec_fname: str) -> List[float]:
        """Get the embedding of the code."""
        # Read the evaluated code
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""
        if evaluated_code != "":
            # Get the embedding of the initial program
            try:
                if self.embedding is not None:
                    redacted_code = redact_immutable(evaluated_code, no_state=True)
                    if self.verbose:
                        logger.debug(
                            "=> EMBED: Code length - "
                            f"Original: {len(evaluated_code)} - "
                            f"Redacted: {len(redacted_code)}"
                        )

                    embedding_result, _ = self.embedding.get_embedding(redacted_code)
                else:
                    if self.verbose:
                        logger.debug("=> EMBED: No embedding model configured.")
                    embedding_result = []
                code_embedding = cast(List[float], embedding_result)
            except Exception as e:
                logger.warning(f"Could not embed code for job {exec_fname}. Error: {e}")
                code_embedding = []
        else:
            code_embedding = []
        return code_embedding
