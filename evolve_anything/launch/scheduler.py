"""Simplified job scheduler with microsandbox and SLURM multi-node support.

This module provides a unified job scheduling interface that:
- Executes code in isolated microsandbox microVMs
- Supports local and SLURM-based multi-node execution
- Handles async job submission and monitoring
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evolve_anything.launch.sandbox import SandboxConfig, SandboxExecutor
from evolve_anything.utils import load_results, parse_time_to_seconds

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for job execution.

    Attributes:
        eval_program_path: Path to the evaluation program.
        extra_cmd_args: Additional command-line arguments.
        image: Microsandbox image to use.
        packages: List of Python packages to install in the sandbox.
        cpus: Number of CPUs per job.
        gpus: Number of GPUs per job.
        memory: Memory in MB.
        timeout: Job timeout (e.g., "01:00:00" or seconds).
        server_url: Microsandbox server URL (None for local).
        partition: SLURM partition (for SLURM jobs).
        nodes: Number of SLURM nodes (for multi-node jobs).
    """

    eval_program_path: str = "evaluate.py"
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)
    image: str = "microsandbox/python"
    packages: List[str] = field(default_factory=list)
    cpus: int = 1
    gpus: int = 0
    memory: int = 8192
    timeout: str = "00:10:00"
    server_url: Optional[str] = None
    partition: str = "gpu"
    nodes: int = 1
    use_sandbox: bool = False  # Disabled by default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def timeout_seconds(self) -> int:
        """Get timeout in seconds."""
        return parse_time_to_seconds(self.timeout)


@dataclass
class JobHandle:
    """Handle to a running job."""

    job_id: str
    job_type: str  # "local" or "slurm"
    results_dir: str
    start_time: float
    process: Optional[subprocess.Popen] = None

    def __str__(self) -> str:
        return f"JobHandle({self.job_type}:{self.job_id})"


# SLURM sbatch template for microsandbox execution
SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/job_log.out
#SBATCH --error={log_dir}/job_log.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
{gpu_directive}
{mem_directive}

echo "Job running on $(hostname) under Slurm job $SLURM_JOB_ID"
echo "Nodes allocated: $SLURM_JOB_NODELIST"

# Start microsandbox server if not already running
if ! pgrep -x "msb" > /dev/null; then
    echo "Starting microsandbox server..."
    msb server start --dev &
    sleep 2
fi

# Pull image if needed
msb pull {image} 2>/dev/null || true

# Run command in sandbox
{sandbox_cmd}

exit $?
"""


class JobScheduler:
    """Unified job scheduler with microsandbox support.

    Supports two job types:
    - "local": Run jobs locally in microsandbox
    - "slurm": Submit jobs to SLURM cluster with microsandbox on each node
    """

    def __init__(
        self,
        job_type: str = "local",
        config: Optional[JobConfig] = None,
        verbose: bool = False,
        max_workers: int = 4,
    ):
        """Initialize the job scheduler.

        Args:
            job_type: Either "local" or "slurm".
            config: Job configuration.
            verbose: Enable verbose logging.
            max_workers: Max concurrent workers for async operations.
        """
        if job_type not in ["local", "slurm"]:
            raise ValueError(
                f"Unknown job type: {job_type}. Must be 'local' or 'slurm'"
            )

        self.job_type = job_type
        self.config = config or JobConfig()
        self.verbose = verbose
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._local_jobs: Dict[str, JobHandle] = {}

    def _submit_local(self, exec_fname: str, results_dir: str) -> JobHandle:
        """Submit a local job using microsandbox or subprocess."""
        log_dir = Path(results_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        job_id = f"local-{uuid.uuid4().hex[:8]}"
        stdout_path = log_dir / "job_log.out"
        stderr_path = log_dir / "job_log.err"

        # Set up environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Try to use microsandbox if enabled, fall back to direct execution
        use_sandbox = False
        if self.config.use_sandbox:
            try:
                subprocess.run(
                    ["msb", "--version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                use_sandbox = True
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                if self.verbose:
                    logger.warning("Microsandbox not available, using direct execution")

        if use_sandbox:
            # Use microsandbox Python API - run in a separate thread via executor
            return self._submit_sandbox_job(
                job_id, exec_fname, results_dir, log_dir, env
            )
        else:
            # Direct execution
            with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
                cmd = [
                    "./" + self.config.eval_program_path,
                    "--program_path",
                    exec_fname,
                    "--results_dir",
                    results_dir,
                ]
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=env,
                    start_new_session=True,
                )

            handle = JobHandle(
                job_id=job_id,
                job_type="local",
                results_dir=results_dir,
                start_time=time.time(),
                process=process,
            )
            self._local_jobs[job_id] = handle

            if self.verbose:
                logger.info(f"Submitted local job {job_id} (PID: {process.pid})")

            return handle

    def _submit_sandbox_job(
        self,
        job_id: str,
        exec_fname: str,
        results_dir: str,
        log_dir: Path,
        env: Dict[str, str],
    ) -> JobHandle:
        """Submit a job using microsandbox Python API."""
        stdout_path = log_dir / "job_log.out"
        stderr_path = log_dir / "job_log.err"

        # Run sandbox job in background thread
        future = self.executor.submit(
            self._run_sandbox_job_sync,
            job_id,
            exec_fname,
            results_dir,
            stdout_path,
            stderr_path,
            env,
        )

        handle = JobHandle(
            job_id=job_id,
            job_type="local",
            results_dir=results_dir,
            start_time=time.time(),
            process=None,  # No subprocess for sandbox jobs
        )
        # Store future for status checking
        handle._sandbox_future = future  # type: ignore
        self._local_jobs[job_id] = handle

        if self.verbose:
            logger.info(f"Submitted sandbox job {job_id}")

        return handle

    def _run_sandbox_job_sync(
        self,
        job_id: str,
        exec_fname: str,
        results_dir: str,
        stdout_path: Path,
        stderr_path: Path,
        env: Dict[str, str],
    ) -> int:
        """Run a sandbox job synchronously (called from thread pool)."""
        return asyncio.run(
            self._run_sandbox_job_async(
                job_id,
                exec_fname,
                results_dir,
                stdout_path,
                stderr_path,
                env,
            )
        )

    async def _run_sandbox_job_async(
        self,
        job_id: str,
        exec_fname: str,
        results_dir: str,
        stdout_path: Path,
        stderr_path: Path,
        env: Dict[str, str],
    ) -> int:
        """Run a job in microsandbox asynchronously using SandboxExecutor."""
        try:
            # Create SandboxConfig from JobConfig
            sandbox_config = SandboxConfig(
                image=self.config.image,
                packages=self.config.packages,
                cpus=self.config.cpus,
                memory=self.config.memory,
                timeout=self.config.timeout_seconds,
            )

            executor = SandboxExecutor(sandbox_config, verbose=self.verbose)

            async with executor:
                # Read content from host files
                eval_content = Path(self.config.eval_program_path).read_text()
                exec_content = Path(exec_fname).read_text()

                # Write program into sandbox
                await executor.write_file("program.py", exec_content)
                await executor.write_file("evaluate.py", eval_content)

                # Make evaluate.py executable
                exit_code, stdout, stderr = await executor.run_command(
                    ["`which pip`", "install", "numpy"]
                )
                print(stdout)
                exit_code, stdout, stderr = await executor.run_command(
                    ["chmod", "+x", "evaluate.py"]
                )
                exit_code, stdout, stderr = await executor.run_command(
                    [
                        "./evaluate.py",
                        "--program_path",
                        "program.py",
                        "--results_dir",
                        "results",
                    ]
                )

                # Retrieve results
                if exit_code == 0:
                    _, metrics_json, _ = await executor.run_command(
                        ["cat", "results/metrics.json"]
                    )
                    _, correct_json, _ = await executor.run_command(
                        ["cat", "results/correct.json"]
                    )

            # Write logs to files
            with open(stdout_path, "w") as f:
                f.write(stdout)
            with open(stderr_path, "w") as f:
                f.write(stderr)

            return exit_code

        except Exception as e:
            with open(stderr_path, "w") as f:
                f.write(f"Sandbox error: {e}\n")
            return 1

    def _submit_slurm(self, results_dir: str, cmd: List[str]) -> JobHandle:
        """Submit a SLURM job with microsandbox."""
        log_dir = Path(results_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        job_name = f"msb-{uuid.uuid4().hex[:6]}"

        # Build sandbox command for SLURM
        cmd_str = " ".join(cmd)
        if self.config.packages:
            pkg_list = " ".join(self.config.packages)
            cmd_str = f"pip install -q {pkg_list} && {cmd_str}"
        sandbox_cmd = f'msb exe --image {self.config.image} -- bash -c "{cmd_str}"'

        gpu_directive = ""
        if self.config.gpus > 0:
            gpu_directive = f"#SBATCH --gres=gpu:{self.config.gpus}"

        mem_directive = f"#SBATCH --mem={self.config.memory}M"

        sbatch_script = SBATCH_TEMPLATE.format(
            job_name=job_name,
            log_dir=str(log_dir),
            time=self.config.timeout,
            partition=self.config.partition,
            nodes=self.config.nodes,
            cpus=self.config.cpus,
            gpu_directive=gpu_directive,
            mem_directive=mem_directive,
            image=self.config.image,
            sandbox_cmd=sandbox_cmd,
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
            f.write(sbatch_script)
            sbatch_path = f.name

        try:
            result = subprocess.run(
                ["sbatch", sbatch_path],
                capture_output=True,
                check=True,
                text=True,
            )
            job_id = result.stdout.strip().split()[-1]
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e.stderr}")
            raise
        finally:
            os.unlink(sbatch_path)

        handle = JobHandle(
            job_id=job_id,
            job_type="slurm",
            results_dir=results_dir,
            start_time=time.time(),
        )

        if self.verbose:
            logger.info(f"Submitted SLURM job {job_id}")

        return handle

    def submit_async(self, exec_fname: str, results_dir: str) -> JobHandle:
        """Submit a job asynchronously.

        Args:
            exec_fname: Path to the program to evaluate.
            results_dir: Directory for results.

        Returns:
            JobHandle for the submitted job.
        """

        if self.job_type == "local":
            return self._submit_local(exec_fname, results_dir)
        else:
            return self._submit_slurm(exec_fname, results_dir)

    def run(self, exec_fname: str, results_dir: str) -> Tuple[Dict[str, Any], float]:
        """Submit a job and wait for completion.

        Args:
            exec_fname: Path to the program to evaluate.
            results_dir: Directory for results.

        Returns:
            Tuple of (results_dict, runtime_seconds).
        """
        start_time = time.time()
        handle = self.submit_async(exec_fname, results_dir)

        while self.check_job_status(handle):
            time.sleep(2)

        results = self.get_job_results(handle.job_id, results_dir)
        runtime = time.time() - start_time

        if results is None:
            results = {"correct": {"correct": False}, "metrics": {}}

        return results, runtime

    def check_job_status(self, job) -> bool:
        """Check if a job is still running.

        Args:
            job: JobHandle or object with job_id field containing a JobHandle.

        Returns:
            True if running, False if completed.
        """
        # Handle objects with job_id attribute (e.g., RunningJob)
        if hasattr(job, "job_id") and isinstance(job.job_id, JobHandle):
            job = job.job_id

        if not isinstance(job, JobHandle):
            logger.warning(f"check_job_status received unexpected type: {type(job)}")
            return False

        if job.job_type == "local":
            return self._check_local_status(job)
        else:
            return self._check_slurm_status(job.job_id)

    def _check_local_status(self, handle: JobHandle) -> bool:
        """Check status of a local job (subprocess or sandbox)."""
        timeout = self.config.timeout_seconds

        # Check for sandbox job (has _sandbox_future attribute)
        if hasattr(handle, "_sandbox_future"):
            future = handle._sandbox_future
            if time.time() - handle.start_time > timeout:
                if self.verbose:
                    logger.warning(f"Sandbox job {handle.job_id} exceeded timeout")
                future.cancel()
                return False
            return not future.done()

        # Regular subprocess job
        if handle.process is None:
            return False

        if time.time() - handle.start_time > timeout:
            if self.verbose:
                logger.warning(f"Job {handle.job_id} exceeded timeout, killing...")
            try:
                handle.process.kill()
            except Exception:
                pass
            return False

        return handle.process.poll() is None

    def _check_slurm_status(self, job_id: str) -> bool:
        """Check status of a SLURM job."""
        try:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "--noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def get_job_results(
        self,
        job_id,
        results_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Get results from a completed job.

        Args:
            job_id: Job ID string or JobHandle object.
            results_dir: Directory containing results.

        Returns:
            Results dictionary or None.
        """
        # Handle JobHandle objects
        if isinstance(job_id, JobHandle):
            results_dir = job_id.results_dir
        return load_results(results_dir)

    async def submit_async_nonblocking(
        self, exec_fname: str, results_dir: str
    ) -> JobHandle:
        """Submit a job without blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.submit_async, exec_fname, results_dir
        )

    async def check_job_status_async(self, job: JobHandle) -> bool:
        """Async version of check_job_status."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.check_job_status, job)

    async def get_job_results_async(
        self,
        job_id: str,
        results_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_job_results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.get_job_results, job_id, results_dir
        )

    async def batch_check_status_async(self, jobs: List[JobHandle]) -> List[bool]:
        """Check status of multiple jobs concurrently."""
        tasks = [self.check_job_status_async(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_job_async(self, job: JobHandle) -> bool:
        """Cancel a running job."""
        try:
            if job.job_type == "local":
                if job.process:
                    job.process.kill()
                return True
            else:
                result = subprocess.run(
                    ["scancel", job.job_id],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to cancel job {job.job_id}: {e}")
            return False

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
