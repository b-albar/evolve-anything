"""Unit tests for SLURM job scheduler functionality.

Tests the SLURM-related methods in JobScheduler using mocked subprocess calls.
"""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from evolve_anything.launch.scheduler import (
    JobConfig,
    JobHandle,
    JobScheduler,
    SBATCH_TEMPLATE,
)


class TestJobConfig:
    """Tests for JobConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JobConfig()
        assert config.eval_program_path == "evaluate.py"
        assert config.image == "microsandbox/python"
        assert config.cpus == 1
        assert config.gpus == 0
        assert config.memory == 8192
        assert config.timeout == "01:00:00"
        assert config.partition == "gpu"
        assert config.nodes == 1
        assert config.use_sandbox is False

    def test_timeout_seconds(self):
        """Test timeout conversion to seconds."""
        config = JobConfig(timeout="01:00:00")
        assert config.timeout_seconds == 3600

        config = JobConfig(timeout="00:30:00")
        assert config.timeout_seconds == 1800

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = JobConfig(cpus=4, gpus=2)
        d = config.to_dict()
        assert d["cpus"] == 4
        assert d["gpus"] == 2
        assert "eval_program_path" in d


class TestJobHandle:
    """Tests for JobHandle dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        handle = JobHandle(
            job_id="12345",
            job_type="slurm",
            results_dir="/tmp/results",
            start_time=time.time(),
        )
        assert "slurm" in str(handle)
        assert "12345" in str(handle)


class TestSlurmScheduler:
    """Tests for SLURM-related JobScheduler methods."""

    @pytest.fixture
    def slurm_scheduler(self):
        """Create a SLURM job scheduler for testing."""
        config = JobConfig(
            cpus=4,
            gpus=1,
            memory=16384,
            timeout="02:00:00",
            partition="gpu",
            nodes=2,
            packages=["numpy", "pandas"],
        )
        return JobScheduler(job_type="slurm", config=config, verbose=False)

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create a temporary results directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        return str(results_dir)

    def test_scheduler_initialization_slurm(self, slurm_scheduler):
        """Test SLURM scheduler initialization."""
        assert slurm_scheduler.job_type == "slurm"
        assert slurm_scheduler.config.cpus == 4
        assert slurm_scheduler.config.gpus == 1
        assert slurm_scheduler.config.nodes == 2

    def test_scheduler_invalid_job_type(self):
        """Test that invalid job type raises error."""
        with pytest.raises(ValueError, match="Unknown job type"):
            JobScheduler(job_type="invalid")

    @patch("subprocess.run")
    def test_submit_slurm_success(self, mock_run, slurm_scheduler, temp_results_dir):
        """Test successful SLURM job submission."""
        # Mock sbatch returning a job ID
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 12345678\n",
            stderr="",
        )

        handle = slurm_scheduler._submit_slurm(
            results_dir=temp_results_dir,
            cmd=["python", "evaluate.py", "--program_path", "test.py"],
        )

        assert handle.job_id == "12345678"
        assert handle.job_type == "slurm"
        assert handle.results_dir == temp_results_dir

        # Verify sbatch was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "sbatch"

    @patch("subprocess.run")
    def test_submit_slurm_failure(self, mock_run, slurm_scheduler, temp_results_dir):
        """Test SLURM job submission failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["sbatch"],
            stderr="sbatch: error: Invalid partition specified",
        )

        with pytest.raises(subprocess.CalledProcessError):
            slurm_scheduler._submit_slurm(
                results_dir=temp_results_dir,
                cmd=["python", "test.py"],
            )

    @patch("subprocess.run")
    @patch("os.unlink")
    def test_submit_slurm_script_content(
        self, mock_unlink, mock_run, slurm_scheduler, temp_results_dir
    ):
        """Test that SLURM script contains correct directives."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 99999\n",
        )

        # Submit the job - this will create a temp file with the script
        handle = slurm_scheduler._submit_slurm(
            results_dir=temp_results_dir,
            cmd=["python", "test.py"],
        )

        assert handle.job_id == "99999"

        # Verify sbatch was called with a .sbatch file
        call_args = mock_run.call_args
        sbatch_file = call_args[0][0][1]  # Second arg is the script path
        assert sbatch_file.endswith(".sbatch")

        # Verify unlink was called to clean up
        mock_unlink.assert_called_once()

    @patch("subprocess.run")
    def test_check_slurm_status_running(self, mock_run, slurm_scheduler):
        """Test checking status of a running SLURM job."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345678  gpu  job_name  user  R  0:05  1  node01\n",
            stderr="",
        )

        is_running = slurm_scheduler._check_slurm_status("12345678")
        assert is_running is True

        mock_run.assert_called_once_with(
            ["squeue", "-j", "12345678", "--noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_check_slurm_status_completed(self, mock_run, slurm_scheduler):
        """Test checking status of a completed SLURM job."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",  # Empty output means job is not in queue
            stderr="",
        )

        is_running = slurm_scheduler._check_slurm_status("12345678")
        assert is_running is False

    @patch("subprocess.run")
    def test_check_slurm_status_timeout(self, mock_run, slurm_scheduler):
        """Test handling timeout when checking SLURM status."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["squeue"], timeout=30)

        is_running = slurm_scheduler._check_slurm_status("12345678")
        assert is_running is False

    @patch("subprocess.run")
    def test_check_slurm_status_error(self, mock_run, slurm_scheduler):
        """Test handling error when checking SLURM status."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["squeue"]
        )

        is_running = slurm_scheduler._check_slurm_status("12345678")
        assert is_running is False

    def test_check_job_status_slurm(self, slurm_scheduler, temp_results_dir):
        """Test check_job_status with SLURM job handle."""
        handle = JobHandle(
            job_id="12345",
            job_type="slurm",
            results_dir=temp_results_dir,
            start_time=time.time(),
        )

        with patch.object(
            slurm_scheduler, "_check_slurm_status", return_value=True
        ) as mock_check:
            is_running = slurm_scheduler.check_job_status(handle)
            assert is_running is True
            mock_check.assert_called_once_with("12345")

    @patch("subprocess.run")
    def test_cancel_slurm_job(self, mock_run, slurm_scheduler, temp_results_dir):
        """Test cancelling a SLURM job."""
        mock_run.return_value = MagicMock(returncode=0)

        handle = JobHandle(
            job_id="12345",
            job_type="slurm",
            results_dir=temp_results_dir,
            start_time=time.time(),
        )

        # Use asyncio to run the async method
        import asyncio

        result = asyncio.run(slurm_scheduler.cancel_job_async(handle))
        assert result is True

        mock_run.assert_called_with(
            ["scancel", "12345"],
            capture_output=True,
            timeout=30,
        )


class TestSbatchTemplate:
    """Tests for SBATCH script template."""

    def test_template_contains_required_directives(self):
        """Test that template has required SLURM directives."""
        assert "#SBATCH --job-name=" in SBATCH_TEMPLATE
        assert "#SBATCH --output=" in SBATCH_TEMPLATE
        assert "#SBATCH --error=" in SBATCH_TEMPLATE
        assert "#SBATCH --time=" in SBATCH_TEMPLATE
        assert "#SBATCH --partition=" in SBATCH_TEMPLATE
        assert "#SBATCH --nodes=" in SBATCH_TEMPLATE
        assert "#SBATCH --cpus-per-task=" in SBATCH_TEMPLATE

    def test_template_formatting(self):
        """Test that template can be formatted with expected parameters."""
        formatted = SBATCH_TEMPLATE.format(
            job_name="test-job",
            log_dir="/tmp/logs",
            time="01:00:00",
            partition="gpu",
            nodes=1,
            cpus=4,
            gpu_directive="#SBATCH --gres=gpu:1",
            mem_directive="#SBATCH --mem=8192M",
            image="microsandbox/python",
            sandbox_cmd="python test.py",
        )

        assert "test-job" in formatted
        assert "/tmp/logs" in formatted
        assert "01:00:00" in formatted
        assert "gpu" in formatted
        assert "#SBATCH --gres=gpu:1" in formatted
        assert "python test.py" in formatted

    def test_template_gpu_directive_optional(self):
        """Test that GPU directive can be empty."""
        formatted = SBATCH_TEMPLATE.format(
            job_name="test-job",
            log_dir="/tmp/logs",
            time="01:00:00",
            partition="cpu",
            nodes=1,
            cpus=4,
            gpu_directive="",  # No GPU
            mem_directive="#SBATCH --mem=8192M",
            image="microsandbox/python",
            sandbox_cmd="python test.py",
        )

        assert "--gres=gpu" not in formatted


class TestSlurmIntegration:
    """Integration-style tests for SLURM workflow (still mocked)."""

    @pytest.fixture
    def scheduler_with_mocks(self, tmp_path):
        """Create scheduler with common mocks set up."""
        config = JobConfig(
            cpus=2,
            gpus=1,
            memory=8192,
            timeout="00:30:00",
        )
        scheduler = JobScheduler(job_type="slurm", config=config, verbose=True)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        return scheduler, str(results_dir)

    @patch("subprocess.run")
    def test_submit_and_check_workflow(self, mock_run, scheduler_with_mocks):
        """Test submitting a job and checking its status."""
        scheduler, results_dir = scheduler_with_mocks

        # First call: sbatch for submission
        # Second call: squeue for status check
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Submitted batch job 55555\n"),
            MagicMock(returncode=0, stdout="55555  gpu  job  user  R  0:01  1  node\n"),
            MagicMock(returncode=0, stdout=""),  # Job completed
        ]

        # Submit
        handle = scheduler._submit_slurm(
            results_dir=results_dir,
            cmd=["python", "eval.py"],
        )
        assert handle.job_id == "55555"

        # Check status - running
        assert scheduler._check_slurm_status("55555") is True

        # Check status - completed
        assert scheduler._check_slurm_status("55555") is False
