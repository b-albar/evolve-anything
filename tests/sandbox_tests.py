import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from evolve_anything.launch.sandbox import SandboxExecutor, SandboxConfig


# Helper to mock microsandbox module since it might not be installed
@pytest.fixture
def mock_microsandbox():
    with patch.dict(sys.modules, {"microsandbox": MagicMock()}):
        mock_msb = sys.modules["microsandbox"]
        yield mock_msb


@pytest.fixture
def sandbox_executor():
    return SandboxExecutor(SandboxConfig())


def test_init():
    async def _run():
        config = SandboxConfig(image="test/image", cpus=2)
        executor = SandboxExecutor(config)
        assert executor.config.image == "test/image"
        assert executor.config.cpus == 2
        assert executor._sandbox is None

    asyncio.run(_run())


def test_start_creates_sandbox(sandbox_executor, mock_microsandbox):
    async def _run():
        mock_sandbox_context = MagicMock()
        mock_sandbox_instance = AsyncMock()

        # Setup the context manager return
        mock_sandbox_context.__aenter__.return_value = mock_sandbox_instance
        mock_microsandbox.PythonSandbox.create.return_value = mock_sandbox_context

        await sandbox_executor.start()

        assert sandbox_executor._sandbox is mock_sandbox_instance
        mock_microsandbox.PythonSandbox.create.assert_called_once()
        mock_sandbox_context.__aenter__.assert_called_once()

    asyncio.run(_run())


def test_start_idempotent(sandbox_executor, mock_microsandbox):
    async def _run():
        mock_sandbox_context = MagicMock()
        mock_sandbox_instance = AsyncMock()
        mock_sandbox_context.__aenter__.return_value = mock_sandbox_instance
        mock_microsandbox.PythonSandbox.create.return_value = mock_sandbox_context

        await sandbox_executor.start()
        sandbox_executor._sandbox = mock_sandbox_instance  # verify it's set

        # Call start again
        await sandbox_executor.start()

        # Should only be called once
        assert mock_microsandbox.PythonSandbox.create.call_count == 1

    asyncio.run(_run())


def test_stop_cleans_up(sandbox_executor, mock_microsandbox):
    async def _run():
        # Actually in the code:
        # self._sandbox_context = PythonSandbox.create(...)
        # self._sandbox = await self._sandbox_context.__aenter__()

        # Let's mock create returning a mock that supports __aenter__ and __aexit__
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_ctx.__aexit__ = AsyncMock()

        mock_microsandbox.PythonSandbox.create.return_value = mock_ctx

        await sandbox_executor.start()
        await sandbox_executor.stop()

        mock_ctx.__aexit__.assert_called_once()
        assert sandbox_executor._sandbox is None
        assert sandbox_executor._sandbox_context is None
        assert sandbox_executor._packages_installed is False

    asyncio.run(_run())


def test_write_file(sandbox_executor):
    async def _run():
        mock_sandbox = AsyncMock()
        sandbox_executor._sandbox = mock_sandbox

        path = "/workspace/data.txt"
        content = "some content\nwith newlines"

        await sandbox_executor.write_file(path, content)

        mock_sandbox.run.assert_called_once()
        code_sent = mock_sandbox.run.call_args[0][0]

        assert path in code_sent
        assert "some content" in code_sent
        assert "os.makedirs" in code_sent

    asyncio.run(_run())


def test_run_command_success(sandbox_executor):
    async def _run():
        mock_sandbox = AsyncMock()
        # Mock the command chain: sandbox.command.run() -> execution object
        mock_execution = AsyncMock()
        mock_execution.output.return_value = "stdout output"
        mock_execution.error.return_value = ""
        mock_execution.exit_code = 0

        mock_sandbox.command.run.return_value = mock_execution
        sandbox_executor._sandbox = mock_sandbox

        exit_code, stdout, stderr = await sandbox_executor.run_command(["ls", "-la"])

        assert exit_code == 0
        assert stdout == "stdout output"
        assert stderr == ""

        mock_sandbox.command.run.assert_called_once()
        call_args = mock_sandbox.command.run.call_args
        assert call_args[0][0] == "bash"
        assert "ls -la" in call_args[0][1][1]  # ["-c", "cmd"]

    asyncio.run(_run())


def test_run_command_failure(sandbox_executor):
    async def _run():
        mock_sandbox = AsyncMock()
        mock_execution = AsyncMock()
        mock_execution.output.return_value = ""
        mock_execution.error.return_value = "command not found"
        mock_execution.exit_code = 127

        mock_sandbox.command.run.return_value = mock_execution
        sandbox_executor._sandbox = mock_sandbox

        exit_code, stdout, stderr = await sandbox_executor.run_command(["invalid_cmd"])

        assert exit_code == 127
        assert stderr == "command not found"

    asyncio.run(_run())


def test_run_python_code(sandbox_executor):
    async def _run():
        mock_sandbox = AsyncMock()

        mock_result = MagicMock()
        mock_result.output = "42"
        mock_result.error = ""
        mock_result.has_error = False

        mock_sandbox.run.return_value = mock_result
        sandbox_executor._sandbox = mock_sandbox

        code = "print(6 * 7)"
        exit_code, stdout, stderr = await sandbox_executor.run_python_code(code)

        assert exit_code == 0
        assert stdout == "42"
        mock_sandbox.run.assert_called_once()
        assert code in mock_sandbox.run.call_args[0][0]

    asyncio.run(_run())


def test_run_python_code_with_args(sandbox_executor):
    async def _run():
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = MagicMock(output="", error="", has_error=False)
        sandbox_executor._sandbox = mock_sandbox

        code = "import sys; print(sys.argv)"
        args = ["--arg1", "val1"]

        await sandbox_executor.run_python_code(code, args=args)

        sent_code = mock_sandbox.run.call_args[0][0]
        assert "sys.argv =" in sent_code
        assert "--arg1" in sent_code
        assert "val1" in sent_code

    asyncio.run(_run())


def test_run_python_file(sandbox_executor, tmp_path):
    async def _run():
        mock_sandbox = AsyncMock()

        mock_result = MagicMock()
        mock_result.output = "file executed"
        mock_result.error = ""
        mock_result.has_error = False
        mock_sandbox.run.return_value = mock_result
        sandbox_executor._sandbox = mock_sandbox

        # Create temp file
        script_file = tmp_path / "script.py"
        script_file.write_text("print('file executed')")

        exit_code, stdout, stderr = await sandbox_executor.run_python_file(
            str(script_file)
        )

        assert exit_code == 0
        assert stdout == "file executed"

        sent_code = mock_sandbox.run.call_args[0][0]
        assert "print('file executed')" in sent_code

    asyncio.run(_run())


def test_ensure_packages(sandbox_executor):
    async def _run():
        config = SandboxConfig(packages=["numpy", "pandas"])
        sandbox_executor.config = config
        mock_sandbox = AsyncMock()
        sandbox_executor._sandbox = mock_sandbox

        # Run a command, which triggers _ensure_packages
        # We need to mock run_command's usage of _ensure_packages or call it directly
        # Let's call it directly
        await sandbox_executor._ensure_packages()

        mock_sandbox.command.run.assert_called_once()
        call_args = mock_sandbox.command.run.call_args
        cmd_sent = call_args[0][1][1]
        assert "pip install -q numpy pandas" in cmd_sent
        assert sandbox_executor._packages_installed is True

        # Call again - should invoke nothing
        mock_sandbox.command.run.reset_mock()
        await sandbox_executor._ensure_packages()
        mock_sandbox.command.run.assert_not_called()

    asyncio.run(_run())
