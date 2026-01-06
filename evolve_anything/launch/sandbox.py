"""Microsandbox-based code execution.

This module provides sandboxed code execution using microsandbox microVMs.
See: https://github.com/zerocore-ai/microsandbox
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for microsandbox execution.

    Attributes:
        image: Microsandbox image to use (e.g., "microsandbox/python").
        packages: List of Python packages to install before execution.
        cpus: Number of CPUs to allocate to the sandbox.
        memory: Memory in MB to allocate to the sandbox.
        timeout: Timeout for command execution in seconds.
        server_url: URL of the microsandbox server (default: local).
        workdir: Working directory inside the sandbox.
        env: Environment variables to set in the sandbox.
        mounts: List of (host_path, sandbox_path) tuples for volume mounts.
    """

    image: str = "microsandbox/python"
    packages: List[str] = field(default_factory=list)
    cpus: int = 1
    memory: int = 4096
    timeout: int = 3600
    server_url: Optional[str] = None
    workdir: str = "."
    env: Dict[str, str] = field(default_factory=dict)
    mounts: List[Tuple[str, str]] = field(default_factory=list)


class SandboxExecutor:
    """Execute code in a microsandbox environment.

    This executor uses the microsandbox Python SDK to run code in isolated
    microVM environments with hardware-level isolation.

    The sandbox is created once and reused for all commands.
    """

    def __init__(self, config: SandboxConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self._sandbox: Optional[Any] = None
        self._sandbox_context: Optional[Any] = None
        self._packages_installed = False

        # Set server URL from config or environment
        if config.server_url:
            os.environ["MSB_SERVER_URL"] = config.server_url

    async def start(self) -> None:
        """Start the sandbox (create it once)."""
        if self._sandbox is not None:
            return  # Already started

        try:
            from microsandbox import PythonSandbox
        except ImportError:
            raise ImportError(
                "microsandbox is required for sandbox execution. "
                "Install with: pip install microsandbox"
            )

        # PythonSandbox.create() only accepts: server_url, namespace, name, api_key
        self._sandbox_context = PythonSandbox.create(
            name=f"evolve-{os.getpid()}",
            server_url=self.config.server_url,
        )
        self._sandbox = await self._sandbox_context.__aenter__()

        if self.verbose:
            logger.info(f"Started sandbox: evolve-{os.getpid()}")

    async def stop(self) -> None:
        """Stop and cleanup the sandbox."""
        if self._sandbox_context is not None:
            await self._sandbox_context.__aexit__(None, None, None)
            self._sandbox = None
            self._sandbox_context = None
            self._packages_installed = False

            if self.verbose:
                logger.info("Stopped sandbox")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def _ensure_packages(self) -> None:
        """Install packages if not already installed."""
        if self._packages_installed or not self.config.packages:
            return

        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        if self.verbose:
            logger.info(f"Installing packages: {self.config.packages}")

        execution = await self._sandbox.command.run(
            "bash", ["-c", "pip install -q " + " ".join(self.config.packages)]
        )
        await execution.output()

        if execution.exit_code != 0:
            stderr = await execution.error()
            raise RuntimeError(f"Failed to install packages: {stderr}")

        self._packages_installed = True

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file inside the sandbox.

        Args:
            path: Path to the file inside the sandbox.
            content: Content to write to the file.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        import json

        # Use Python code to write the file inside the sandbox
        escaped_content = json.dumps(content)
        code = f'''
import os
os.makedirs(os.path.dirname("{path}") or ".", exist_ok=True)
with open("{path}", "w") as f:
    f.write({escaped_content})
'''
        await self._sandbox.run(code)

        if self.verbose:
            logger.info(f"Wrote file to sandbox: {path}")

    def write_file_sync(self, path: str, content: str) -> None:
        """Synchronous wrapper for write_file."""

        async def _run():
            async with self:
                return await self.write_file(path, content)

        return asyncio.run(_run())

    async def run_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Run a command in the sandbox.

        Args:
            cmd: Command and arguments to run.
            cwd: Working directory (defaults to config.workdir).
            timeout: Timeout in seconds (defaults to config.timeout).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        timeout = timeout or self.config.timeout
        workdir = cwd or self.config.workdir

        try:
            # Ensure packages are installed
            await self._ensure_packages()

            # Build the command
            cmd_str = " ".join(cmd)
            if workdir:
                cmd_str = f"cd {workdir} && {cmd_str}"

            if self.verbose:
                logger.info(f"Running command in sandbox: {cmd_str}")

            # Execute command
            execution = await self._sandbox.command.run(
                "bash", ["-c", cmd_str], timeout=timeout
            )

            stdout = await execution.output() or ""
            stderr = await execution.error() or ""
            exit_code = execution.exit_code or 0

            return exit_code, stdout, stderr

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return 1, "", str(e)

    def run_command_sync(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Synchronous wrapper for run_command.

        Note: This creates a new event loop, so the sandbox must be started
        within the same asyncio.run() call or use run_with_sandbox().
        """

        async def _run():
            async with self:
                return await self.run_command(cmd, cwd, timeout)

        return asyncio.run(_run())

    async def run_python_file(
        self,
        file_path: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Run a Python file in the sandbox.

        The file is read from the host filesystem and its content is executed
        in the sandbox using sb.run().

        Args:
            file_path: Path to the Python file on the host.
            args: Additional arguments (sets sys.argv in sandbox).
            timeout: Timeout in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        try:
            # Read the file from host filesystem
            code = Path(file_path).read_text()

            # If args provided, set sys.argv
            if args:
                import json

                argv_setup = (
                    f"import sys; sys.argv = {json.dumps([file_path] + args)}\n"
                )
                code = argv_setup + code

            if self.verbose:
                logger.info(f"Running Python file in sandbox: {file_path}")

            # Execute the code in sandbox
            exec_result = await self._sandbox.run(code)

            stdout = await exec_result.output() or ""
            stderr = await exec_result.error() or ""
            exit_code = 0 if not exec_result.has_error else 1

            return exit_code, stdout, stderr

        except FileNotFoundError:
            return 1, "", f"File not found: {file_path}"
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return 1, "", str(e)

    async def run_python_code(
        self,
        code: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Run Python code (as a string) in the sandbox.

        Args:
            code: Python code to execute.
            args: Additional arguments (sets sys.argv in sandbox).
            timeout: Timeout in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        try:
            # Ensure packages are installed
            await self._ensure_packages()

            # If args provided, set sys.argv
            if args:
                import json

                argv_setup = f"import sys; sys.argv = {json.dumps(args)}\n"
                code = argv_setup + code

            if self.verbose:
                logger.info("Running Python code in sandbox")

            # Execute the code in sandbox
            exec_result = await self._sandbox.run(code)

            stdout = await exec_result.output() or ""
            stderr = await exec_result.error() or ""
            exit_code = 0 if not exec_result.has_error else 1

            return exit_code, stdout, stderr

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return 1, "", str(e)

    def run_python_code_sync(
        self,
        code: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Synchronous wrapper for run_python_code."""

        async def _run():
            async with self:
                return await self.run_python_code(code, args, timeout)

        return asyncio.run(_run())

    def run_python_file_sync(
        self,
        file_path: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Synchronous wrapper for run_python_file."""

        async def _run():
            async with self:
                return await self.run_python_file(file_path, args, timeout)

        return asyncio.run(_run())


async def run_in_sandbox_async(
    cmd: List[str],
    config: Optional[SandboxConfig] = None,
    verbose: bool = False,
) -> Tuple[int, str, str]:
    """Run a command in a sandbox asynchronously.

    Args:
        cmd: Command and arguments to run.
        config: Sandbox configuration (uses defaults if None).
        verbose: Enable verbose logging.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    config = config or SandboxConfig()
    async with SandboxExecutor(config, verbose=verbose) as executor:
        return await executor.run_command(cmd)


def run_in_sandbox(
    cmd: List[str],
    config: Optional[SandboxConfig] = None,
    verbose: bool = False,
) -> Tuple[int, str, str]:
    """Convenience function to run a command in a sandbox.

    Args:
        cmd: Command and arguments to run.
        config: Sandbox configuration (uses defaults if None).
        verbose: Enable verbose logging.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    return asyncio.run(run_in_sandbox_async(cmd, config, verbose))
