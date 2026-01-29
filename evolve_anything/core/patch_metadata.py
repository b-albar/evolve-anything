"""Patch metadata module.

This module contains the PatchMetadata dataclass for tracking patch generation
metadata and results.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import rich.box

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class PatchMetadata:
    """Metadata from a patch generation attempt."""

    patch_type: str
    patch_name: Optional[str] = None
    patch_description: Optional[str] = None
    num_applied: int = 0
    error_attempt: Optional[str] = None
    novelty_attempt: int = 1
    resample_attempt: int = 1
    patch_attempt: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    llm_kwargs: dict = field(default_factory=dict)
    llm_result: Optional[dict] = None
    diff_summary: dict = field(default_factory=dict)
    validation_attempts: int = 0
    # Optional fields added after patch generation
    meta_recommendations: Optional[str] = None
    meta_summary: Optional[str] = None
    meta_scratch_pad: Optional[str] = None
    novelty_checks_performed: int = 0
    novelty_explanation: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """Convert to dict for storage in Program metadata."""
        result = {
            "patch_type": self.patch_type,
            "patch_name": self.patch_name,
            "patch_description": self.patch_description,
            "num_applied": self.num_applied,
            "error_attempt": self.error_attempt,
            "novelty_attempt": self.novelty_attempt,
            "resample_attempt": self.resample_attempt,
            "patch_attempt": self.patch_attempt,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_result": self.llm_result,
            "diff_summary": self.diff_summary,
            **self.llm_kwargs,  # Spread llm_kwargs at top level for compatibility
        }
        # Add optional meta fields if present
        if self.meta_recommendations:
            result["meta_recommendations"] = self.meta_recommendations
        if self.meta_summary:
            result["meta_summary"] = self.meta_summary
        if self.meta_scratch_pad:
            result["meta_scratch_pad"] = self.meta_scratch_pad
        if self.novelty_checks_performed > 0:
            result["novelty_checks_performed"] = self.novelty_checks_performed
        if self.novelty_explanation:
            result["novelty_explanation"] = self.novelty_explanation
        return result

    def print_table(
        self,
        generation: int,
        num_generations: int,
        max_novelty_attempts: int,
        max_patch_resamples: int,
        max_patch_attempts: int,
        console: Optional["Console"] = None,
    ) -> None:
        """Display metadata in a formatted rich table.

        Args:
            generation: Current generation number.
            num_generations: Total number of generations.
            max_novelty_attempts: Max novelty attempts from config.
            max_patch_resamples: Max patch resamples from config.
            max_patch_attempts: Max patch attempts from config.
            console: Rich Console instance (creates one if not provided).
        """
        from rich.table import Table
        from rich.console import Console as RichConsole

        if console is None:
            console = RichConsole()

        # Create title with generation and attempt information
        title = (
            f"[bold magenta]Patch Metadata - "
            f"Gen {generation}/{num_generations} - "
            f"Novelty: {self.novelty_attempt}/{max_novelty_attempts} - "
            f"Resample: {self.resample_attempt}/{max_patch_resamples} - "
            f"Patch: {self.patch_attempt}/{max_patch_attempts}[/bold magenta]"
        )

        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=120,
        )
        table.add_column("Field", style="cyan bold", no_wrap=True, width=25)
        table.add_column("Value", style="green", overflow="fold", width=90)

        # Display core fields in order
        table.add_row("patch_type", str(self.patch_type))
        table.add_row(
            "patch_name",
            str(self.patch_name) if self.patch_name else "[dim]None[/dim]",
        )

        if self.patch_description:
            desc = str(self.patch_description)
            table.add_row(
                "patch_description", desc[:100] + "..." if len(desc) > 100 else desc
            )

        table.add_row("num_applied", str(self.num_applied))
        table.add_row("total_tokens", str(self.total_tokens))
        table.add_row("input_tokens", str(self.input_tokens))
        table.add_row("output_tokens", str(self.output_tokens))

        # Error handling
        if self.error_attempt is None:
            table.add_row("error_attempt", "[green]Success[/green]")
        else:
            error_str = str(self.error_attempt)
            table.add_row(
                "error_attempt",
                f"[red]{error_str[:100]}...[/red]"
                if len(error_str) > 100
                else f"[red]{error_str}[/red]",
            )

        if self.validation_attempts > 0:
            table.add_row("validation_attempts", str(self.validation_attempts))

        # Add llm_kwargs fields
        for key, value in self.llm_kwargs.items():
            if value is not None:
                formatted = (
                    str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                )
                table.add_row(key, formatted)

        # Add diff summary if available
        if self.diff_summary:
            summary_text = "; ".join(f"{k}: {v}" for k, v in self.diff_summary.items())
            table.add_row("diff_summary", summary_text.strip())

        console.print(table)
