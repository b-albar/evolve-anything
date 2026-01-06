"""
MAP-Elites Style Behavioral Archive

Implements quality-diversity optimization by maintaining an archive of programs
across behavioral dimensions. Each cell in the behavioral space keeps the best
program found for that behavioral niche.

Reference: "Illuminating Search Spaces by Mapping Elites" (Mouret & Clune, 2015)
https://arxiv.org/abs/1504.04909
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BehaviorDimension:
    """Defines a single behavioral dimension for the MAP-Elites grid."""

    name: str
    min_value: float
    max_value: float
    num_bins: int
    source: str = "metadata"  # "metadata", "public_metrics", "program"
    key_path: Optional[str] = None  # Dot-separated path for nested access
    minimize: bool = False  # If True, lower values are better for this dimension

    def get_bin_index(self, value: float) -> int:
        """Get the bin index for a given value."""
        if value <= self.min_value:
            return 0
        if value >= self.max_value:
            return self.num_bins - 1

        # Normalize to [0, 1] and then to bin index
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        bin_idx = int(normalized * self.num_bins)
        return min(bin_idx, self.num_bins - 1)

    def get_bin_range(self, bin_idx: int) -> Tuple[float, float]:
        """Get the value range for a bin."""
        bin_width = (self.max_value - self.min_value) / self.num_bins
        low = self.min_value + bin_idx * bin_width
        high = low + bin_width
        return (low, high)

    def extract_value(self, program: Any) -> Optional[float]:
        """Extract the dimension value from a program."""
        try:
            if self.source == "program":
                # Direct attribute on program
                return getattr(program, self.name, None)

            elif self.source == "metadata":
                # From metadata dict, possibly nested
                data = getattr(program, "metadata", {}) or {}
                if self.key_path:
                    for key in self.key_path.split("."):
                        if isinstance(data, dict):
                            data = data.get(key, {})
                        else:
                            return None
                    return float(data) if data else None
                return data.get(self.name)

            elif self.source == "public_metrics":
                metrics = getattr(program, "public_metrics", {}) or {}
                if self.key_path:
                    for key in self.key_path.split("."):
                        if isinstance(metrics, dict):
                            metrics = metrics.get(key, {})
                        else:
                            return None
                    return float(metrics) if metrics else None
                return metrics.get(self.name)

            return None
        except (TypeError, ValueError, AttributeError):
            return None


@dataclass
class MapElitesConfig:
    """Configuration for MAP-Elites behavioral archive."""

    # Behavioral dimensions
    dimensions: List[BehaviorDimension] = field(default_factory=list)

    # Whether to use MAP-Elites archive for parent selection
    enabled: bool = True

    # Probability of sampling from MAP-Elites archive vs regular archive
    selection_probability: float = 0.3

    # Whether MAP-Elites replaces or supplements the regular archive
    replace_archive: bool = False

    # Minimum score threshold for adding to MAP-Elites (relative to best)
    min_score_ratio: float = 0.5

    @classmethod
    def default_for_code(cls) -> "MapElitesConfig":
        """Create a default config with code complexity dimensions."""
        return cls(
            dimensions=[
                BehaviorDimension(
                    name="complexity_score",
                    min_value=0.0,
                    max_value=1.0,
                    num_bins=5,
                    source="metadata",
                    key_path="code_analysis_metrics.complexity_score",
                    minimize=True,
                ),
                BehaviorDimension(
                    name="lines_of_code",
                    min_value=10,
                    max_value=500,
                    num_bins=5,
                    source="metadata",
                    key_path="code_analysis_metrics.lines_of_code",
                    minimize=True,
                ),
            ],
            enabled=True,
            selection_probability=0.3,
        )


class MapElitesArchive:
    """
    MAP-Elites behavioral archive that maintains the best program for each
    behavioral niche (grid cell).
    """

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: MapElitesConfig,
        minimize_score: bool = False,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.minimize_score = minimize_score

        # Create the MAP-Elites archive table
        self._create_tables()

    def _create_tables(self):
        """Create the MAP-Elites archive table if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS map_elites_archive (
                cell_key TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                combined_score REAL,
                behavior_values TEXT,  -- JSON serialized behavior values
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        # Index for faster program lookups
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_map_elites_program_id
            ON map_elites_archive(program_id)
            """
        )

        self.conn.commit()
        logger.debug("MAP-Elites archive tables created/verified.")

    def get_cell_key(self, program: Any) -> Optional[str]:
        """
        Compute the grid cell key for a program based on its behavioral values.
        Returns None if any dimension value cannot be extracted.
        """
        if not self.config.dimensions:
            return None

        bin_indices = []
        for dim in self.config.dimensions:
            value = dim.extract_value(program)
            if value is None:
                logger.debug(
                    f"Could not extract dimension '{dim.name}' "
                    f"from program {getattr(program, 'id', 'unknown')}"
                )
                return None
            bin_idx = dim.get_bin_index(value)
            bin_indices.append(str(bin_idx))

        return "_".join(bin_indices)

    def get_behavior_values(self, program: Any) -> Dict[str, float]:
        """Extract all behavior dimension values from a program."""
        values = {}
        for dim in self.config.dimensions:
            value = dim.extract_value(program)
            if value is not None:
                values[dim.name] = value
        return values

    def add(self, program: Any, verbose: bool = False) -> bool:
        """
        Attempt to add a program to the MAP-Elites archive.

        Returns True if the program was added (either new cell or better score).
        """
        if not self.config.enabled:
            return False

        # Only add correct programs
        if not getattr(program, "correct", False):
            return False

        cell_key = self.get_cell_key(program)
        if cell_key is None:
            logger.debug(
                f"Could not compute cell key for program {program.id}, "
                "skipping MAP-Elites add"
            )
            return False

        program_score = getattr(program, "combined_score", None)
        if program_score is None:
            return False

        # Check if cell already has a program
        self.cursor.execute(
            "SELECT program_id, combined_score FROM map_elites_archive WHERE cell_key = ?",
            (cell_key,),
        )
        existing = self.cursor.fetchone()

        should_add = False

        if existing is None:
            # New cell - add the program
            should_add = True
            logger.info(
                f"MAP-Elites: New cell {cell_key} filled by program "
                f"{program.id[:8]}... (score: {program_score:.4f})"
            )
        else:
            existing_score = existing["combined_score"]

            # Compare scores (handle minimize vs maximize)
            if self.minimize_score:
                is_better = program_score < existing_score
            else:
                is_better = program_score > existing_score

            if is_better:
                should_add = True
                logger.info(
                    f"MAP-Elites: Cell {cell_key} improved from "
                    f"{existing_score:.4f} to {program_score:.4f} "
                    f"by program {program.id[:8]}..."
                )

        if should_add:
            behavior_values = self.get_behavior_values(program)
            behavior_json = json.dumps(behavior_values)

            self.cursor.execute(
                """
                INSERT OR REPLACE INTO map_elites_archive
                (cell_key, program_id, combined_score, behavior_values)
                VALUES (?, ?, ?, ?)
                """,
                (cell_key, program.id, program_score, behavior_json),
            )
            self.conn.commit()

        return should_add

    def sample(
        self,
        n: int = 1,
        exclude_ids: Optional[List[str]] = None,
        strategy: str = "uniform",
    ) -> List[str]:
        """
        Sample program IDs from the MAP-Elites archive.

        Args:
            n: Number of programs to sample
            exclude_ids: Program IDs to exclude from sampling
            strategy: Sampling strategy - "uniform", "roulette", or "curiosity"

        Returns:
            List of program IDs
        """
        exclude_ids = exclude_ids or []

        if exclude_ids:
            placeholders = ",".join(["?"] * len(exclude_ids))
            query = f"""
                SELECT program_id, combined_score, cell_key
                FROM map_elites_archive
                WHERE program_id NOT IN ({placeholders})
            """
            self.cursor.execute(query, exclude_ids)
        else:
            self.cursor.execute(
                "SELECT program_id, combined_score, cell_key FROM map_elites_archive"
            )

        rows = self.cursor.fetchall()

        if not rows:
            return []

        if strategy == "uniform":
            # Uniform random sampling across cells
            selected_indices = np.random.choice(
                len(rows), size=min(n, len(rows)), replace=False
            )
            return [rows[i]["program_id"] for i in selected_indices]

        elif strategy == "roulette":
            # Fitness-proportional sampling
            scores = np.array([row["combined_score"] or 0.0 for row in rows])
            if self.minimize_score:
                # Invert for minimization (add small epsilon to avoid division by zero)
                scores = 1.0 / (scores + 1e-8)

            # Normalize to probabilities
            scores = scores - scores.min() + 1e-8  # Shift to positive
            probs = scores / scores.sum()

            selected_indices = np.random.choice(
                len(rows), size=min(n, len(rows)), replace=False, p=probs
            )
            return [rows[i]["program_id"] for i in selected_indices]

        elif strategy == "curiosity":
            # Prefer less-explored cells (cells with older programs or fewer visits)
            # For now, just use uniform - could track visit counts later
            return self.sample(n, exclude_ids, strategy="uniform")

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about the MAP-Elites archive."""
        self.cursor.execute("SELECT COUNT(*) FROM map_elites_archive")
        filled_cells = self.cursor.fetchone()[0]

        # Total possible cells
        total_cells = 1
        for dim in self.config.dimensions:
            total_cells *= dim.num_bins

        self.cursor.execute(
            "SELECT MIN(combined_score), MAX(combined_score), AVG(combined_score) "
            "FROM map_elites_archive"
        )
        row = self.cursor.fetchone()

        stats = {
            "filled_cells": filled_cells,
            "total_cells": total_cells,
            "coverage": filled_cells / total_cells if total_cells > 0 else 0.0,
            "min_score": row[0] if row else None,
            "max_score": row[1] if row else None,
            "avg_score": row[2] if row else None,
            "dimensions": [
                {
                    "name": dim.name,
                    "bins": dim.num_bins,
                    "range": (dim.min_value, dim.max_value),
                }
                for dim in self.config.dimensions
            ],
        }

        return stats

    def get_archive_heatmap(self) -> Optional[np.ndarray]:
        """
        Get a 2D heatmap of scores for visualization (only works with 2 dimensions).
        Returns None if not exactly 2 dimensions.
        """
        if len(self.config.dimensions) != 2:
            return None

        dim1, dim2 = self.config.dimensions
        heatmap = np.full((dim1.num_bins, dim2.num_bins), np.nan)

        self.cursor.execute("SELECT cell_key, combined_score FROM map_elites_archive")
        for row in self.cursor.fetchall():
            parts = row["cell_key"].split("_")
            if len(parts) == 2:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    heatmap[i, j] = row["combined_score"]
                except (ValueError, IndexError):
                    continue

        return heatmap

    def get_all_programs(self) -> List[str]:
        """Get all program IDs in the MAP-Elites archive."""
        self.cursor.execute("SELECT program_id FROM map_elites_archive")
        return [row["program_id"] for row in self.cursor.fetchall()]

    def clear(self):
        """Clear the MAP-Elites archive."""
        self.cursor.execute("DELETE FROM map_elites_archive")
        self.conn.commit()
        logger.info("MAP-Elites archive cleared.")

    def print_summary(self):
        """Print a summary of the MAP-Elites archive."""
        from rich.console import Console
        from rich.table import Table
        import rich.box

        stats = self.get_archive_stats()
        console = Console()

        table = Table(
            title="[bold cyan]MAP-Elites Archive Summary[/bold cyan]",
            box=rich.box.ROUNDED,
            border_style="cyan",
        )

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row(
            "Filled Cells", f"{stats['filled_cells']} / {stats['total_cells']}"
        )
        table.add_row("Coverage", f"{stats['coverage'] * 100:.1f}%")
        table.add_row(
            "Score Range",
            f"{stats['min_score']:.4f} - {stats['max_score']:.4f}"
            if stats["min_score"] is not None
            else "N/A",
        )
        table.add_row(
            "Avg Score",
            f"{stats['avg_score']:.4f}" if stats["avg_score"] is not None else "N/A",
        )

        console.print(table)

        # Print dimension info
        dim_table = Table(
            title="[bold]Behavioral Dimensions[/bold]",
            box=rich.box.SIMPLE,
        )
        dim_table.add_column("Dimension", style="cyan")
        dim_table.add_column("Bins", style="yellow")
        dim_table.add_column("Range", style="green")

        for dim in stats["dimensions"]:
            dim_table.add_row(
                dim["name"],
                str(dim["bins"]),
                f"[{dim['range'][0]:.2f}, {dim['range'][1]:.2f}]",
            )

        console.print(dim_table)


class MapElitesParentSelector:
    """
    Parent selection strategy that incorporates MAP-Elites archive.
    Can be used alongside or instead of the regular parent selection.
    """

    def __init__(
        self,
        map_elites_archive: MapElitesArchive,
        get_program_func: Callable[[str], Any],
    ):
        self.archive = map_elites_archive
        self.get_program = get_program_func

    def sample_parent(
        self,
        exclude_ids: Optional[List[str]] = None,
        strategy: str = "roulette",
    ) -> Optional[Any]:
        """
        Sample a parent from the MAP-Elites archive.

        Args:
            exclude_ids: Program IDs to exclude
            strategy: Sampling strategy

        Returns:
            A Program object or None if archive is empty
        """
        program_ids = self.archive.sample(
            n=1, exclude_ids=exclude_ids, strategy=strategy
        )

        if not program_ids:
            return None

        return self.get_program(program_ids[0])

    def sample_inspirations(
        self,
        n: int,
        exclude_ids: Optional[List[str]] = None,
        strategy: str = "uniform",
    ) -> List[Any]:
        """
        Sample inspiration programs from diverse behavioral niches.

        Args:
            n: Number of inspirations to sample
            exclude_ids: Program IDs to exclude
            strategy: Sampling strategy

        Returns:
            List of Program objects
        """
        program_ids = self.archive.sample(
            n=n, exclude_ids=exclude_ids, strategy=strategy
        )
        programs = []

        for pid in program_ids:
            prog = self.get_program(pid)
            if prog is not None:
                programs.append(prog)

        return programs
