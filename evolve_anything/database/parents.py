import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

import numpy as np  # type: ignore
from .map_elites import MapElitesArchive, MapElitesParentSelector

logger = logging.getLogger(__name__)


def sample_with_powerlaw(items: list, alpha: float = 1.0) -> int:
    """
    Sample an index from a list of items using a power law distribution
    based on their rank (order in the list).

    Parameters
    ----------
    items : list
        List of items to sample from (order implies rank, e.g., best first).
    alpha : float, default=1.0
        Power law exponent.
        - alpha = 0: uniform sampling
        - alpha > 0: items earlier in the list (higher rank) are sampled more.
        - alpha < 0: items later in the list (lower rank) are sampled more.

    Returns
    -------
    int
        Index of the sampled item from the input list.
    """
    if not items:
        raise ValueError("Empty items list for power-law sampling")

    # Probabilities based on rank (index + 1)
    probs = np.array([(i + 1) ** (-alpha) for i in range(len(items))])
    if np.sum(probs) == 0:  # Avoid div by zero if all probs are zero
        # Fallback to uniform if power law results in all zero probabilities
        probs = np.ones(len(items))

    probs = probs / probs.sum()  # Normalize
    logger.info(f"Power law probs: {probs.tolist()}")
    return np.random.choice(len(items), p=probs)


def stable_sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function that avoids overflow.

    Standard sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    This can overflow when x is very large (positive or negative).

    For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))

    Args:
        x: Input value

    Returns:
        Sigmoid value between 0 and 1
    """
    if x >= 0:
        exp_neg_x = np.exp(-x)
        return 1.0 / (1.0 + exp_neg_x)
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


class ParentSamplingStrategy(ABC):
    """Abstract base class for parent sampling strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        island_idx: Optional[int] = None,
        program_from_row_func: Optional[Callable] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_program = get_program_func
        self.best_program_id = best_program_id
        self.island_idx = island_idx
        self.program_from_row = program_from_row_func

    @abstractmethod
    def sample_parent(self) -> Any:
        """Sample and return a parent program."""
        pass

    @property
    def minimize(self) -> bool:
        """Whether to minimize (True) or maximize (False) combined_score."""
        return getattr(self.config, "minimize", False)

    @property
    def sort_order(self) -> str:
        """SQL sort order: 'ASC' for minimize, 'DESC' for maximize."""
        return "ASC" if self.minimize else "DESC"

    def _get_island_idx(self, program_id: str) -> Optional[int]:
        """Get the island index for a given program ID."""
        self.cursor.execute(
            "SELECT island_idx FROM programs WHERE id = ?", (program_id,)
        )
        row = self.cursor.fetchone()
        return row["island_idx"] if row else None

    def _programs_from_rows(self, rows) -> list:
        """Convert DB rows to Program objects using program_from_row if available,
        otherwise fall back to get_program (N+1 queries)."""
        if self.program_from_row:
            return [p for row in rows if (p := self.program_from_row(row))]
        return [p for row in rows if (p := self.get_program(row["id"]))]

    def _fallback_to_best(self, island_idx: Optional[int] = None) -> Any:
        """Try best program, then random correct program. Returns Program or None."""
        if self.best_program_id:
            best_prog = self.get_program(self.best_program_id)
            if (
                best_prog
                and best_prog.correct
                and (island_idx is None or best_prog.island_idx == island_idx)
            ):
                logger.info(
                    f"Fallback: Using best program {self.best_program_id}"
                )
                return best_prog

        return self._random_correct_program(island_idx)

    def _random_correct_program(self, island_idx: Optional[int] = None) -> Any:
        """Get a random correct program, optionally constrained to an island."""
        if island_idx is not None:
            self.cursor.execute(
                """SELECT id FROM programs
                   WHERE correct = 1 AND island_idx = ?
                   ORDER BY RANDOM() LIMIT 1""",
                (island_idx,),
            )
        else:
            self.cursor.execute(
                """SELECT id FROM programs WHERE correct = 1
                   ORDER BY RANDOM() LIMIT 1"""
            )
        row = self.cursor.fetchone()
        if row:
            prog = self.get_program(row["id"])
            if prog:
                logger.info(f"Fallback: Random correct program {row['id']}")
                return prog
        return None


class PowerLawSamplingStrategy(ParentSamplingStrategy):
    """Power law sampling strategy for parent selection."""

    def sample_parent(self) -> Any:
        if not hasattr(self.config, "exploitation_ratio"):
            raise ValueError("Config missing exploitation_ratio for parent sampling.")

        pid: Optional[str] = None
        # Try elite archive for exploitation (archive only contains correct programs)
        if hasattr(self.config, "exploitation_ratio"):
            if np.random.random() < self.config.exploitation_ratio:
                if self.island_idx is not None:
                    self.cursor.execute(
                        """SELECT p.* FROM programs p
                           JOIN archive a ON a.program_id = p.id
                           WHERE p.island_idx = ?""",
                        (self.island_idx,),
                    )
                else:
                    self.cursor.execute(
                        "SELECT p.* FROM programs p JOIN archive a ON a.program_id = p.id"
                    )
                archived_rows = self.cursor.fetchall()
                if archived_rows:
                    archived_programs = self._programs_from_rows(archived_rows)

                    if archived_programs:
                        # Sort by combined_score descending (best first) unless minimizing
                        # reverse=True means higher score first (maximizing)
                        # reverse=False means lower score first (minimizing)
                        archived_programs.sort(
                            key=lambda p: p.combined_score or 0.0,
                            reverse=not self.minimize,
                        )
                        logger.info(
                            f"Island {self.island_idx} => Archived program scores: {[p.combined_score for p in archived_programs]}"
                        )

                        alpha = getattr(self.config, "exploitation_alpha", 1.0)
                        sampled_idx = sample_with_powerlaw(archived_programs, alpha)
                        selected_prog = archived_programs[sampled_idx]
                        pid = selected_prog.id

                        logger.info(
                            f"Exploitation: Sampled from archive: {pid} "
                            f"(Gen: {selected_prog.generation}, "
                            f"Score: {selected_prog.combined_score or 0.0:.4f}, "
                            f"Island: {selected_prog.island_idx})"
                        )

        # Exploration from all correct programs (sorted by performance)
        if not pid:
            if self.island_idx is not None:
                self.cursor.execute(
                    f"""SELECT p.* FROM programs p
                       WHERE p.correct = 1 AND p.island_idx = ?
                       ORDER BY p.combined_score {self.sort_order}""",
                    (self.island_idx,),
                )
            else:
                self.cursor.execute(
                    f"""SELECT p.* FROM programs p
                       WHERE p.correct = 1
                       ORDER BY p.combined_score {self.sort_order}"""
                )
            correct_rows = self.cursor.fetchall()
            if correct_rows:
                correct_programs = self._programs_from_rows(correct_rows)

                if correct_programs:
                    # Use exploration_alpha for more diverse parent selection
                    alpha = getattr(self.config, "exploration_alpha", 0.5)
                    logger.info(
                        f"Island {self.island_idx} => Correct program scores: {[p.combined_score for p in correct_programs]}"
                    )
                    sampled_idx = sample_with_powerlaw(correct_programs, alpha)
                    selected_prog = correct_programs[sampled_idx]
                    pid = selected_prog.id

                    logger.info(
                        f"Exploration (alpha={alpha}): Sampled from all correct: {pid} "
                        f"(Gen: {selected_prog.generation}, "
                        f"Score: {selected_prog.combined_score or 0.0:.4f}, "
                        f"Island: {selected_prog.island_idx})"
                    )

        # Exploration from different islands (only correct programs)
        if (
            not pid
            and hasattr(self.config, "num_islands")
            and self.config.num_islands > 0
            and self.island_idx is None  # Only do this if no island constraint
        ):
            self.cursor.execute("SELECT DISTINCT island_idx FROM programs")
            island_indices = [r["island_idx"] for r in self.cursor.fetchall()]
            if island_indices:
                idx = np.random.choice(island_indices)
                self.cursor.execute(
                    """SELECT p.id FROM programs p
                       WHERE p.island_idx = ? AND p.correct = 1
                       ORDER BY RANDOM() LIMIT 1""",
                    (idx,),
                )
                row = self.cursor.fetchone()
                if row:
                    pid = row["id"]
                    prog = self.get_program(pid)
                    if prog:
                        score = prog.combined_score or 0.0
                        logger.info(
                            f"Exploration: Sampled from island {idx}: {pid} "
                            f"(Gen: {prog.generation}, Score: {score:.4f})"
                        )

        # Fallbacks (only correct programs)
        if not pid:
            fallback = self._fallback_to_best(self.island_idx)
            return fallback

        return self.get_program(pid)


class WeightedSamplingStrategy(ParentSamplingStrategy):
    """Weighted sampling strategy for parent selection."""

    def sample_parent(self) -> Any:
        # Fetch all programs from the archive.
        if self.island_idx is not None:
            self.cursor.execute(
                """
                SELECT p.*
                FROM programs p
                JOIN archive a ON p.id = a.program_id
                WHERE p.correct = 1 AND p.island_idx = ?
                """,
                (self.island_idx,),
            )
        else:
            self.cursor.execute(
                """
                SELECT p.*
                FROM programs p
                JOIN archive a ON p.id = a.program_id
                WHERE p.correct = 1
                """
            )
        archive_rows = self.cursor.fetchall()

        if not archive_rows:
            logger.warning("No archived programs found for weighted sampling.")
            return self._fallback_to_best(self.island_idx)

        eligible_programs = []
        for row in archive_rows:
            p_dict = dict(row)
            # Only need id, combined_score, children_count, generation, island_idx
            # for weighted sampling — skip full deserialization
            eligible_programs.append(p_dict)

        # Calculate baseline performance (alpha_0) as the median
        scores = [p.get("combined_score") or 0.0 for p in eligible_programs]
        alpha_0 = np.median(scores) if scores else 0.0

        # Calculate scale-robust normalization factor (MAD)
        score_deviations = [abs(score - alpha_0) for score in scores]
        mad = np.median(score_deviations) if score_deviations else 1.0
        scale_factor = max(mad, 1e-6)

        # Calculate weights for each program
        weights = []
        lambda_ = self.config.parent_selection_lambda

        for i, p in enumerate(eligible_programs):
            alpha_i = p.get("combined_score") or 0.0
            n_i = p.get("children_count") or 0

            diff = alpha_i - alpha_0
            if self.minimize:
                diff = -diff
            normalized_diff = diff / scale_factor
            s_i = stable_sigmoid(lambda_ * normalized_diff)
            h_i = 1 / (1 + n_i)
            w_i = s_i * h_i
            weights.append(w_i)
            logger.debug(
                f"I-{self.island_idx} => P-{i}: w_i: {w_i:.2f}, s_i: {s_i:.2f}, h_i: {h_i:.2f}, alpha_i: {alpha_i:.2f}, alpha_0: {alpha_0:.2f}, "
                f"normalized_diff: {normalized_diff:.2f}, scale_factor: {scale_factor:.2f}"
            )

        # Normalize weights to get probabilities
        weights_sum = sum(weights)
        if weights_sum > 0:
            probabilities = [w / weights_sum for w in weights]
        else:
            logger.warning(
                "All parent selection weights are zero, falling back to uniform sampling."
            )
            num_eligible = len(eligible_programs)
            probabilities = [1.0 / num_eligible] * num_eligible
        logger.info(
            f"Island {self.island_idx} => Probabilities: {np.array(probabilities).tolist()}"
        )
        logger.info(
            f"Island {self.island_idx} => Scores: {scores}"
        )
        # Sample one parent based on probabilities
        selected_idx = np.random.choice(len(eligible_programs), p=probabilities)
        selected = eligible_programs[selected_idx]

        logger.info(
            f"Sampled parent {selected['id']} "
            f"(Gen: {selected.get('generation')}, "
            f"Score: {selected.get('combined_score', 0.0):.4f}, "
            f"Children: {selected.get('children_count', 0)}, "
            f"Island: {selected.get('island_idx')})"
        )

        return self.get_program(selected["id"])


class BeamSearchSamplingStrategy(ParentSamplingStrategy):
    """Beam search sampling strategy that locks onto a parent for multiple generations."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        island_idx: Optional[int] = None,
        beam_search_parent_id: Optional[str] = None,
        last_iteration: int = 0,
        update_metadata_func: Optional[Callable[[str, Optional[str]], None]] = None,
        get_best_program_func: Optional[Callable[[], Any]] = None,
        program_from_row_func: Optional[Callable] = None,
    ):
        super().__init__(
            cursor, conn, config, get_program_func, best_program_id, island_idx,
            program_from_row_func=program_from_row_func,
        )
        self.beam_search_parent_id = beam_search_parent_id
        self.last_iteration = last_iteration
        self.update_metadata = update_metadata_func
        self.get_best_program_func = get_best_program_func

    def sample_parent(self) -> Any:
        num_beams = getattr(self.config, "num_beams", 5)

        # If no current beam search parent, select one
        if not self.beam_search_parent_id:
            # Get top programs and select one
            if self.get_best_program_func:
                best_program = self.get_best_program_func()
                if best_program:
                    self.beam_search_parent_id = best_program.id
                    if self.update_metadata:
                        self.update_metadata(
                            "beam_search_parent_id", self.beam_search_parent_id
                        )
                    logger.info(
                        f"Beam search: Selected new parent {self.beam_search_parent_id} "
                        f"(Gen: {best_program.generation}, "
                        f"Score: {best_program.combined_score or 0.0:.4f})"
                    )

        # Use the current beam search parent
        if self.beam_search_parent_id:
            parent = self.get_program(self.beam_search_parent_id)
            if parent:
                # Check if we should continue with this parent based on num_beams
                self.cursor.execute(
                    "SELECT COUNT(*) FROM programs WHERE parent_id = ?",
                    (self.beam_search_parent_id,),
                )
                children_count = (self.cursor.fetchone() or [0])[0]

                if children_count < num_beams:
                    logger.info(
                        f"Beam search: Continue with parent {self.beam_search_parent_id} "
                        f"({children_count}/{num_beams} children)"
                    )
                    return parent
                else:
                    # This parent has enough children, select a new one
                    if self.get_best_program_func:
                        best_program = self.get_best_program_func()
                        if best_program:
                            self.beam_search_parent_id = best_program.id
                            if self.update_metadata:
                                self.update_metadata(
                                    "beam_search_parent_id", self.beam_search_parent_id
                                )
                            logger.info(
                                f"Beam search: Switch to new parent {self.beam_search_parent_id} "
                                f"(Gen: {best_program.generation}, "
                                f"Score: {best_program.combined_score or 0.0:.4f})"
                            )
                            return best_program

        # Fallback
        return self._fallback_to_best(self.island_idx)


class BestOfNSamplingStrategy(ParentSamplingStrategy):
    """Best-of-N sampling strategy that always returns the initial program as parent."""

    def sample_parent(self) -> Any:
        # Find the initial program (generation 0) in the specified island or globally
        if self.island_idx is not None:
            self.cursor.execute(
                """SELECT id FROM programs
                   WHERE generation = 0 AND island_idx = ? AND correct = 1
                   ORDER BY id LIMIT 1""",
                (self.island_idx,),
            )
        else:
            self.cursor.execute(
                """SELECT id FROM programs
                   WHERE generation = 0 AND correct = 1
                   ORDER BY id LIMIT 1"""
            )

        row = self.cursor.fetchone()
        if row:
            pid = row["id"]
            prog = self.get_program(pid)
            if prog:
                logger.info(
                    f"Best-of-N: Selected initial program {pid} "
                    f"(Gen: {prog.generation}, "
                    f"Score: {prog.combined_score or 0.0:.4f}, "
                    f"Island: {prog.island_idx})"
                )
                return prog

        # Fallback: if no generation 0 program found, try any correct program
        logger.warning(
            "No generation 0 program found, falling back to any correct program"
        )
        return self._random_correct_program(self.island_idx)


class CombinedParentSelector:
    """Combined parent selector that handles all parent sampling strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        beam_search_parent_id: Optional[str] = None,
        last_iteration: int = 0,
        update_metadata_func: Optional[Callable[[str, Optional[str]], None]] = None,
        get_best_program_func: Optional[Callable[[], Any]] = None,
        map_elites_archive: Optional[MapElitesArchive] = None,
        program_from_row_func: Optional[Callable] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_program = get_program_func
        self.best_program_id = best_program_id
        self.beam_search_parent_id = beam_search_parent_id
        self.last_iteration = last_iteration
        self.update_metadata = update_metadata_func
        self.get_best_program_func = get_best_program_func
        self.map_elites_archive = map_elites_archive
        self.program_from_row = program_from_row_func

    def sample_parent(self, island_idx: Optional[int] = None) -> Any:
        """Sample a parent using the configured sampling strategy."""
        strategy_name = self.config.parent_selection_strategy

        _from_row = self.program_from_row

        if strategy_name == "power_law":
            strategy = PowerLawSamplingStrategy(
                self.cursor,
                self.conn,
                self.config,
                self.get_program,
                self.best_program_id,
                island_idx,
                program_from_row_func=_from_row,
            )
        elif strategy_name == "weighted":
            strategy = WeightedSamplingStrategy(
                self.cursor,
                self.conn,
                self.config,
                self.get_program,
                self.best_program_id,
                island_idx,
                program_from_row_func=_from_row,
            )
        elif strategy_name == "beam_search":
            strategy = BeamSearchSamplingStrategy(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                get_program_func=self.get_program,
                best_program_id=self.best_program_id,
                island_idx=island_idx,
                beam_search_parent_id=self.beam_search_parent_id,
                last_iteration=self.last_iteration,
                update_metadata_func=self.update_metadata,
                get_best_program_func=self.get_best_program_func,
                program_from_row_func=_from_row,
            )
        elif strategy_name == "best_of_n":
            strategy = BestOfNSamplingStrategy(
                self.cursor,
                self.conn,
                self.config,
                self.get_program,
                self.best_program_id,
                island_idx,
                program_from_row_func=_from_row,
            )
        elif strategy_name == "map_elites":
            if not self.map_elites_archive:
                logger.warning(
                    "MAP-Elites strategy selected but archive not initialized. Falling back to weighted."
                )
                strategy = WeightedSamplingStrategy(
                    self.cursor,
                    self.conn,
                    self.config,
                    self.get_program,
                    self.best_program_id,
                    island_idx,
                    program_from_row_func=_from_row,
                )
            else:
                # MapElitesParentSelector needs to look like ParentSamplingStrategy
                # The existing class in map_elites.py has a different signiture for sample_parent
                # We need to wrap it or adapt it.

                # Let's create a temporary adapter class locally or just use logic inline
                # Since MapElitesParentSelector is simple, we can just use it directly
                # passing arguments it needs.

                selector = MapElitesParentSelector(
                    self.map_elites_archive, self.get_program
                )
                # It returns Optional[Program]
                parent = selector.sample_parent()
                if parent:
                    return parent
                # If it failed (empty archive), fall back
                return None
        else:
            raise ValueError(f"Unknown parent selection strategy: {strategy_name}")

        parent = strategy.sample_parent()

        if not parent:
            # Use the strategy's fallback helpers
            parent = strategy._fallback_to_best(island_idx)

        if not parent:
            raise ValueError("Database empty or parent sampling failed.")

        return parent
