import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, List, Set
from .map_elites import MapElitesArchive

logger = logging.getLogger(__name__)


class ContextSelectorStrategy(ABC):
    """Abstract base class for context selection strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        get_island_idx_func: Optional[Callable[[str], Optional[int]]] = None,
        program_from_row_func: Optional[Callable[[sqlite3.Row], Any]] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_program = get_program_func
        self.best_program_id = best_program_id
        self.get_island_idx = get_island_idx_func
        self.program_from_row = program_from_row_func

    @property
    def minimize(self) -> bool:
        """Whether to minimize (True) or maximize (False) combined_score."""
        return getattr(self.config, "minimize", False)

    @property
    def sort_order(self) -> str:
        """SQL sort order: 'ASC' for minimize, 'DESC' for maximize."""
        return "ASC" if self.minimize else "DESC"

    @abstractmethod
    def sample_context(self, parent: Any, n: int) -> List[Any]:
        """Sample context programs for the given parent."""
        pass


class ArchiveInspirationSelector(ContextSelectorStrategy):
    """Strategy for selecting archive inspirations."""

    def sample_context(self, parent: Any, n: int) -> List[Any]:
        """Sample archive inspirations based on elites, best program, and random selection."""
        if n <= 0:
            return []
        if not hasattr(self.config, "elite_selection_ratio"):
            raise ConnectionError(
                "Config missing elite_selection_ratio for inspiration sampling."
            )

        parent_island_idx = (
            self.get_island_idx(parent.id) if self.get_island_idx else None
        )

        inspirations: List[Any] = []
        insp_ids: Set[str] = {parent.id}

        enforce_separation = getattr(self.config, "enforce_island_separation", False)

        # 1. Best program (only if correct)
        if self.best_program_id and self.best_program_id not in insp_ids:
            prog = self.get_program(self.best_program_id)
            if prog and prog.correct:
                if enforce_separation:
                    if prog.island_idx == parent_island_idx:
                        inspirations.append(prog)
                        insp_ids.add(prog.id)
                else:
                    inspirations.append(prog)
                    insp_ids.add(prog.id)

        # 2. Elites from parent's island
        num_elites = max(0, int(n * self.config.elite_selection_ratio))
        if num_elites > 0 and len(inspirations) < n and parent_island_idx is not None:
            self.cursor.execute(
                f"""
                SELECT p.id FROM programs p
                JOIN archive a ON p.id = a.program_id
                WHERE p.island_idx = ? AND p.correct = 1
                ORDER BY p.combined_score {self.sort_order}
                LIMIT ?
                """,
                (parent_island_idx, num_elites + len(insp_ids)),
            )
            for row in self.cursor.fetchall():
                if len(inspirations) >= n:
                    break
                prog = self.get_program(row["id"])
                if prog and prog.id not in insp_ids:
                    inspirations.append(prog)
                    insp_ids.add(prog.id)

        # 3. Random correct programs from parent's island
        if len(inspirations) < n and parent_island_idx is not None:
            needed = n - len(inspirations)
            if needed > 0:
                placeholders_rand = ",".join("?" * len(insp_ids))
                sql_rand = f"""
                    SELECT p.id FROM programs p
                    JOIN archive a ON p.id = a.program_id
                    WHERE p.island_idx = ? AND p.correct = 1
                    AND p.id NOT IN ({placeholders_rand})
                    ORDER BY RANDOM() LIMIT ?
                """
                params_rand = [parent_island_idx] + list(insp_ids) + [needed]

                self.cursor.execute(sql_rand, params_rand)
                for row in self.cursor.fetchall():
                    prog = self.get_program(row["id"])
                    if prog:  # id is already not in insp_ids from query
                        inspirations.append(prog)

        # 4. Fallback to global random sampling if not enough inspirations
        # found on island
        if len(inspirations) < n and not enforce_separation:
            needed = n - len(inspirations)
            if needed > 0:
                placeholders_rand = ",".join("?" * len(insp_ids))
                sql_rand = f"""SELECT p.id FROM programs p
                                 JOIN archive a ON p.id = a.program_id
                                 WHERE p.correct = 1
                                 AND p.id NOT IN ({placeholders_rand})
                                 ORDER BY RANDOM() LIMIT ?
                                 """
                params_rand = list(insp_ids) + [needed]
                self.cursor.execute(sql_rand, params_rand)
                for row in self.cursor.fetchall():
                    prog = self.get_program(row["id"])
                    if prog:
                        inspirations.append(prog)

        if inspirations:
            inspiration_details = [
                f"{p.id} (Gen: {p.generation}, "
                f"Score: {p.combined_score or 0.0:.4f}, "
                f"Island: {p.island_idx})"
                for p in inspirations
            ]
            logger.info(
                f"Sampled {len(inspirations)} archive inspirations: "
                f"{inspiration_details}"
            )
        return inspirations


class TopKInspirationSelector(ContextSelectorStrategy):
    """Strategy for selecting top-k inspirations from archive."""

    def sample_context(
        self, parent: Any, excluded_programs: List[Any], k: int
    ) -> List[Any]:
        """
        Get the top-k best performing programs from the archive excluding the parent
        and already selected archive inspirations.
        """
        if k <= 0:
            return []

        enforce_separation = getattr(self.config, "enforce_island_separation", False)
        parent_island_idx = parent.island_idx

        if enforce_separation and parent_island_idx is None:
            logger.debug(
                f"Parent {parent.id} has no island and island separation is enforced, "
                "skipping top-k inspirations."
            )
            return []

        # Build set of IDs to exclude (parent + archive inspirations)
        excluded_ids: Set[str] = {parent.id}
        excluded_ids.update(prog.id for prog in excluded_programs)

        if not hasattr(self.config, "archive_size") or self.config.archive_size <= 0:
            return []

        # Query archive for programs
        placeholders = ",".join("?" * len(excluded_ids))

        if enforce_separation and parent_island_idx is not None:
            # Only search within parent's island
            query = f"""
                SELECT p.*
                FROM programs p
                JOIN archive a ON p.id = a.program_id
                WHERE p.island_idx = ? AND p.id NOT IN ({placeholders})
                AND p.correct = 1
            """
            params = [parent_island_idx] + list(excluded_ids)
            search_scope = f"island {parent_island_idx}"
        else:
            # Search globally across all islands
            query = f"""
                SELECT p.*
                FROM programs p
                JOIN archive a ON p.id = a.program_id
                WHERE p.id NOT IN ({placeholders})
                AND p.correct = 1
            """
            params = list(excluded_ids)
            search_scope = "all islands"

        self.cursor.execute(query, params)
        archive_rows = self.cursor.fetchall()

        if not archive_rows:
            logger.debug(
                f"No eligible archived programs for top-k selection on {search_scope}."
            )
            return []

        archive_programs = [
            self.program_from_row(row) for row in archive_rows if self.program_from_row
        ]
        archive_programs = [p for p in archive_programs if p]

        if not archive_programs:
            return []

        # Sort by performance - prioritize combined_score, then average metrics
        # When minimizing, lower is better; when maximizing, higher is better
        def sort_key(prog: Any) -> float:
            if prog.combined_score is not None:
                return prog.combined_score
            elif prog.public_metrics:
                return sum(prog.public_metrics.values()) / len(prog.public_metrics)
            else:
                # Fallback: use worst possible value based on optimization direction
                return float("inf") if self.minimize else -float("inf")

        # reverse=True for maximize (highest first), reverse=False for minimize (lowest first)
        sorted_programs = sorted(
            archive_programs, key=sort_key, reverse=not self.minimize
        )

        # Return top-k programs
        top_k = sorted_programs[:k]

        if top_k:
            inspiration_details = [
                f"{p.id} (Gen: {p.generation}, "
                f"Score: {p.combined_score or 0.0:.4f}, "
                f"Island: {p.island_idx})"
                for p in top_k
            ]
            logger.info(
                f"Selected {len(top_k)} top-k inspirations from archive on "
                f"{search_scope}: {inspiration_details}"
            )

        return top_k


class MapElitesInspirationSelector(ContextSelectorStrategy):
    """Strategy for selecting inspirations from MAP-Elites archive."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        map_elites_archive: Optional[MapElitesArchive] = None,
        **kwargs: Any,
    ):
        super().__init__(cursor, conn, config, get_program_func, **kwargs)
        self.map_elites_archive = map_elites_archive

    def sample_context(
        self, parent: Any, n: int, exclude_programs: Optional[List[Any]] = None
    ) -> List[Any]:
        """Sample diverse inspirations from the MAP-Elites archive."""
        if not self.map_elites_archive or n <= 0:
            return []

        exclude_ids = [parent.id]
        if exclude_programs:
            exclude_ids.extend([p.id for p in exclude_programs])

        # Use the selector from map_elites.py logic
        # We need to manually invoke sampling logic on the archive
        program_ids = self.map_elites_archive.sample(
            n=n, exclude_ids=exclude_ids, strategy="uniform"
        )

        inspirations = []
        for pid in program_ids:
            prog = self.get_program(pid)
            if prog:
                inspirations.append(prog)

        if inspirations:
            logger.info(f"Sampled {len(inspirations)} MAP-Elites inspirations")

        return inspirations


class CombinedContextSelector:
    """Combined context selector that handles both archive inspirations and top-k selection."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        get_island_idx_func: Optional[Callable[[str], Optional[int]]] = None,
        program_from_row_func: Optional[Callable[[sqlite3.Row], Any]] = None,
        map_elites_archive: Optional[MapElitesArchive] = None,
    ):
        self.archive_selector = ArchiveInspirationSelector(
            cursor=cursor,
            conn=conn,
            config=config,
            get_program_func=get_program_func,
            best_program_id=best_program_id,
            get_island_idx_func=get_island_idx_func,
            program_from_row_func=program_from_row_func,
        )
        self.topk_selector = TopKInspirationSelector(
            cursor=cursor,
            conn=conn,
            config=config,
            get_program_func=get_program_func,
            best_program_id=best_program_id,
            get_island_idx_func=get_island_idx_func,
            program_from_row_func=program_from_row_func,
        )
        self.map_elites_selector = MapElitesInspirationSelector(
            cursor=cursor,
            conn=conn,
            config=config,
            get_program_func=get_program_func,
            best_program_id=best_program_id,
            get_island_idx_func=get_island_idx_func,
            program_from_row_func=program_from_row_func,
            map_elites_archive=map_elites_archive,
        )

    def sample_context(
        self, parent: Any, num_archive: int, num_topk: int
    ) -> tuple[List[Any], List[Any]]:
        """Sample archive, top-k, and potentially MAP-Elites inspirations."""

        # Determine if we should replace some archive slots with MAP-Elites samples
        # This simple logic allocates half of num_archive to MAP-Elites if available
        num_map_elites = 0
        if self.map_elites_selector.map_elites_archive and num_archive > 0:
            # Check config for probability/ratio if desired, for now hardcode 50% split
            # or use a config value if present
            if (
                getattr(
                    self.map_elites_selector.map_elites_archive.config,
                    "selection_probability",
                    0,
                )
                > 0
            ):
                num_map_elites = num_archive // 2
                num_archive = num_archive - num_map_elites

        archive_inspirations = self.archive_selector.sample_context(parent, num_archive)

        # Sample MAP-Elites inspirations
        map_elites_inspirations = []
        if num_map_elites > 0:
            map_elites_inspirations = self.map_elites_selector.sample_context(
                parent, num_map_elites, exclude_programs=archive_inspirations
            )

        # Combine them
        combined_archive = archive_inspirations + map_elites_inspirations

        top_k_inspirations = self.topk_selector.sample_context(
            parent, combined_archive, num_topk
        )
        return combined_archive, top_k_inspirations
