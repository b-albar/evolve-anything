from typing import List, Dict
from evolve_anything.database import Program


BASE_SYSTEM_MSG = (
    "You are an expert software engineer tasked with improving the "
    "performance of a given program. Your job is to analyze the current "
    "program and suggest improvements based on the collected feedback from "
    "previous attempts."
)


def perf_str(
    combined_score: float,
    public_metrics: Dict[str, float],
    minimize: bool = False,
) -> str:
    """Format performance metrics as a string.

    Args:
        combined_score: The combined score value.
        public_metrics: Dictionary of public metric names to values.
        minimize: If True, goal is to minimize; if False, goal is to maximize.
    """
    goal = "minimize" if minimize else "maximize"
    result = f"Combined score to {goal}: {combined_score:.2f}\n"
    for key, value in public_metrics.items():
        if isinstance(value, float):
            result += f"{key}: {value:.2f}; "
        else:
            result += f"{key}: {value}; "
    return result[:-2] if result.endswith("; ") else result


def format_text_feedback_section(text_feedback) -> str:
    """Format text feedback for inclusion in prompts."""
    if not text_feedback or not text_feedback.strip():
        return ""

    feedback_text = text_feedback
    if isinstance(feedback_text, list):
        feedback_text = "\n".join(feedback_text)

    return f"""
Here is additional text feedback about the current program:

{feedback_text.strip()}
"""


def construct_eval_history_msg(
    inspiration_programs: List[Program],
    language: str = "python",
    include_text_feedback: bool = False,
) -> str:
    """Construct an edit message for the given parent program and
    inspiration programs."""
    inspiration_str = (
        "Here are the performance metrics of a set of previously "
        "implemented programs:\n\n"
    )
    for i, prog in enumerate(inspiration_programs):
        if i == 0:
            inspiration_str += "# Prior programs\n\n"
        inspiration_str += f"```{language}\n{prog.code}\n```\n\n"
        inspiration_str += (
            f"Performance metrics:\n"
            f"{perf_str(prog.combined_score, prog.public_metrics)}\n\n"
        )

        # Add text feedback if available and requested
        if include_text_feedback and prog.text_feedback:
            feedback_text = prog.text_feedback
            if isinstance(feedback_text, list):
                feedback_text = "\n".join(feedback_text)
            if feedback_text.strip():
                inspiration_str += f"Text feedback:\n{feedback_text.strip()}\n\n"

    return inspiration_str


def format_learning_log(
    ancestor_chain: List["Program"],
    minimize: bool = False,
) -> str:
    """Format ancestor mutations as a learning log for the LLM.

    The ancestor_chain is [parent, grandparent, ...] (nearest first).
    We reverse it so the LLM reads oldest-to-newest (chronological).
    Each entry shows what was tried and whether it helped.
    """
    if not ancestor_chain:
        return ""

    entries = []
    # Reverse for chronological order (oldest ancestor first)
    chronological = list(reversed(ancestor_chain))
    for i, program in enumerate(chronological):
        meta = program.metadata or {}
        patch_name = meta.get("patch_name", "unnamed")
        patch_desc = meta.get("patch_description", "unknown change")
        score = program.combined_score
        correct = program.correct
        status = "correct" if correct else "incorrect"

        # Compute score delta: compare this program to the next one in chain
        # (its child, which is the next item chronologically)
        if i + 1 < len(chronological):
            child_score = chronological[i + 1].combined_score
            delta = child_score - score
            better = (delta > 0) != minimize  # XOR: positive delta = better when maximizing
            direction = "improved" if better else "worsened"
            outcome = f"child score {direction} by {abs(delta):.4f}"
        else:
            # Last in chain = direct parent of the current program
            outcome = f"current parent, score: {score:.4f}"

        entries.append(f"- **{patch_name}**: {patch_desc} -> {outcome} ({status})")

    header = (
        "# Learning Log (Recent Ancestor Mutations)\n"
        "These are recent changes in this lineage and their outcomes. "
        "Avoid repeating approaches that worsened the score.\n\n"
    )
    return header + "\n".join(entries) + "\n"


def format_program_files(files: Dict[str, str], language: str) -> str:
    """Format program files for LLM display.

    Single-file programs get a plain fenced code block.
    Multi-file programs get per-file headers with separate fences.
    """
    if len(files) <= 1:
        code = next(iter(files.values()), "")
        return f"```{language}\n{code}\n```"
    parts = []
    for path, content in sorted(files.items()):
        parts.append(f"### FILE: {path}\n```{language}\n{content}\n```")
    return "\n\n".join(parts)


def construct_individual_program_msg(
    program: Program,
    language: str = "python",
    include_text_feedback: bool = False,
) -> str:
    """Construct a message for a single program for individual analysis."""
    program_str = "# Program to Analyze\n\n"
    program_str += f"```{language}\n{program.code}\n```\n\n"
    program_str += (
        f"Performance metrics:\n"
        f"{perf_str(program.combined_score, program.public_metrics)}\n\n"
    )
    # Include program correctness if available
    if program.correct:
        program_str += "The program is correct and passes all validation tests.\n\n"
    else:
        program_str += (
            "The program is incorrect and does not pass all validation tests.\n\n"
        )

    # Add text feedback if available and requested
    if include_text_feedback and program.text_feedback:
        feedback_text = program.text_feedback
        if isinstance(feedback_text, list):
            feedback_text = "\n".join(feedback_text)
        if feedback_text.strip():
            program_str += f"Text feedback:\n{feedback_text.strip()}\n\n"

    return program_str
