"""
Format constants for LLM response parsing.

This module contains customizable constants for parsing LLM responses,
including XML tags, code block markers, and language-specific comment characters.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ResponseFormat:
    """Configuration for parsing LLM responses.

    Attributes:
        name_start_tag: Opening tag for patch name.
        name_end_tag: Closing tag for patch name.
        description_start_tag: Opening tag for patch description.
        description_end_tag: Closing tag for patch description.
        code_start_tag: Opening tag for code blocks (e.g., "```python").
        code_end_tag: Closing tag for code blocks (e.g., "```").
        evolve_block_start: Marker for start of evolvable code section.
        evolve_block_end: Marker for end of evolvable code section.
        json_start_tag: Opening tag for JSON content.
        json_end_tag: Closing tag for JSON content.
    """

    name_start_tag: str = "<NAME>"
    name_end_tag: str = "</NAME>"
    description_start_tag: str = "<DESCRIPTION>"
    description_end_tag: str = "</DESCRIPTION>"
    code_start_tag: str = "```{language}"  # {language} is replaced at runtime
    code_end_tag: str = "```"
    evolve_block_start: str = "EVOLVE-BLOCK-START"
    evolve_block_end: str = "EVOLVE-BLOCK-END"
    evolve_block_comment: str = "EVOLVE-BLOCK"
    json_start_tag: str = "<json>"
    json_end_tag: str = "</json>"


# Comment characters for different programming languages
LANGUAGE_COMMENT_CHARS: Dict[str, str] = {
    "python": "#",
    "cuda": "//",
    "cpp": "//",
    "c": "//",
    "rust": "//",
    "swift": "//",
    "java": "//",
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "json": "//",  # JSON doesn't have comments, but we use // as placeholder
    "json5": "//",
}

# File extensions for different programming languages
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    "cuda": "cu",
    "cpp": "cpp",
    "c": "c",
    "python": "py",
    "rust": "rs",
    "swift": "swift",
    "java": "java",
    "javascript": "js",
    "typescript": "ts",
    "go": "go",
    "json": "json",
    "json5": "json",
}


def get_comment_char(language: str) -> str:
    """Get the comment character for a programming language.

    Args:
        language: Programming language name (lowercase).

    Returns:
        Comment character string (e.g., "#" for Python, "//" for C++).
    """
    return LANGUAGE_COMMENT_CHARS.get(language.lower(), "//")


def get_file_extension(language: str) -> str:
    """Get the file extension for a programming language.

    Args:
        language: Programming language name (lowercase).

    Returns:
        File extension without leading dot (e.g., "py" for Python).

    Raises:
        ValueError: If language is not supported.
    """
    lang = language.lower()
    if lang not in LANGUAGE_EXTENSIONS:
        raise ValueError(f"Language {language} not supported")
    return LANGUAGE_EXTENSIONS[lang]


# Default format instance
DEFAULT_FORMAT = ResponseFormat()
