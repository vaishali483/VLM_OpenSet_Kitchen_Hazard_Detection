"""Utilities for loading prompts stored outside Python files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.settings import PROMPTS_DIR


@lru_cache(maxsize=16)
def load_prompt(filename: str) -> str:
    """
    Load and cache one UTF-8 prompt file.

    Restart FastAPI after editing a prompt because prompts are
    deliberately cached after their first load.
    """

    if not filename:
        raise ValueError(
            "Prompt filename cannot be empty."
        )

    prompts_root = PROMPTS_DIR.resolve()

    prompt_path = (
        PROMPTS_DIR
        / filename
    ).resolve()

    if (
        prompt_path != prompts_root
        and prompts_root not in prompt_path.parents
    ):
        raise ValueError(
            "Prompt path must remain inside the prompts directory."
        )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file was not found: {prompt_path}"
        )

    if not prompt_path.is_file():
        raise ValueError(
            f"Prompt path is not a file: {prompt_path}"
        )

    prompt_text = prompt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt_text:
        raise ValueError(
            f"Prompt file is empty: {prompt_path}"
        )

    return prompt_text