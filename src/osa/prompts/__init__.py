"""System prompt loader for OSA.

The prompt lives in ``system_ru.txt`` next to this module. A user may override it
without rebuilding the package by setting ``OSA_SYSTEM_PROMPT`` to a custom file path.
"""

from __future__ import annotations

import os
from pathlib import Path


_FALLBACK = (
    "Ты — OSA (Оса), персональный ассистент с доступом к компьютеру пользователя. "
    "Отвечай кратко, вежливо и по-русски."
)


def _default_prompt_path() -> Path:
    return Path(__file__).with_name("system_ru.txt")


def load_system_prompt(path: Path | None = None) -> str:
    """Return the system prompt text.

    Precedence:
        1. explicit ``path`` argument
        2. ``$OSA_SYSTEM_PROMPT`` env var (if the file exists)
        3. bundled ``system_ru.txt``
        4. short Russian fallback string
    """
    candidates: list[Path] = []

    if path is not None:
        candidates.append(path)
    else:
        env_path = os.environ.get("OSA_SYSTEM_PROMPT", "").strip()

        if env_path:
            candidates.append(Path(env_path))

        candidates.append(_default_prompt_path())

    for candidate in candidates:
        try:
            if candidate.is_file():
                content = candidate.read_text(encoding="utf-8").strip()

                if content:
                    return content
        except OSError:
            continue

    return _FALLBACK
