"""The user's extra wake name (OSA 1.5.2.4).

The built-in name (Оса) is permanent. The user may add one more name ("теперь буду
звать тебя Джарвис"); it is validated, stored in a small JSON settings file and pushed
to subscribers (the background listener). Pure standard-library code that behaves the
same on every platform.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DISPLAY_NAME = "Оса"
SETTINGS_KEY = "wake_name"

_NAME = re.compile(r"^[^\W\d_]{3,24}(?:-[^\W\d_]{2,12})?$", re.UNICODE)
_QUOTES = " \t\r\n\"'«»“”„.,;:!?()[]"
_RESERVED = frozenset(
    {"выход", "стоп", "закройся", "да", "нет", "yes", "exit", "quit", "stop", "оса", "osa"}
)

_LATIN_TO_CYRILLIC = (
    ("shch", "щ"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("yu", "ю"),
    ("ya", "я"),
    ("yo", "ё"),
    ("ph", "ф"),
    ("th", "т"),
    ("ck", "к"),
    ("ee", "и"),
    ("oo", "у"),
)
_LATIN_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "й", "z": "з",
}
_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def latin_to_cyrillic(text: str) -> str:
    """Spell a Latin word with Russian letters (a plain alphabet map, no word list)."""
    lowered = text.casefold()
    result: list[str] = []
    index = 0

    while index < len(lowered):
        for latin, cyrillic in _LATIN_TO_CYRILLIC:
            if lowered.startswith(latin, index):
                result.append(cyrillic)
                index += len(latin)
                break
        else:
            char = lowered[index]
            result.append(_LATIN_SINGLE.get(char, char))
            index += 1

    return "".join(result)


def cyrillic_to_latin(text: str) -> str:
    """Spell a Russian word with Latin letters ("дж" is written as "j")."""
    lowered = text.casefold().replace("дж", "j")

    return "".join(_CYRILLIC_TO_LATIN.get(char, char) for char in lowered)


def _is_latin(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]

    return bool(letters) and all(char.isascii() for char in letters)


def alias_forms(name: str) -> tuple[str, ...]:
    """Return the name plus its spelling in the other alphabet."""
    other = latin_to_cyrillic(name) if _is_latin(name) else cyrillic_to_latin(name)
    forms = [name]

    if other and other != name:
        forms.append(other)

    return tuple(forms)


def normalize_wake_name(raw: str) -> str:
    """Validate and normalize a name; raise ValueError with a user-facing reason."""
    name = str(raw).strip(_QUOTES).casefold().replace("ё", "е")

    if not name:
        return _reject("Имя не может быть пустым.")

    if name in _RESERVED:
        if name in ("оса", "osa"):
            return _reject("Имя «Оса» уже основное и откликается всегда.")

        return _reject(f"Имя «{name}» зарезервировано, выберите другое.")

    if not _NAME.match(name):
        return _reject(
            "Имя должно быть одним словом из 3–24 букв, без цифр и пробелов. "
            "Например: Джарвис."
        )

    return name


def _reject(message: str) -> str:
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class NameChange:
    """Result of changing the extra name."""

    name: str
    persisted: bool


class WakeNameService:
    """Holds the user's extra name, persists it and notifies subscribers."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._name: str | None = None
        self._subscribers: list[Callable[[tuple[str, ...]], None]] = []
        self._load()

    @property
    def custom_name(self) -> str | None:
        with self._lock:
            return self._name

    def alias_names(self) -> tuple[str, ...]:
        """Spellings to listen for (the name and its other-alphabet form)."""
        with self._lock:
            return alias_forms(self._name) if self._name else ()

    def subscribe(
        self,
        callback: Callable[[tuple[str, ...]], None],
        *,
        notify: bool = True,
    ) -> None:
        """Register a callback for name changes; by default call it right away."""
        with self._lock:
            self._subscribers.append(callback)
            current = self.alias_names()

        if notify:
            self._call(callback, current)

    def set_custom_name(self, raw: str) -> NameChange:
        """Set the extra name (replacing the previous one); raises ValueError if invalid."""
        name = normalize_wake_name(raw)

        with self._lock:
            self._name = name
            persisted = self._save()

        self._notify()

        return NameChange(name, persisted)

    def clear_custom_name(self) -> bool:
        """Remove the extra name; return False when there was none."""
        with self._lock:
            had_name = self._name is not None
            self._name = None
            self._save()

        self._notify()

        return had_name

    def _notify(self) -> None:
        with self._lock:
            callbacks = list(self._subscribers)
            current = self.alias_names()

        for callback in callbacks:
            self._call(callback, current)

    @staticmethod
    def _call(callback: Callable[[tuple[str, ...]], None], names: tuple[str, ...]) -> None:
        try:
            callback(names)
        except Exception:
            pass

    def _read_settings(self) -> dict[str, Any]:
        if self._path is None:
            return {}

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

        return data if isinstance(data, dict) else {}

    def _load(self) -> None:
        stored = self._read_settings().get(SETTINGS_KEY)

        if not isinstance(stored, str):
            return

        try:
            self._name = normalize_wake_name(stored)
        except ValueError:
            self._name = None

    def _save(self) -> bool:
        if self._path is None:
            return False

        settings = self._read_settings()

        if self._name is None:
            settings.pop(SETTINGS_KEY, None)
        else:
            settings[SETTINGS_KEY] = self._name

        settings["version"] = 1
        temporary = self._path.with_name(self._path.name + ".tmp")

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self._path)
        except OSError:
            return False

        return True
