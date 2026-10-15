"""Application and window tools built on DesktopAutomationInterface (OSA 1.5.1.2).

Application names are resolved against the real list of installed or running
applications (fuzzy matching plus a generic Cyrillic transliteration); there is no
hardcoded alias dictionary. Results are verified against the observed desktop state:
a tool never reports an application as launched or closed unless that state was
seen, and it says so explicitly when verification is impossible. Nothing here
passes model-provided text to a shell.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import sys
import time
from typing import Any

from osa.desktop.automation import (
    DesktopApplication,
    DesktopAutomationInterface,
    DesktopWindow,
)
from osa.desktop.factory import create_desktop_automation
from osa.tools.registry import ToolInterface, ToolResult

_CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_FUZZY_MIN_RATIO = 0.7
_FUZZY_WEIGHT = 0.85
_CONFIDENT_SCORE = 0.9
_CONFIDENT_MARGIN = 0.05
_FUZZY_MARGIN = 0.08
_MAX_LISTED = 25


def transliterate(text: str) -> str:
    """Transliterate Cyrillic letters to Latin (a generic alphabet map, not an app list)."""
    return "".join(_CYRILLIC_TO_LATIN.get(char, char) for char in text.casefold())


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.casefold()).strip()


def _query_forms(query: str) -> tuple[str, ...]:
    forms: list[str] = []

    for candidate in (_normalize(query), _normalize(transliterate(query))):
        if candidate and candidate not in forms:
            forms.append(candidate)

    return tuple(forms)


def _score(forms: Sequence[str], candidate: str) -> float:
    cand = _normalize(candidate)

    if not cand:
        return 0.0

    words = cand.split()
    best = 0.0

    for query in forms:
        if query == cand:
            return 1.0

        if len(query) >= 3:
            if query in words:
                best = max(best, 0.95)
            elif cand.startswith(query):
                best = max(best, 0.92)
            elif query in cand:
                best = max(best, 0.9)
            elif len(cand) >= 3 and cand in query:
                best = max(best, 0.9)

        ratios = [difflib.SequenceMatcher(None, query, cand).ratio()]
        ratios.extend(difflib.SequenceMatcher(None, query, word).ratio() for word in words)
        fuzzy = max(ratios)

        if fuzzy >= _FUZZY_MIN_RATIO:
            best = max(best, fuzzy * _FUZZY_WEIGHT)

    return best


@dataclass(frozen=True, slots=True)
class NameMatch:
    """Result of resolving a user-provided name against real candidates."""

    name: str | None
    suggestions: tuple[str, ...] = ()


def resolve_name(query: str, candidates: Iterable[str]) -> NameMatch:
    """Resolve a name against real candidates; ambiguous or weak matches resolve to None."""
    forms = _query_forms(query)
    unique: dict[str, str] = {}

    for candidate in candidates:
        key = _normalize(candidate)

        if key and key not in unique:
            unique[key] = candidate

    if not forms or not unique:
        return NameMatch(None)

    scored = sorted(
        ((_score(forms, name), name) for name in unique.values()),
        key=lambda item: (-item[0], len(item[1]), item[1].casefold()),
    )
    ranked = [(score, name) for score, name in scored if score > 0]

    if not ranked:
        return NameMatch(None)

    suggestions = tuple(name for _, name in ranked[:5])
    best_score, best_name = ranked[0]
    top = [name for score, name in ranked if score >= best_score - 1e-9]

    if len(top) > 1:
        base = _normalize(top[0])

        if not all(base in _normalize(other) for other in top[1:]):
            return NameMatch(None, suggestions)

    lower = [score for score, _ in ranked if score < best_score - 1e-9]
    runner_up = lower[0] if lower else 0.0
    margin = _CONFIDENT_MARGIN if best_score >= _CONFIDENT_SCORE else _FUZZY_MARGIN

    if best_score - runner_up >= margin:
        return NameMatch(best_name, suggestions)

    return NameMatch(None, suggestions)


def installed_application_names() -> tuple[str, ...]:
    """Return installed application names (macOS only; empty elsewhere)."""
    if sys.platform != "darwin":
        return ()

    roots = (
        Path("/Applications"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    )
    names: set[str] = set()

    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue

        for item in entries:
            if item.suffix == ".app":
                names.add(item.stem)
            elif item.is_dir() and not item.name.startswith("."):
                try:
                    for sub in item.iterdir():
                        if sub.suffix == ".app":
                            names.add(sub.stem)
                except OSError:
                    continue

    return tuple(sorted(names, key=str.casefold))


def _default_announce(message: str) -> None:
    print(f" [⚡ {message}]", flush=True)


def _name_present(target: str, names: Iterable[str]) -> bool:
    wanted = _normalize(target)

    for name in names:
        key = _normalize(name)

        if not key:
            continue

        if key == wanted or (len(key) >= 3 and (key in wanted or wanted in key)):
            return True

    return False


def _limited(names: Sequence[str]) -> str:
    shown = ", ".join(names[:_MAX_LISTED])

    if len(names) > _MAX_LISTED:
        shown += f" и ещё {len(names) - _MAX_LISTED}"

    return shown


class _DesktopAppTool(ToolInterface):
    """Shared plumbing: lazy backend, injectable catalog, clock and announcer."""

    def __init__(
        self,
        automation: DesktopAutomationInterface | None = None,
        *,
        installed_provider: Callable[[], Sequence[str]] | None = None,
        announce: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        verify_timeout: float = 6.0,
        poll_interval: float = 0.3,
    ) -> None:
        if verify_timeout < 0 or poll_interval <= 0:
            raise ValueError("verify_timeout must be >= 0 and poll_interval must be > 0.")

        self._automation = automation
        self._installed_provider = installed_provider or installed_application_names
        self._announce = announce or _default_announce
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._verify_timeout = verify_timeout
        self._poll_interval = poll_interval

    def _backend(self) -> DesktopAutomationInterface:
        if self._automation is None:
            self._automation = create_desktop_automation()

        return self._automation

    def _running_applications(self) -> tuple[DesktopApplication, ...]:
        try:
            return tuple(self._backend().applications())
        except Exception:
            return ()

    def _installed_names(self) -> tuple[str, ...]:
        try:
            return tuple(self._installed_provider())
        except Exception:
            return ()

    def _wait_until(self, predicate: Callable[[], bool], timeout: float) -> bool:
        deadline = self._monotonic() + timeout

        while True:
            if predicate():
                return True

            if self._monotonic() >= deadline:
                return False

            self._sleep(self._poll_interval)


class ListApplicationsTool(_DesktopAppTool):
    """List running or installed applications."""

    @property
    def name(self) -> str:
        return "list_applications"

    @property
    def description(self) -> str:
        return (
            "List applications on this computer. scope 'running' (default) shows open "
            "applications, 'installed' shows applications that can be launched."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["running", "installed"],
                    "description": "Which applications to list. Defaults to 'running'.",
                },
            },
            "required": [],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        scope = str(arguments.get("scope", "running") or "running").strip().lower()

        if scope == "installed":
            names = self._installed_names()

            if not names:
                return ToolResult(
                    success=False,
                    error="Список установленных приложений недоступен на этой платформе.",
                )

            return ToolResult(
                success=True,
                output=f"Установленные приложения: {_limited(list(names))}.",
                metadata={"scope": "installed", "count": len(names)},
            )

        if scope != "running":
            return ToolResult(success=False, error="scope must be 'running' or 'installed'.")

        names = [app.name for app in self._running_applications()]

        if not names:
            return ToolResult(
                success=True,
                output=(
                    "Запущенные приложения не обнаружены или список недоступен "
                    "(возможно, нет разрешения на управление компьютером)."
                ),
                metadata={"scope": "running", "count": 0},
            )

        return ToolResult(
            success=True,
            output=f"Запущенные приложения: {_limited(names)}.",
            metadata={"scope": "running", "count": len(names)},
        )


def _window_label(window: DesktopWindow) -> str:
    app = (window.application_id or "").strip()
    title = (window.title or "").strip()

    if app and title and _normalize(app) != _normalize(title):
        return f"{app} — {title}"

    return title or app or window.window_id


class ListWindowsTool(_DesktopAppTool):
    """List open windows."""

    @property
    def name(self) -> str:
        return "list_windows"

    @property
    def description(self) -> str:
        return "List open windows (application and window title)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            windows = tuple(self._backend().windows())
        except Exception as exc:
            return ToolResult(success=False, error=f"Не удалось получить список окон: {exc}")

        if not windows:
            return ToolResult(
                success=True,
                output=(
                    "Открытых окон не найдено или список недоступен "
                    "(возможно, нет разрешения на управление компьютером)."
                ),
                metadata={"count": 0},
            )

        labels = [_window_label(window) for window in windows]

        return ToolResult(
            success=True,
            output="Открытые окна:\n" + "\n".join(labels[:_MAX_LISTED * 2]),
            metadata={"count": len(labels)},
        )


class FocusWindowTool(_DesktopAppTool):
    """Bring a window to the foreground by application name or title."""

    @property
    def name(self) -> str:
        return "focus_window"

    @property
    def description(self) -> str:
        return (
            "Bring an open window to the foreground. query is a part of the application "
            "name or of the window title, as shown by list_windows."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Application name or part of the window title.",
                },
            },
            "required": ["query"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()

        if not query:
            return ToolResult(success=False, error="Window query cannot be empty.")

        backend = self._backend()

        try:
            windows = tuple(backend.windows())
        except Exception as exc:
            return ToolResult(success=False, error=f"Не удалось получить список окон: {exc}")

        if not windows:
            return ToolResult(
                success=False,
                error="Открытых окон не найдено или список недоступен.",
            )

        by_label: dict[str, DesktopWindow] = {}

        for window in windows:
            by_label.setdefault(_window_label(window), window)

        match = resolve_name(query, by_label)

        if match.name is None:
            hint = match.suggestions or tuple(by_label)
            return ToolResult(
                success=False,
                error=f"Не нашёл окно '{query}'. Открыты: {_limited(list(hint))}.",
                metadata={"candidates": list(hint)},
            )

        window = by_label[match.name]
        self._announce(f"Фокус: {match.name}")

        try:
            backend.activate_window(window.window_id)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Не удалось переключиться на '{match.name}': {exc}",
            )

        return ToolResult(
            success=True,
            output=f"Окно '{match.name}' выведено на передний план.",
            metadata={"window": match.name},
        )


class LaunchApplicationTool(_DesktopAppTool):
    """Launch an application and verify that it is running."""

    @property
    def name(self) -> str:
        return "launch_application"

    @property
    def description(self) -> str:
        return (
            "Launch an installed application and bring it to the foreground. Use the "
            "application's real name, preferably in Latin letters (for example "
            "'Google Chrome'). Use list_applications with scope 'installed' to see names."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "application_name": {
                    "type": "string",
                    "description": "Real name of the application to launch.",
                },
            },
            "required": ["application_name"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_name = str(arguments.get("application_name", "")).strip()

        if not raw_name:
            return ToolResult(success=False, error="Application name cannot be empty.")

        backend = self._backend()
        installed = self._installed_names()
        before = self._running_applications()
        before_keys = {_normalize(app.name) for app in before}

        if installed:
            match = resolve_name(raw_name, installed)
            target = match.name

            if target is None:
                running_match = resolve_name(raw_name, [app.name for app in before])
                target = running_match.name

            if target is None:
                hint = match.suggestions or installed
                return ToolResult(
                    success=False,
                    error=(
                        f"Не нашёл приложение '{raw_name}'. "
                        f"Возможные варианты: {_limited(list(hint))}."
                    ),
                    metadata={"candidates": list(hint)},
                )
        else:
            running_match = resolve_name(raw_name, [app.name for app in before])
            target = running_match.name or raw_name

        self._announce(f"Запуск: {target}")

        try:
            backend.launch_application(target)
        except Exception as exc:
            return ToolResult(success=False, error=f"Не удалось запустить '{target}': {exc}")

        observed = False

        def appeared() -> bool:
            nonlocal observed
            names = [app.name for app in self._running_applications()]

            if names:
                observed = True

            return _name_present(target, names) or any(
                _normalize(name) not in before_keys for name in names
            )

        if self._wait_until(appeared, self._verify_timeout):
            return ToolResult(
                success=True,
                output=f"Приложение '{target}' запущено.",
                metadata={"application": target, "verified": True},
            )

        if not observed:
            return ToolResult(
                success=True,
                output=(
                    f"Команда запуска '{target}' выполнена, но проверить результат не удалось: "
                    "список запущенных приложений недоступен."
                ),
                metadata={"application": target, "verified": False},
            )

        return ToolResult(
            success=False,
            error=(
                f"Команда запуска '{target}' выполнена, но приложение не появилось "
                "среди запущенных."
            ),
            metadata={"application": target, "verified": False},
        )


class CloseApplicationTool(_DesktopAppTool):
    """Close a running application and verify that it is gone."""

    def __init__(
        self,
        automation: DesktopAutomationInterface | None = None,
        *,
        verify_timeout: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(automation, verify_timeout=verify_timeout, **kwargs)

    @property
    def name(self) -> str:
        return "close_application"

    @property
    def description(self) -> str:
        return (
            "Quit a running application. Use the application's real name as shown by "
            "list_applications."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "application_name": {
                    "type": "string",
                    "description": "Name of the running application to quit.",
                },
            },
            "required": ["application_name"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_name = str(arguments.get("application_name", "")).strip()

        if not raw_name:
            return ToolResult(success=False, error="Application name cannot be empty.")

        backend = self._backend()
        running = self._running_applications()

        if not running:
            return ToolResult(
                success=False,
                error="Не удалось получить список запущенных приложений, закрывать нечего.",
            )

        by_name = {app.name: app for app in running}
        match = resolve_name(raw_name, by_name)

        if match.name is None:
            hint = match.suggestions or tuple(by_name)
            return ToolResult(
                success=False,
                error=f"Приложение '{raw_name}' не запущено. Запущены: {_limited(list(hint))}.",
                metadata={"candidates": list(hint)},
            )

        app = by_name[match.name]
        self._announce(f"Закрытие: {app.name}")

        try:
            backend.close_application(app.application_id)
        except Exception as first_error:
            try:
                backend.close_application(app.name)
            except Exception:
                return ToolResult(
                    success=False,
                    error=f"Не удалось закрыть '{app.name}': {first_error}",
                )

        def gone() -> bool:
            names = [item.name for item in self._running_applications()]
            return not _name_present(app.name, names)

        if self._wait_until(gone, self._verify_timeout):
            return ToolResult(
                success=True,
                output=f"Приложение '{app.name}' закрыто.",
                metadata={"application": app.name, "verified": True},
            )

        return ToolResult(
            success=False,
            error=(
                f"Приложение '{app.name}' не закрылось "
                "(возможно, оно ждёт подтверждения сохранения)."
            ),
            metadata={"application": app.name, "verified": False},
        )
