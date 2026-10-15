"""Tests for application and window tools built on the desktop interface (OSA 1.5.1.2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from osa.desktop.automation import (
    DesktopApplication,
    DesktopAutomationActionError,
    DesktopAutomationInterface,
    DesktopWindow,
)
from osa.tools import (
    CloseApplicationTool,
    FocusWindowTool,
    LaunchApplicationTool,
    ListApplicationsTool,
    ListWindowsTool,
    desktop_apps,
)
from osa.tools.desktop_apps import resolve_name, transliterate

INSTALLED = (
    "Google Chrome",
    "Google Chrome Canary",
    "Google Drive",
    "Telegram",
    "Calculator",
    "Notes",
    "Music",
    "Terminal",
    "Visual Studio Code",
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


def _app(name: str) -> DesktopApplication:
    return DesktopApplication(
        application_id=f"app-{name.lower().replace(' ', '-')}",
        name=name,
        running=True,
    )


class _ScriptedDesktop(DesktopAutomationInterface):
    """Deterministic desktop backend that records calls and simulates state changes."""

    def __init__(
        self,
        running: tuple[str, ...] = (),
        windows: tuple[DesktopWindow, ...] = (),
        *,
        launch_error: Exception | None = None,
        appear_after_polls: int | None = None,
        quit_after_polls: int | None = None,
        fail_close_by_id: bool = False,
        hide_applications: bool = False,
    ) -> None:
        self.running = [_app(name) for name in running]
        self._windows = windows
        self.launch_error = launch_error
        self.appear_after_polls = appear_after_polls
        self.quit_after_polls = quit_after_polls
        self.fail_close_by_id = fail_close_by_id
        self.hide_applications = hide_applications
        self.launched: list[str] = []
        self.closed: list[str] = []
        self.activated: list[str] = []
        self._pending_launch: list | None = None
        self._pending_quit: list | None = None

    def _tick(self) -> None:
        if self._pending_launch is not None:
            if self._pending_launch[1] <= 0:
                self.running.append(_app(self._pending_launch[0]))
                self._pending_launch = None
            else:
                self._pending_launch[1] -= 1

        if self._pending_quit is not None:
            if self._pending_quit[1] <= 0:
                self.running = [a for a in self.running if a is not self._pending_quit[0]]
                self._pending_quit = None
            else:
                self._pending_quit[1] -= 1

    def applications(self):
        if self.hide_applications:
            return ()

        self._tick()
        return tuple(self.running)

    def launch_application(self, application):
        self.launched.append(application)

        if self.launch_error is not None:
            raise self.launch_error

        if self.appear_after_polls is not None:
            self._pending_launch = [application, self.appear_after_polls]

        return _app(application)

    def close_application(self, application_id):
        self.closed.append(application_id)

        if self.fail_close_by_id and application_id.startswith("app-"):
            raise DesktopAutomationActionError("unknown application id")

        if self.quit_after_polls is not None:
            for app in self.running:
                if application_id in (app.application_id, app.name):
                    self._pending_quit = [app, self.quit_after_polls]

    def windows(self):
        return self._windows

    def activate_window(self, window_id):
        self.activated.append(window_id)
        return DesktopWindow(window_id=window_id, title="x", focused=True)

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError

    close_window = find_element = click_element = click_point = _unsupported
    type_text = press = hotkey = screenshot = _unsupported

    def health_check(self):
        return True


def _kwargs(clock: _Clock, installed=INSTALLED) -> dict:
    return {
        "installed_provider": lambda: installed,
        "announce": lambda message: None,
        "sleep": clock.sleep,
        "monotonic": clock.monotonic,
    }


# --- name resolution ---------------------------------------------------------------


def test_resolve_exact_and_partial_names() -> None:
    assert resolve_name("Chrome", INSTALLED).name == "Google Chrome"
    assert resolve_name("google chrome", INSTALLED).name == "Google Chrome"
    assert resolve_name("NOTES", INSTALLED).name == "Notes"
    assert resolve_name("Notes app", INSTALLED).name == "Notes"


def test_resolve_cyrillic_by_transliteration() -> None:
    assert resolve_name("хром", INSTALLED).name == "Google Chrome"
    assert resolve_name("телеграм", INSTALLED).name == "Telegram"
    assert resolve_name("калькулятор", INSTALLED).name == "Calculator"
    assert resolve_name("терминал", INSTALLED).name == "Terminal"


def test_resolve_ambiguous_query_returns_suggestions_only() -> None:
    match = resolve_name("google", INSTALLED)

    assert match.name is None
    assert "Google Drive" in match.suggestions


def test_resolve_unknown_name_is_not_guessed() -> None:
    assert resolve_name("заметки", INSTALLED).name is None
    assert resolve_name("zzzz", INSTALLED).name is None
    assert resolve_name("", INSTALLED).name is None
    assert resolve_name("Chrome", ()).name is None


def test_transliterate_is_a_plain_alphabet_map() -> None:
    assert transliterate("Хром") == "khrom"
    assert transliterate("Chrome") == "chrome"


# --- launch_application ------------------------------------------------------------


def test_launch_uses_real_installed_name_and_verifies() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",), appear_after_polls=2)
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "chrome"})

    assert result.success is True
    assert backend.launched == ["Google Chrome"]
    assert result.metadata["verified"] is True


def test_launch_resolves_cyrillic_name() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",), appear_after_polls=1)
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "хром"})

    assert result.success is True
    assert backend.launched == ["Google Chrome"]


def test_launch_unknown_application_is_not_guessed() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",))
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "заметки"})

    assert result.success is False
    assert backend.launched == []
    assert "Google Chrome" in (result.error or "")


def test_launch_not_appearing_is_reported_as_failure() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",))
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Telegram"})

    assert result.success is False
    assert backend.launched == ["Telegram"]
    assert clock.now >= 6.0
    assert result.metadata["verified"] is False


def test_launch_without_observable_state_is_success_but_unverified() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(hide_applications=True)
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Telegram"})

    assert result.success is True
    assert result.metadata["verified"] is False
    assert "проверить" in result.output


def test_launch_already_running_application_succeeds_immediately() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Google Chrome",))
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Chrome"})

    assert result.success is True
    assert clock.sleeps == 0


def test_launch_backend_error_is_reported() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(
        running=("Finder",),
        launch_error=DesktopAutomationActionError("open failed"),
    )
    tool = LaunchApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Notes"})

    assert result.success is False
    assert "open failed" in (result.error or "")


def test_launch_without_catalog_passes_raw_name_to_backend() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",), appear_after_polls=1)
    tool = LaunchApplicationTool(backend, **_kwargs(clock, installed=()))

    result = tool.execute({"application_name": "Some Tool"})

    assert result.success is True
    assert backend.launched == ["Some Tool"]


def test_launch_rejects_empty_name() -> None:
    clock = _Clock()
    tool = LaunchApplicationTool(_ScriptedDesktop(), **_kwargs(clock))

    assert tool.execute({"application_name": "  "}).success is False
    assert tool.execute({}).success is False


# --- close_application -------------------------------------------------------------


def test_close_resolves_running_app_and_verifies_exit() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Google Chrome", "Telegram"), quit_after_polls=1)
    tool = CloseApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "хром"})

    assert result.success is True
    assert backend.closed == ["app-google-chrome"]
    assert result.metadata["verified"] is True


def test_close_that_does_not_take_effect_is_reported_as_failure() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Google Chrome",))
    tool = CloseApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Chrome"})

    assert result.success is False
    assert clock.now >= 5.0
    assert result.metadata["verified"] is False


def test_close_application_that_is_not_running_does_nothing() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder",))
    tool = CloseApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Telegram"})

    assert result.success is False
    assert backend.closed == []
    assert "Finder" in (result.error or "")


def test_close_never_forwards_model_text_to_the_backend() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Notes",), quit_after_polls=0)
    tool = CloseApplicationTool(backend, **_kwargs(clock))
    hostile = 'Notes" & do shell script "touch /tmp/pwned'

    tool.execute({"application_name": hostile})

    assert backend.closed == ["app-notes"]
    assert all('"' not in item and "shell" not in item for item in backend.closed)


def test_close_falls_back_to_name_when_id_is_rejected() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(
        running=("Google Chrome",),
        quit_after_polls=0,
        fail_close_by_id=True,
    )
    tool = CloseApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Chrome"})

    assert result.success is True
    assert backend.closed == ["app-google-chrome", "Google Chrome"]


def test_close_without_observable_state_does_nothing() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Notes",), hide_applications=True)
    tool = CloseApplicationTool(backend, **_kwargs(clock))

    result = tool.execute({"application_name": "Notes"})

    assert result.success is False
    assert backend.closed == []


# --- list_applications / list_windows / focus_window -------------------------------


def test_list_applications_running_and_installed() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(running=("Finder", "Telegram"))
    tool = ListApplicationsTool(backend, **_kwargs(clock))

    running = tool.execute({})
    installed = tool.execute({"scope": "installed"})

    assert running.success is True and "Telegram" in running.output
    assert installed.success is True and "Google Chrome" in installed.output


def test_list_applications_installed_unavailable_is_failure() -> None:
    clock = _Clock()
    tool = ListApplicationsTool(_ScriptedDesktop(), **_kwargs(clock, installed=()))

    assert tool.execute({"scope": "installed"}).success is False


def test_list_applications_rejects_unknown_scope() -> None:
    clock = _Clock()
    tool = ListApplicationsTool(_ScriptedDesktop(), **_kwargs(clock))

    assert tool.execute({"scope": "everything"}).success is False


def _windows() -> tuple[DesktopWindow, ...]:
    return (
        DesktopWindow(
            window_id="win-1-google-chrome",
            title="YouTube",
            application_id="Google Chrome",
        ),
        DesktopWindow(
            window_id="win-2-telegram",
            title="Telegram",
            application_id="Telegram",
        ),
        DesktopWindow(
            window_id="win-3-notes",
            title="Shopping list",
            application_id="Notes",
        ),
    )


def test_list_windows_shows_application_and_title() -> None:
    clock = _Clock()
    tool = ListWindowsTool(_ScriptedDesktop(windows=_windows()), **_kwargs(clock))

    result = tool.execute({})

    assert result.success is True
    assert "Google Chrome — YouTube" in result.output
    assert result.metadata["count"] == 3


def test_list_windows_without_windows_is_not_an_error() -> None:
    clock = _Clock()
    tool = ListWindowsTool(_ScriptedDesktop(), **_kwargs(clock))

    result = tool.execute({})

    assert result.success is True
    assert result.metadata["count"] == 0


def test_focus_window_by_application_name_and_by_title() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(windows=_windows())
    tool = FocusWindowTool(backend, **_kwargs(clock))

    by_app = tool.execute({"query": "хром"})
    by_title = tool.execute({"query": "shopping"})

    assert by_app.success is True and by_title.success is True
    assert backend.activated == ["win-1-google-chrome", "win-3-notes"]


def test_focus_window_unknown_query_does_not_activate_anything() -> None:
    clock = _Clock()
    backend = _ScriptedDesktop(windows=_windows())
    tool = FocusWindowTool(backend, **_kwargs(clock))

    result = tool.execute({"query": "photoshop"})

    assert result.success is False
    assert backend.activated == []
    assert "Telegram" in (result.error or "")


def test_focus_window_requires_query_and_windows() -> None:
    clock = _Clock()

    assert FocusWindowTool(_ScriptedDesktop(), **_kwargs(clock)).execute({}).success is False
    assert (
        FocusWindowTool(_ScriptedDesktop(), **_kwargs(clock)).execute({"query": "x"}).success
        is False
    )


# --- structure ---------------------------------------------------------------------


def test_tools_expose_expected_names_and_schemas_without_a_backend() -> None:
    assert LaunchApplicationTool().name == "launch_application"
    assert CloseApplicationTool().name == "close_application"
    assert ListApplicationsTool().name == "list_applications"
    assert ListWindowsTool().name == "list_windows"
    assert FocusWindowTool().name == "focus_window"
    assert LaunchApplicationTool().parameters["required"] == ["application_name"]
    assert FocusWindowTool().parameters["required"] == ["query"]


def test_desktop_apps_module_never_uses_a_shell() -> None:
    source = Path(desktop_apps.__file__).read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "shell=True" not in source
    assert "os.system" not in source


def test_main_requires_confirmation_for_close_application() -> None:
    spec = importlib.util.find_spec("osa.main")

    if spec is None or spec.origin is None:
        return

    source = Path(spec.origin).read_text(encoding="utf-8")

    assert '"close_application": PermissionLevel.CONFIRM' in source
    assert '"focus_window": PermissionLevel.ALLOW' in source
