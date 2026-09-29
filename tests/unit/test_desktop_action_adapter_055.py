import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.actions.desktop import (
    DESKTOP_ACTION_NAMES,
    DesktopActionAdapter,
    DesktopActionAdapterError,
)
from osa.desktop.automation import (
    DesktopAutomationInterface,
    DesktopElementLocator,
    DesktopPoint,
)


class FakeDesktopBackend(DesktopAutomationInterface):
    """Deterministic backend for desktop action adapter tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def applications(self):
        self.calls.append(("applications", None))
        return [{"name": "Calculator"}]

    def launch_application(self, command: str):
        self.calls.append(("launch_application", command))
        return f"launched:{command}"

    def close_application(self, name: str):
        self.calls.append(("close_application", name))
        return f"closed_application:{name}"

    def windows(self):
        self.calls.append(("windows", None))
        return [{"title": "Main"}]

    def activate_window(self, title: str):
        self.calls.append(("activate_window", title))
        return f"activated:{title}"

    def close_window(self, title: str):
        self.calls.append(("close_window", title))
        return f"closed_window:{title}"

    def find_element(self, locator: DesktopElementLocator):
        self.calls.append(("find_element", locator))
        return {
            "kind": locator.kind,
            "value": locator.value,
        }

    def click_element(self, element):
        self.calls.append(("click_element", element))
        return "clicked_element"

    def click_point(self, point: DesktopPoint):
        self.calls.append(("click_point", point))
        return "clicked_point"

    def type_text(self, text: str):
        self.calls.append(("type_text", text))
        return f"typed:{text}"

    def press(self, key: str):
        self.calls.append(("press", key))
        return f"pressed:{key}"

    def hotkey(self, *keys: str):
        self.calls.append(("hotkey", keys))
        return f"hotkey:{','.join(keys)}"

    def screenshot(self):
        self.calls.append(("screenshot", None))
        return "desktop-screenshot"

    def health_check(self):
        self.calls.append(("health_check", None))
        return True


def request(
    name: str,
    arguments: dict | None = None,
) -> ActionRequest:
    """Build a desktop action request."""
    return ActionRequest(
        kind=ActionKind.DESKTOP,
        name=name,
        arguments=arguments or {},
    )


def test_action_names_are_complete_and_stable() -> None:
    expected = (
        "desktop_applications",
        "desktop_launch_application",
        "desktop_close_application",
        "desktop_windows",
        "desktop_activate_window",
        "desktop_close_window",
        "desktop_find_element",
        "desktop_click_element",
        "desktop_click_point",
        "desktop_type",
        "desktop_press",
        "desktop_hotkey",
        "desktop_screenshot",
        "desktop_health_check",
    )

    assert DESKTOP_ACTION_NAMES == expected
    assert len(DESKTOP_ACTION_NAMES) == len(set(DESKTOP_ACTION_NAMES))


def test_adapter_accepts_valid_backend() -> None:
    backend = FakeDesktopBackend()

    adapter = DesktopActionAdapter(backend)

    assert adapter is not None


def test_adapter_rejects_invalid_backend() -> None:
    with pytest.raises(DesktopActionAdapterError):
        DesktopActionAdapter(object())  # type: ignore[arg-type]


def test_applications_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request("desktop_applications")
    )

    assert result.success is True
    assert backend.calls == [
        ("applications", None),
    ]


def test_launch_application_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_launch_application",
            {"command": "Calculator"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("launch_application", "Calculator"),
    ]


def test_close_application_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_close_application",
            {"name": "Calculator"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("close_application", "Calculator"),
    ]


def test_windows_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request("desktop_windows")
    )

    assert result.success is True
    assert backend.calls == [
        ("windows", None),
    ]


def test_activate_window_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_activate_window",
            {"title": "Main"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("activate_window", "Main"),
    ]


def test_close_window_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_close_window",
            {"title": "Main"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("close_window", "Main"),
    ]


def test_find_element_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_find_element",
            {
                "kind": "name",
                "value": "Save",
            },
        )
    )

    assert result.success is True
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "find_element"

    locator = backend.calls[0][1]
    assert isinstance(locator, DesktopElementLocator)
    assert locator.kind == "name"
    assert locator.value == "Save"


def test_click_element_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_click_element",
            {
                "kind": "name",
                "value": "Save",
            },
        )
    )

    assert result.success is True
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "click_element"

    locator = backend.calls[0][1]
    assert isinstance(locator, DesktopElementLocator)
    assert locator.kind == "name"
    assert locator.value == "Save"


def test_click_point_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_click_point",
            {
                "x": 100,
                "y": 200,
            },
        )
    )

    assert result.success is True
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "click_point"

    point = backend.calls[0][1]
    assert isinstance(point, DesktopPoint)
    assert point.x == 100
    assert point.y == 200


def test_type_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_type",
            {"text": "hello"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("type_text", "hello"),
    ]


def test_press_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_press",
            {"key": "ENTER"},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("press", "ENTER"),
    ]


def test_hotkey_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_hotkey",
            {"keys": ["CTRL", "C"]},
        )
    )

    assert result.success is True
    assert backend.calls == [
        ("hotkey", ("CTRL", "C")),
    ]


def test_screenshot_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request("desktop_screenshot")
    )

    assert result.success is True
    assert backend.calls == [
        ("screenshot", None),
    ]


def test_health_check_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request("desktop_health_check")
    )

    assert result.success is True
    assert backend.calls == [
        ("health_check", None),
    ]


def test_execute_rejects_wrong_action_kind() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        ActionRequest(
            kind=ActionKind.BROWSER,
            name="browser_open",
            arguments={"url": "https://example.com"},
        )
    )

    assert result.success is False
    assert result.error


def test_execute_rejects_unsupported_action() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request("desktop_unknown")
    )

    assert result.success is False
    assert result.error
    assert backend.calls == []


def test_backend_exception_is_converted_to_failed_result() -> None:
    class FailingBackend(FakeDesktopBackend):
        def press(self, key: str):
            self.calls.append(("press", key))
            raise RuntimeError("backend exploded")

    backend = FailingBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            "desktop_press",
            {"key": "ENTER"},
        )
    )

    assert result.success is False
    assert result.error
    assert "backend exploded" in result.error


@pytest.mark.parametrize(
    ("action_name", "arguments"),
    [
        ("desktop_launch_application", {}),
        ("desktop_close_application", {}),
        ("desktop_activate_window", {}),
        ("desktop_close_window", {}),
        ("desktop_find_element", {}),
        (
            "desktop_find_element",
            {"kind": "name"},
        ),
        (
            "desktop_find_element",
            {"value": "Save"},
        ),
        ("desktop_click_element", {}),
        (
            "desktop_click_element",
            {"kind": "name"},
        ),
        (
            "desktop_click_element",
            {"value": "Save"},
        ),
        ("desktop_click_point", {}),
        (
            "desktop_click_point",
            {"x": 10},
        ),
        (
            "desktop_click_point",
            {"y": 20},
        ),
        ("desktop_type", {}),
        ("desktop_press", {}),
        ("desktop_hotkey", {}),
    ],
)
def test_parameter_validation_rejects_invalid_arguments(
    action_name: str,
    arguments: dict,
) -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    result = adapter.execute(
        request(
            action_name,
            arguments,
        )
    )

    assert result.success is False
    assert result.error
    assert backend.calls == []


def test_dispatch_alias_matches_execute() -> None:
    backend = FakeDesktopBackend()
    adapter = DesktopActionAdapter(backend)

    execute_result = adapter.execute(
        request("desktop_health_check")
    )

    backend.calls.clear()

    dispatch_result = adapter.dispatch(
        request("desktop_health_check")
    )

    assert execute_result.success is True
    assert dispatch_result.success is True
    assert execute_result.output == dispatch_result.output
    assert backend.calls == [
        ("health_check", None),
    ]
