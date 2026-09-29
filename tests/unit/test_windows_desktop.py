from __future__ import annotations

import sys

import pytest

from osa.desktop import (
    DesktopAutomationConnectionError,
    WindowsDesktopAutomation,
    WindowsDesktopAutomationConfig,
)


def test_windows_config_defaults() -> None:
    config = WindowsDesktopAutomationConfig()

    assert config.backend == "uia"
    assert config.application_start_timeout_ms == 30_000
    assert config.action_timeout_ms == 10_000
    assert config.screenshot_all_screens is True


def test_windows_config_rejects_wrong_backend() -> None:
    with pytest.raises(
        ValueError,
        match="'uia'",
    ):
        WindowsDesktopAutomationConfig(
            backend="win32",
        )


def test_windows_config_rejects_invalid_timeout() -> None:
    with pytest.raises(
        ValueError,
        match="application_start_timeout_ms",
    ):
        WindowsDesktopAutomationConfig(
            application_start_timeout_ms=0,
        )

    with pytest.raises(
        ValueError,
        match="action_timeout_ms",
    ):
        WindowsDesktopAutomationConfig(
            action_timeout_ms=0,
        )


def test_backend_can_be_constructed_on_all_platforms() -> None:
    backend = WindowsDesktopAutomation()

    assert backend.config.backend == "uia"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Non-Windows platform guard is not applicable.",
)
def test_non_windows_health_check_is_false() -> None:
    backend = WindowsDesktopAutomation()

    assert backend.health_check() is False


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Non-Windows platform guard is not applicable.",
)
def test_non_windows_operations_fail_closed() -> None:
    backend = WindowsDesktopAutomation()

    with pytest.raises(
        DesktopAutomationConnectionError,
        match="only on Windows",
    ):
        backend.windows()

    with pytest.raises(
        DesktopAutomationConnectionError,
        match="only on Windows",
    ):
        backend.type_text("OSA")


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Live Windows UI Automation test.",
)
def test_windows_health_check() -> None:
    backend = WindowsDesktopAutomation()

    assert backend.health_check() is True
