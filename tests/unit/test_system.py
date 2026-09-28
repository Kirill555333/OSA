import platform

import pytest

from osa.system import (
    MacOSSystemProvider,
    SystemProvider,
    WindowsSystemProvider,
)
from osa.tools import SystemInfoTool


def test_create_system_provider_selects_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osa.system import create_system_provider

    monkeypatch.setattr(
        platform,
        "system",
        lambda: "Darwin",
    )

    assert isinstance(
        create_system_provider(),
        MacOSSystemProvider,
    )


def test_create_system_provider_selects_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osa.system import create_system_provider

    monkeypatch.setattr(
        platform,
        "system",
        lambda: "Windows",
    )

    assert isinstance(
        create_system_provider(),
        WindowsSystemProvider,
    )


def test_create_system_provider_uses_generic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osa.system import create_system_provider

    monkeypatch.setattr(
        platform,
        "system",
        lambda: "Linux",
    )

    assert type(create_system_provider()) is SystemProvider


def test_system_provider_returns_basic_information() -> None:
    provider = SystemProvider()

    info = provider.get_system_info()

    assert info.operating_system
    assert info.os_release
    assert info.os_version
    assert info.machine
    assert info.architecture
    assert info.cpu_count >= 1
    assert info.python_version
    assert info.platform_name


def test_macos_provider_normalizes_operating_system() -> None:
    provider = MacOSSystemProvider()

    info = provider.get_system_info()

    assert info.operating_system == "macOS"


def test_windows_provider_normalizes_operating_system() -> None:
    provider = WindowsSystemProvider()

    info = provider.get_system_info()

    assert info.operating_system == "Windows"


def test_system_info_tool_returns_json() -> None:
    tool = SystemInfoTool(
        SystemProvider()
    )

    result = tool.execute({})

    assert result.success is True
    assert '"operating_system"' in result.output
    assert '"python_version"' in result.output
    assert result.metadata["cpu_count"] >= 1


def test_system_info_tool_rejects_arguments() -> None:
    tool = SystemInfoTool(
        SystemProvider()
    )

    result = tool.execute(
        {"unexpected": True}
    )

    assert result.success is False
    assert "does not accept any arguments" in result.error
