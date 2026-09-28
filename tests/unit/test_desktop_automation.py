from __future__ import annotations

import pytest

from osa.desktop import (
    DesktopElement,
    DesktopElementLocator,
    DesktopPoint,
    DesktopRect,
    DesktopWindow,
    create_fake_desktop_automation,
)


def test_desktop_rect_rejects_negative_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="width",
    ):
        DesktopRect(
            x=0,
            y=0,
            width=-1,
            height=10,
        )

    with pytest.raises(
        ValueError,
        match="height",
    ):
        DesktopRect(
            x=0,
            y=0,
            width=10,
            height=-1,
        )


def test_element_locator_rejects_empty_value() -> None:
    with pytest.raises(
        ValueError,
        match="locator value",
    ):
        DesktopElementLocator(
            kind="name",
            value="   ",
        )


def test_launch_application() -> None:
    desktop = create_fake_desktop_automation()

    app = desktop.launch_application(
        "Calculator"
    )

    assert app.name == "Calculator"
    assert app.running is True
    assert desktop.applications() == (app,)


def test_launch_existing_application_activates_it() -> None:
    desktop = create_fake_desktop_automation()

    first = desktop.launch_application(
        "Calculator"
    )

    desktop.close_application(
        first.application_id
    )

    second = desktop.launch_application(
        "Calculator"
    )

    assert second.application_id == first.application_id
    assert second.running is True
    assert len(desktop.applications()) == 1


def test_window_lifecycle() -> None:
    desktop = create_fake_desktop_automation()

    window = desktop.add_window(
        title="Calculator",
        bounds=DesktopRect(
            x=0,
            y=0,
            width=500,
            height=700,
        ),
    )

    assert isinstance(window, DesktopWindow)
    assert desktop.windows() == (window,)

    active = desktop.activate_window(
        window.window_id
    )

    assert active.focused is True

    desktop.close_window(
        window.window_id
    )

    assert desktop.windows() == ()


def test_activate_window_focuses_only_selected_window() -> None:
    desktop = create_fake_desktop_automation()

    first = desktop.add_window(
        title="First"
    )
    second = desktop.add_window(
        title="Second"
    )

    desktop.activate_window(
        second.window_id
    )

    windows = {
        item.window_id: item
        for item in desktop.windows()
    }

    assert windows[first.window_id].focused is False
    assert windows[second.window_id].focused is True


def test_find_and_click_element() -> None:
    desktop = create_fake_desktop_automation()

    window = desktop.add_window(
        title="Example"
    )

    locator = DesktopElementLocator(
        kind="name",
        value="Submit",
        exact=True,
    )

    element = desktop.add_element(
        locator,
        window_id=window.window_id,
        name="Submit",
        role="button",
        bounds=DesktopRect(
            x=100,
            y=200,
            width=100,
            height=40,
        ),
    )

    found = desktop.find_element(
        locator,
        window_id=window.window_id,
    )

    assert isinstance(found, DesktopElement)
    assert found == element

    desktop.click_element(
        locator,
        window_id=window.window_id,
    )

    assert any(
        action[0] == "click_element"
        for action in desktop.actions
    )


def test_click_missing_element_fails() -> None:
    desktop = create_fake_desktop_automation()

    locator = DesktopElementLocator(
        kind="name",
        value="Missing",
    )

    with pytest.raises(
        Exception,
        match="not found",
    ):
        desktop.click_element(locator)


def test_click_point() -> None:
    desktop = create_fake_desktop_automation()

    point = DesktopPoint(
        x=100,
        y=200,
    )

    desktop.click_point(point)

    assert (
        "click_point",
        point,
    ) in desktop.actions


def test_keyboard_actions() -> None:
    desktop = create_fake_desktop_automation()

    desktop.type_text("OSA")
    desktop.press("Enter")
    desktop.hotkey(
        "Control",
        "A",
    )

    assert (
        "type_text",
        "OSA",
    ) in desktop.actions

    assert (
        "press",
        "Enter",
    ) in desktop.actions

    assert (
        "hotkey",
        ("Control", "A"),
    ) in desktop.actions


def test_keyboard_validation() -> None:
    desktop = create_fake_desktop_automation()

    with pytest.raises(
        ValueError,
        match="key cannot be empty",
    ):
        desktop.press("")

    with pytest.raises(
        ValueError,
        match="at least one key",
    ):
        desktop.hotkey()


def test_screenshot_and_health_check() -> None:
    desktop = create_fake_desktop_automation()

    assert desktop.screenshot() == (
        b"fake-desktop-screenshot"
    )
    assert desktop.health_check() is True


def test_unknown_window_fails() -> None:
    desktop = create_fake_desktop_automation()

    with pytest.raises(
        Exception,
        match="Unknown window",
    ):
        desktop.activate_window(
            "missing"
        )


def test_unknown_application_fails() -> None:
    desktop = create_fake_desktop_automation()

    with pytest.raises(
        Exception,
        match="Unknown application",
    ):
        desktop.close_application(
            "missing"
        )
