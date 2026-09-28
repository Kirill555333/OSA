from __future__ import annotations

from osa.browser import create_fake_browser_automation
from osa.tools.browser_automation import (
    BrowserClickTool,
    BrowserFindTool,
    BrowserOpenTool,
    BrowserPressTool,
    BrowserReadTool,
    BrowserScreenshotTool,
    BrowserTypeTool,
    BrowserWaitTool,
    create_browser_action_tools,
)


def browser():
    instance = create_fake_browser_automation()
    instance.open("https://example.com")
    return instance


def locator():
    return {
        "kind": "role",
        "value": "button",
        "exact": True,
    }


def test_create_browser_action_tools() -> None:
    instance = browser()

    tools = create_browser_action_tools(instance)

    assert [tool.name for tool in tools] == [
        "browser_open",
        "browser_find",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_read",
        "browser_wait",
        "browser_observe",
        "browser_screenshot",
    ]


def test_browser_open() -> None:
    instance = create_fake_browser_automation()
    tool = BrowserOpenTool(instance)

    result = tool.execute(
        {
            "url": "https://example.com",
        }
    )

    assert result.success is True
    assert "Tab ID:" in result.output
    assert result.metadata["url"] == "https://example.com"


def test_browser_open_rejects_non_http() -> None:
    instance = browser()
    tool = BrowserOpenTool(instance)

    result = tool.execute(
        {
            "url": "file:///tmp/test.html",
        }
    )

    assert result.success is False
    assert "HTTP and HTTPS" in result.error


def test_browser_open_rejects_embedded_credentials() -> None:
    instance = browser()
    tool = BrowserOpenTool(instance)

    result = tool.execute(
        {
            "url": "https://user:pass@example.com",
        }
    )

    assert result.success is False
    assert "credentials" in result.error


def test_browser_find() -> None:
    instance = browser()
    tool = BrowserFindTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
        }
    )

    assert result.success is True
    assert "button" in result.output
    assert result.metadata["visible"] is True


def test_browser_find_missing_locator() -> None:
    instance = browser()
    tool = BrowserFindTool(instance)

    result = tool.execute({})

    assert result.success is False
    assert "locator" in result.error


def test_browser_click() -> None:
    instance = browser()
    tool = BrowserClickTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
        }
    )

    assert result.success is True
    assert "Clicked element" in result.output
    assert any(
        action[0] == "click"
        for action in instance.actions
    )


def test_browser_type() -> None:
    instance = browser()
    tool = BrowserTypeTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
            "text": "OSA",
        }
    )

    assert result.success is True
    assert result.metadata["clear"] is True


def test_browser_type_rejects_invalid_text() -> None:
    instance = browser()
    tool = BrowserTypeTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
            "text": 123,
        }
    )

    assert result.success is False
    assert result.error == "Argument 'text' must be a string."


def test_browser_press() -> None:
    instance = browser()
    tool = BrowserPressTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
            "key": "Enter",
        }
    )

    assert result.success is True
    assert result.metadata["key"] == "Enter"


def test_browser_read_page() -> None:
    instance = browser()
    tool = BrowserReadTool(instance)

    result = tool.execute({})

    assert result.success is True
    assert result.output == "fake page text"
    assert result.metadata["scope"] == "page"


def test_browser_read_element() -> None:
    instance = browser()
    tool = BrowserReadTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
        }
    )

    assert result.success is True
    assert result.output == "fake page text"
    assert result.metadata["scope"] == "element"


def test_browser_wait() -> None:
    instance = browser()
    tool = BrowserWaitTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
            "state": "visible",
            "timeout_ms": 5000,
        }
    )

    assert result.success is True
    assert result.metadata["state"] == "visible"
    assert result.metadata["timeout_ms"] == 5000


def test_browser_wait_rejects_invalid_timeout() -> None:
    instance = browser()
    tool = BrowserWaitTool(instance)

    result = tool.execute(
        {
            "locator": locator(),
            "timeout_ms": -1,
        }
    )

    assert result.success is False
    assert "cannot be negative" in result.error


def test_browser_screenshot() -> None:
    instance = browser()
    tool = BrowserScreenshotTool(instance)

    result = tool.execute({})

    assert result.success is True
    assert result.metadata["content_type"] == "image/png"
    assert result.metadata["byte_length"] > 0
    assert "Screenshot captured successfully" in result.output


def test_all_tools_have_valid_schemas() -> None:
    instance = browser()

    tools = create_browser_action_tools(instance)

    for tool in tools:
        schema = tool.parameters

        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert isinstance(tool.name, str)
        assert tool.name.strip()
        assert isinstance(tool.description, str)
        assert tool.description.strip()
