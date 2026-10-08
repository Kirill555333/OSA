"""Tests for safe link opening (OSA 1.5.1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from osa.tools import web_link
from osa.tools.web_link import OpenUrlTool, normalize_url

# --- address validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://github.com", "https://github.com"),
        ("youtube.com/watch?v=abc", "https://youtube.com/watch?v=abc"),
        ("  «https://google.com»  ", "https://google.com"),
        ("HTTPS://Example.COM/Path?q=1#frag", "https://example.com/Path?q=1#frag"),
        ("http://127.0.0.1:8080/health", "http://127.0.0.1:8080/health"),
        ("localhost:3000/a b", "https://localhost:3000/a%20b"),
        ("[::1]:8080/x", "https://[::1]:8080/x"),
        ("https://a.com/%D0%B0?x=%20y", "https://a.com/%D0%B0?x=%20y"),
        (
            "https://www.youtube.com/results?search_query=обзоры на ноутбуки",
            "https://www.youtube.com/results?search_query="
            "%D0%BE%D0%B1%D0%B7%D0%BE%D1%80%D1%8B%20%D0%BD%D0%B0%20"
            "%D0%BD%D0%BE%D1%83%D1%82%D0%B1%D1%83%D0%BA%D0%B8",
        ),
        (
            "https://пример.рф/страница",
            "https://xn--e1afmkfd.xn--p1ai/%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B0",
        ),
    ],
)
def test_valid_addresses_are_normalized(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "vbscript:x",
        "data:text/html,<b>x</b>",
        "mailto:someone@example.com",
        "tel:+123456789",
        "ftp://example.com/file",
        "http://",
        "https://user:secret@evil.example",
        "https://example.com/\x00",
        "https://example.com/a\nb",
        "https://example.com:99999",
        "https://exa mple.com",
        "https://" + "a" * 3000,
    ],
)
def test_dangerous_or_malformed_addresses_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_url(raw)


def test_rejection_reasons_are_explained_in_russian() -> None:
    with pytest.raises(ValueError) as scheme:
        normalize_url("file:///etc/passwd")

    with pytest.raises(ValueError) as empty:
        normalize_url("")

    assert "http" in str(scheme.value)
    assert "пуст" in str(empty.value)


# --- the tool ----------------------------------------------------------------------


def _tool(result: bool | Exception = True):
    opened: list[str] = []
    announced: list[str] = []

    def opener(url: str) -> bool:
        if isinstance(result, Exception):
            raise result

        opened.append(url)
        return result

    tool = OpenUrlTool(opener=opener, announce=announced.append)

    return tool, opened, announced


def test_tool_hands_the_normalized_link_to_the_browser_once() -> None:
    tool, opened, announced = _tool()

    result = tool.execute({"url": "youtube.com/results?search_query=привет мир"})

    assert result.success is True
    expected = (
        "https://youtube.com/results?search_query="
        "%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82%20%D0%BC%D0%B8%D1%80"
    )
    assert opened == [expected]
    assert opened[0] in result.output
    assert result.metadata == {"url": opened[0], "verified": False}
    assert len(announced) == 1 and opened[0] in announced[0]


def test_tool_does_not_claim_the_page_was_verified() -> None:
    tool, _, _ = _tool()

    result = tool.execute({"url": "https://github.com"})

    assert "передана" in result.output
    assert result.metadata["verified"] is False


def test_tool_refuses_dangerous_links_without_calling_the_browser() -> None:
    tool, opened, announced = _tool()

    for url in ("file:///etc/passwd", "javascript:alert(1)", "", "https://u:p@evil.example"):
        assert tool.execute({"url": url}).success is False

    assert tool.execute({}).success is False
    assert opened == []
    assert announced == []


def test_tool_reports_a_browser_that_could_not_be_opened() -> None:
    tool, _, _ = _tool(result=False)

    result = tool.execute({"url": "https://github.com"})

    assert result.success is False
    assert "браузер" in (result.error or "")


def test_tool_reports_an_opener_failure() -> None:
    tool, _, _ = _tool(result=OSError("no display"))

    result = tool.execute({"url": "https://github.com"})

    assert result.success is False
    assert "no display" in (result.error or "")


def test_tool_schema_and_default_opener() -> None:
    tool = OpenUrlTool()

    assert tool.name == "open_url"
    assert tool.parameters["required"] == ["url"]
    assert "search" in tool.description


# --- security properties of the module ---------------------------------------------


def test_link_module_never_uses_a_shell_or_a_fixed_browser() -> None:
    source = Path(web_link.__file__).read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "shell=True" not in source
    assert "os.system" not in source
    assert "Google Chrome" not in source
