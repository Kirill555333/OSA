from __future__ import annotations

import pytest

from osa.browser import (
    BrowserLocator,
    BrowserSafetyError,
    BrowserSafetyPolicy,
)


def test_https_url_is_allowed() -> None:
    policy = BrowserSafetyPolicy()

    policy.validate_url(
        "https://example.com"
    )


def test_http_url_is_allowed() -> None:
    policy = BrowserSafetyPolicy()

    policy.validate_url(
        "http://example.com"
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/test.html",
        "ftp://example.com/file",
        "javascript:alert(1)",
    ],
)
def test_non_http_urls_are_blocked(url: str) -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(
        BrowserSafetyError,
        match="Only HTTP and HTTPS",
    ):
        policy.validate_url(url)


def test_embedded_credentials_are_blocked() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(
        BrowserSafetyError,
        match="embedded credentials",
    ):
        policy.validate_url(
            "https://user:password@example.com"
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://10.0.0.5",
        "http://192.168.1.10",
        "http://[::1]:8080",
    ],
)
def test_local_destinations_are_blocked(url: str) -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(
        BrowserSafetyError,
    ):
        policy.validate_url(url)


def test_local_networks_can_be_explicitly_allowed() -> None:
    policy = BrowserSafetyPolicy(
        block_local_networks=False,
    )

    policy.validate_url(
        "http://127.0.0.1:8080"
    )


def test_blocked_domain() -> None:
    policy = BrowserSafetyPolicy(
        blocked_domains=frozenset(
            {"example.com"}
        )
    )

    with pytest.raises(
        BrowserSafetyError,
        match="blocked",
    ):
        policy.validate_url(
            "https://example.com"
        )


def test_subdomain_of_blocked_domain_is_blocked() -> None:
    policy = BrowserSafetyPolicy(
        blocked_domains=frozenset(
            {"example.com"}
        )
    )

    with pytest.raises(
        BrowserSafetyError,
        match="blocked",
    ):
        policy.validate_url(
            "https://accounts.example.com"
        )


def test_allowed_domains_work() -> None:
    policy = BrowserSafetyPolicy(
        allowed_domains=frozenset(
            {"example.com"}
        )
    )

    policy.validate_url(
        "https://example.com"
    )

    policy.validate_url(
        "https://docs.example.com"
    )


def test_non_allowed_domain_is_blocked() -> None:
    policy = BrowserSafetyPolicy(
        allowed_domains=frozenset(
            {"example.com"}
        )
    )

    with pytest.raises(
        BrowserSafetyError,
        match="not allowed",
    ):
        policy.validate_url(
            "https://other.example"
        )


def test_sensitive_css_input_is_blocked() -> None:
    policy = BrowserSafetyPolicy()

    locator = BrowserLocator(
        kind="css",
        value="#password",
    )

    with pytest.raises(
        BrowserSafetyError,
        match="sensitive field",
    ):
        policy.validate_interaction(
            locator,
            url="https://example.com",
        )


def test_sensitive_input_can_be_enabled_explicitly() -> None:
    policy = BrowserSafetyPolicy(
        allow_sensitive_input=True,
    )

    locator = BrowserLocator(
        kind="css",
        value="#password",
    )

    policy.validate_interaction(
        locator,
        url="https://example.com",
    )
