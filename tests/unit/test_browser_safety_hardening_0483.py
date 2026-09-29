from __future__ import annotations

import pytest

from osa.browser.safety import BrowserSafetyError, BrowserSafetyPolicy


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.168.1.1",
        "http://169.254.169.254",
        "http://0.0.0.0",
        "http://[::1]",
    ),
)
def test_default_policy_rejects_private_and_loopback_addresses(
    url: str,
) -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url(url)


def test_default_policy_rejects_non_http_schemes() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("file:///etc/hosts")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("javascript:alert(1)")


def test_default_policy_rejects_embedded_credentials() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://user:password@example.com")


def test_blocked_domain_matches_nested_subdomains() -> None:
    policy = BrowserSafetyPolicy(
        blocked_domains={"example.com"},
    )

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://example.com")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://login.example.com")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://deep.login.example.com")


def test_allowed_domain_matches_nested_subdomains() -> None:
    policy = BrowserSafetyPolicy(
        allowed_domains={"example.com"},
    )

    policy.validate_url("https://example.com")
    policy.validate_url("https://www.example.com")
    policy.validate_url("https://deep.www.example.com")


def test_unrelated_domain_is_rejected_when_allowlist_is_configured() -> None:
    policy = BrowserSafetyPolicy(
        allowed_domains={"example.com"},
    )

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://example.org")
