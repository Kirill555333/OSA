"""Safety policy for interactive browser automation."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse

from osa.browser.automation import BrowserLocator


class BrowserSafetyError(RuntimeError):
    """Raised when a browser action violates safety policy."""


@dataclass(frozen=True)
class BrowserSafetyPolicy:
    """
    Browser-specific safety restrictions.

    The policy is deliberately conservative:
    - only HTTP/HTTPS navigation is allowed;
    - URLs containing credentials are rejected;
    - localhost and local-network destinations can be blocked;
    - optional domain allow/block lists are supported;
    - sensitive form fields are blocked by default.
    """

    allowed_domains: frozenset[str] = frozenset()
    blocked_domains: frozenset[str] = frozenset()
    block_local_networks: bool = True
    allow_sensitive_input: bool = False

    def validate_url(self, url: str) -> None:
        """Validate a URL before interactive navigation."""
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            raise BrowserSafetyError(
                "Only HTTP and HTTPS URLs are allowed."
            )

        if parsed.username or parsed.password:
            raise BrowserSafetyError(
                "URLs with embedded credentials are not allowed."
            )

        host = (parsed.hostname or "").strip().lower()

        if not host:
            raise BrowserSafetyError(
                "Browser URL must contain a hostname."
            )

        normalized_host = host.rstrip(".")

        if self._domain_matches(
            normalized_host,
            self.blocked_domains,
        ):
            raise BrowserSafetyError(
                f"Domain is blocked by browser safety policy: {host}"
            )

        if self.allowed_domains:
            if not self._domain_matches(
                normalized_host,
                self.allowed_domains,
            ):
                raise BrowserSafetyError(
                    f"Domain is not allowed by browser safety policy: {host}"
                )

        if self.block_local_networks and self._is_local_host(
            normalized_host
        ):
            raise BrowserSafetyError(
                f"Local-network destination is blocked: {host}"
            )

    def validate_interaction(
        self,
        locator: BrowserLocator,
        *,
        url: str,
    ) -> None:
        """Validate an interaction against the current page."""
        self.validate_url(url)

        if (
            locator.kind == "css"
            and locator.value.strip()
        ):
            lowered = locator.value.casefold()

            sensitive_tokens = (
                "password",
                "passwd",
                "creditcard",
                "card-number",
                "cardnumber",
                "cvv",
                "cvc",
                "security-code",
            )

            if (
                not self.allow_sensitive_input
                and any(
                    token in lowered
                    for token in sensitive_tokens
                )
            ):
                raise BrowserSafetyError(
                    "Interaction with a sensitive field "
                    "is blocked by browser safety policy."
                )

    @staticmethod
    def _domain_matches(
        host: str,
        domains: frozenset[str],
    ) -> bool:
        for domain in domains:
            normalized = domain.strip().lower().rstrip(".")

            if not normalized:
                continue

            if (
                host == normalized
                or host.endswith("." + normalized)
            ):
                return True

        return False

    @staticmethod
    def _is_local_host(host: str) -> bool:
        if host in {
            "localhost",
            "localhost.localdomain",
        }:
            return True

        if host.endswith(".localhost"):
            return True

        if host.endswith(".local"):
            return True

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False

        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        )


def create_default_browser_safety_policy(
    *,
    allowed_domains: frozenset[str] = frozenset(),
    blocked_domains: frozenset[str] = frozenset(),
    allow_sensitive_input: bool = False,
) -> BrowserSafetyPolicy:
    """Create OSA's default conservative browser safety policy."""
    return BrowserSafetyPolicy(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        block_local_networks=True,
        allow_sensitive_input=allow_sensitive_input,
    )
