"""Browser abstractions for OSA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class BrowserError(RuntimeError):
    """Base exception for browser-related errors."""


class BrowserConnectionError(BrowserError):
    """Raised when a page cannot be reached."""


class BrowserResponseError(BrowserError):
    """Raised when a browser response cannot be processed."""


@dataclass(frozen=True, slots=True)
class BrowserPage:
    """Normalized page returned by a browser backend."""

    url: str
    title: str
    text: str
    status_code: int


class BrowserInterface(ABC):
    """Abstract interface implemented by OSA browser backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the browser backend name."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, url: str) -> BrowserPage:
        """Fetch a web page and return normalized content."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the browser backend is available."""
        raise NotImplementedError
