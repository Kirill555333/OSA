"""Structured event logging for OSA."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class EventLogger:
    """Write structured OSA events to a JSON Lines file."""

    def __init__(
        self,
        path: Path,
        *,
        session_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._path = path
        self._session_id = session_id or str(uuid.uuid4())
        self._enabled = enabled
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the event log path."""
        return self._path

    @property
    def session_id(self) -> str:
        """Return the current logging session ID."""
        return self._session_id

    @property
    def enabled(self) -> bool:
        """Return whether event logging is enabled."""
        return self._enabled

    def log(
        self,
        event: str,
        data: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Write one structured event without interrupting OSA."""
        if not self._enabled:
            return

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "event": event,
            "data": dict(data or {}),
        }

        if extra:
            payload["data"].update(extra)

        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )

            with self._lock:
                self._path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with self._path.open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(serialized)
                    file.write("\n")

        except (OSError, TypeError, ValueError):
            # Logging must never crash or interrupt OSA.
            return
