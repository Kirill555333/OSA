from __future__ import annotations

from dataclasses import dataclass

from osa.tasks.manager import Task


@dataclass(frozen=True, slots=True)
class TaskRecoveryPolicy:
    """Bounded automatic retry policy for transient task failures."""

    max_retries: int = 2

    def should_retry(
        self,
        task: Task,
        error: Exception,
        attempt: int,
    ) -> bool:
        del task

        if self.max_retries <= 0:
            return False

        if attempt > self.max_retries:
            return False

        return self._is_transient(error)

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        current: BaseException | None = error
        seen: set[int] = set()

        transient_types = (
            ConnectionError,
            TimeoutError,
            BrokenPipeError,
        )

        transient_names = {
            "ModelConnectionError",
            "ModelResponseError",
            "BrowserConnectionError",
        }

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if isinstance(current, transient_types):
                return True

            if type(current).__name__ in transient_names:
                return True

            current = current.__cause__ or current.__context__

        return False
