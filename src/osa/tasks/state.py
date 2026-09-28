from __future__ import annotations

from dataclasses import dataclass


class WorkingMemoryError(RuntimeError):
    """Raised for invalid working-memory operations."""


@dataclass(frozen=True, slots=True)
class TaskWorkingMemorySnapshot:
    """Immutable snapshot of one active task run."""

    run_id: str
    goal: str
    active_task_id: str | None
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    attempts: dict[str, int]
    outputs: dict[str, str]
    errors: dict[str, str]
    last_error: str | None

    @property
    def progress_count(self) -> int:
        return len(self.completed_task_ids)


@dataclass(slots=True)
class _MutableState:
    goal: str
    active_task_id: str | None
    completed_task_ids: list[str]
    failed_task_ids: list[str]
    attempts: dict[str, int]
    outputs: dict[str, str]
    errors: dict[str, str]
    last_error: str | None


class TaskWorkingMemory:
    """Transient working memory for task execution state."""

    def __init__(self) -> None:
        self._states: dict[str, _MutableState] = {}

    def start_run(self, run_id: str, goal: str) -> None:
        if not run_id.strip():
            raise WorkingMemoryError("run_id must not be empty.")

        if not goal.strip():
            raise WorkingMemoryError("goal must not be empty.")

        self._states[run_id] = _MutableState(
            goal=goal,
            active_task_id=None,
            completed_task_ids=[],
            failed_task_ids=[],
            attempts={},
            outputs={},
            errors={},
            last_error=None,
        )

    def task_started(
        self,
        run_id: str,
        task_id: str,
        attempt: int,
    ) -> None:
        state = self._get(run_id)

        if attempt <= 0:
            raise WorkingMemoryError(
                "attempt must be greater than zero."
            )

        state.active_task_id = task_id
        state.attempts[task_id] = attempt
        state.last_error = None

    def task_completed(
        self,
        run_id: str,
        task_id: str,
        output: str,
    ) -> None:
        state = self._get(run_id)

        state.active_task_id = None

        if task_id not in state.completed_task_ids:
            state.completed_task_ids.append(task_id)

        if task_id in state.failed_task_ids:
            state.failed_task_ids.remove(task_id)

        state.outputs[task_id] = output
        state.errors.pop(task_id, None)
        state.last_error = None

    def task_failed(
        self,
        run_id: str,
        task_id: str,
        error: str,
    ) -> None:
        state = self._get(run_id)

        state.active_task_id = None

        if task_id not in state.failed_task_ids:
            state.failed_task_ids.append(task_id)

        state.errors[task_id] = error
        state.last_error = error

    def task_retried(
        self,
        run_id: str,
        task_id: str,
    ) -> None:
        state = self._get(run_id)

        if task_id in state.failed_task_ids:
            state.failed_task_ids.remove(task_id)

        state.active_task_id = None

    def snapshot(
        self,
        run_id: str,
    ) -> TaskWorkingMemorySnapshot:
        state = self._get(run_id)

        return TaskWorkingMemorySnapshot(
            run_id=run_id,
            goal=state.goal,
            active_task_id=state.active_task_id,
            completed_task_ids=tuple(state.completed_task_ids),
            failed_task_ids=tuple(state.failed_task_ids),
            attempts=dict(state.attempts),
            outputs=dict(state.outputs),
            errors=dict(state.errors),
            last_error=state.last_error,
        )

    def clear(self, run_id: str) -> None:
        self._states.pop(run_id, None)

    def contains(self, run_id: str) -> bool:
        return run_id in self._states

    def _get(self, run_id: str) -> _MutableState:
        try:
            return self._states[run_id]
        except KeyError as exc:
            raise WorkingMemoryError(
                f"Unknown task run: {run_id!r}"
            ) from exc
