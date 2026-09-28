from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from osa.tasks.manager import Task, TaskManager
from osa.tasks.state import TaskWorkingMemory


class TaskExecutorError(RuntimeError):
    """Raised when a task run cannot make further progress."""


TaskHandler = Callable[[Task, Mapping[str, str]], str]


@dataclass(frozen=True, slots=True)
class TaskExecutionReport:
    run_id: str
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    blocked: tuple[str, ...]
    outputs: dict[str, str]
    errors: dict[str, str]
    attempts: dict[str, int]

    @property
    def success(self) -> bool:
        return not self.failed and not self.blocked


class TaskExecutor:
    def __init__(
        self,
        manager: TaskManager,
        recovery=None,
        working_memory: TaskWorkingMemory | None = None,
    ) -> None:
        self.manager = manager
        self.recovery = recovery
        self.working_memory = working_memory or TaskWorkingMemory()

    def execute(
        self,
        run_id: str,
        handler: TaskHandler,
    ) -> TaskExecutionReport:
        from osa.tasks.recovery import TaskRecoveryPolicy

        recovery = self.recovery or TaskRecoveryPolicy(max_retries=0)

        run = self.manager.get_run(run_id)

        if not self.working_memory.contains(run_id):
            self.working_memory.start_run(
                run_id,
                run.goal,
            )

        outputs: dict[str, str] = {}
        errors: dict[str, str] = {}
        attempts: dict[str, int] = {}

        while True:
            ready = self.manager.ready_tasks(run_id)

            if not ready:
                if self.manager.is_complete(run_id):
                    break

                if self.manager.has_failed(run_id):
                    break

                raise TaskExecutorError(
                    f"Task run {run_id!r} cannot make progress"
                )

            for task in ready:
                attempt = attempts.get(task.task_id, 0) + 1
                attempts[task.task_id] = attempt

                self.manager.start_task(run_id, task.task_id)
                self.working_memory.task_started(
                    run_id,
                    task.task_id,
                    attempt,
                )

                try:
                    output = handler(task, outputs)
                    if not isinstance(output, str):
                        raise TypeError("Task handler must return a string.")
                except Exception as exc:
                    error_text = str(exc) or type(exc).__name__

                    self.manager.fail_task(
                        run_id,
                        task.task_id,
                        error_text,
                    )
                    self.working_memory.task_failed(
                        run_id,
                        task.task_id,
                        error_text,
                    )

                    if recovery.should_retry(task, exc, attempt):
                        self.manager.retry_task(
                            run_id,
                            task.task_id,
                        )
                        self.working_memory.task_retried(
                            run_id,
                            task.task_id,
                        )
                        continue

                    errors[task.task_id] = error_text
                    break

                self.manager.complete_task(
                    run_id,
                    task.task_id,
                )
                self.working_memory.task_completed(
                    run_id,
                    task.task_id,
                    output,
                )

                outputs[task.task_id] = output
                errors.pop(task.task_id, None)

        run = self.manager.get_run(run_id)

        return TaskExecutionReport(
            run_id=run_id,
            completed=tuple(
                task.task_id
                for task in run.tasks.values()
                if task.status.value == "completed"
            ),
            failed=tuple(
                task.task_id
                for task in run.tasks.values()
                if task.status.value == "failed"
            ),
            blocked=tuple(
                task.task_id
                for task in self.manager.blocked_tasks(run_id)
            ),
            outputs=dict(outputs),
            errors=dict(errors),
            attempts=dict(attempts),
        )
