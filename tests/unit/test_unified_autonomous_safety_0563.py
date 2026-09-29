from __future__ import annotations

import pytest

from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousTaskResult,
)
from osa.tasks.safety import AutonomousAuthorization
from osa.tasks.unified_safety import (
    UnifiedAutonomousTaskAuthorizer,
)


class FakeSafetyGate:
    def __init__(
        self,
        authorization: AutonomousAuthorization | None = None,
        error: Exception | None = None,
    ) -> None:
        self.authorization = authorization
        self.error = error
        self.calls: list[
            tuple[str, str]
        ] = []

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousAuthorization:
        self.calls.append(
            (
                run_id,
                task_id,
            )
        )

        if self.error is not None:
            raise self.error

        return self.authorization or AutonomousAuthorization(
            allowed=True,
            reason="allowed",
        )


class FakeBackend:
    def __init__(self) -> None:
        self.executed: list[
            tuple[str, str]
        ] = []
        self.complete = False

    def create_run(
        self,
        goal: str,
    ) -> str:
        assert goal == "test goal"
        return "run-1"

    def ready_task_ids(
        self,
        run_id: str,
    ) -> tuple[str, ...]:
        return (
            "task-1",
            "task-2",
        )

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.executed.append(
            (
                run_id,
                task_id,
            )
        )

        if task_id == "task-1":
            self.complete = True

        return AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output="done",
        )

    def is_complete(
        self,
        run_id: str,
    ) -> bool:
        return self.complete

    def has_failed(
        self,
        run_id: str,
    ) -> bool:
        return False


def test_allows_authorized_task() -> None:
    gate = FakeSafetyGate(
        AutonomousAuthorization(
            allowed=True,
            reason="allowed",
        )
    )

    authorizer = UnifiedAutonomousTaskAuthorizer(
        gate
    )

    result = authorizer.authorize(
        "run-1",
        "task-1",
    )

    assert result.allowed is True
    assert result.reason == "allowed"
    assert gate.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_denies_task_from_underlying_gate() -> None:
    gate = FakeSafetyGate(
        AutonomousAuthorization(
            allowed=False,
            reason="task is not allowlisted",
        )
    )

    authorizer = UnifiedAutonomousTaskAuthorizer(
        gate
    )

    result = authorizer.authorize(
        "run-1",
        "task-1",
    )

    assert result.allowed is False
    assert result.reason == "task is not allowlisted"


def test_gate_exception_fails_closed() -> None:
    gate = FakeSafetyGate(
        error=RuntimeError(
            "safety service unavailable"
        )
    )

    authorizer = UnifiedAutonomousTaskAuthorizer(
        gate
    )

    result = authorizer.authorize(
        "run-1",
        "task-1",
    )

    assert result.allowed is False
    assert (
        result.reason
        == (
            "autonomous_safety_gate_error: "
            "safety service unavailable"
        )
    )


def test_invalid_gate_result_fails_closed() -> None:
    class InvalidGate:
        def authorize(
            self,
            run_id: str,
            task_id: str,
        ):
            return {
                "allowed": True,
            }

    authorizer = UnifiedAutonomousTaskAuthorizer(
        InvalidGate()
    )

    result = authorizer.authorize(
        "run-1",
        "task-1",
    )

    assert result.allowed is False
    assert (
        result.reason
        == (
            "autonomous_safety_gate_error: "
            "invalid authorization result."
        )
    )


def test_identifiers_are_normalized() -> None:
    gate = FakeSafetyGate()
    authorizer = UnifiedAutonomousTaskAuthorizer(
        gate
    )

    authorizer.authorize(
        "  run-1  ",
        "  task-1  ",
    )

    assert gate.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_empty_run_id_is_rejected() -> None:
    authorizer = UnifiedAutonomousTaskAuthorizer(
        FakeSafetyGate()
    )

    with pytest.raises(
        ValueError,
        match="run_id cannot be empty",
    ):
        authorizer.authorize(
            "   ",
            "task-1",
        )


def test_empty_task_id_is_rejected() -> None:
    authorizer = UnifiedAutonomousTaskAuthorizer(
        FakeSafetyGate()
    )

    with pytest.raises(
        ValueError,
        match="task_id cannot be empty",
    ):
        authorizer.authorize(
            "run-1",
            "   ",
        )


def test_non_string_run_id_is_rejected() -> None:
    authorizer = UnifiedAutonomousTaskAuthorizer(
        FakeSafetyGate()
    )

    with pytest.raises(
        TypeError,
        match="run_id must be a string",
    ):
        authorizer.authorize(
            123,
            "task-1",
        )


def test_non_string_task_id_is_rejected() -> None:
    authorizer = UnifiedAutonomousTaskAuthorizer(
        FakeSafetyGate()
    )

    with pytest.raises(
        TypeError,
        match="task_id must be a string",
    ):
        authorizer.authorize(
            "run-1",
            123,
        )


def test_authorizer_requires_safety_gate() -> None:
    with pytest.raises(
        ValueError,
        match="safety_gate is required",
    ):
        UnifiedAutonomousTaskAuthorizer(
            None
        )


def test_authorizer_rejects_invalid_safety_gate() -> None:
    with pytest.raises(
        TypeError,
        match="must provide authorize",
    ):
        UnifiedAutonomousTaskAuthorizer(
            object()
        )


def test_autonomous_loop_blocks_denied_tasks_before_backend() -> None:
    gate = FakeSafetyGate(
        AutonomousAuthorization(
            allowed=False,
            reason="blocked by autonomous policy",
        )
    )

    authorizer = UnifiedAutonomousTaskAuthorizer(
        gate
    )

    backend = FakeBackend()

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=1,
            max_tasks_per_cycle=2,
        ),
        authorizer=authorizer,
    )

    report = loop.run(
        "test goal"
    )

    assert report.success is False
    assert report.cycle_count == 1
    assert report.failed_task_ids == (
        "task-1",
        "task-2",
    )
    assert backend.executed == []
