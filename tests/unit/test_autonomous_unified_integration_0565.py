from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.tasks.action_bridge import (
    AutonomousActionBridge,
    CallbackAutonomousActionResolver,
)
from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousTaskResult,
    CallbackAutonomousBackend,
)
from osa.tasks.autonomous_executor import UnifiedAutonomousExecutor
from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    CallbackAutonomousRecoveryPolicy,
)
from osa.tasks.safety import AutonomousAuthorization
from osa.tasks.unified_safety import UnifiedAutonomousTaskAuthorizer


class RecordingPipeline:
    def __init__(self, results: Sequence[ActionResult]) -> None:
        self.results = list(results)
        self.requests: list[ActionRequest] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.requests.append(request)

        if not self.results:
            return ActionResult.failed(
                request.request_id,
                error="pipeline_exhausted",
            )

        result = self.results.pop(0)

        if result.request_id != request.request_id:
            return ActionResult.failed(
                request.request_id,
                error="invalid_test_result",
            )

        return result


class RecordingSafetyGate:
    def __init__(self, authorization: AutonomousAuthorization) -> None:
        self.authorization = authorization
        self.calls: list[tuple[str, str]] = []

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousAuthorization:
        self.calls.append((run_id, task_id))
        return self.authorization


class TaskState:
    def __init__(self, task_ids: Sequence[str]) -> None:
        self.task_ids = tuple(task_ids)
        self.executed: list[str] = []
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self.task_executor: Callable[
            [str, str], AutonomousTaskResult
        ] | None = None

    def create_run(self, goal: str) -> str:
        assert goal == "finish integration test"
        return "run-0565"

    def ready_task_ids(self, run_id: str) -> tuple[str, ...]:
        if self.is_complete(run_id):
            return ()

        return tuple(
            task_id
            for task_id in self.task_ids
            if task_id not in self.completed_task_ids
            and task_id not in self.failed_task_ids
        )

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        if self.task_executor is None:
            raise RuntimeError("task_executor_not_configured")

        self.executed.append(task_id)

        result = self.task_executor(run_id, task_id)

        if result.status == "completed":
            self.completed_task_ids.add(task_id)
        elif result.status == "failed":
            self.failed_task_ids.add(task_id)

        return result

    def is_complete(self, run_id: str) -> bool:
        return set(self.task_ids).issubset(self.completed_task_ids)

    def has_failed(self, run_id: str) -> bool:
        return bool(self.failed_task_ids)

    def set_executor(
        self,
        executor: Callable[[str, str], AutonomousTaskResult],
    ) -> None:
        self.task_executor = executor


def make_request(
    task_id: str,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="integration_tool",
        arguments={
            "task_id": task_id,
        },
    )


def build_stack(
    pipeline: RecordingPipeline,
    task_ids: Sequence[str] = ("task-1",),
    requests: dict[str, ActionRequest] | None = None,
) -> tuple[
    AutonomousLoop,
    TaskState,
    RecordingSafetyGate,
    AutonomousRecoveryExecutor,
]:
    normalized_task_ids = tuple(task_ids)

    request_map = (
        dict(requests)
        if requests is not None
        else {
            task_id: make_request(task_id)
            for task_id in normalized_task_ids
        }
    )

    resolver = CallbackAutonomousActionResolver(
        lambda run_id, task_id: request_map[task_id]
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    unified_executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    recovery_policy = CallbackAutonomousRecoveryPolicy(
        lambda run_id, task_id, result, attempt:
            AutonomousRecoveryDecision(
                retry=False,
                reason="integration default",
            )
    )

    recovery_executor = AutonomousRecoveryExecutor(
        unified_executor,
        policy=recovery_policy,
        max_attempts=1,
    )

    safety_gate = RecordingSafetyGate(
        AutonomousAuthorization(
            True,
            "integration allowed",
        )
    )

    authorizer = UnifiedAutonomousTaskAuthorizer(
        safety_gate
    )

    state = TaskState(
        normalized_task_ids
    )

    state.set_executor(
        recovery_executor.execute
    )

    backend = CallbackAutonomousBackend(
        create_run=state.create_run,
        ready_task_ids=state.ready_task_ids,
        execute_task=state.execute_task,
        is_complete=state.is_complete,
        has_failed=state.has_failed,
    )

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=2,
            max_tasks_per_cycle=4,
        ),
        authorizer=authorizer,
    )

    return (
        loop,
        state,
        safety_gate,
        recovery_executor,
    )


def test_full_success_path_uses_unified_pipeline() -> None:
    request = make_request(
        "task-1"
    )

    pipeline = RecordingPipeline(
        [
            ActionResult.succeeded(
                request.request_id,
                output="completed",
            )
        ]
    )

    loop, state, safety_gate, _ = build_stack(
        pipeline,
        requests={
            "task-1": request,
        },
    )

    report = loop.run(
        "finish integration test"
    )

    assert report.success is True
    assert report.run_id == "run-0565"
    assert report.completed_task_ids == (
        "task-1",
    )
    assert report.failed_task_ids == ()

    assert state.executed == [
        "task-1",
    ]

    assert safety_gate.calls == [
        (
            "run-0565",
            "task-1",
        )
    ]

    assert pipeline.requests == [
        request,
    ]


def test_recovery_reexecutes_through_same_unified_pipeline() -> None:
    request = make_request(
        "task-1"
    )

    resolver = CallbackAutonomousActionResolver(
        lambda run_id, task_id: request
    )

    pipeline = RecordingPipeline(
        [
            ActionResult.failed(
                request.request_id,
                error="transient failure",
            ),
            ActionResult.succeeded(
                request.request_id,
                output="recovered",
            ),
        ]
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    unified_executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    recovery_executor = AutonomousRecoveryExecutor(
        unified_executor,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="transient",
                )
        ),
        max_attempts=2,
    )

    result = recovery_executor.execute(
        "run-0565",
        "task-1",
    )

    assert result.status == "completed"
    assert result.output == "recovered"
    assert result.error is None

    assert pipeline.requests == [
        request,
        request,
    ]


def test_denied_task_never_reaches_pipeline() -> None:
    request = make_request(
        "task-1"
    )

    pipeline = RecordingPipeline(
        [
            ActionResult.succeeded(
                request.request_id,
                output="must not execute",
            )
        ]
    )

    loop, state, safety_gate, _ = build_stack(
        pipeline,
        requests={
            "task-1": request,
        },
    )

    safety_gate.authorization = AutonomousAuthorization(
        False,
        "denied by integration test",
    )

    report = loop.run(
        "finish integration test"
    )

    assert report.success is False
    assert report.completed_task_ids == ()
    assert report.failed_task_ids == (
        "task-1",
    )

    assert state.executed == []
    assert pipeline.requests == []

    assert safety_gate.calls == [
        (
            "run-0565",
            "task-1",
        )
    ]


def test_multiple_tasks_preserve_loop_order() -> None:
    request_one = make_request(
        "task-1"
    )
    request_two = make_request(
        "task-2"
    )

    pipeline = RecordingPipeline(
        [
            ActionResult.succeeded(
                request_one.request_id,
                output="one",
            ),
            ActionResult.succeeded(
                request_two.request_id,
                output="two",
            ),
        ]
    )

    loop, state, safety_gate, _ = build_stack(
        pipeline,
        (
            "task-1",
            "task-2",
        ),
        requests={
            "task-1": request_one,
            "task-2": request_two,
        },
    )

    report = loop.run(
        "finish integration test"
    )

    assert report.success is True
    assert report.completed_task_ids == (
        "task-1",
        "task-2",
    )
    assert report.failed_task_ids == ()

    assert state.executed == [
        "task-1",
        "task-2",
    ]

    assert safety_gate.calls == [
        (
            "run-0565",
            "task-1",
        ),
        (
            "run-0565",
            "task-2",
        ),
    ]

    assert pipeline.requests == [
        request_one,
        request_two,
    ]


def test_failed_pipeline_result_propagates_to_autonomous_loop() -> None:
    request = make_request(
        "task-1"
    )

    pipeline = RecordingPipeline(
        [
            ActionResult.failed(
                request.request_id,
                error="permanent failure",
            )
        ]
    )

    loop, state, _, _ = build_stack(
        pipeline,
        requests={
            "task-1": request,
        },
    )

    report = loop.run(
        "finish integration test"
    )

    assert report.success is False
    assert report.failed_task_ids == (
        "task-1",
    )

    assert state.executed == [
        "task-1",
    ]

    assert pipeline.requests == [
        request,
    ]
