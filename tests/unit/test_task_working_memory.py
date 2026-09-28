from osa.tasks.state import TaskWorkingMemory, WorkingMemoryError


def test_working_memory_tracks_task_lifecycle() -> None:
    memory = TaskWorkingMemory()

    memory.start_run(
        "run-1",
        "Complete project task",
    )

    memory.task_started(
        "run-1",
        "task-1",
        1,
    )

    snapshot = memory.snapshot("run-1")

    assert snapshot.goal == "Complete project task"
    assert snapshot.active_task_id == "task-1"
    assert snapshot.attempts == {"task-1": 1}

    memory.task_completed(
        "run-1",
        "task-1",
        "done",
    )

    snapshot = memory.snapshot("run-1")

    assert snapshot.active_task_id is None
    assert snapshot.completed_task_ids == ("task-1",)
    assert snapshot.outputs == {"task-1": "done"}
    assert snapshot.errors == {}


def test_working_memory_tracks_failure_and_retry() -> None:
    memory = TaskWorkingMemory()

    memory.start_run(
        "run-1",
        "Recover failed task",
    )

    memory.task_started(
        "run-1",
        "task-1",
        1,
    )

    memory.task_failed(
        "run-1",
        "task-1",
        "temporary failure",
    )

    snapshot = memory.snapshot("run-1")

    assert snapshot.failed_task_ids == ("task-1",)
    assert snapshot.errors["task-1"] == "temporary failure"
    assert snapshot.last_error == "temporary failure"

    memory.task_retried(
        "run-1",
        "task-1",
    )

    snapshot = memory.snapshot("run-1")

    assert snapshot.failed_task_ids == ()
    assert snapshot.active_task_id is None


def test_working_memory_rejects_unknown_run() -> None:
    memory = TaskWorkingMemory()

    try:
        memory.snapshot("missing")
    except WorkingMemoryError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected WorkingMemoryError")


def test_working_memory_can_be_cleared() -> None:
    memory = TaskWorkingMemory()

    memory.start_run(
        "run-1",
        "Temporary task",
    )

    assert memory.contains("run-1")

    memory.clear("run-1")

    assert not memory.contains("run-1")
