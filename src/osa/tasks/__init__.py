"""Task management components for OSA."""

from osa.tasks.safety import (
    AllowlistedAutonomousSafetyGate,
    AutonomousAuthorization,
    AutonomousSafetyError,
    AutonomousSafetyGate,
)

from osa.tasks.autonomous import (
    AutonomousCycle,
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousLoopError,
    AutonomousRunReport,
    AutonomousTaskResult,
    CallbackAutonomousBackend,
)

from osa.tasks.verification_rules import (
    create_default_task_verifier,
)

from osa.tasks.state import (
    TaskWorkingMemory,
    TaskWorkingMemorySnapshot,
    WorkingMemoryError,
)

from osa.tasks.verification import (
    TaskVerificationError,
    TaskVerifier,
    TaskVerifierFunction,
    VerificationResult,
)

from osa.tasks.recovery import (
    TaskRecoveryPolicy,
)

from osa.tasks.runner import (
    create_default_planned_task_runner,
    PlannedTaskRunner,
    PlanCreator,
    ActionResolver,
    TaskRunnerError,
    ToolExecutor,
)

from osa.tasks.action import (
    TaskAction,
    TaskActionError,
    TaskActionResolver,
)

from osa.tasks.executor import (
    TaskExecutionReport,
    TaskExecutor,
    TaskExecutorError,
    TaskHandler,
)

from osa.tasks.manager import (
    Task,
    TaskManager,
    TaskManagerError,
    TaskRun,
    TaskStatus,
)

from osa.tasks.action_bridge import (
    AutonomousActionBridge,
    AutonomousActionBridgeError,
    AutonomousActionResolver,
    CallbackAutonomousActionResolver,
)

from osa.tasks.autonomous_executor import (
    UnifiedAutonomousExecutor,
    UnifiedAutonomousExecutorError,
)

from osa.tasks.unified_safety import (
    UnifiedAutonomousSafetyError,
    UnifiedAutonomousTaskAuthorizer,
)

from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    AutonomousRecoveryPolicy,
    CallbackAutonomousRecoveryPolicy,
    NeverRetryAutonomousRecoveryPolicy,
)


__all__ = [
    "AllowlistedAutonomousSafetyGate",
    "AutonomousAuthorization",
    "AutonomousSafetyError",
    "AutonomousSafetyGate",
    "AutonomousCycle",
    "AutonomousLoop",
    "AutonomousLoopConfig",
    "AutonomousLoopError",
    "AutonomousRunReport",
    "AutonomousTaskResult",
    "CallbackAutonomousBackend",
    "create_default_planned_task_runner",
    "TaskRecoveryPolicy",
    "TaskWorkingMemory",
    "TaskWorkingMemorySnapshot",
    "WorkingMemoryError",
    "create_default_task_verifier",
    "VerificationResult",
    "TaskVerifierFunction",
    "TaskVerifier",
    "TaskVerificationError",
    "ToolExecutor",
    "TaskRunnerError",
    "ActionResolver",
    "PlanCreator",
    "PlannedTaskRunner",
    "TaskActionResolver",
    "TaskActionError",
    "TaskAction",
    "TaskExecutionReport",
    "TaskExecutor",
    "TaskExecutorError",
    "TaskHandler",
    "Task",
    "TaskManager",
    "TaskManagerError",
    "TaskRun",
    "TaskStatus",
    "AutonomousActionBridge",
    "AutonomousActionBridgeError",
    "AutonomousActionResolver",
    "CallbackAutonomousActionResolver",
    "UnifiedAutonomousExecutor",
    "UnifiedAutonomousExecutorError",
    "UnifiedAutonomousSafetyError",
    "UnifiedAutonomousTaskAuthorizer",
    "AutonomousRecoveryDecision",
    "AutonomousRecoveryExecutor",
    "AutonomousRecoveryPolicy",
    "CallbackAutonomousRecoveryPolicy",
    "NeverRetryAutonomousRecoveryPolicy",
]
