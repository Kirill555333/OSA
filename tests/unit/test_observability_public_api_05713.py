from __future__ import annotations

import importlib

import osa.observability as observability


EXPECTED_EXPORTS = {
    "ActionPipelineLike",
    "ObservableActionRouter",
    "ObservableActionSafetyPipeline",
    "ActionRouterLike",
    "ActionRouterObservabilityError",
    "AgentLike",
    "AgentObservabilityError",
    "ObservableAgent",
    "AutonomousObservabilityError",
    "ObservableAutonomousLoop",
    "EMPTY_OBSERVABILITY_CONTEXT",
    "ObservabilityContext",
    "ObservabilityContextError",
    "ObservabilityEvent",
    "ObservabilityEventError",
    "OBSERVABLE_EXECUTION_SOURCES",
    "ActionExecutionHandler",
    "ActionExecutionObservabilityError",
    "ObservableActionExecution",
    "InMemoryObservabilityLogger",
    "NullObservabilityLogger",
    "ObservabilityLogger",
    "ObservabilityLoggerError",
    "ObservabilityEventQuery",
    "ObservabilityEventQueryAPI",
    "ObservabilityEventSource",
    "ObservabilityQueryError",
    "ObservableAutonomousRecovery",
    "RecoveryObservabilityError",
    "DEFAULT_SENSITIVE_KEYS",
    "ObservabilityRedactionError",
    "ObservabilityRedactor",
    "RedactingObservabilityLogger",
}


def test_public_exports_are_present() -> None:
    exported = set(observability.__all__)

    assert exported == EXPECTED_EXPORTS


def test_public_exports_are_unique() -> None:
    assert len(
        observability.__all__
    ) == len(
        set(observability.__all__)
    )


def test_every_public_export_resolves() -> None:
    for name in observability.__all__:
        assert hasattr(
            observability,
            name,
        ), name


def test_public_exports_are_importable_by_name() -> None:
    for name in observability.__all__:
        namespace = {}

        exec(
            f"from osa.observability import {name}",
            {},
            namespace,
        )

        assert name in namespace
        assert namespace[name] is getattr(
            observability,
            name,
        )


def test_observability_module_is_importable() -> None:
    module = importlib.import_module(
        "osa.observability"
    )

    assert module is observability


def test_public_api_points_to_expected_modules() -> None:
    expected_modules = {
        "ObservableActionRouter": (
            "osa.observability.action_router"
        ),
        "ObservableActionSafetyPipeline": (
            "osa.observability.action_pipeline"
        ),
        "ObservableAgent": (
            "osa.observability.agent"
        ),
        "ObservableAutonomousLoop": (
            "osa.observability.autonomous"
        ),
        "ObservabilityEvent": (
            "osa.observability.events"
        ),
        "InMemoryObservabilityLogger": (
            "osa.observability.logger"
        ),
        "ObservableActionExecution": (
            "osa.observability.execution"
        ),
        "ObservabilityEventQueryAPI": (
            "osa.observability.query"
        ),
        "ObservableAutonomousRecovery": (
            "osa.observability.recovery"
        ),
        "ObservabilityRedactor": (
            "osa.observability.redaction"
        ),
        "RedactingObservabilityLogger": (
            "osa.observability.redaction"
        ),
    }

    for name, module_name in expected_modules.items():
        value = getattr(
            observability,
            name,
        )

        assert value.__module__ == module_name


def test_sensitive_keys_public_constant_is_immutable() -> None:
    assert isinstance(
        observability.DEFAULT_SENSITIVE_KEYS,
        frozenset,
    )


def test_execution_sources_public_constant_is_immutable() -> None:
    assert isinstance(
        observability.OBSERVABLE_EXECUTION_SOURCES,
        tuple,
    )


def test_empty_context_is_public_and_empty() -> None:
    context = (
        observability.EMPTY_OBSERVABILITY_CONTEXT
    )

    assert context.empty is True
    assert context.run_id is None
    assert context.task_id is None
    assert context.request_id is None


def test_query_api_can_be_constructed_from_public_api() -> None:
    logger = (
        observability.InMemoryObservabilityLogger()
    )

    api = (
        observability.ObservabilityEventQueryAPI(
            logger
        )
    )

    assert api.source is logger


def test_redacting_logger_can_be_constructed_from_public_api() -> None:
    logger = (
        observability.InMemoryObservabilityLogger()
    )

    safe_logger = (
        observability.RedactingObservabilityLogger(
            logger
        )
    )

    assert safe_logger.logger is logger
    assert isinstance(
        safe_logger.redactor,
        observability.ObservabilityRedactor,
    )
