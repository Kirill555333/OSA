from __future__ import annotations

import pytest

from osa.actions import (
    ActionKind,
    ActionRequest,
    ActionResult,
    ActionRouter,
    ActionRouterError,
)


class FakeHandler:
    def __init__(self, result: ActionResult | None = None) -> None:
        self.calls: list[ActionRequest] = []
        self.result = result

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls.append(request)

        if self.result is not None:
            return self.result

        return ActionResult.succeeded(
            request.request_id,
            output=f"handled:{request.name}",
        )


class RaisingHandler:
    def execute(self, request: ActionRequest) -> ActionResult:
        raise RuntimeError("simulated backend failure")


class InvalidResultHandler:
    def execute(self, request: ActionRequest) -> object:
        return object()


def make_request(
    kind: ActionKind = ActionKind.BROWSER,
    name: str = "browser_open",
) -> ActionRequest:
    return ActionRequest(
        kind=kind,
        name=name,
    )


def test_router_dispatches_to_matching_backend() -> None:
    browser = FakeHandler()
    desktop = FakeHandler()

    router = ActionRouter(
        {
            ActionKind.BROWSER: browser,
            "desktop": desktop,
        }
    )

    request = make_request(ActionKind.BROWSER)
    result = router.dispatch(request)

    assert result.success is True
    assert result.output == "handled:browser_open"
    assert browser.calls == [request]
    assert desktop.calls == []


def test_router_returns_registered_kinds_in_stable_order() -> None:
    router = ActionRouter(
        {
            ActionKind.TOOL: FakeHandler(),
            ActionKind.BROWSER: FakeHandler(),
            ActionKind.DESKTOP: FakeHandler(),
        }
    )

    assert router.supported_kinds == (
        ActionKind.BROWSER,
        ActionKind.DESKTOP,
        ActionKind.TOOL,
    )


def test_router_reports_missing_handler_as_failed_result() -> None:
    router = ActionRouter()
    request = make_request(ActionKind.TOOL, "tool_call")

    result = router.dispatch(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == (
        "No action handler is registered for 'tool'."
    )
    assert result.metadata["router_error"] == "handler_not_registered"


def test_router_converts_backend_exception_to_failed_result() -> None:
    router = ActionRouter(
        {ActionKind.BROWSER: RaisingHandler()}
    )
    request = make_request()

    result = router.dispatch(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == "Action handler failed: simulated backend failure"
    assert result.metadata["router_error"] == "handler_exception"
    assert result.metadata["exception_type"] == "RuntimeError"


def test_router_rejects_invalid_handler_registration() -> None:
    router = ActionRouter()

    with pytest.raises(
        ActionRouterError,
        match=r"Handler for 'browser' must expose execute",
    ):
        router.register(ActionKind.BROWSER, object())


def test_router_rejects_invalid_request_type() -> None:
    router = ActionRouter()

    with pytest.raises(
        ActionRouterError,
        match="request must be an ActionRequest",
    ):
        router.dispatch(object())  # type: ignore[arg-type]


def test_router_rejects_invalid_handler_result_type() -> None:
    router = ActionRouter(
        {ActionKind.BROWSER: InvalidResultHandler()}
    )
    request = make_request()

    result = router.dispatch(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == "Action handler returned an invalid result type."
    assert result.metadata["router_error"] == "invalid_handler_result"
    assert result.metadata["result_type"] == "object"


def test_router_rejects_mismatched_request_id() -> None:
    router = ActionRouter(
        {
            ActionKind.BROWSER: FakeHandler(
                ActionResult.succeeded(
                    "different-request-id",
                    output="wrong correlation",
                )
            )
        }
    )
    request = make_request()

    result = router.dispatch(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == (
        "Action handler returned a mismatched request_id."
    )
    assert result.metadata["router_error"] == "request_id_mismatch"


def test_router_unregister_removes_handler() -> None:
    router = ActionRouter(
        {ActionKind.TOOL: FakeHandler()}
    )

    router.unregister("tool")

    assert router.handler_for(ActionKind.TOOL) is None
    assert router.supported_kinds == ()


def test_router_missing_unregister_is_idempotent() -> None:
    router = ActionRouter()

    router.unregister(ActionKind.DESKTOP)

    assert router.supported_kinds == ()


def test_router_handler_for_normalizes_string_kind() -> None:
    handler = FakeHandler()
    router = ActionRouter({"desktop": handler})

    assert router.handler_for("desktop") is handler


def test_router_rejects_invalid_kind() -> None:
    router = ActionRouter()

    with pytest.raises(
        ActionRouterError,
        match="kind must be a valid ActionKind",
    ):
        router.handler_for("unknown")
