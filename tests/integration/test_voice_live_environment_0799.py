from __future__ import annotations

import functools
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from osa.actions import (
    ActionKind,
    ActionRequest,
    ActionRouter,
    ActionSafetyPipeline,
    BrowserActionAdapter,
    DesktopActionAdapter,
)
from osa.actions.browser import BrowserActionAdapterError
from osa.actions.contracts import ActionResult
from osa.browser import (
    BrowserAutomationError,
    BrowserLocator,
    PlaywrightBrowser,
    PlaywrightBrowserConfig,
)
from osa.browser.playwright import PlaywrightBrowserConfig
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import AgentVoiceActionAdapter
from osa.desktop import WindowsDesktopAutomation
from osa.recovery_action import RecoverableActionExecutor
from osa.voice.contracts import VoiceInput, VoiceTranscript
from osa.voice.session import VoiceSession
from osa.tools.browser_automation import create_browser_action_tools


class FakeVAD:
    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        del audio, sample_rate_hz, channels
        return True

    def reset(self) -> None:
        return None


class FakeSTT:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        del voice_input
        return VoiceTranscript(
            text=self.transcript,
            language="en",
        )


class FakeTTS:
    def synthesize(self, output: Any) -> bytes:
        return output.text.encode("utf-8")

    def stop(self) -> None:
        return None


class StaticVoiceResolver:
    def __init__(
        self,
        *,
        kind: ActionKind,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.kind = kind
        self.name = name
        self.arguments = dict(arguments)

    def resolve(self, command: str) -> ActionRequest:
        del command

        return ActionRequest(
            kind=self.kind,
            name=self.name,
            arguments=dict(self.arguments),
            metadata={
                "source": "voice-live",
            },
        )


class RecoveringAgent:
    def __init__(
        self,
        recovery: AgentRecoveryIntegration,
    ) -> None:
        self._recovery = recovery

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ):
        return self._recovery.execute(
            request,
            run_id=run_id,
            task_id=task_id,
        )


def _build_voice_session(
    *,
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    pipeline: ActionSafetyPipeline,
) -> VoiceSession:
    recovery_executor = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=1,
    )
    recovery = AgentRecoveryIntegration(
        recovery_executor,
    )
    agent = RecoveringAgent(recovery)

    resolver = StaticVoiceResolver(
        kind=kind,
        name=name,
        arguments=arguments,
    )

    voice_agent = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

    return VoiceSession(
        voice_agent,
        FakeVAD(),
        FakeSTT("execute the requested action"),
        FakeTTS(),
    )


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        del format, args


@pytest.fixture
def local_web_server(
    tmp_path: Path,
):
    html = """
<!doctype html>
<html>
<head>
    <title>OSA Live Test</title>
</head>
<body>
    <h1>OSA live browser test</h1>
    <button id="submit" type="button"
            onclick="document.getElementById('status').textContent='submitted';">
        Submit
    </button>
    <div id="status">waiting</div>
</body>
</html>
""".strip()

    (tmp_path / "index.html").write_text(
        html,
        encoding="utf-8",
    )

    handler = functools.partial(
        QuietHTTPRequestHandler,
        directory=str(tmp_path),
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/index.html"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_voice_reaches_real_playwright_browser_backend(
    local_web_server: str,
) -> None:
    browser = PlaywrightBrowser(
        config=PlaywrightBrowserConfig(
            headless=True,
            browser_name="chromium",
        )
    )

    try:
        if not browser.health_check():
            pytest.skip(
                "Playwright Chromium is not available in this environment."
            )

        adapter = BrowserActionAdapter(
            create_browser_action_tools(browser)
        )
        router = ActionRouter(
            {
                ActionKind.BROWSER: adapter,
            }
        )
        pipeline = ActionSafetyPipeline(router)

        open_session = _build_voice_session(
            kind=ActionKind.BROWSER,
            name="browser_open",
            arguments={
                "url": local_web_server,
            },
            pipeline=pipeline,
        )

        open_result = open_session.process_audio(
            b"audio",
            sample_rate_hz=16_000,
        )

        assert open_result is not None
        assert "OSA Live Test" in open_result.response.text

        click_session = _build_voice_session(
            kind=ActionKind.BROWSER,
            name="browser_click",
            arguments={
                "locator": {
                    "kind": "css",
                    "value": "#submit",
                },
            },
            pipeline=pipeline,
        )

        click_result = click_session.process_audio(
            b"audio",
            sample_rate_hz=16_000,
        )

        assert click_result is not None
        assert click_result.response.text == (
            "Clicked element: css=#submit"
        )

        read_session = _build_voice_session(
            kind=ActionKind.BROWSER,
            name="browser_read",
            arguments={
                "locator": {
                    "kind": "css",
                    "value": "#status",
                },
            },
            pipeline=pipeline,
        )

        read_result = read_session.process_audio(
            b"audio",
            sample_rate_hz=16_000,
        )

        assert read_result is not None
        assert read_result.response.text == "submitted"

        current_tab = browser.current_tab()
        assert current_tab.url == local_web_server
        assert current_tab.title == "OSA Live Test"

    except BrowserAutomationError as exc:
        pytest.fail(
            f"Playwright live browser interaction failed: {exc}"
        )
    finally:
        browser.close()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Live Windows UI Automation test.",
)
def test_voice_reaches_real_windows_desktop_backend() -> None:
    backend = WindowsDesktopAutomation()
    adapter = DesktopActionAdapter(backend)

    router = ActionRouter(
        {
            ActionKind.DESKTOP: adapter,
        }
    )
    pipeline = ActionSafetyPipeline(router)

    session = _build_voice_session(
        kind=ActionKind.DESKTOP,
        name="desktop_health_check",
        arguments={},
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == (
        "Desktop backend is healthy."
    )
