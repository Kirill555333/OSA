"""Background worker for GUI: agent tasks + background voice listening (OSA 1.5.3.0)."""

from __future__ import annotations

import threading
import time
import traceback

from PySide6.QtCore import QObject, QTimer, Signal

from osa.core import Agent
from osa.shell.input_hub import InputHub
from osa.utils.paths import resolve_osa_home
from osa.voice import OSAVoiceEngine
from osa.voice.background import BackgroundListener
from osa.voice.background_audio import FasterWhisperTranscriber, SoundDeviceFrameSource
from osa.voice.wake import ListenMode, WakeConfig, WakeWordGate


def _log_voice(message: str) -> None:
    """Append a line to ~/.osa/logs/gui-voice.log (best effort)."""
    try:
        path = resolve_osa_home() / "logs" / "gui-voice.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except Exception:
        pass


class AgentWorker(QObject):
    """Runs Agent thinking, tools and background voice listening asynchronously."""

    state_changed = Signal(str)
    message_received = Signal(str, str)
    action_occurred = Signal(str, str, str)
    screen_captured = Signal(str)
    task_finished = Signal()
    voice_text_received = Signal(str)
    listening_state_changed = Signal(bool)

    def __init__(self, agent: Agent, *, follow_up_seconds: float = 60.0) -> None:
        super().__init__()
        self._agent = agent
        self._voice_engine = OSAVoiceEngine()
        self._voice_enabled = True
        self._follow_up_seconds = follow_up_seconds

        self._background_listener: BackgroundListener | None = None
        self._input_hub: InputHub | None = None
        self._listening = False
        self._starting = False

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_voice_events)
        self._poll_timer.start()

    @property
    def is_listening(self) -> bool:
        return self._listening or self._starting

    # --- background listening ------------------------------------------------------

    def _build_listener(
        self, model_size: str
    ) -> tuple[BackgroundListener, InputHub]:
        transcriber = FasterWhisperTranscriber(model_size)
        transcriber.preload()
        hub = InputHub()
        gate = WakeWordGate(
            WakeConfig(follow_up_seconds=self._follow_up_seconds),
            mode=ListenMode.WAKE,
        )
        listener = BackgroundListener(
            SoundDeviceFrameSource(),
            transcriber,
            gate,
            hub.post_voice,
        )
        return listener, hub

    def start_listening(self, model_size: str = "small") -> None:
        if self._listening or self._starting:
            return
        self._starting = True
        _log_voice(f"start_listening requested (model={model_size})")
        threading.Thread(
            target=self._start_listening_bg,
            args=(model_size,),
            daemon=True,
            name="osa-gui-listener-start",
        ).start()

    def _start_listening_bg(self, model_size: str) -> None:
        try:
            self.action_occurred.emit("🎙", "Загрузка Whisper", f"Модель {model_size}")
            _log_voice(f"preload({model_size}) beginning")
            listener, hub = self._build_listener(model_size)
            _log_voice("preload done")

            if not self._starting:
                _log_voice("start aborted (user cancelled during preload)")
                try:
                    listener.stop()
                except Exception:
                    pass
                return

            self.action_occurred.emit("🎙", "Открываю микрофон…", "")
            listener.start()
            _log_voice(f"listener started, running={listener.running}")

            self._background_listener = listener
            self._input_hub = hub
            self._listening = True
            self._starting = False

            self.listening_state_changed.emit(True)
            self.state_changed.emit("listening")
            self.action_occurred.emit(
                "🎙", "Слушаю", "Скажите «Оса, …» или продолжите разговор"
            )
            _log_voice("start_listening succeeded")
        except Exception as exc:
            _log_voice(f"start_listening FAILED: {exc!r}\n{traceback.format_exc()}")
            self._starting = False
            self._cleanup_listener()
            self.listening_state_changed.emit(False)
            self.state_changed.emit("error")
            self.action_occurred.emit(
                "❌", "Голос не запустился", f"{type(exc).__name__}: {exc}"
            )

    def stop_listening(self) -> None:
        if not self._listening and not self._starting:
            return
        self._listening = False
        self._starting = False
        _log_voice("stop_listening requested")
        self._cleanup_listener()
        self.listening_state_changed.emit(False)
        self.state_changed.emit("idle")
        self.action_occurred.emit("⏹", "Микрофон выключен", "")

    def _cleanup_listener(self) -> None:
        listener = self._background_listener
        self._background_listener = None
        self._input_hub = None
        if listener is not None:
            try:
                listener.stop()
                _log_voice("listener stopped")
            except Exception as exc:
                _log_voice(f"listener.stop() failed: {exc!r}")

    def _poll_voice_events(self) -> None:
        hub = self._input_hub
        if hub is None:
            return
        while True:
            event = hub.next_event(timeout=0)
            if event is None:
                return
            if event.kind == "voice" and event.text.strip():
                text = event.text.strip()
                _log_voice(f"voice received: {text!r} (reason={event.reason})")
                self.action_occurred.emit("🗣", "Распознано", text[:60])
                self.voice_text_received.emit(text)

    # --- agent execution -----------------------------------------------------------

    def execute_text_query(self, user_query: str) -> None:
        query = user_query.strip()
        if not query:
            self.task_finished.emit()
            return
        self.message_received.emit("user", query)
        self.state_changed.emit("thinking")
        self.action_occurred.emit("🧠", "Анализ задачи", query)
        try:
            response = self._agent.chat(query)
            output = getattr(response, "content", "") or ""
            if not output and hasattr(response, "reasoning_content"):
                output = getattr(response, "reasoning_content", "") or ""
            if not output:
                output = "Готово, сэр."
            self.message_received.emit("assistant", output)
            latest_screen = resolve_osa_home() / "data" / "screenshots" / "latest.png"
            if latest_screen.exists() and latest_screen.stat().st_size > 1024:
                self.screen_captured.emit(str(latest_screen))
            if self._voice_enabled:
                self._speak(output)
            if self._listening:
                self.state_changed.emit("listening")
                listener = self._background_listener
                if listener is not None:
                    try:
                        listener.open_follow_up()
                    except Exception:
                        pass
            else:
                self.state_changed.emit("idle")
        except Exception as exc:
            self.state_changed.emit("error")
            self.action_occurred.emit("❌", "Сбой выполнения", str(exc))
            self.message_received.emit(
                "assistant", f"Произошла ошибка при выполнении: {exc}"
            )
        finally:
            self.task_finished.emit()

    def _speak(self, text: str) -> None:
        listener = self._background_listener
        self.state_changed.emit("speaking")
        self.action_occurred.emit("🔊", "Озвучивание ответа", text[:40] + "...")
        if listener is not None:
            try:
                listener.mute()
            except Exception:
                pass
        try:
            self._voice_engine.speak(text)
            deadline = time.monotonic() + 120.0
            try:
                while self._voice_engine.is_speaking() and time.monotonic() < deadline:
                    time.sleep(0.05)
            except Exception:
                pass
        except Exception as exc:
            self.action_occurred.emit("⚠️", "Ошибка TTS", str(exc))
        finally:
            if listener is not None:
                try:
                    listener.unmute()
                except Exception:
                    pass

    def set_voice_enabled(self, enabled: bool) -> None:
        self._voice_enabled = bool(enabled)
