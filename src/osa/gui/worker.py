"""Background worker for GUI: agent tasks + background voice listening (OSA 1.5.3.0)."""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer, Signal

from osa.core import Agent
from osa.shell.input_hub import InputHub
from osa.utils.paths import resolve_osa_home
from osa.voice import OSAVoiceEngine
from osa.voice.background import BackgroundListener
from osa.voice.background_audio import FasterWhisperTranscriber, SoundDeviceFrameSource
from osa.voice.wake import ListenMode, WakeConfig, WakeWordGate


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
        self._poll_timer: QTimer | None = None
        self._listening = False

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self, model_size: str = "small") -> bool:
        if self._listening:
            return True

        try:
            self.action_occurred.emit("🎙", "Загрузка Whisper", f"Модель {model_size}")
            transcriber = FasterWhisperTranscriber(model_size)
            transcriber.preload()
            self._input_hub = InputHub()
            gate = WakeWordGate(
                WakeConfig(follow_up_seconds=self._follow_up_seconds),
                mode=ListenMode.WAKE,
            )
            self._background_listener = BackgroundListener(
                SoundDeviceFrameSource(),
                transcriber,
                gate,
                self._input_hub.post_voice,
            )
            self._background_listener.start()
        except Exception as exc:
            self.action_occurred.emit("❌", "Микрофон недоступен", str(exc))
            self._cleanup_listener()
            return False

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_voice_events)
        self._poll_timer.start()

        self._listening = True
        self.listening_state_changed.emit(True)
        self.state_changed.emit("listening")
        self.action_occurred.emit(
            "🎙", "Слушаю", "Скажите «Оса, …» или продолжите разговор"
        )
        return True

    def stop_listening(self) -> None:
        if not self._listening:
            return

        self._listening = False

        timer = self._poll_timer
        self._poll_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

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
            except Exception:
                pass

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
                self.action_occurred.emit("🗣", "Распознано", text[:60])
                self.voice_text_received.emit(text)

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
