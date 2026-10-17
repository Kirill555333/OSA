"""Background worker executing Agent tasks and Voice loop without blocking GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from osa.core import Agent
from osa.voice import OSAVoiceEngine, OSAVoiceListener


class AgentWorker(QObject):
    """Runs Agent thinking, tools execution and voice interactions asynchronously."""

    state_changed = Signal(str)             # 'idle', 'listening', 'thinking', 'speaking', 'error'
    message_received = Signal(str, str)     # sender ('user' / 'assistant'), text
    action_occurred = Signal(str, str, str) # icon, title, details
    screen_captured = Signal(str)           # screenshot path
    task_finished = Signal()

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent
        self._voice_engine = OSAVoiceEngine()
        self._voice_listener: OSAVoiceListener | None = None
        self._voice_enabled = True

    def preload_voice(self) -> None:
        try:
            self._voice_listener = OSAVoiceListener(model_size="medium")
            self._voice_listener.preload_model()
        except Exception as exc:
            self.action_occurred.emit("⚠️", "Слух Whisper", f"Ошибка предзагрузки: {exc}")

    def execute_text_query(self, user_query: str) -> None:
        """Process user text input asynchronously with rolling memory window."""
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

            # Check if screenshot was taken
            latest_screen = Path.cwd() / "data" / "screenshots" / "latest.png"
            if latest_screen.exists() and latest_screen.stat().st_size > 1024:
                self.screen_captured.emit(str(latest_screen))

            if self._voice_enabled:
                self.state_changed.emit("speaking")
                self.action_occurred.emit("🔊", "Озвучивание ответа", output[:40] + "...")
                self._voice_engine.speak(output)

            self.state_changed.emit("idle")
        except Exception as exc:
            self.state_changed.emit("error")
            self.action_occurred.emit("❌", "Сбой выполнения", str(exc))
            self.message_received.emit("assistant", f"Произошла ошибка при выполнении: {exc}")
        finally:
            self.task_finished.emit()

    def listen_and_process(self) -> None:
        """Trigger single voice input phrase."""
        if self._voice_listener is None:
            self.preload_voice()

        self.state_changed.emit("listening")
        self.action_occurred.emit("🎙", "Слушаю речь", "Говорите команду...")

        try:
            assert self._voice_listener is not None
            text = self._voice_listener.listen_phrase()
            if text and text.strip():
                self.execute_text_query(text)
            else:
                self.state_changed.emit("idle")
                self.action_occurred.emit("ℹ️", "Микрофон", "Фраза не распознана")
                self.task_finished.emit()
        except Exception as exc:
            self.state_changed.emit("error")
            self.action_occurred.emit("❌", "Ошибка звука", str(exc))
            self.task_finished.emit()
