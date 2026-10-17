"""Main window for OSA Desktop HUD."""

from __future__ import annotations

import threading

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from osa.core import Agent
from osa.gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    MAIN_STYLESHEET,
    TEXT_MUTED,
)
from osa.gui.widgets import ActionFeedWidget, ChatViewWidget, CoreOrbWidget
from osa.gui.worker import AgentWorker


class JarvisMainWindow(QMainWindow):
    """Primary HUD window combining Chat, Status Orb, and Live Telemetry for OSA."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.setWindowTitle("OSA — Personal AI Assistant HUD")
        self.resize(1100, 720)
        self.setMinimumSize(850, 550)
        self.setStyleSheet(MAIN_STYLESHEET)

        self._worker = AgentWorker(agent)

        self._setup_ui()
        self._connect_signals()

        # Emit initial status
        self._worker.action_occurred.emit("⚡", "Инициализация", "OSA v1.5 Ядро готово к работе")

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(18, 14, 18, 18)
        root_layout.setSpacing(14)

        # 1. Top Header with Core Orb
        header_frame = QFrame()
        header_frame.setObjectName("PanelFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 10, 16, 10)

        # Left title info
        title_box = QVBoxLayout()
        title_label = QLabel("OSA AUTONOMOUS DESKTOP")
        title_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {ACCENT_CYAN}; letter-spacing: 1px;")
        self._status_desc = QLabel("СПИТ (ЖДУ СЛОВО 'ОСА')")
        self._status_desc.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {TEXT_MUTED};")
        title_box.addWidget(title_label)
        title_box.addWidget(self._status_desc)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Center animated Orb
        self._orb = CoreOrbWidget()
        header_layout.addWidget(self._orb)
        header_layout.addStretch()

        # Right quick controls
        controls_box = QHBoxLayout()
        self._btn_voice_toggle = QPushButton("🔊 ГОЛОС: ВКЛ")
        self._btn_voice_toggle.setCheckable(True)
        self._btn_voice_toggle.setChecked(True)
        self._btn_voice_toggle.setStyleSheet(f"color: {ACCENT_GREEN}; border-color: {ACCENT_GREEN};")
        controls_box.addWidget(self._btn_voice_toggle)

        header_layout.addLayout(controls_box)
        root_layout.addWidget(header_frame)

        # 2. Main Work Area (Chat on Left/Center, Action Feed on Right)
        work_layout = QHBoxLayout()
        work_layout.setSpacing(14)

        # Chat
        self._chat_view = ChatViewWidget()
        work_layout.addWidget(self._chat_view, stretch=3)

        # Action Telemetry Feed
        self._action_feed = ActionFeedWidget()
        work_layout.addWidget(self._action_feed, stretch=1)

        root_layout.addLayout(work_layout)

        # 3. Bottom Input Bar
        input_frame = QFrame()
        input_frame.setObjectName("PanelFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 8, 10, 8)
        input_layout.setSpacing(10)

        self._btn_mic = QPushButton("🎙 Говорить")
        self._btn_mic.setFixedWidth(120)
        self._btn_mic.setStyleSheet(f"font-weight: bold; border-color: {ACCENT_CYAN};")
        input_layout.addWidget(self._btn_mic)

        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("Напишите команду или вопрос для OSA (например: 'хочу куплинова посмотреть')...")
        input_layout.addWidget(self._input_field)

        self._btn_send = QPushButton("Отправить")
        self._btn_send.setFixedWidth(100)
        input_layout.addWidget(self._btn_send)

        root_layout.addWidget(input_frame)

    def _connect_signals(self) -> None:
        self._btn_send.clicked.connect(self._on_send_clicked)
        self._input_field.returnPressed.connect(self._on_send_clicked)
        self._btn_mic.clicked.connect(self._on_mic_clicked)
        self._btn_voice_toggle.clicked.connect(self._on_voice_toggled)

        # Worker signals
        self._worker.state_changed.connect(self._on_state_changed)
        self._worker.message_received.connect(self._on_message_received)
        self._worker.action_occurred.connect(self._action_feed.add_action)
        self._worker.screen_captured.connect(self._action_feed.update_screenshot_preview)
        self._worker.task_finished.connect(self._on_task_finished)

    def _on_send_clicked(self) -> None:
        text = self._input_field.text().strip()
        if not text:
            return
        self._input_field.clear()
        self._input_field.setEnabled(False)
        self._btn_send.setEnabled(False)

        # Safe asynchronous execution
        threading.Thread(target=self._worker.execute_text_query, args=(text,), daemon=True).start()

    def _on_mic_clicked(self) -> None:
        self._btn_mic.setEnabled(False)
        self._btn_mic.setObjectName("MicButtonActive")
        self._btn_mic.style().unpolish(self._btn_mic)
        self._btn_mic.style().polish(self._btn_mic)

        # Safe asynchronous execution
        threading.Thread(target=self._worker.listen_and_process, daemon=True).start()

    def _on_voice_toggled(self) -> None:
        is_on = self._btn_voice_toggle.isChecked()
        self._worker._voice_enabled = is_on
        self._btn_voice_toggle.setText("🔊 ГОЛОС: ВКЛ" if is_on else "🔇 ГОЛОС: ВЫКЛ")
        color = ACCENT_GREEN if is_on else TEXT_MUTED
        self._btn_voice_toggle.setStyleSheet(f"color: {color}; border-color: {color};")
        self._chat_view.append_system_notice("Голосовой вывод включен" if is_on else "Голосовой вывод отключен")

    def _on_state_changed(self, state: str) -> None:
        self._orb.set_state(state)
        self._status_desc.setText(self._orb.current_state_label)

    def _on_message_received(self, sender: str, text: str) -> None:
        if sender == "user":
            self._chat_view.append_user_message(text)
        else:
            self._chat_view.append_assistant_message(text)

    def _on_task_finished(self) -> None:
        self._input_field.setEnabled(True)
        self._btn_send.setEnabled(True)
        self._btn_mic.setEnabled(True)
        self._btn_mic.setObjectName("")
        self._btn_mic.style().unpolish(self._btn_mic)
        self._btn_mic.style().polish(self._btn_mic)
        self._input_field.setFocus()
