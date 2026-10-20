"""Custom interactive widgets: Core Orb, Action Feed, and Chat View for OSA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from osa.gui.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_CARD,
    BORDER_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class CoreOrbWidget(QWidget):
    """Animated central status orb representing OSA state."""

    STATE_COLORS = {
        "idle": (QColor("#4B5563"), "СПИТ (ЖДУ СЛОВО 'ОСА')"),
        "listening": (QColor(ACCENT_CYAN), "СЛУШАЮ РЕЧЬ..."),
        "thinking": (QColor(ACCENT_AMBER), "РАССУЖДЕНИЕ И ДЕЙСТВИЕ..."),
        "speaking": (QColor(ACCENT_GREEN), "ОТВЕЧАЮ ВСЛУХ"),
        "error": (QColor(ACCENT_RED), "ОШИБКА ДЕЙСТВИЯ"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self._state = "idle"
        self._angle = 0.0
        self._pulse = 0.0

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_tick)
        self._anim_timer.start(30)

    def set_state(self, state: str) -> None:
        if state in self.STATE_COLORS:
            self._state = state
            self.update()

    @property
    def current_state_label(self) -> str:
        return self.STATE_COLORS.get(self._state, (QColor("#4B5563"), "ГОТОВ"))[1]

    def _on_tick(self) -> None:
        self._angle = (self._angle + 2.5) % 360.0
        self._pulse = (self._pulse + 0.08) % (2 * math.pi)
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color, _ = self.STATE_COLORS.get(self._state, (QColor("#4B5563"), ""))
        center = QPointF(self.width() / 2.0, self.height() / 2.0)

        pulse_delta = math.sin(self._pulse) * 4.0 if self._state != "idle" else 0.0
        radius = 42.0 + pulse_delta

        glow_color = QColor(color.red(), color.green(), color.blue(), 35)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(center, radius + 14.0, radius + 14.0)

        pen_dash = QPen(color, 2.5, Qt.PenStyle.DashLine)
        pen_dash.setDashPattern([6, 6])
        painter.setPen(pen_dash)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.save()
        painter.translate(center)
        painter.rotate(self._angle)
        painter.drawEllipse(QRectF(-radius - 4, -radius - 4, (radius + 4) * 2, (radius + 4) * 2))
        painter.restore()

        core_color = QColor(
            color.red(),
            color.green(),
            color.blue(),
            180 if self._state != "idle" else 90,
        )
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(core_color))
        painter.drawEllipse(center, radius - 8.0, radius - 8.0)

        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Helvetica", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(0, 0, self.width(), self.height()),
            Qt.AlignmentFlag.AlignCenter,
            "OSA",
        )


class ActionFeedWidget(QFrame):
    """Right-side live telemetry feed showing tools executed in real time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setFixedWidth(290)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        title = QLabel("LIVE TELEMETRY")
        title.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-weight: bold; "
            "font-size: 12px; letter-spacing: 1px;"
        )
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                border: none;
            }}
            QListWidget::item {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 8px 10px;
                margin-bottom: 6px;
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
        """)
        layout.addWidget(self._list)

        self._preview_label = QLabel("Нет активного скриншота")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_CARD};
                border: 1px dashed {BORDER_COLOR};
                border-radius: 8px;
                color: {TEXT_MUTED};
                font-size: 11px;
                padding: 6px;
            }}
        """)
        self._preview_label.setFixedHeight(120)
        layout.addWidget(self._preview_label)

    def add_action(self, icon: str, title: str, details: str = "") -> None:
        item = QListWidgetItem()
        item.setText(f"{icon} {title}\n  {details}" if details else f"{icon} {title}")
        self._list.insertItem(0, item)
        if self._list.count() > 30:
            self._list.takeItem(self._list.count() - 1)

    def update_screenshot_preview(self, image_path: str) -> None:
        path = Path(image_path)
        # Require file to exist and be at least 1 KB to prevent libpng errors
        if not path.exists() or path.stat().st_size < 1024:
            return

        try:
            pixmap = QPixmap()
            if pixmap.load(str(path)):
                scaled = pixmap.scaled(
                    self._preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._preview_label.setPixmap(scaled)
        except Exception:
            pass


class ChatViewWidget(QFrame):
    """Central conversation stream with styled message bubbles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        layout.addWidget(self._browser)
        self._append_system_banner()

    def _append_system_banner(self) -> None:
        banner_html = f"""
        <div style="text-align: center; margin-bottom: 16px;">
            <span style="color: {ACCENT_CYAN}; font-size: 16px; font-weight: bold;">OSA HUD (v1.5)</span><br/>
            <span style="color: {TEXT_MUTED}; font-size: 12px;">Автономный ассистент готов к работе. Напишите команду или нажмите микрофон.</span>
        </div>
        """
        self._browser.append(banner_html)

    def append_user_message(self, text: str) -> None:
        html = f"""
        <div style="margin-top: 10px; margin-bottom: 10px; text-align: right;">
            <div style="display: inline-block; background-color: #1E3A8A; color: #FFFFFF; padding: 10px 14px; border-radius: 12px; max-width: 80%; text-align: left;">
                <b style="color: {ACCENT_CYAN}; font-size: 11px;">ВЫ</b><br/>
                {text}
            </div>
        </div>
        """
        self._browser.append(html)
        self._scroll_to_bottom()

    def append_assistant_message(self, text: str) -> None:
        html = f"""
        <div style="margin-top: 10px; margin-bottom: 10px; text-align: left;">
            <div style="display: inline-block; background-color: {BG_CARD}; color: {TEXT_PRIMARY}; padding: 10px 14px; border-radius: 12px; border: 1px solid {BORDER_COLOR}; max-width: 85%;">
                <b style="color: {ACCENT_GREEN}; font-size: 11px;">OSA</b><br/>
                {text.replace(chr(10), '<br/>')}
            </div>
        </div>
        """
        self._browser.append(html)
        self._scroll_to_bottom()

    def append_system_notice(self, notice: str) -> None:
        html = f"""
        <div style="text-align: center; margin: 6px 0;">
            <span style="color: {TEXT_SECONDARY}; font-size: 11px; background-color: {BG_CARD}; padding: 4px 10px; border-radius: 6px;">
                {notice}
            </span>
        </div>
        """
        self._browser.append(html)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
