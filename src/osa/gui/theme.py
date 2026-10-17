"""Modern dark OSA theme styles and color constants."""

from __future__ import annotations

BG_MAIN = "#0B0F19"
BG_PANEL = "#111827"
BG_CARD = "#1F2937"
BORDER_COLOR = "#374151"

TEXT_PRIMARY = "#F3F4F6"
TEXT_SECONDARY = "#9CA3AF"
TEXT_MUTED = "#6B7280"

ACCENT_CYAN = "#00D2FF"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"

MAIN_STYLESHEET = f"""
QMainWindow {{
    background-color: {BG_MAIN};
}}

QWidget {{
    font-family: Arial, Helvetica;
    color: {TEXT_PRIMARY};
}}

QFrame#PanelFrame {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
}}

QLineEdit {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT_CYAN};
}}

QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: #374151;
    border: 1px solid {ACCENT_CYAN};
}}

QPushButton:pressed {{
    background-color: #4B5563;
}}

QPushButton#MicButtonActive {{
    background-color: #064E3B;
    border: 1px solid {ACCENT_GREEN};
    color: #A7F3D0;
}}

QTextBrowser {{
    background-color: transparent;
    border: none;
    font-size: 14px;
    line-height: 1.5;
}}

QScrollBar:vertical {{
    border: none;
    background: {BG_PANEL};
    width: 6px;
    border-radius: 3px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_COLOR};
    border-radius: 3px;
    min-height: 20px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
