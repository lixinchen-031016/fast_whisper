#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""苹果风格主题：全局 QSS 与字体设置。

设计要点（macOS Human Interface Guidelines 风格）：
- 背景 #F5F5F7（苹果浅灰），卡片纯白 + 12px 圆角 + 细边框
- 强调色 #0A84FF（系统蓝），文字 #1D1D1F，次要文字 #86868B
- 字体栈按平台自动选择（macOS 苹方 / Windows 微软雅黑 / Linux Noto）
"""
import sys

from PySide6.QtGui import QFont

ACCENT = "#0A84FF"
ACCENT_HOVER = "#2492FF"
ACCENT_PRESSED = "#006EDB"
BG = "#F5F5F7"
CARD = "#FFFFFF"
TEXT = "#1D1D1F"
TEXT_SECOND = "#86868B"
BORDER = "#E5E5EA"
DANGER = "#FF3B30"

# 跨平台字体栈：macOS 用苹方，Windows 用微软雅黑。
# 不引用当前系统不存在的字体，避免 qt.qpa.fonts 字体别名填充警告。
if sys.platform == "darwin":
    FONT_STACK = '"PingFang SC", "Helvetica Neue"'
elif sys.platform == "win32":
    FONT_STACK = '"Microsoft YaHei", "Segoe UI"'
else:
    FONT_STACK = '"Noto Sans CJK SC", "WenQuanYi Micro Hei"'

QSS = f"""
QMainWindow, QDialog {{ background: {BG}; }}

/* ---------- 通用文字 ---------- */
QWidget {{
    color: {TEXT};
    font-family: {FONT_STACK};
}}
QLabel#titleLabel {{ font-size: 22px; font-weight: 600; }}
QLabel#subtitleLabel {{ font-size: 13px; color: {TEXT_SECOND}; }}
QLabel#sectionLabel {{ font-size: 15px; font-weight: 600; }}
QLabel#hintLabel {{ font-size: 12px; color: {TEXT_SECOND}; }}

/* ---------- 圆角卡片 ---------- */
QFrame#card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 13px;
}}
QPushButton:hover {{ background: #FAFAFC; }}
QPushButton:pressed {{ background: #F0F0F2; }}
QPushButton:disabled {{ color: {TEXT_SECOND}; background: {BG}; }}

QPushButton#primaryButton {{
    background: {ACCENT};
    border: none;
    color: white;
    font-size: 15px;
    font-weight: 600;
    padding: 11px 20px;
    border-radius: 10px;
}}
QPushButton#primaryButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primaryButton:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#primaryButton:disabled {{ background: #B7D6FF; color: white; }}

QPushButton#dangerButton {{ color: {DANGER}; }}

/* ---------- 输入控件 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1.5px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: white;
    outline: none;
}}

/* ---------- 表格 ---------- */
QTableWidget {{
    background: {CARD};
    border: none;
    gridline-color: transparent;
    selection-background-color: #E8F1FF;
    selection-color: {TEXT};
    alternate-background-color: #FAFAFC;
}}
QTableWidget::item {{ padding: 6px 10px; border-bottom: 1px solid {BORDER}; }}
QHeaderView::section {{
    background: {CARD};
    color: {TEXT_SECOND};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {CARD}; border: none; }}

/* ---------- 滚动条（细窄悬浮风） ---------- */
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #C7C7CC; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #AEAEB2; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: #C7C7CC; border-radius: 4px; min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: #E8E8ED;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_SECOND};
    padding: 8px 18px;
    font-size: 13px;
    border: none;
}}
QTabBar::tab:selected {{ color: {TEXT}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QPlainTextEdit, QTextEdit {{
    background: {CARD};
    border: none;
    font-size: 13px;
    selection-background-color: #CCE4FF;
}}

QStatusBar {{ background: transparent; color: {TEXT_SECOND}; font-size: 12px; }}
QToolTip {{
    background: {TEXT}; color: white; border: none;
    border-radius: 6px; padding: 6px 10px; font-size: 12px;
}}
"""


def apply_theme(app):
    """设置应用字体与全局样式。"""
    # 取字体栈的第一项作为应用默认字体
    default_family = FONT_STACK.split(",")[0].strip().strip('"')
    font = QFont(default_family, 13)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setStyleSheet(QSS)
