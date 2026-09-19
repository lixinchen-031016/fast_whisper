#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastWhisper 入口：初始化应用、应用主题、启动主窗口。

用法：
    .venv/bin/python main.py
"""
import os
import sys

from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.theme import apply_theme
from app.main_window import MainWindow


def _ensure_dirs():
    """启动时确保默认输出目录存在（模型目录在下载/扫描时按需创建）。"""
    from app import settings
    try:
        os.makedirs(settings.get_models_dir(), exist_ok=True)
    except OSError:
        pass  # 目录不可创建时由具体功能再行报错，不阻断启动


def main():
    # 高分屏支持（Windows 缩放 / macOS Retina）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("FastWhisper")
    app.setApplicationDisplayName("FastWhisper 语音转文字")

    apply_theme(app)
    _ensure_dirs()

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
