#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastWhisper 入口：初始化应用、应用主题、启动主窗口。

用法：
    .venv/bin/python main.py
"""
import sys

from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.theme import apply_theme
from app.main_window import MainWindow


def main():
    # 高分屏支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("FastWhisper")
    app.setApplicationDisplayName("FastWhisper 语音转文字")

    apply_theme(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
