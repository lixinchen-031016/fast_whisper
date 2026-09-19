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


def _setup_frozen_env():
    """打包态（onefile）：把解压目录加入 DLL 搜索路径。

    Windows 内嵌的 CUDA 运行库（cublas/cudnn DLL）随 exe 解压到 _MEIPASS，
    CTranslate2 的 C++ 层按名称加载它们，必须让该目录处于搜索路径中。
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            os.environ["PATH"] = meipass + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(meipass)
            except (AttributeError, OSError):
                pass  # 非 Windows 系统无 add_dll_directory，忽略


def main():
    # 高分屏支持（Windows 缩放 / macOS Retina）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    _setup_frozen_env()
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
