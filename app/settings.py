#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户偏好设置：导出路径、模型存放位置等，跨平台持久化。

存储后端使用 Qt QSettings（QSettings.IniFormat 统一用 ini 文件）：
- macOS:   ~/.config/FastWhisper/FastWhisper.ini
- Windows: %APPDATA%\\FastWhisper\\FastWhisper.ini
统一用 ini 而非原生后端（macOS plist / Windows 注册表），便于用户查看、
备份与在测试中隔离（指向临时文件即可）。
"""
import os
import sys

from PySide6.QtCore import QSettings

from .config import APP_ROOT

_ORG = "FastWhisper"
_APP = "FastWhisper"

# 模块级单例；测试可通过 set_settings_file() 指向临时文件隔离
_settings: QSettings = None


def _qsettings() -> QSettings:
    global _settings
    if _settings is None:
        _settings = QSettings(QSettings.IniFormat, QSettings.UserScope, _ORG, _APP)
    return _settings


def set_settings_file(path):
    """重定向设置存储到指定 ini 文件（测试/便携模式用）；传 None 恢复默认后端。"""
    global _settings
    _settings = QSettings(path, QSettings.IniFormat) if path else None


def default_models_dir() -> str:
    """默认模型目录：开发态用项目内 models/；打包态 exe/app 目录可能只读
    （如 Program Files / /Applications），改用用户主目录。"""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), "FastWhisper", "models")
    return os.path.join(APP_ROOT, "models")


def get_models_dir() -> str:
    """模型存放目录（自定义优先，未设置用默认值）。"""
    v = str(_qsettings().value("paths/models_dir", "") or "").strip()
    return v or default_models_dir()


def set_models_dir(path: str):
    if path:
        _qsettings().setValue("paths/models_dir", os.path.normpath(path))
    else:
        _qsettings().remove("paths/models_dir")
    _qsettings().sync()


def get_export_dir() -> str:
    """默认导出目录；空字符串表示“跟随媒体文件所在目录”。"""
    return str(_qsettings().value("paths/export_dir", "") or "").strip()


def set_export_dir(path: str):
    if path:
        _qsettings().setValue("paths/export_dir", os.path.normpath(path))
    else:
        _qsettings().remove("paths/export_dir")
    _qsettings().sync()
