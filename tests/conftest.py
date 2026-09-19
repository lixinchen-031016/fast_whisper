# -*- coding: utf-8 -*-
import os
import tempfile

import pytest

# 界面测试全程离屏，CI 的 Ubuntu 容器无显示器也可运行
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from app.theme import apply_theme
    apply_theme(app)
    return app


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    """把用户设置重定向到临时 ini 文件，避免污染真实用户配置。"""
    from app import settings
    ini = os.path.join(str(tmp_path), "settings.ini")
    settings.set_settings_file(ini)
    yield settings
    settings.set_settings_file(None)  # 恢复默认后端
