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


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """离屏测试中一律不弹模态对话框。

    `QMessageBox` 的静态方法内部会走 `QDialog::exec()`，在无头（offscreen）环境
    没有用户可点击，会**永久阻塞**——把一次断言失败伪装成“测试卡死”，极难定位
    （本仓库踩过：预检走到「模型架构不匹配 / 请先选择模型」分支后整轮挂起）。
    需要验证弹窗内容与次数的测试可自行 monkeypatch 覆盖本夹具。
    """
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
