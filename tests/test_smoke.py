# -*- coding: utf-8 -*-
"""跨平台冒烟测试：路径与设置持久化、硬件探测、引擎过滤、界面构造。

运行：pytest -q（需 QT_QPA_PLATFORM=offscreen，conftest 已自动设置）
"""
import os

from app import settings
from app.config import ENGINES, MODELS_DIR
from app.model_manager import filter_by_engine
from app.workers import list_local_models, local_model_kind


# ---------- 设置持久化 ----------
def test_settings_roundtrip(isolated_settings, tmp_path):
    ini = os.path.join(str(tmp_path), "settings.ini")
    models_dir = os.path.join(str(tmp_path), "my_models")
    export_dir = os.path.join(str(tmp_path), "exports")

    settings.set_models_dir(models_dir)
    settings.set_export_dir(export_dir)

    # 重新加载同一 ini 文件 → 持久化生效
    settings.set_settings_file(ini)
    assert settings.get_models_dir() == os.path.normpath(models_dir)
    assert settings.get_export_dir() == os.path.normpath(export_dir)

    # 清空导出目录 → 回退“跟随媒体文件”
    settings.set_export_dir("")
    assert settings.get_export_dir() == ""

    # 清空模型目录 → 回退默认值（非空且指向有效位置）
    settings.set_models_dir("")
    d = settings.get_models_dir()
    assert d and os.path.normpath(d) == os.path.normpath(settings.default_models_dir())


def test_default_models_dir_platform():
    """开发态默认模型目录 = 项目内 models/；两个分隔符写法均应正常。"""
    d = settings.default_models_dir()
    assert os.path.isabs(d)
    if not getattr(__import__("sys"), "frozen", False):
        assert os.path.normpath(d) == os.path.normpath(MODELS_DIR)


# ---------- 硬件探测与引擎过滤 ----------
def test_engine_registry_complete():
    for key in ("cpu", "cuda", "mlx"):
        spec = ENGINES[key]
        assert spec["kind"] in ("fw", "mlx")
        assert spec["builtin"], f"{key} 缺少内置模型列表"


def test_filter_by_engine_matrix():
    assert filter_by_engine("Systran/faster-whisper-medium", "cpu")
    assert filter_by_engine("Systran/faster-whisper-medium", "cuda")
    assert not filter_by_engine("Systran/faster-whisper-medium", "mlx")
    assert filter_by_engine("mlx-community/whisper-tiny", "mlx")
    assert not filter_by_engine("mlx-community/whisper-tiny", "cpu")
    # openai/whisper 原始权重不能被 CPU/CUDA 引擎加载，必须过滤
    assert not filter_by_engine("openai/whisper-large-v3", "cpu")
    assert not filter_by_engine("openai/whisper-large-v3", "cuda")


def test_model_kind_detection(tmp_path):
    # CTranslate2 格式：model.bin
    fw_dir = tmp_path / "m1"
    fw_dir.mkdir()
    (fw_dir / "model.bin").write_bytes(b"x")
    assert local_model_kind(str(fw_dir)) == "fw"
    # MLX 格式：safetensors
    mlx_dir = tmp_path / "m2"
    mlx_dir.mkdir()
    (mlx_dir / "model.safetensors").write_bytes(b"x")
    assert local_model_kind(str(mlx_dir)) == "mlx"
    # 空目录：无法判定
    empty = tmp_path / "m3"
    empty.mkdir()
    assert local_model_kind(str(empty)) is None
    # 扫描过滤
    assert list_local_models(str(tmp_path), kind="fw") == [str(fw_dir)]
    assert list_local_models(str(tmp_path), kind="mlx") == [str(mlx_dir)]


# ---------- 界面构造（offscreen） ----------
def test_main_window_construct(qapp, isolated_settings):
    from app.main_window import MainWindow
    w = MainWindow()
    assert w.model_combo.count() > 0
    assert w.engine_combo.count() == 4   # 自动 + cpu + cuda + mlx（含不可用占位）
    # 自动模式必须解析出具体引擎
    assert w._current_engine() in ("cpu", "cuda", "mlx")


def test_settings_dialog_construct(qapp, isolated_settings):
    from app.widgets import SettingsDialog
    dlg = SettingsDialog()
    assert dlg.models_edit.text()
    dlg._save()
    assert settings.get_models_dir() == os.path.normpath(dlg.models_edit.text())


def test_model_search_dialog_construct(qapp, isolated_settings):
    from app.widgets import ModelSearchDialog
    for engine, arch in (("fw", "cpu"), ("mlx", "mlx")):
        dlg = ModelSearchDialog(engine=engine)
        assert dlg.arch_combo.currentData() == arch