# -*- coding: utf-8 -*-
"""针对本轮优化项的单测：下载（并行/续传/校验）、模型缓存、提示词、批量扫描。

下载测试用 fake requests.Session 模拟 Range/206/416/长度不符，不触网。
"""
import os

import pytest
import requests

from app import model_manager, settings, workers
from app.config import HF_MIRROR


# ==================== fake 下载后端 ====================
class _FakeResp:
    def __init__(self, status_code, headers, body=b""):
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSession:
    """按 URL 返回完整内容，自动模拟 HTTP Range 语义；可选模拟异常场景。"""

    def __init__(self, contents, lie=None, ignore_range=False, record=None):
        self.contents = contents          # {url: bytes}
        self.lie = lie or {}              # {url: 声明的 content-length}（用于模拟长度不符）
        self.ignore_range = ignore_range  # True → 忽略 Range 一律返回 200 全量
        self.record = record if record is not None else []

    def get(self, url, headers=None, stream=False, timeout=None, allow_redirects=True):
        data = self.contents.get(url, b"")
        total = len(data)
        declared = self.lie.get(url, total)
        rng = (headers or {}).get("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= total:
                self.record.append(("416", url))
                return _FakeResp(416, {"content-range": f"bytes */{total}"})
            body = data[start:]
            self.record.append(("206", url))
            return _FakeResp(206, {
                "content-length": str(len(body)),
                "content-range": f"bytes {start}-{total - 1}/{total}",
            }, body)
        self.record.append(("200", url))
        return _FakeResp(200, {"content-length": str(declared)}, data)

    def head(self, *a, **k):
        raise AssertionError("优化后不应再发起 HEAD 请求")

    def close(self):
        pass


def _factory(contents, **kw):
    """返回 (session 工厂, 记录列表)。"""
    record = []
    kw["record"] = record
    return (lambda: _FakeSession(contents, **kw)), record


def _url(repo, name):
    return f"{HF_MIRROR}/{repo}/resolve/main/{name}"


# ==================== 下载：全量 / 并行 ====================
def test_download_full_parallel(isolated_settings, tmp_path, monkeypatch):
    repo = "Systran/faster-whisper-tiny"
    files = ["model.bin", "config.json", "vocabulary.txt"]
    contents = {_url(repo, n): (n.encode() * 5000) for n in files}
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    factory, _rec = _factory(contents)
    monkeypatch.setattr(model_manager.requests, "Session", factory)

    dest = model_manager.download_model(repo)
    for n in files:
        p = os.path.join(dest, n)
        assert os.path.isfile(p)
        assert open(p, "rb").read() == contents[_url(repo, n)]


# ==================== 下载：断点续传（206） ====================
def test_download_resume_206(isolated_settings, tmp_path, monkeypatch):
    repo = "repo/model"
    files = ["model.bin"]
    full = b"abcdefghij" * 1000
    url = _url(repo, "model.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    factory, rec = _factory({url: full})
    monkeypatch.setattr(model_manager.requests, "Session", factory)

    dest_dir = model_manager.ModelInfo(repo).local_dir
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "model.bin"), "wb") as f:
        f.write(full[:4000])          # 预置前半段，触发续传

    dest = model_manager.download_model(repo)
    assert open(os.path.join(dest, "model.bin"), "rb").read() == full
    assert ("206", url) in rec        # 确实走了 206 续传分支


# ==================== 下载：已完成跳过（416） ====================
def test_download_skip_when_complete(isolated_settings, tmp_path, monkeypatch):
    repo = "repo/model"
    files = ["m.bin"]
    full = b"x" * 2048
    url = _url(repo, "m.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    factory, rec = _factory({url: full})
    monkeypatch.setattr(model_manager.requests, "Session", factory)

    dest_dir = model_manager.ModelInfo(repo).local_dir
    os.makedirs(dest_dir, exist_ok=True)
    p = os.path.join(dest_dir, "m.bin")
    with open(p, "wb") as f:
        f.write(full)                 # 已完整下载

    model_manager.download_model(repo)
    assert open(p, "rb").read() == full
    assert ("416", url) in rec


# ==================== 下载：服务器忽略 Range（200 全量）不重复计数 ====================
def test_download_ignore_range_no_overcount(isolated_settings, tmp_path, monkeypatch):
    repo = "repo/model"
    files = ["m.bin"]
    full = b"y" * 3000
    url = _url(repo, "m.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    factory, _rec = _factory({url: full}, ignore_range=True)
    monkeypatch.setattr(model_manager.requests, "Session", factory)

    dest_dir = model_manager.ModelInfo(repo).local_dir
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "m.bin"), "wb") as f:
        f.write(full[:1000])          # 预置残片

    progress = []
    dest = model_manager.download_model(
        repo, progress_cb=lambda n, d, t: progress.append((d, t)))

    assert open(os.path.join(dest, "m.bin"), "rb").read() == full
    # 关键：任何时刻进度都不应超过声明的总大小（修复前会因 +resume_from 溢出）
    assert all(t == 0 or d <= t for d, t in progress)


# ==================== 下载：长度不符 → 报错 ====================
def test_download_size_mismatch_raises(isolated_settings, tmp_path, monkeypatch):
    repo = "repo/model"
    files = ["m.bin"]
    url = _url(repo, "m.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    # 声明 5000 字节，实际只返回 100，触发完成后校验失败
    factory, _rec = _factory({url: b"z" * 100}, lie={url: 5000})
    monkeypatch.setattr(model_manager.requests, "Session", factory)

    with pytest.raises(IOError):
        model_manager.download_model(repo)


def test_content_range_total_parsing():
    assert model_manager._content_range_total("bytes 0-99/100") == 100
    assert model_manager._content_range_total("bytes */1234") == 1234
    assert model_manager._content_range_total(None) is None
    assert model_manager._content_range_total("garbage") is None


# ==================== 模型缓存 ====================
def test_model_cache_hit_and_evict(tmp_path):
    workers.clear_model_cache()
    calls = []

    def loader():
        calls.append(1)
        return object()

    p = str(tmp_path / "m")
    m1, cached1 = workers._load_cached(p, "cpu", "int8", loader)
    assert cached1 is False and len(calls) == 1

    m2, cached2 = workers._load_cached(p, "cpu", "int8", loader)
    assert cached2 is True and m1 is m2 and len(calls) == 1   # 命中，不重复加载

    # 不同精度 → 新键；上限 1 → 淘汰旧键
    m3, cached3 = workers._load_cached(p, "cpu", "float32", loader)
    assert cached3 is False and len(calls) == 2

    _, cached4 = workers._load_cached(p, "cpu", "int8", loader)
    assert cached4 is False and len(calls) == 3   # 旧键已淘汰，需重新加载
    workers.clear_model_cache()


# ==================== 提示词 / 时长 / 扫描 ====================
def test_initial_prompt_by_language():
    assert "普通话" in workers._initial_prompt(None)
    assert "普通话" in workers._initial_prompt("auto")
    assert "English" in workers._initial_prompt("en")
    assert workers._initial_prompt("xx") is None    # 未知语言交由模型判断


def test_probe_duration_missing_file():
    assert workers._probe_duration("/no/such/file.mp3") == 0.0


def test_scan_media_files(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"1")
    (tmp_path / "b.txt").write_bytes(b"1")
    (tmp_path / "c.MP4").write_bytes(b"1")     # 大写扩展名也应识别
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.wav").write_bytes(b"1")   # 子目录不扫描

    found = [os.path.basename(p) for p in workers.scan_media_files(str(tmp_path))]
    assert found == ["a.mp3", "c.MP4"]


# ==================== 批量导出 ====================
def test_batch_worker_export(qapp, tmp_path):
    out_dir = str(tmp_path / "out")
    w = workers.BatchTranscribeWorker([], "model", out_dir)
    segs = [{"start": 0.0, "end": 1.0, "text": "你好"},
            {"start": 1.0, "end": 2.0, "text": "世界"}]
    out = w._export(str(tmp_path / "音频.mp3"), segs)
    assert out.endswith("音频_转写稿.txt")
    assert open(out, encoding="utf-8").read() == "你好\n世界\n"


# ==================== 转写核心：模型缓存复用 + 提示词 ====================
class _FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    duration = 12.5
    language = "zh"
    language_probability = 0.987


class _FakeWhisperModel:
    instances = 0
    init_args = None
    transcribe_kwargs = None

    def __init__(self, path, device=None, compute_type=None, **kw):
        _FakeWhisperModel.instances += 1
        _FakeWhisperModel.init_args = (path, device, compute_type)

    def transcribe(self, media, **kw):
        _FakeWhisperModel.transcribe_kwargs = kw
        return ([_FakeSeg(0.0, 1.0, " 你好 "), _FakeSeg(1.0, 2.0, "世界 ")],
                _FakeInfo())


def test_transcribe_core_reuses_model_cache(monkeypatch):
    """同一模型连续转写只构造一次（缓存命中），且按语言传正确 initial_prompt。"""
    import sys
    import types

    workers.clear_model_cache()
    _FakeWhisperModel.instances = 0
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    segs1, info1 = workers.transcribe_media("/a.mp3", "/model", engine="cpu", language="zh")
    assert [s["text"] for s in segs1] == ["你好", "世界"]
    assert info1["language"] == "zh" and info1["engine"] == "cpu"
    assert _FakeWhisperModel.instances == 1
    assert _FakeWhisperModel.transcribe_kwargs["initial_prompt"] == "以下是普通话语音转写。"

    # 第二次同模型同精度 → 命中缓存，不再构造新实例
    workers.transcribe_media("/b.mp3", "/model", engine="cpu", language="zh")
    assert _FakeWhisperModel.instances == 1

    # 切英文 → 提示词随语言变化
    workers.transcribe_media("/c.mp3", "/model", engine="cpu", language="en")
    assert "English" in _FakeWhisperModel.transcribe_kwargs["initial_prompt"]
    workers.clear_model_cache()


def test_main_window_has_batch_button(qapp, isolated_settings):
    from app.main_window import MainWindow
    w = MainWindow()
    assert w.batch_btn.isEnabled() in (True, False)   # 存在且可判定状态
    assert "批量" in w.batch_btn.text()


def test_segment_batching_flush(qapp, isolated_settings):
    """分段结果按阈值批量落表，最后一次性补刷，行数与数据一致。"""
    from app.main_window import MainWindow
    w = MainWindow()
    for i in range(95):
        w._on_segment(float(i), float(i) + 1, f"t{i}")
    # 阈值 40：到 40、80 各刷一次，表格行数应不少于 80，其余仍在缓冲
    assert w.seg_table.rowCount() >= 80
    assert w.seg_table.rowCount() < 95
    w._flush_rows()                       # 收尾补刷
    assert w.seg_table.rowCount() == 95
    assert len(w.segments) == 95
    assert w.seg_table.item(94, 2).text() == "t94"


# ==================== 第二轮：已下载判定 / 重试 / 导出清洗 ====================
def test_model_info_is_downloaded_supports_mlx(isolated_settings, tmp_path):
    """is_downloaded 必须同时识别 CTranslate2(model.bin) 与 MLX(safetensors)。"""
    isolated_settings.set_models_dir(str(tmp_path))
    fw = tmp_path / "Org__fw-model"; fw.mkdir(); (fw / "model.bin").write_bytes(b"x")
    mlx = tmp_path / "Org__mlx-model"; mlx.mkdir()
    (mlx / "model.safetensors").write_bytes(b"x")
    empty = tmp_path / "Org__empty"; empty.mkdir()

    assert model_manager.ModelInfo("Org/fw-model").is_downloaded is True
    assert model_manager.ModelInfo("Org/mlx-model").is_downloaded is True
    assert model_manager.ModelInfo("Org/empty").is_downloaded is False


def test_get_retries_then_success(monkeypatch):
    state = {"n": 0}

    class _Ok:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None, **kw):
        state["n"] += 1
        if state["n"] < 3:
            raise requests.ConnectionError("boom")
        return _Ok()

    monkeypatch.setattr(model_manager.requests, "get", fake_get)
    monkeypatch.setattr(model_manager.time, "sleep", lambda s: None)
    model_manager._get("http://example/x")
    assert state["n"] == 3        # 前两次失败、第三次成功


def test_get_exhausts_retries(monkeypatch):
    def fake_get(url, timeout=None, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(model_manager.requests, "get", fake_get)
    monkeypatch.setattr(model_manager.time, "sleep", lambda s: None)
    with pytest.raises(requests.RequestException):
        model_manager._get("http://example/x", retries=2)


def test_exporter_drops_empty_segments(tmp_path):
    from app import exporter
    segs = [{"start": 0, "end": 1, "text": "你好"},
            {"start": 1, "end": 2, "text": "   "},      # 空段应被丢弃
            {"start": 2, "end": 3, "text": " 世界 "}]
    p = str(tmp_path / "o.txt")
    exporter.export_txt(segs, p)
    assert open(p, encoding="utf-8").read() == "你好\n世界\n"

    ps = str(tmp_path / "o.srt")
    exporter.export_srt(segs, ps)
    srt = open(ps, encoding="utf-8").read()
    assert srt.startswith("1\n")
    assert srt.count("-->") == 2        # 空段不产生字幕条目


# ==================== 第二轮：界面偏好持久化 ====================
def test_ui_pref_roundtrip(isolated_settings):
    isolated_settings.set_ui("engine", "cuda")
    assert isolated_settings.get_ui("engine") == "cuda"
    isolated_settings.set_ui("vad", None)               # None → 空串
    assert isolated_settings.get_ui("vad", "1") == "1"  # 回退默认


def test_main_window_prefs_persist(qapp, isolated_settings):
    from app.main_window import MainWindow
    w = MainWindow()
    w.lang_combo.setCurrentIndex(1)      # 中文
    w.beam_combo.setCurrentIndex(0)      # beam=1
    w.vad_check.setChecked(False)
    w._save_prefs()

    w2 = MainWindow()                    # 重新构造 → 应恢复上次选择
    assert w2.lang_combo.currentData() == w.lang_combo.currentData()
    assert w2.beam_combo.currentData() == 1
    assert w2.vad_check.isChecked() is False


def test_cancel_button_visibility_toggles(qapp, isolated_settings):
    from app.main_window import MainWindow
    w = MainWindow()
    assert w.cancel_btn.isHidden() is True
    w._set_busy(True)
    assert w.cancel_btn.isHidden() is False
    assert w.start_btn.isEnabled() is False
    w._set_busy(False)
    assert w.cancel_btn.isHidden() is True


# ==================== 第二轮：转写进度回调 ====================
def test_transcribe_reports_progress(monkeypatch):
    import sys
    import types

    workers.clear_model_cache()
    _FakeWhisperModel.instances = 0
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    prog = []
    workers.transcribe_media("/a.mp3", "/m", engine="cpu", language="zh",
                             on_progress=prog.append)
    # info.duration=12.5，段落 end=1.0/2.0 → 8% / 16%
    assert prog == [8, 16]
    workers.clear_model_cache()


# ==================== 第二轮：搜索线程成功/失败 ====================
def test_search_worker_success_and_failure(qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from app import widgets

    monkeypatch.setattr(model_manager, "search_models", lambda *a, **k: ["m1", "m2"])
    ok = []
    w = widgets.SearchWorker("q", "cpu")
    w.finished_ok.connect(ok.append, Qt.ConnectionType.DirectConnection)
    w.start(); w.wait(2000)
    assert ok == [["m1", "m2"]]

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_manager, "search_models", _boom)
    err = []
    w2 = widgets.SearchWorker("q", "cpu")
    w2.failed.connect(err.append, Qt.ConnectionType.DirectConnection)
    w2.start(); w2.wait(2000)
    assert err and "boom" in err[0]
