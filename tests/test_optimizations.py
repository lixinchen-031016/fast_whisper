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
            # 与真实 requests 一致：HTTPError.response 指向响应本身，
            # 下载重试逻辑据此区分 4xx（不重试）与 5xx/429（重试）
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

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
                # 416 也按 lie 声明大小，便于模拟“服务器始终谎报长度”的场景
                return _FakeResp(416, {"content-range": f"bytes */{declared}"})
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
    """服务器始终谎报长度（200 与 416 都报 5000，实际只有 100）→ 重试后仍报错。"""
    repo = "repo/model"
    files = ["m.bin"]
    url = _url(repo, "m.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: files)
    # 声明 5000 字节，实际只返回 100，触发完成后校验失败
    factory, _rec = _factory({url: b"z" * 100}, lie={url: 5000})
    monkeypatch.setattr(model_manager.requests, "Session", factory)
    monkeypatch.setattr(model_manager.time, "sleep", lambda s: None)   # 跳过退避等待

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


# ==================== 第三轮：MLX 绕开系统 ffmpeg ====================
def _write_wav(path, seconds=0.5, rate=16000, freq=8.0):
    """生成一段可被 PyAV 解码的单声道 16k WAV（不依赖任何外部可执行文件）。"""
    import math
    import struct
    import wave

    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(i / freq))) for i in range(n)))
    return str(path)


def test_decode_audio_pyav(tmp_path):
    """内置 PyAV 解码：单声道 float32、16k 采样、幅值归一化到 [-1,1]。"""
    import numpy as np

    wav = _write_wav(tmp_path / "a.wav", seconds=0.5, rate=16000)
    audio = workers.decode_audio_pyav(wav)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert abs(audio.shape[0] - 8000) <= 2          # 0.5s * 16000
    assert float(abs(audio).max()) <= 1.0


def test_mlx_path_passes_waveform_not_path(tmp_path, monkeypatch):
    """MLX 引擎应把 PyAV 解码出的波形（ndarray）传给 mlx_whisper，而非文件路径。

    这正是修复 `No such file or directory: 'ffmpeg'` 的关键——
    mlx_whisper 收到字符串时会 subprocess 调用系统 ffmpeg。
    """
    import sys
    import types

    import numpy as np

    wav = _write_wav(tmp_path / "b.wav")

    captured = {}

    def _fake_transcribe(audio, path_or_hf_repo=None, language=None, task=None,
                         verbose=None, **kw):
        captured["audio"] = audio
        return {"segments": [{"start": 0.0, "end": 1.0, "text": " 你好 "}],
                "language": "zh"}

    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = _fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)

    segs, info = workers.transcribe_media(wav, "/model", engine="mlx", language="zh")
    assert [s["text"] for s in segs] == ["你好"]
    assert info["engine"] == "mlx"
    assert isinstance(captured["audio"], np.ndarray)   # 传的是波形，不是路径


def test_mlx_path_falls_back_to_path_when_decoder_unavailable(tmp_path, monkeypatch):
    """内置解码器不可用（缺 PyAV）时回退为传路径，保证不回归。"""
    import sys
    import types

    captured = {}

    def _fake_transcribe(audio, **kw):
        captured["audio"] = audio
        return {"segments": [], "language": "?"}

    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = _fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)

    def _no_decoder(*a, **k):
        raise ImportError("no module named av")

    monkeypatch.setattr(workers, "decode_audio_pyav", _no_decoder)
    workers.transcribe_media("/tmp/whatever.mp4", "/model", engine="mlx")
    assert captured["audio"] == "/tmp/whatever.mp4"    # 回退为路径


def test_mlx_path_surfaces_decode_error_for_invalid_media(tmp_path, monkeypatch):
    """真正的解码错误（文件损坏/格式不支持）应直接抛出，而非误判为『无音频流』。

    回归保护：PyAV 的 InvalidDataError 继承自 ValueError，不能用 ValueError 兜底。
    """
    import sys
    import types

    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda *a, **k: {"segments": [], "language": "?"}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"definitely not media")
    with pytest.raises(Exception) as ei:
        workers.transcribe_media(str(bad), "/model", engine="mlx")
    assert not isinstance(ei.value, workers.NoAudioStreamError)


def test_friendly_error_ffmpeg_missing():
    """ffmpeg 缺失要给出准确提示，而非误导性的『路径不存在』。"""
    err = FileNotFoundError("[Errno 2] No such file or directory: 'ffmpeg'")
    msg = workers._friendly_media_error(err)
    assert "ffmpeg" in msg
    assert "解码器不可用" in msg
    assert "路径不存在" not in msg


def test_friendly_error_no_audio_stream():
    msg = workers._friendly_media_error(ValueError("媒体文件中没有可用的音频流"))
    assert "音频轨道" in msg


# ==================== 第四轮：线程保活 / 下载重试 / 批量进度 ====================
def test_retain_worker_keeps_running_thread_alive(qapp, monkeypatch):
    """QThread 运行期间必须被保活，否则对象被回收时 Qt 会 qFatal 崩溃。

    回归场景：搜索框里发起搜索后立刻关闭对话框——原先 SearchWorker 只被对话框
    属性引用，对话框销毁即丢掉最后一个引用，而线程可能还在跑（镜像站慢时有 15s
    超时 + 重试）。保活后线程对象在结束前一直有强引用。
    """
    import threading as _threading

    from app import widgets

    class _BlockingWorker(workers.RetainedThread):
        def __init__(self):
            super().__init__()
            self.gate = _threading.Event()

        def run(self):
            self.gate.wait(5)          # 停在 run() 里，状态可确定地断言

    w = _BlockingWorker()
    w.start()
    assert w in workers._LIVE_WORKERS              # 启动即登记，运行期间不会被回收
    w.gate.set()
    w.wait(3000)
    qapp.processEvents()                           # 释放回调经事件队列派发
    assert w not in workers._LIVE_WORKERS          # 结束后自动解除，不会长期滞留

    monkeypatch.setattr(model_manager, "search_models", lambda *a, **k: [])
    s = widgets.SearchWorker("q", "cpu")
    s.start()
    s.wait(3000)
    qapp.processEvents()
    assert s not in workers._LIVE_WORKERS

    # RetainedThread 判定：所有工作线程都继承保活基类
    for cls in (workers.DownloadWorker, workers.TranscribeWorker,
                workers.BatchTranscribeWorker, widgets.SearchWorker):
        assert issubclass(cls, workers.RetainedThread), cls


def test_on_file_chosen_keeps_start_disabled_while_busy(qapp, isolated_settings, tmp_path):
    """转写进行中拖入新文件不得重新启用「开始转写」（否则会并发启动第二个任务）。

    回归：原实现直接 start_btn.setEnabled(bool(currentData()))，绕过了忙碌判定。
    """
    from app.main_window import MainWindow

    media = tmp_path / "a.mp3"
    media.write_bytes(b"x")

    class _Running:
        def isRunning(self):
            return True

    w = MainWindow()
    w.worker = _Running()          # 模拟正在转写的线程
    w._on_file_chosen(str(media))
    assert w.start_btn.isEnabled() is False
    assert w.batch_btn.isEnabled() is False

    w.worker = None                # 任务结束 → 恢复可用
    w._on_file_chosen(str(media))
    assert w.start_btn.isEnabled() is (bool(w.model_combo.currentData()))


def test_resolve_model_path(isolated_settings, tmp_path):
    """内置仓库 id 映射到模型目录；已是目录则原样返回（预检与扫描共用）。"""
    isolated_settings.set_models_dir(str(tmp_path))
    p = workers.resolve_model_path("Systran/faster-whisper-tiny", "cpu")
    assert p == os.path.join(str(tmp_path), "Systran__faster-whisper-tiny")

    real = tmp_path / "MyModel"
    real.mkdir()
    assert workers.resolve_model_path(str(real), "cpu") == str(real)
    # 未知取值原样返回，不做猜测
    assert workers.resolve_model_path("/tmp/nope", "cpu") == "/tmp/nope"


def test_preflight_accepts_manually_placed_builtin_model(qapp, isolated_settings, tmp_path):
    """内置推荐模型只要本地目录已有完整权重，就应该可直接转写（不再要求重新下载）。"""
    from app.main_window import MainWindow

    isolated_settings.set_models_dir(str(tmp_path))
    d = tmp_path / "Systran__faster-whisper-tiny"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x")

    w = MainWindow()
    # 显式固定 CPU 引擎：默认“自动识别”在 Apple Silicon 上会选中 Metal，
    # 而本用例要验证的是 CTranslate2 内置模型（必须与引擎 kind 一致）
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("cpu"))
    assert w._current_engine() == "cpu"
    # 模拟“下载刚完成、列表还没刷新”的瞬间：下拉项仍是内置仓库 id，
    # 但本地目录里已经有完整权重 → 预检应解析为本地目录并放行
    w.model_combo.addItem("○ Systran/faster-whisper-tiny（未下载）",
                          "Systran/faster-whisper-tiny")
    w.model_combo.setCurrentIndex(w.model_combo.count() - 1)

    pf = w._preflight()
    assert pf is not None, "内置模型已有本地权重时应通过预检"
    model_path, engine = pf
    assert model_path == str(d) and engine == "cpu"


def test_preflight_reports_missing_builtin_model(qapp, isolated_settings, tmp_path, monkeypatch):
    """未下载的内置模型仍然要给出「需要先下载模型」的提示（返回 None）。"""
    from app.main_window import MainWindow

    isolated_settings.set_models_dir(str(tmp_path))
    w = MainWindow()
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("cpu"))
    idx = w.model_combo.findData("Systran/faster-whisper-tiny")
    assert idx >= 0, "CPU 引擎下应列出 CTranslate2 内置推荐模型"
    w.model_combo.setCurrentIndex(idx)

    shown = []
    monkeypatch.setattr("app.main_window.QMessageBox.information",
                        lambda *a, **k: shown.append(a))
    assert w._preflight() is None
    assert shown


def test_download_retries_after_dropped_stream(isolated_settings, tmp_path, monkeypatch):
    """第一次下载流到一半断连 → 自动重试并续传，最终文件完整。"""
    repo = "repo/model"
    full = b"abcdefghij" * 500
    url = _url(repo, "model.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: ["model.bin"])
    monkeypatch.setattr(model_manager.time, "sleep", lambda s: None)

    record = []
    state = {"dropped": False}

    class _DroppingResp(_FakeResp):
        def iter_content(self, chunk_size=1):
            half = len(self._body) // 2
            yield self._body[:half]                 # 只写一半就断
            raise requests.ConnectionError("connection reset by peer")

    class _FlakySession(_FakeSession):
        def get(self, url, headers=None, **kw):
            # 注意：每次重试都会新建 Session，抖动状态必须放在 Session 之外
            if not state["dropped"]:
                state["dropped"] = True
                record.append(("drop", url))
                body = self.contents.get(url, b"")
                return _DroppingResp(200, {"content-length": str(len(body))}, body)
            return super().get(url, headers=headers, **kw)

    monkeypatch.setattr(model_manager.requests, "Session",
                        lambda: _FlakySession({url: full}, record=record))

    dest = model_manager.download_model(repo)
    assert open(os.path.join(dest, "model.bin"), "rb").read() == full
    assert ("drop", url) in record and ("206", url) in record   # 断连后走了续传


def test_download_does_not_retry_client_error(isolated_settings, tmp_path, monkeypatch):
    """404 之类的客户端错误不重试（避免把确定性失败放大成 3 倍等待）。"""
    repo = "repo/model"
    url = _url(repo, "model.bin")
    isolated_settings.set_models_dir(str(tmp_path))
    monkeypatch.setattr(model_manager, "get_model_files", lambda r: ["model.bin"])
    monkeypatch.setattr(model_manager.time, "sleep", lambda s: None)

    calls = []

    class _Session404(_FakeSession):
        def get(self, url, headers=None, **kw):
            calls.append(url)
            return _FakeResp(404, {})

    monkeypatch.setattr(model_manager.requests, "Session", lambda: _Session404({url: b""}))
    with pytest.raises(requests.HTTPError):
        model_manager.download_model(repo)
    assert len(calls) == 1


def test_is_retryable_matrix():
    assert model_manager._is_retryable(requests.ConnectionError("x")) is True
    assert model_manager._is_retryable(IOError("size mismatch")) is True
    assert model_manager._is_retryable(requests.HTTPError(response=_FakeResp(503, {}))) is True
    assert model_manager._is_retryable(requests.HTTPError(response=_FakeResp(429, {}))) is True
    assert model_manager._is_retryable(requests.HTTPError(response=_FakeResp(404, {}))) is False
    assert model_manager._is_retryable(ValueError("nope")) is False


def test_batch_worker_reports_whole_batch_progress(qapp, monkeypatch, tmp_path):
    """批量进度应为整批百分比（含当前文件内部进度），单调不减直到 100。"""
    from PySide6.QtCore import Qt

    def _fake_transcribe(media, model, engine="cpu", language="auto", task="transcribe",
                         beam_size=5, use_vad=True, batched=False, on_segment=None,
                         on_status=None, on_progress=None, cancel_check=None):
        for pct in (25, 50, 100):
            if on_progress:
                on_progress(pct)
        if on_segment:
            on_segment(0.0, 1.0, "文字")
        return ([{"start": 0.0, "end": 1.0, "text": "文字"}], {"duration": 1.0})

    monkeypatch.setattr(workers, "transcribe_media", _fake_transcribe)

    files = [str(tmp_path / f"f{i}.mp3") for i in range(2)]
    w = workers.BatchTranscribeWorker(files, "model", str(tmp_path / "out"))
    seen = []
    w.progress_pct.connect(seen.append, Qt.ConnectionType.DirectConnection)
    done = []
    w.finished_ok.connect(lambda a, b: done.append((a, b)), Qt.ConnectionType.DirectConnection)
    w.start()
    w.wait(5000)

    assert seen and seen == sorted(seen)        # 单调不减
    assert seen[-1] == 100                      # 结束时报满
    assert done == [(2, 2)]
    # 第 1 个文件 50% → 整批 25%；两个文件各占 50%
    assert 25 in seen


def test_cuda_failure_is_remembered(monkeypatch):
    """CUDA 加载失败后应记住不可用，批量转写不再逐文件重试 CUDA 初始化。"""
    from app import hardware

    hardware._cache.pop("cuda", None)
    try:
        class _CT2:
            @staticmethod
            def get_cuda_device_count():
                return 1                 # 探测说有显卡

        monkeypatch.setitem(__import__("sys").modules, "ctranslate2", _CT2)
        assert hardware.cuda_available() is True
        hardware.mark_cuda_unusable()
        assert hardware.cuda_available() is False    # 失败已被记住
        assert hardware.detect_best() in ("cpu", "mlx")
    finally:
        hardware._cache.pop("cuda", None)


def test_fw_cuda_load_failure_falls_back_and_memoizes(monkeypatch):
    """CUDA 模型加载失败 → 回退 CPU，并把 CUDA 记为不可用。"""
    import sys
    import types

    from app import hardware

    hardware._cache.pop("cuda", None)
    workers.clear_model_cache()
    attempts = []

    class _Model:
        def __init__(self, path, device=None, compute_type=None, **kw):
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("cublas64_12.dll not found")

        def transcribe(self, media, **kw):
            return ([_FakeSeg(0.0, 1.0, " 你好 ")], _FakeInfo())

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    status = []
    segs, info = workers.transcribe_media("/a.mp3", "/model", engine="cuda",
                                          language="zh", on_status=status.append)
    assert attempts == ["cuda", "cpu"]          # 先试 CUDA，失败回退 CPU
    assert info["engine"] == "cuda"             # 展示上仍是用户所选引擎
    assert [s["text"] for s in segs] == ["你好"]
    assert any("回退 CPU" in m for m in status)
    assert hardware._cache.get("cuda") is False  # 已记住，后续不再重试 CUDA
    workers.clear_model_cache()
    hardware._cache.pop("cuda", None)


def test_batched_mode_passes_prompt_and_keeps_vad(monkeypatch):
    """批量推理必须同样带上语种提示词，并强制 VAD 切块（否则长音频直接报错）。"""
    import sys
    import types

    workers.clear_model_cache()
    captured = {}

    class _Model:
        def __init__(self, path, device=None, compute_type=None, **kw):
            pass

    class _Batched:
        def __init__(self, model=None):
            pass

        def transcribe(self, media, **kw):
            captured.update(kw)
            return ([_FakeSeg(0.0, 1.0, " hi ")], _FakeInfo())

    pkg = types.ModuleType("faster_whisper")
    pkg.WhisperModel = _Model
    tp = types.ModuleType("faster_whisper.transcribe")
    tp.BatchedInferencePipeline = _Batched
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.transcribe", tp)

    workers.transcribe_media("/a.mp3", "/model", engine="cpu", language="en", batched=True)
    assert captured["vad_filter"] is True                     # 切块必需
    assert captured["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert "English" in captured["initial_prompt"]            # 与逐段模式一致
    workers.clear_model_cache()


def test_status_label_updates_are_batched(qapp, isolated_settings):
    """段数文案随表格一起按批刷新，不再每段改一次 QLabel。"""
    from app.main_window import MainWindow

    w = MainWindow()
    w.status_label.setText("就绪")
    for i in range(5):
        w._on_segment(float(i), float(i) + 1, f"t{i}")
    assert w.status_label.text() == "就绪"          # 未达阈值 → 不刷
    w._flush_rows()
    assert "已完成 5 段" in w.status_label.text()


def test_dialog_download_progress_is_aggregated(qapp, isolated_settings):
    """多文件并行下载时，进度条按字节总和计算（单调、不回退），并显示总体量。"""
    from app.widgets import ModelSearchDialog

    dlg = ModelSearchDialog(engine="fw")
    dlg._file_progress = {}
    dlg._on_progress("a.bin", 50, 100)       # 单文件 → 50%
    assert dlg.progress.value() == 50
    assert "总体" not in dlg.status_label.text()

    dlg._on_progress("b.bin", 0, 100)        # 另一个大文件开始 → 总体回落到 25%
    assert dlg.progress.value() == 25
    dlg._on_progress("a.bin", 100, 100)
    dlg._on_progress("b.bin", 100, 100)      # 全部完成 → 100%
    assert dlg.progress.value() == 100
    assert "总体" in dlg.status_label.text()


# ==================== 第四轮：真实转写端到端（需要本地 tiny 模型） ====================
_TINY_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "Systran__faster-whisper-tiny")


@pytest.mark.skipif(not os.path.isdir(_TINY_MODEL), reason="本地未下载 tiny 模型")
def test_main_window_real_transcribe_end_to_end(qapp, isolated_settings, tmp_path):
    """真实跑通「界面 → 工作线程 → 分段落表 → 状态收尾」全链路（无 mock）。

    用仓库自带的 tiny 模型转写一段静音 WAV：不校验识别内容（静音无语音），
    只验证管线能正常结束、忙碌态能复位、取消按钮会收起——这几处正是并发/崩溃类
    缺陷的高发区（例如转写中拖入新文件会重新启用「开始转写」）。
    """
    from PySide6.QtCore import QTimer

    from app.main_window import MainWindow

    isolated_settings.set_models_dir(os.path.dirname(_TINY_MODEL))
    wav = _write_wav(tmp_path / "silence.wav", seconds=2.0)

    w = MainWindow()
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("cpu"))   # 固定 CPU 引擎
    w.refresh_models()
    idx = w.model_combo.findData(_TINY_MODEL)
    assert idx >= 0, "本地 tiny 模型应出现在下拉列表中"
    w.model_combo.setCurrentIndex(idx)

    w._on_file_chosen(wav)
    assert w.start_btn.isEnabled() is True

    w.start_transcribe()
    assert w.start_btn.isEnabled() is False          # 启动后进入忙碌态
    assert w.cancel_btn.isHidden() is False

    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(qapp.quit)
    deadline.start(120000)
    w.worker.finished.connect(qapp.quit)
    qapp.exec()

    assert w.worker.isFinished()
    assert w.progress.isVisible() is False           # 收尾隐藏进度条
    assert w.cancel_btn.isHidden() is True           # 收尾收起取消按钮
    assert "转写完成" in w.status_label.text() or "失败" in w.status_label.text()
    assert w._pending_rows == []                          # 分段缓冲已排空
    assert w.seg_table.rowCount() == len(w.segments)      # 表格与数据一致
