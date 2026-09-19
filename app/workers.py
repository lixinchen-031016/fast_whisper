#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台工作线程：模型下载与语音转写，均通过信号与界面通信。

转写核心（transcribe_media）与 Qt 线程解耦：单文件（TranscribeWorker）与
批量（BatchTranscribeWorker）复用同一实现，也便于脱离 GUI 做单元测试。
"""
import os
import sys
import threading
import traceback

from PySide6.QtCore import QThread, Signal

from . import model_manager
from .model_manager import CancelledError


# ==================== 模型缓存 ====================
# 连续转写（同模型多文件 / 批量 / 重试）时复用已加载模型，避免每次都重新
# 从磁盘加载权重（CPU 上大模型加载可达 10s+）。
_MODEL_CACHE = {}          # key -> model
_MODEL_CACHE_ORDER = []    # 插入顺序，用于淘汰
_MODEL_CACHE_LIMIT = 1     # 上限 1：只保留最近使用的模型，控制显存/内存占用
_MODEL_CACHE_LOCK = threading.Lock()


def _load_cached(model_path: str, device: str, compute_type: str, loader):
    """按 (路径, 设备, 精度) 缓存模型实例。

    命中返回 (model, True)；未命中则调用 loader() 加载并缓存，返回 (model, False)。
    超出上限时淘汰最旧的模型（释放其内存/显存）。
    """
    key = (os.path.abspath(model_path), device, compute_type)
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key], True
    model = loader()   # 加载放在锁外，避免长时间阻塞其它线程
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = model
            _MODEL_CACHE_ORDER.append(key)
            while len(_MODEL_CACHE_ORDER) > _MODEL_CACHE_LIMIT:
                old = _MODEL_CACHE_ORDER.pop(0)
                if old != key:
                    _MODEL_CACHE.pop(old, None)
    return model, False


def clear_model_cache():
    """清空模型缓存（切换模型目录/引擎或测试时调用），释放已加载模型。"""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
        _MODEL_CACHE_ORDER.clear()


# ==================== 转写提示词 ====================
# 按语言给出 initial_prompt，避免统一用中文提示对非中文音频产生偏置。
# 未指定语言（自动检测）时沿用中文提示：本项目主要面向中文转写场景。
_INITIAL_PROMPTS = {
    "zh": "以下是普通话语音转写。",
    "en": "The following is an English speech transcription.",
    "ja": "以下は日本語の文字起こしです。",
    "ko": "다음은 한국어 받아쓰기입니다.",
    "de": "Im Folgenden handelt es sich um eine deutsche Transkription.",
    "fr": "Voici une transcription en français.",
    "es": "A continuación se muestra una transcripción en español.",
}


def _initial_prompt(language) -> str:
    """按语言返回转写初始提示词（未知语言返回 None，交由模型自行判断）。"""
    if language is None or language == "auto":
        language = "zh"
    return _INITIAL_PROMPTS.get(language)


def _probe_duration(path: str) -> float:
    """用 PyAV 读取媒体真实时长（秒）；失败返回 0.0。"""
    try:
        import av
        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / av.time_base
            for stream in container.streams:
                if getattr(stream, "duration", None) and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


# ==================== 转写核心（与线程解耦） ====================
def transcribe_media(media_path: str, model_path: str, engine: str = "cpu",
                     language="auto", task: str = "transcribe",
                     beam_size: int = 5, use_vad: bool = True, batched: bool = False,
                     on_segment=None, on_status=None, cancel_check=None):
    """执行一次转写，返回 (segments, info)。

    on_segment(start, end, text) —— 每识别出一段即回调（可为 None）。
    on_status(text)             —— 状态文案回调（可为 None）。
    cancel_check() -> bool      —— 返回 True 时中止并抛出 CancelledError。
    engine: "cpu" / "cuda"（faster-whisper）/ "mlx"（mlx-whisper，仅 macOS）。
    language: None 或 "auto" 表示自动检测。
    """
    lang = None if language in (None, "auto", "") else language
    if engine == "mlx":
        return _transcribe_mlx(media_path, model_path, lang, task,
                               on_segment, on_status, cancel_check)
    return _transcribe_fw(media_path, model_path, engine, lang, task, beam_size,
                          use_vad, batched, on_segment, on_status, cancel_check)


def _transcribe_fw(media_path, model_path, engine, lang, task, beam_size,
                   use_vad, batched, on_segment, on_status, cancel_check):
    """faster-whisper 路径（CPU / NVIDIA CUDA）。"""
    from .config import ENGINES
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(f"缺少 faster-whisper：{e}")

    spec = ENGINES.get(engine, ENGINES["cpu"])
    device, compute = spec["device"], spec["compute_type"]

    def _emit(msg):
        if on_status:
            on_status(msg)

    _emit(f"正在加载模型（{spec['label']}，首次加载需要一些时间）…")
    try:
        model, cached = _load_cached(
            model_path, device, compute,
            lambda: WhisperModel(model_path, device=device, compute_type=compute))
        _emit(f"已复用已加载的模型（{spec['label']}）" if cached
              else f"已使用 {spec['label']} 推理")
    except Exception as e:
        # CUDA 环境不完整（缺 cuBLAS/cuDNN 等）时自动回退 CPU
        if device == "cuda":
            _emit(f"NVIDIA GPU 加载失败（{e}），已自动回退 CPU 推理")
            model, _ = _load_cached(
                model_path, "cpu", "int8",
                lambda: WhisperModel(model_path, device="cpu", compute_type="int8"))
        else:
            raise

    _emit("正在转写（批量模式）…" if batched else "正在转写…")
    if batched:
        # 批量推理：VAD 切块后并行解码（CPU 实测约 1.8x）。
        # 注意其参数集与逐段推理不同，不能混传 condition_on_previous_text 等。
        from faster_whisper.transcribe import BatchedInferencePipeline
        runner = BatchedInferencePipeline(model=model)
        segments, info = runner.transcribe(
            media_path, language=lang, task=task,
            beam_size=beam_size, batch_size=8)
    else:
        segments, info = model.transcribe(
            media_path, language=lang, task=task, beam_size=beam_size,
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,      # 避免重复幻觉
            initial_prompt=_initial_prompt(lang),
        )

    segs = []
    for seg in segments:
        if cancel_check and cancel_check():
            raise CancelledError()
        text = seg.text.strip()
        segs.append({"start": seg.start, "end": seg.end, "text": text})
        if on_segment:
            on_segment(seg.start, seg.end, text)

    info_obj = {
        "duration": info.duration,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "engine": engine,
    }
    return segs, info_obj


def _transcribe_mlx(media_path, model_path, lang, task, on_segment, on_status, cancel_check):
    """mlx-whisper 路径（Apple Metal GPU）。"""
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(f"缺少 mlx-whisper（Metal 引擎）：{e}")

    if on_status:
        on_status("正在加载模型到 Metal GPU …")
    result = mlx_whisper.transcribe(
        media_path, path_or_hf_repo=model_path,
        language=lang, task=task, verbose=False)
    if cancel_check and cancel_check():
        raise CancelledError()

    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])]
    for s in segs:
        if on_segment:
            on_segment(s["start"], s["end"], s["text"])

    # mlx 返回无总时长字段：优先用 PyAV 探测真实时长，失败才退回末段结束时间
    duration = _probe_duration(media_path) or (segs[-1]["end"] if segs else 0)
    info_obj = {
        "duration": duration,
        "language": result.get("language", "?"),
        "language_probability": 0,
        "engine": "mlx",
    }
    return segs, info_obj


# ==================== Qt 线程封装 ====================
class DownloadWorker(QThread):
    """模型下载线程。"""
    # 注意用 float 而非 int：大模型（如 large-v3 约 3GB）字节数会超出
    # Qt 信号 C++ int（4 字节）的表示范围，触发 libshiboken Overflow
    progress = Signal(str, float, float)  # filename, done_bytes, total_bytes
    finished_ok = Signal(str)             # 本地目录
    failed = Signal(str)

    def __init__(self, repo_id: str, parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            path = model_manager.download_model(
                self.repo_id,
                progress_cb=lambda f, d, t: self.progress.emit(f, float(d), float(t)),
                cancel_check=lambda: self._cancelled,
            )
            self.finished_ok.emit(path)
        except CancelledError:
            self.failed.emit("已取消下载")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{e}")


class TranscribeWorker(QThread):
    """单文件转写线程：加载模型 → 推理 → 逐段回报。"""
    model_loading = Signal()
    segment_ready = Signal(float, float, str)   # start, end, text
    progress_text = Signal(str)
    finished_ok = Signal(list, object)          # segments(list[dict]), info
    failed = Signal(str)

    def __init__(self, media_path: str, model_path: str,
                 engine: str = "cpu",
                 language: str = "auto", task: str = "transcribe",
                 beam_size: int = 5, use_vad: bool = True,
                 batched: bool = False, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.model_path = model_path
        self.engine = engine if engine in ("cpu", "cuda", "mlx") else "cpu"
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.use_vad = use_vad
        self.batched = batched
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        self.model_loading.emit()
        try:
            segs, info = transcribe_media(
                self.media_path, self.model_path, engine=self.engine,
                language=self.language, task=self.task, beam_size=self.beam_size,
                use_vad=self.use_vad, batched=self.batched,
                on_segment=lambda s, e, t: self.segment_ready.emit(s, e, t),
                on_status=lambda m: self.progress_text.emit(m),
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                self.progress_text.emit("已取消")
                return
            self.finished_ok.emit(segs, info)
        except CancelledError:
            self.progress_text.emit("已取消")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(_friendly_media_error(e))


class BatchTranscribeWorker(QThread):
    """批量转写线程：顺序处理多个媒体文件，逐个自动导出为 TXT。

    复用模型缓存：全部文件共用一次模型加载，避免逐文件重复加载。
    """
    file_started = Signal(int, int, str)      # index(1-based), total, path
    segment_ready = Signal(float, float, str)
    file_finished = Signal(int, str, str)     # index, path, export_path
    file_failed = Signal(int, str, str)       # index, path, error
    progress_text = Signal(str)
    finished_ok = Signal(int, int)            # done, total
    failed = Signal(str)                      # 致命错误（如缺少依赖）

    def __init__(self, media_files: list, model_path: str, out_dir: str,
                 engine: str = "cpu", language: str = "auto",
                 task: str = "transcribe", beam_size: int = 5,
                 use_vad: bool = True, batched: bool = False, parent=None):
        super().__init__(parent)
        self.media_files = list(media_files)
        self.model_path = model_path
        self.out_dir = out_dir
        self.engine = engine if engine in ("cpu", "cuda", "mlx") else "cpu"
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.use_vad = use_vad
        self.batched = batched
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.media_files)
        done = 0
        for i, path in enumerate(self.media_files, 1):
            if self._cancelled:
                break
            self.file_started.emit(i, total, path)
            try:
                segs, _info = transcribe_media(
                    path, self.model_path, engine=self.engine,
                    language=self.language, task=self.task, beam_size=self.beam_size,
                    use_vad=self.use_vad, batched=self.batched,
                    on_segment=lambda s, e, t: self.segment_ready.emit(s, e, t),
                    on_status=lambda m: self.progress_text.emit(m),
                    cancel_check=lambda: self._cancelled,
                )
                if self._cancelled:
                    break
                out = self._export(path, segs)
                done += 1
                self.file_finished.emit(i, path, out)
            except CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                self.file_failed.emit(i, path, _friendly_media_error(e))
        self.finished_ok.emit(done, total)

    def _export(self, media_path: str, segs: list) -> str:
        from .exporter import export_txt
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(media_path))[0]
        out = os.path.join(self.out_dir, f"{base}_转写稿.txt")
        return export_txt(segs, out)


# ==================== 环境探测与辅助 ====================
def mlx_available() -> bool:
    """当前环境是否可使用 mlx-whisper（Metal 引擎）。"""
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def check_media_decode() -> str:
    """预检媒体解码能力（PyAV 内置 FFmpeg）。

    返回空字符串表示可用；否则返回面向用户的中文错误说明。
    PyAV 的 FFmpeg 库在 import av 时即加载，导入成功即可用。
    """
    try:
        import av  # noqa: F401
        return ""
    except ImportError as e:
        if getattr(sys, "frozen", False):
            return (
                "内置解码器（FFmpeg）初始化失败，视频/音频转写不可用。\n"
                f"详细信息：{e}\n"
                "请重新下载安装包；若反复出现请附本提示反馈。"
            )
        return (
            "缺少 PyAV（内置 FFmpeg 的解码库），视频/音频转写不可用。\n"
            f"详细信息：{e}\n"
            "开发环境请执行：pip install av"
        )
    except Exception as e:
        return (
            "内置解码器（FFmpeg）加载失败，视频/音频转写不可用。\n"
            f"详细信息：{e}\n"
            "安装包可能不完整，请重新下载。"
        )


def _friendly_media_error(e: Exception) -> str:
    """把底层解码/转写异常翻译成用户能看懂的提示。"""
    msg = str(e)
    type_name = type(e).__name__
    # PyAV 打不开文件：路径不存在、容器/编码不支持、文件损坏
    if type_name.startswith("FileNotFoundError") or "No such file" in msg:
        return f"无法打开媒体文件：路径不存在或文件已被移动。\n{msg}"
    if type(e).__module__.startswith("av") or type_name.startswith("InvalidData"):
        return (
            "无法解码该媒体文件：可能是格式不支持或文件损坏。\n"
            f"详细信息：{msg}"
        )
    if "DLL load failed" in msg or "dylib" in msg or "ffmpeg" in msg.lower():
        return (
            "内置 FFmpeg 库加载失败，安装可能不完整。\n"
            f"详细信息：{msg}\n"
            "请重新下载安装包；若反复出现请附本提示反馈。"
        )
    return msg


def local_model_kind(path: str):
    """按目录内文件判断模型属于哪个引擎：
    返回 "fw"（含 model.bin，CTranslate2 格式）/ "mlx"（含 safetensors）/ None。"""
    if os.path.isfile(os.path.join(path, "model.bin")):
        return "fw"
    for name in os.listdir(path) if os.path.isdir(path) else []:
        if name.endswith(".safetensors") or name == "weights.npz":
            return "mlx"
    return None


def list_local_models(models_dir: str, kind: str = None) -> list:
    """扫描本地已下载完成的模型目录；kind 过滤引擎类型（None 不过滤）。"""
    out = []
    if not os.path.isdir(models_dir):
        return out
    for name in sorted(os.listdir(models_dir)):
        d = os.path.join(models_dir, name)
        if not os.path.isdir(d):
            continue
        k = local_model_kind(d)
        if k is None:
            continue
        if kind and k != kind:
            continue
        out.append(d)
    return out


def scan_media_files(folder: str) -> list:
    """扫描文件夹内的可转写媒体文件（不含子目录），按文件名排序。"""
    from .config import MEDIA_EXTS
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in MEDIA_EXTS:
            out.append(p)
    return out
