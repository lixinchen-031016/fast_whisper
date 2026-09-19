#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台工作线程：模型下载与语音转写，均通过信号与界面通信。"""
import os
import traceback

from PySide6.QtCore import QThread, Signal

from . import model_manager


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
        except model_manager.CancelledError:
            self.failed.emit("已取消下载")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{e}")


class TranscribeWorker(QThread):
    """转写线程：加载模型 → 推理 → 逐段回报。

    engine: "cpu"  = faster-whisper CPU int8（可开批量推理）
            "cuda" = faster-whisper NVIDIA GPU float16（加载失败自动回退 CPU）
            "mlx"  = mlx-whisper Apple Metal GPU（仅 macOS）
    """
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
        self.language = None if language == "auto" else language
        self.task = task
        self.beam_size = beam_size
        self.use_vad = use_vad
        self.batched = batched
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self.engine == "mlx":
            self._run_mlx()
        else:
            self._run_fw()

    # ---------- faster-whisper（CPU / CUDA） ----------
    def _run_fw(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            self.failed.emit(f"缺少 faster-whisper：{e}")
            return

        from .config import ENGINES
        spec = ENGINES.get(self.engine, ENGINES["cpu"])
        device, compute = spec["device"], spec["compute_type"]

        try:
            self.model_loading.emit()
            self.progress_text.emit(
                f"正在加载模型（{spec['label']}，首次加载需要一些时间）…")
            try:
                model = WhisperModel(self.model_path,
                                     device=device, compute_type=compute)
                self.progress_text.emit(f"已使用 {spec['label']} 推理")
            except Exception as e:
                # CUDA 环境不完整（缺 cuBLAS/cuDNN 等）时自动回退 CPU
                if device == "cuda":
                    self.progress_text.emit(
                        f"NVIDIA GPU 加载失败（{e}），已自动回退 CPU 推理")
                    model = WhisperModel(self.model_path,
                                         device="cpu", compute_type="int8")
                else:
                    raise

            self.progress_text.emit(
                "正在转写（批量模式）…" if self.batched else "正在转写…")
            if self.batched:
                # 批量推理：VAD 切块后并行解码（CPU 实测约 1.8x）。
                # 注意其参数集与逐段推理不同，不能混传 condition_on_previous_text 等。
                from faster_whisper.transcribe import BatchedInferencePipeline
                runner = BatchedInferencePipeline(model=model)
                segments, info = runner.transcribe(
                    self.media_path,
                    language=self.language,
                    task=self.task,
                    beam_size=self.beam_size,
                    batch_size=8,
                )
            else:
                segments, info = model.transcribe(
                    self.media_path,
                    language=self.language,
                    task=self.task,
                    beam_size=self.beam_size,
                    vad_filter=self.use_vad,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=False,   # 避免重复幻觉
                    initial_prompt="以下是普通话语音转写。",
                )

            segs = []
            for seg in segments:
                if self._cancelled:
                    self.progress_text.emit("已取消")
                    return
                item = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                segs.append(item)
                self.segment_ready.emit(seg.start, seg.end, seg.text.strip())

            info_obj = {
                "duration": info.duration,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "engine": self.engine,
            }
            self.finished_ok.emit(segs, info_obj)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{e}")

    # ---------- mlx-whisper（Metal GPU，仅 macOS） ----------
    def _run_mlx(self):
        try:
            import mlx_whisper
        except ImportError as e:
            self.failed.emit(f"缺少 mlx-whisper（Metal 引擎）：{e}")
            return

        try:
            self.model_loading.emit()
            self.progress_text.emit("正在加载模型到 Metal GPU …")
            result = mlx_whisper.transcribe(
                self.media_path,
                path_or_hf_repo=self.model_path,
                language=self.language,
                task=self.task,
                verbose=False,
            )
            if self._cancelled:
                self.progress_text.emit("已取消")
                return

            segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in result.get("segments", [])]
            for s in segs:
                self.segment_ready.emit(s["start"], s["end"], s["text"])

            info_obj = {
                # mlx 返回无总时长字段，用最后一段结束时间近似
                "duration": segs[-1]["end"] if segs else 0,
                "language": result.get("language", "?"),
                "language_probability": 0,
                "engine": "mlx",
            }
            self.finished_ok.emit(segs, info_obj)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{e}")


def mlx_available() -> bool:
    """当前环境是否可使用 mlx-whisper（Metal 引擎）。"""
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def local_model_kind(path: str):
    """按目录内文件判断模型属于哪个引擎：
    返回 "fw"（含 model.bin，CTranslate2 格式）/ "mlx"（含 safetensors）/ None。"""
    import os
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
