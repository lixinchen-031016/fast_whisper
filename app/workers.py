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

from . import homophone, model_manager, settings
from .model_manager import CancelledError


# ==================== 工作线程保活 ====================
# Qt 在 QThread 仍运行时析构会直接 qFatal 崩溃（"QThread: Destroyed while thread
# is still running"）。工作线程对象常常只被局部变量或对话框属性引用，一旦用户在
# 任务进行中关闭窗口、关闭搜索框或再次触发任务，引用丢失就会触发该崩溃。
# 这里统一保活：线程启动时登记强引用，结束后释放，与业务逻辑无关。
_LIVE_WORKERS = set()
_LIVE_WORKERS_LOCK = threading.Lock()


def _release_worker(worker):
    with _LIVE_WORKERS_LOCK:
        _LIVE_WORKERS.discard(worker)


def retain_worker(worker):
    """登记正在运行的 QThread，避免其在运行期间被 GC/作用域回收而崩溃。

    幂等：重复调用不会重复连接 finished 信号。线程结束后自动解除引用。
    """
    with _LIVE_WORKERS_LOCK:
        if worker in _LIVE_WORKERS:
            return
        _LIVE_WORKERS.add(worker)
    worker.finished.connect(lambda w=worker: _release_worker(w))


class RetainedThread(QThread):
    """启动即自动保活的 QThread 基类（见 retain_worker 的说明）。"""

    def start(self, *args, **kwargs):
        retain_worker(self)
        super().start(*args, **kwargs)


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


# 术语表提示词的长度上限：Whisper 的提示词窗口只有 224 个 token（上下文的一半
# 留给前序文本），过长会被截断甚至挤掉语种/任务标记，反而降低准确率。
# 实测精简术语表与冗长术语表效果相同，故按 token 量截断（约 3 字符/token）。
TERM_PROMPT_MAX_CHARS = 220


def _terms_prompt(language, terms) -> str:
    """把用户填写的术语表并进 initial_prompt。

    背景（真实采访素材实测）：Whisper 对专有名词极不稳定——「德国管理应用技术
    大学」在两台引擎上分别被识别为「国安利用技术大学」「管理用技术大学」，
    「生源质量」被识别为「声援质量」。而把正确写法放进提示词后，两台引擎都能
    稳定纠正（术语命中 6/10 → 7/10，且不增加耗时）。故术语表是投入产出比最高
    的准确率手段。

    返回 None 表示无提示词（不传该参数，让模型自行判断）。
    """
    base = _initial_prompt(language)
    terms = (terms or "").strip() if isinstance(terms, str) else ""
    if not terms:
        return base
    # 术语表里逗号/顿号/换行混用都可能，统一成中文顿号，避免模型照抄分隔符。
    # 拆分口径与近音词纠正共用 homophone.split_terms，避免两处规则漂移。
    flat = "、".join(homophone.split_terms(terms))
    flat = flat[:TERM_PROMPT_MAX_CHARS]
    hint = f"专业术语（请按此写法输出）：{flat}。"
    return f"{base}{hint}" if base else hint


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


# Whisper 系列模型固定要求的采样率
WHISPER_SAMPLE_RATE = 16000


class NoAudioStreamError(Exception):
    """媒体文件不含音频轨道（可明确提示用户，区别于其它解码错误）。"""


# 预分配缓冲的上限：容器时长元数据可能被构造得离谱（损坏文件/恶意样本），
# 无上限时一次 np.empty 就可能申请到数 GB。30 分钟（约 115 MB）已覆盖绝大多数
# 采访素材，超出部分走「按需增长」分支，行为与预分配路径一致。
_PREALLOC_LIMIT_SECONDS = 1800


def _estimate_sample_count(container, stream, sample_rate: int) -> int:
    """用容器/流的时长元数据估算重采样后的采样点数（无法估算时返回 0）。

    实测（C8733/C8735/C8736 三个 4K 采访素材、以及多档合成素材）容器时长与
    实际解码出的采样点数**偏差为 0**，故可据此预分配精确大小的缓冲，把峰值内存
    压到波形本体的 1 倍左右。
    """
    import av

    seconds = 0.0
    if container.duration:
        seconds = float(container.duration) / av.time_base
    elif getattr(stream, "duration", None) and stream.time_base:
        seconds = float(stream.duration * stream.time_base)
    if seconds <= 0:
        return 0
    return min(int(seconds * sample_rate), _PREALLOC_LIMIT_SECONDS * sample_rate)


def _resample_into(frame, resampler, append):
    """把重采样结果逐段交给 append（frame 为 None 时表示冲刷尾帧）。"""
    for out in resampler.resample(frame):
        append(out.to_ndarray().reshape(-1))


# 重采样输入的分组粒度（源采样点数）。FFmpeg 的重采样器是有状态的，输入切分粒度
# 会影响输出：逐帧喂入会与 faster-whisper 的 `decode_audio` 差 1 个 int16 LSB。
# 实测在 25 组素材（三种真实 4K 采访、6 档源采样率 × 单/双声道、多档时长）上，
# 取 16000（源采样点，约 1/3 秒）即可与上游**逐位一致**；而按上游的 500000 分组
# 会把额外峰值从 1.05x 抬到 1.29x（FIFO 积压的帧缓冲），故取小分组。
_RESAMPLE_GROUP_SAMPLES = 16000


def decode_audio_pyav(path: str, sample_rate: int = WHISPER_SAMPLE_RATE):
    """用随包内置的 PyAV 把媒体解码为**单声道 float32 波形**（幅值 [-1, 1]）。

    背景：mlx_whisper 自带的 load_audio 通过 subprocess 调用系统 `ffmpeg` 命令行，
    打包成 .app/.exe 后 PATH 通常不含 ffmpeg，会抛
    `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`。
    本项目已内置 PyAV（自带 FFmpeg 共享库），这里统一用它解码，彻底摆脱
    对系统 ffmpeg 可执行文件的依赖，与「无需手动安装 FFmpeg」的设计一致。

    返回一维 float32 ndarray；无音频流时抛 NoAudioStreamError。
    注意：PyAV 的解码错误（如 InvalidDataError）继承自 ValueError，调用方需
    用 `except NoAudioStreamError` 单独区分，切勿用 ValueError 兜底。

    内存与增益（两处都是实测驱动的修正，改前改后输出逐位一致的部分已用测试锁定）：

    1) 峰值内存 = 波形本体（约 1.04x），而非原来的 3.5~3.8x。
       原实现把整个文件的 PyAV 帧对象累积在列表里，再 `np.concatenate`——帧对象
       持有的 C 层缓冲要到循环结束才释放，于是「波形 + 全部帧」同时驻留。
       30 分钟素材实测 312 MB 的额外开销。现改为：**按容器时长预分配一段 float32
       缓冲，逐帧原地写入**，解码过程中不再产生任何整段拷贝。
       注意波形本身是 16 kHz × 4 B = 64 KB/秒（约 230 MB/小时），这是不可压缩的
       下限，预分配只消除其上的叠加开销。

    2) 立体声下混不再有 +3 dB 增益偏差（削波样本从 0.83% 降到 0）。
       重采样输出格式取 `s16` 而非 `flt`：FFmpeg 对整型做下混用 (L+R)/2，对浮点
       用 (L+R)/√2，后者幅度高 3 dB。实测三个采访素材（峰值 0.87~0.89）在 flt
       下会被推到 1.23~1.26，其中 0.83% 的样本削波到 ±1.0 —— 这是实打实的失真，
       且与 ffmpeg 命令行、faster-whisper、mlx-whisper 的行为都不一致。
       取 s16 后与 faster-whisper 的 `decode_audio` 逐位一致（最大差 0），
       代价只是量化到 16 bit —— 这正是上游 Whisper 全系的既有精度。
    """
    import numpy as np
    import av

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise NoAudioStreamError("媒体文件中没有可用的音频流")

        estimate = _estimate_sample_count(container, stream, sample_rate)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        buffer = np.empty(estimate, dtype=np.float32) if estimate else None
        written = 0

        def _append(chunk):
            """把一段 int16 原地换算并写入缓冲，必要时扩容。"""
            nonlocal buffer, written
            end = written + chunk.size
            if buffer is None or buffer.size < end:
                # 元数据缺失或偏小时按 1.5 倍增长，保证摊销复杂度为 O(n)
                capacity = max(end, int((buffer.size if buffer is not None else 0) * 1.5),
                               estimate, sample_rate)
                grown = np.empty(capacity, dtype=np.float32)
                if buffer is not None:
                    grown[:written] = buffer[:written]
                buffer = grown
            np.multiply(chunk, 1.0 / 32768.0, out=buffer[written:end])
            written = end

        # 分组后再重采样：FFmpeg 的重采样器是**有状态**的，输入切分粒度不同会
        # 产生 1 个 int16 LSB 级别的差异（实测 100 万样本里有 5 个）。按上游
        # faster-whisper 的 500000 样本分组，可与其 `decode_audio` 输出逐位一致，
        # 从而保证换成这里解码后转写结果不发生任何变化。
        # 分组缓冲固定上限 16000 个源采样点，与整段波形无关。
        fifo = av.audio.fifo.AudioFifo()
        for frame in container.decode(stream):
            frame.pts = None                       # 忽略时间戳校验，与上游一致
            fifo.write(frame)
            if fifo.samples >= _RESAMPLE_GROUP_SAMPLES:
                _resample_into(fifo.read(), resampler, _append)
        if fifo.samples > 0:
            _resample_into(fifo.read(), resampler, _append)
        _resample_into(None, resampler, _append)   # 冲刷重采样器尾帧

    if buffer is None or written == 0:
        return np.zeros(0, dtype=np.float32)
    if buffer.size != written:
        buffer.resize(written, refcheck=False)     # 原地截断，不产生整段拷贝
    return buffer


# ==================== 转写核心（与线程解耦） ====================
def _pct_reporter(on_progress, duration: float):
    """构造把「已处理到的时间点」换算为百分比并去重的回调。

    duration 未知（<=0）时返回空操作，不产生进度回调。
    """
    if not on_progress or not duration or duration <= 0:
        return lambda end: None
    state = {"last": -1}

    def _report(end):
        pct = int(max(0.0, min(1.0, float(end) / duration)) * 100)
        if pct != state["last"]:
            state["last"] = pct
            on_progress(pct)

    return _report


def transcribe_media(media_path: str, model_path: str, engine: str = "cpu",
                     language="auto", task: str = "transcribe",
                     beam_size: int = 5, use_vad: bool = True, batched: bool = False,
                     terms: str = "", fix_homophones: bool = True,
                     on_segment=None, on_status=None, on_progress=None,
                     cancel_check=None):
    """执行一次转写，返回 (segments, info)。

    on_segment(start, end, text) —— 每识别出一段即回调（可为 None）。
    on_status(text)             —— 状态文案回调（可为 None）。
    on_progress(pct)            —— 进度百分比 0~100（可为 None）。
    cancel_check() -> bool      —— 返回 True 时中止并抛出 CancelledError。
    terms                       —— 用户术语表（专有名词），并进 initial_prompt；
                                   实测能把「德国管理应用技术大学」等错识别纠正回来。
    fix_homophones              —— 是否按术语表做近音词确定性校正。提示词是概率手段，
                                   同一段音频里可能只修好一部分（实测「德国管理应用技术
                                   大学」修好了、「生源质量」仍错），该层用拼音比对把
                                   剩余的同音错别字确定性改回（见 app/homophone.py）。
    beam_size                   —— CPU/CUDA 下即 beam；Metal 下映射为温度策略
                                   （mlx 未实现 beam search，见 _temperatures_for_beam）。
    engine: "cpu" / "cuda"（faster-whisper）/ "mlx"（mlx-whisper，仅 macOS）。
    language: None 或 "auto" 表示自动检测。
    """
    lang = None if language in (None, "auto", "") else language
    prompt = _terms_prompt(lang, terms)
    # 纠错器只构造一次（索引构建是主要开销），逐段复用
    fixer = homophone.make_fixer(homophone.split_terms(terms)) if fix_homophones else None
    if engine == "mlx":
        return _transcribe_mlx(media_path, model_path, lang, task, beam_size, use_vad,
                               prompt, fixer,
                               on_segment, on_status, on_progress, cancel_check)
    return _transcribe_fw(media_path, model_path, engine, lang, task, beam_size,
                          use_vad, batched, prompt, fixer,
                          on_segment, on_status, on_progress,
                          cancel_check)


# ==================== 特征提取（分块 STFT） ====================
# faster-whisper 的 FeatureExtractor 对**整段**音频一次性做 STFT：先把波形切帧成
# (帧数, 400) 的视图，再对整块做 rfft。中间会同时驻留复数频谱与幅度谱，
# 30 分钟素材实测多出 1914 MB —— 这是全链路最大的内存热点（波形本身才 112 MB）。
#
# 分块在数学上完全等价：每帧的 rfft 只依赖该帧的 400 个采样点，帧间无耦合。
# 实测在 40 组用例（0.1s ~ 5min、含真实采访素材、padding/chunk_length 组合）上
# 与上游**逐位一致**，峰值开销从 1914 MB 降到 275 MB。
_FEATURE_FRAME_CHUNK = 2000        # 每块 2000 帧 ≈ 20 秒音频（约 6 MB 中间态）


def _chunked_feature_extractor_class():
    """构造 FeatureExtractor 的子类，把 STFT 改为分块执行。

    必须用子类而非给实例赋值 `__call__`：Python 的特殊方法在**类型**上查找，
    实例属性对 `obj(...)` 不生效（实测被静默忽略）。
    """
    from faster_whisper.feature_extractor import FeatureExtractor

    class ChunkedFeatureExtractor(FeatureExtractor):
        def __call__(self, waveform, padding=160, chunk_length=None):
            import numpy as np

            if chunk_length is not None:
                self.n_samples = chunk_length * self.sampling_rate
                self.nb_max_frames = self.n_samples // self.hop_length
            if waveform.dtype is not np.float32:
                waveform = waveform.astype(np.float32)
            if padding:
                waveform = np.pad(waveform, (0, padding))

            n_fft = self.n_fft
            pad = n_fft // 2
            padded = np.pad(waveform, (pad, pad), mode="reflect")
            n_frames = 1 + (padded.shape[0] - n_fft) // self.hop_length

            # 短音频（不超过一块）直接用上游实现，保持路径完全一致
            if n_frames - 1 <= _FEATURE_FRAME_CHUNK:
                return super().__call__(waveform, padding=0)

            window = np.hanning(n_fft + 1)[:-1].astype("float32")
            magnitudes = np.empty((n_fft // 2 + 1, n_frames - 1), dtype=np.float32)
            for start in range(0, n_frames - 1, _FEATURE_FRAME_CHUNK):
                stop = min(start + _FEATURE_FRAME_CHUNK, n_frames - 1)
                # 零拷贝切帧：stride 技巧与上游一致，只是分块做
                frames = np.lib.stride_tricks.as_strided(
                    padded[start * self.hop_length:], (stop - start, n_fft),
                    (self.hop_length * padded.strides[0], padded.strides[0]))
                spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
                magnitudes[:, start:stop] = np.abs(spectrum.astype("complex64")).T ** 2
            del padded, window

            mel_spec = self.mel_filters @ magnitudes
            del magnitudes
            log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
            del mel_spec
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            return (log_spec + 4.0) / 4.0

    return ChunkedFeatureExtractor


def _patch_feature_extractor(model):
    """把模型的特征提取器换成分块实现（幂等）。

    只替换实例（而非类），作用域限于本次转写所用的模型对象。
    mel 滤波器直接复用原实例已算好的矩阵，避免重复构造。
    """
    fe = getattr(model, "feature_extractor", None)
    if fe is None:
        return                      # 该对象没有特征提取器（如测试替身）→ 无事可做
    try:
        extractor_cls = _get_chunked_feature_extractor()
    except ImportError:
        return                      # 拿不到上游基类就保持原样，绝不影响转写
    if isinstance(fe, extractor_cls):
        return
    chunked = extractor_cls(
        feature_size=fe.mel_filters.shape[0], sampling_rate=fe.sampling_rate,
        hop_length=fe.hop_length, chunk_length=fe.chunk_length, n_fft=fe.n_fft)
    chunked.mel_filters = fe.mel_filters          # 复用已算好的滤波器
    model.feature_extractor = chunked


def _get_chunked_feature_extractor():
    """惰性构造分块特征提取器子类（首次用到时才导入 faster-whisper）。

    不能在模块导入期构造：faster-whisper 是可选依赖，缺失时 `app.workers`
    必须仍能导入（否则「缺少 faster-whisper」的友好提示会退化成导入崩溃）。
    """
    global _CHUNKED_FEATURE_EXTRACTOR
    if _CHUNKED_FEATURE_EXTRACTOR is None:
        _CHUNKED_FEATURE_EXTRACTOR = _chunked_feature_extractor_class()
    return _CHUNKED_FEATURE_EXTRACTOR


# 惰性缓存（见 _get_chunked_feature_extractor）
_CHUNKED_FEATURE_EXTRACTOR = None


def _decode_for_fw(media_path: str):
    """为 faster-whisper 路径预解码波形（失败返回 None，交由库自行解码）。

    收益（30 分钟素材实测）：解码阶段的额外峰值从 181 MB 降到约 0，
    整体峰值 2441 → 1650 MB 量级。波形本身（16 kHz × 4 B ≈ 230 MB/小时）
    是不可压缩的下限，优化只消除其上的叠加开销。

    为什么失败要回退而不是直接报错：
    - PyAV 缺失（打包异常）→ 回退后仍能跑（库自带解码）
    - 文件损坏/路径不存在 → 回退后由库抛出**它自己的**异常，
      经 _friendly_media_error 得到与改动前完全一致的提示文案
    - 唯一例外是「无音频流」：这是我们能给出更准确提示的情形，直接上抛
    """
    try:
        audio = decode_audio_pyav(media_path)
    except NoAudioStreamError:
        raise
    except MemoryError:
        raise                       # 内存不足不是"解码器不可用"，不能掩盖
    except Exception:
        return None
    # 空波形（0 采样点）会让下游 duration=0 出现除零/边界问题，交由库处理
    return audio if getattr(audio, "size", 0) else None


def _transcribe_fw(media_path, model_path, engine, lang, task, beam_size,
                   use_vad, batched, prompt, fixer, on_segment, on_status,
                   on_progress, cancel_check):
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

    # 先解码再加载模型：避免解码器的临时帧缓冲与模型加载临时缓冲叠加；
    # 波形本体仍需保留到转写完成，供 faster-whisper 直接复用。
    audio = _decode_for_fw(media_path)

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
            # 记住这次失败：批量转写/连续转写时不再逐个文件重试 CUDA 初始化
            from . import hardware
            hardware.mark_cuda_unusable()
            _emit(f"NVIDIA GPU 加载失败（{e}），已自动回退 CPU 推理")
            model, _ = _load_cached(
                model_path, "cpu", "int8",
                lambda: WhisperModel(model_path, device="cpu", compute_type="int8"))
        else:
            raise

    # 分块特征提取与音频来源无关（无论波形来自我们预解码还是库自行解码），
    # 故无条件应用；短音频自动走上游原路径，行为不变。
    _patch_feature_extractor(model)

    _emit("正在转写（批量模式）…" if batched else "正在转写…")
    if batched:
        # 批量推理：VAD 切块后并行解码（CPU 实测约 1.8x）。
        # 注意其参数集与逐段推理不同，不能混传 condition_on_previous_text 等。
        # vad_filter 必须为真：批量管线依赖 VAD 把长音频切成 <=30s 的块，
        # 关闭它时超过 30s 的音频会直接抛 RuntimeError（界面上已说明）。
        from faster_whisper.transcribe import BatchedInferencePipeline
        runner = BatchedInferencePipeline(model=model)
        segments, info = runner.transcribe(
            audio if audio is not None else media_path,
            language=lang, task=task,
            beam_size=beam_size, batch_size=8,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=prompt,                  # 与逐段模式保持一致
        )
    else:
        segments, info = model.transcribe(
            audio if audio is not None else media_path,
            language=lang, task=task, beam_size=beam_size,
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,      # 避免重复幻觉
            initial_prompt=prompt,
        )

    # 用 info.duration 与已处理段落的 end 估算进度（percent 去重后回调）
    report = _pct_reporter(on_progress, float(getattr(info, "duration", 0) or 0))
    segs, fixes = [], 0
    for seg in segments:
        if cancel_check and cancel_check():
            raise CancelledError()
        text = seg.text.strip()
        # 逐段纠错（而非等全部转完再改）：界面是流式出字的，用户看到的就已纠正
        if fixer:
            text, spans = fixer(text)
            fixes += len(spans)
        segs.append({"start": seg.start, "end": seg.end, "text": text})
        if on_segment:
            on_segment(seg.start, seg.end, text)
        report(seg.end)

    info_obj = {
        "duration": info.duration,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "engine": engine,
        "homophone_fixes": fixes,
    }
    return segs, info_obj


def _temperatures_for_beam(beam_size: int):
    """把界面的「精细度（beam）」折算为 Metal 引擎的温度策略。

    mlx-whisper 尚未实现 beam search（传 beam_size 直接抛 NotImplementedError），
    所以「精细度」在 Metal 下若原样透传就是**无效选项**。改为映射到温度回退链：
    - 快速（beam=1）  → 只用 0.0 贪心：最快，但长音频偶发重复/幻觉时没有兜底
    - 均衡（beam=5）  → 默认回退链 (0,0.2,…,1.0)：失败自动升温重试，最稳（默认）
    - 精细（beam>=8） → 更密的回退链：多给几次低中温重试机会
    这样同一个下拉框在三种引擎下都真实生效，语义都是“越往上越慢越准”。
    """
    try:
        beam = int(beam_size)
    except (TypeError, ValueError):
        beam = 5
    if beam <= 1:
        return (0.0,)
    if beam >= 8:
        return (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
    return (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _speech_clip_timestamps(audio, min_silence_ms: int = 500):
    """用 faster-whisper 内置的 Silero VAD 算出语音区间（秒），供 mlx 切分音频。

    mlx-whisper 自身没有任何 VAD 能力，只能按 `clip_timestamps` 提供的区间切分，
    因此 Metal 引擎下「过滤静音段」必需由调用方先算好区间再传进去——否则该选项
    在 Metal 上是无效的，而采访素材停顿多，静音段正是幻觉（凭空生成字幕）的温床。

    返回 [(start, end), …]；无语音或 VAD 不可用时返回 None（调用方按整段处理）。
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        chunks = get_speech_timestamps(
            audio, VadOptions(min_silence_duration_ms=min_silence_ms))
    except Exception:
        return None                      # VAD 依赖缺失/推理失败 → 退化为整段转写
    if not chunks:
        return None
    return [(c["start"] / WHISPER_SAMPLE_RATE, c["end"] / WHISPER_SAMPLE_RATE)
            for c in chunks]


def _transcribe_mlx(media_path, model_path, lang, task, beam_size, use_vad,
                    prompt, fixer, on_segment, on_status, on_progress, cancel_check):
    """mlx-whisper 路径（Apple Metal GPU）。

    - 音频解码走内置 PyAV（规避 mlx 对系统 ffmpeg 命令行的依赖）
    - 「精细度」映射为温度策略（mlx 不支持 beam search，见 _temperatures_for_beam）
    - 「过滤静音段」由内置 Silero VAD 算出语音区间后经 clip_timestamps 生效
    - 注意：mlx_whisper.transcribe 一次性返回全部结果，不提供流式进度，
      故本路径不使用 on_progress（界面保持不确定进度动画）。
    """
    try:
        import mlx_whisper
    except ImportError as e:
        # 区分两种完全不同的成因：包没装，vs 装了但当前会话拿不到 Metal 设备
        # （无头/沙箱/虚拟化的 macOS 上 GPU 不可见——此时提示"缺少 mlx-whisper"
        # 是误导，用户重装也没用）。与「无音频流」那次修正是同一类问题。
        low = str(e).lower()
        if "metal" in low or "load_device" in low:
            raise RuntimeError(
                "Metal GPU 当前不可用（未获取到 Metal 设备）。\n"
                "常见于无显示器/沙箱/虚拟化的 macOS 会话，或远程登录环境。\n"
                "请在正常桌面会话中运行，或改用 CPU / NVIDIA 引擎。\n"
                f"详细信息：{e}")
        raise RuntimeError(
            "缺少 mlx-whisper（Metal 引擎）。\n"
            "安装命令：.venv/bin/pip install mlx-whisper\n"
            f"详细信息：{e}")

    # 优先用内置 PyAV 解码为波形传入，避免依赖系统 ffmpeg 可执行文件
    # （mlx_whisper 传文件路径时会 subprocess 调用 `ffmpeg`，打包后常缺失）
    if on_status:
        on_status("正在用内置解码器解码音频 …")
    try:
        audio = decode_audio_pyav(media_path)
    except NoAudioStreamError:
        raise                       # 「无音频流」明确上报，给准确提示
    except (ImportError, ModuleNotFoundError):
        audio = None                # 内置解码器不可用 → 回退为传路径（交由 mlx 处理）
    audio_input = audio if (audio is not None and getattr(audio, "size", 0)) else media_path

    # 静音过滤：算出语音区间交给 mlx 只解码这些片段（mlx 自己不做 VAD）
    clip_timestamps = None
    if use_vad and audio is not None and getattr(audio, "size", 0):
        if on_status:
            on_status("正在检测静音段（VAD）…")
        chunks = _speech_clip_timestamps(audio)
        if chunks:
            clip_timestamps = [ts for span in chunks for ts in span]
            if on_status:
                covered = sum(e - s for s, e in chunks)
                on_status(f"已跳过多余静音（保留语音 {covered:.0f} 秒）")

    if cancel_check and cancel_check():
        raise CancelledError()
    if on_status:
        on_status("正在加载模型到 Metal GPU …")
    mlx_kwargs = {
        "path_or_hf_repo": model_path,
        "language": lang,
        "task": task,
        "verbose": False,
        # temperature 支持元组 → 失败自动升温重试（等价于别处的回退链）
        "temperature": _temperatures_for_beam(beam_size),
        # mlx 默认 True 会跨窗喂前序文本，长音频易陷入重复循环；与 CPU 路径保持一致
        "condition_on_previous_text": False,
    }
    if prompt:
        mlx_kwargs["initial_prompt"] = prompt
    if clip_timestamps:
        mlx_kwargs["clip_timestamps"] = clip_timestamps
    result = mlx_whisper.transcribe(audio_input, **mlx_kwargs)
    if cancel_check and cancel_check():
        raise CancelledError()

    segs, fixes = [], 0
    for s in result.get("segments", []):
        text = s["text"].strip()
        if fixer:
            text, spans = fixer(text)
            fixes += len(spans)
        segs.append({"start": s["start"], "end": s["end"], "text": text})
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
        "homophone_fixes": fixes,
    }
    return segs, info_obj


# ==================== Qt 线程封装 ====================
class DownloadWorker(RetainedThread):
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


class TranscribeWorker(RetainedThread):
    """单文件转写线程：加载模型 → 推理 → 逐段回报。"""
    model_loading = Signal()
    segment_ready = Signal(float, float, str)   # start, end, text
    progress_text = Signal(str)
    progress_pct = Signal(int)                  # 0~100
    finished_ok = Signal(list, object)          # segments(list[dict]), info
    failed = Signal(str)

    def __init__(self, media_path: str, model_path: str,
                 engine: str = "cpu",
                 language: str = "auto", task: str = "transcribe",
                 beam_size: int = 5, use_vad: bool = True,
                 batched: bool = False, terms: str = "",
                 fix_homophones: bool = True, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.model_path = model_path
        self.engine = engine if engine in ("cpu", "cuda", "mlx") else "cpu"
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.use_vad = use_vad
        self.batched = batched
        self.terms = terms
        self.fix_homophones = fix_homophones
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        self.model_loading.emit()
        try:
            segs, info = transcribe_media(
                self.media_path, self.model_path, engine=self.engine,
                language=self.language, task=self.task, beam_size=self.beam_size,
                use_vad=self.use_vad, batched=self.batched, terms=self.terms,
                fix_homophones=self.fix_homophones,
                on_segment=lambda s, e, t: self.segment_ready.emit(s, e, t),
                on_status=lambda m: self.progress_text.emit(m),
                on_progress=lambda p: self.progress_pct.emit(p),
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


class BatchTranscribeWorker(RetainedThread):
    """批量转写线程：顺序处理多个媒体文件，逐个自动导出为 TXT。

    复用模型缓存：全部文件共用一次模型加载，避免逐文件重复加载。
    progress_pct 报整批百分比（把“当前文件内百分比”折算进整批），
    否则每个文件转写期间进度条都停在文件边界不动。
    """
    file_started = Signal(int, int, str)      # index(1-based), total, path
    segment_ready = Signal(float, float, str)
    file_finished = Signal(int, str, str)     # index, path, export_path
    file_failed = Signal(int, str, str)       # index, path, error
    progress_text = Signal(str)
    progress_pct = Signal(int)                # 整批 0~100
    finished_ok = Signal(int, int)            # done, total
    failed = Signal(str)                      # 致命错误（如缺少依赖）

    def __init__(self, media_files: list, model_path: str, out_dir: str,
                 engine: str = "cpu", language: str = "auto",
                 task: str = "transcribe", beam_size: int = 5,
                 use_vad: bool = True, batched: bool = False, terms: str = "",
                 fix_homophones: bool = True, parent=None):
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
        self.terms = terms
        self.fix_homophones = fix_homophones
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.media_files)
        done = 0
        plan = self._plan_outputs()
        for i, path in enumerate(self.media_files, 1):
            if self._cancelled:
                break

            def _on_pct(pct, index=i):
                """把第 index 个文件内部的进度折算为整批百分比。"""
                overall = ((index - 1) + max(0.0, min(100.0, float(pct))) / 100.0) / max(1, total)
                self.progress_pct.emit(int(overall * 100))

            self.file_started.emit(i, total, path)
            try:
                segs, _info = transcribe_media(
                    path, self.model_path, engine=self.engine,
                    language=self.language, task=self.task, beam_size=self.beam_size,
                    use_vad=self.use_vad, batched=self.batched, terms=self.terms,
                    fix_homophones=self.fix_homophones,
                    on_segment=lambda s, e, t: self.segment_ready.emit(s, e, t),
                    on_status=lambda m: self.progress_text.emit(m),
                    on_progress=_on_pct,
                    cancel_check=lambda: self._cancelled,
                )
                if self._cancelled:
                    break
                out = self._export(path, segs, plan.get(path))
                done += 1
                # 该文件必然已完成：即使引擎不报内部进度（Metal），进度条也会前进
                self.progress_pct.emit(int(i * 100 / max(1, total)))
                self.file_finished.emit(i, path, out)
            except CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                self.file_failed.emit(i, path, _friendly_media_error(e))
                self.progress_pct.emit(int(i * 100 / max(1, total)))
        self.finished_ok.emit(done, total)

    def _plan_outputs(self) -> dict:
        """为整批文件预先规划输出路径，避免「同名不同扩展名」互相覆盖。

        仅用文件名主干命名时，同一目录下的 `采访.mp4`、`采访.wav`、`采访.m4a`
        会全部写到 `采访_转写稿.txt` —— 实测 3 个源文件最终只剩 1 份转写稿，
        另外 2 份被静默覆盖（无任何提示）。故对主干重名的条目补上源扩展名区分。
        """
        groups = {}
        for path in self.media_files:
            base = os.path.splitext(os.path.basename(path))[0]
            groups.setdefault(base, []).append(path)

        plan = {}
        for base, paths in groups.items():
            if len(paths) == 1:
                plan[paths[0]] = self._default_out_path(base, None)
                continue
            for path in paths:
                ext = os.path.splitext(os.path.basename(path))[1].lstrip(".").lower()
                out = self._default_out_path(base, ext)
                # 极端兜底：扩展名也相同（同名同扩展名）时再补序号，绝不覆盖
                suffix = 2
                while out in plan.values():
                    out = self._default_out_path(f"{base}({suffix})", ext)
                    suffix += 1
                plan[path] = out
        return plan

    def _default_out_path(self, base: str, ext=None) -> str:
        name = f"{base}_{ext}_转写稿.txt" if ext else f"{base}_转写稿.txt"
        return os.path.join(self.out_dir, name)

    def _export(self, media_path: str, segs: list, out_path: str = None) -> str:
        from .exporter import export_txt
        os.makedirs(self.out_dir, exist_ok=True)
        if out_path is None:      # 单独调用（非批量）时沿用默认命名
            base = os.path.splitext(os.path.basename(media_path))[0]
            out_path = self._default_out_path(base)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return export_txt(segs, out_path)


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
    low = msg.lower()
    # MLX 引擎旧路径会调用系统 ffmpeg 命令行；PATH 中缺失时给出可操作指引
    if "ffmpeg" in low and (type_name.startswith("FileNotFoundError")
                            or "no such file" in low or "not found" in low):
        return (
            "音频解码器不可用：未找到 ffmpeg 可执行文件。\n"
            "本程序已优先使用内置解码器；若仍出现此提示，请改用 CPU / NVIDIA 引擎，"
            "或安装 ffmpeg（macOS: brew install ffmpeg，Windows 见 ffmpeg.org）。\n"
            f"详细信息：{msg}"
        )
    if "没有可用的音频流" in msg:
        return f"该文件不包含音频轨道，无法转写。\n{msg}"
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


def resolve_model_path(selector: str, engine: str) -> str:
    """把界面上的“模型选择项”解析为本地目录路径。

    选择项有两种形态：
    - 已下载模型：字符串本身即本地目录路径，原样返回；
    - 内置推荐模型（未下载）：形如 `Systran/faster-whisper-tiny` 的仓库 id，
      映射到模型目录下的 `Systran__faster-whisper-tiny`。

    主窗口的“模型列表刷新”与“转写前预检”共用这一处推导，避免两边各拼一次路径
    而在模型目录变化后出现不一致（曾导致预检永远认为内置模型未下载）。
    """
    if os.path.isdir(selector):
        return selector
    from .config import ENGINES
    if selector in ENGINES.get(engine, {}).get("builtin", []):
        return os.path.join(settings.get_models_dir(), selector.replace("/", "__"))
    return selector


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
