#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬件探测：识别 CPU / Apple Metal / NVIDIA CUDA 三种架构，推荐最佳引擎。

探测结果进程内缓存，避免重复加载 mlx / ctranslate2 的开销。
"""
import sys

_cache = {}


def cuda_available() -> bool:
    """NVIDIA GPU + CTranslate2 CUDA 支持是否可用。

    注意：即使有显卡，运行时还需系统安装 cuBLAS/cuDNN 才能真正加载模型，
    加载失败时 TranscribeWorker 会自动回退 CPU（并调用 mark_cuda_unusable()）。
    """
    if "cuda" in _cache:
        return _cache["cuda"]
    ok = False
    try:
        import ctranslate2
        ok = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        ok = False
    _cache["cuda"] = ok
    return ok


def mark_cuda_unusable():
    """把 CUDA 记为不可用（探测通过但真正加载模型失败时调用）。

    `get_cuda_device_count()` 只反映“有没有显卡”，并不能说明 cuBLAS/cuDNN 是否齐全：
    缺运行库时每次构造 CUDA 模型都会失败，并付出一次驱动探测 + CUDA 运行库
    加载的代价（实测可达数秒）。批量转写时若每个文件都重试一遍 CUDA，
    整批会白白浪费大量时间，故首次失败后直接记住，后续统一走 CPU。
    """
    _cache["cuda"] = False


def metal_available() -> bool:
    """Apple Silicon + mlx-whisper 是否可用。"""
    if "mlx" in _cache:
        return _cache["mlx"]
    ok = False
    if sys.platform == "darwin":
        try:
            import mlx_whisper  # noqa: F401
            ok = True
        except Exception:
            ok = False
    _cache["mlx"] = ok
    return ok


def detect_best() -> str:
    """返回当前机器的最佳引擎："mlx" / "cuda" / "cpu"。

    优先级：Metal GPU > NVIDIA CUDA > CPU。
    （M 系列芯片上 mlx 推理快且零配置；CUDA 依赖 cuDNN，出问题可手动切 CPU）
    """
    if metal_available():
        return "mlx"
    if cuda_available():
        return "cuda"
    return "cpu"


def available_engines() -> list:
    """返回当前机器可用的引擎列表（"cpu" 永远可用）。"""
    engines = ["cpu"]
    if cuda_available():
        engines.append("cuda")
    if metal_available():
        engines.append("mlx")
    return engines
