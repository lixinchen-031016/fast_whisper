#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""应用全局配置与路径常量。

注意：本模块只提供“默认值”，不做任何磁盘写入；目录的实际创建推迟到使用时
（main.py 启动时、模型下载时），避免导入副作用，也兼容只读安装目录。
"""
import os
import sys

# 项目根目录（app/ 的上一级）；PyInstaller 打包后为 exe/app 所在目录
if getattr(sys, "frozen", False):
    APP_ROOT = os.path.dirname(sys.executable)
else:
    APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认路径（用户可在偏好设置中覆盖，见 app/settings.py）
MODELS_DIR = os.path.join(APP_ROOT, "models")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

# 国内镜像站（HuggingFace 官方站的镜像），模型搜索/下载均走此地址
HF_MIRROR = "https://hf-mirror.com"

# 媒体文件过滤器
MEDIA_FILTER = (
    "媒体文件 (*.mp4 *.mov *.m4v *.mkv *.avi *.webm "
    "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;"
    "视频 (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;"
    "音频 (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;"
    "所有文件 (*)"
)

# 内置推荐模型（CTranslate2 格式，供 faster-whisper CPU / CUDA 引擎共用）
FW_BUILTIN_MODELS = [
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-large-v3",
]

# 内置推荐模型（MLX 格式，供 mlx-whisper Metal GPU 引擎使用，仅 macOS）
MLX_BUILTIN_MODELS = [
    "mlx-community/whisper-tiny",
    "mlx-community/whisper-base",
    "mlx-community/whisper-small",
    "mlx-community/whisper-medium",
    "mlx-community/whisper-large-v3-turbo",
]

# 引擎注册表：三种架构的运行参数与模型格式
# kind 用于模型格式判别：fw = CTranslate2（model.bin），mlx = MLX（safetensors/weights.npz）
ENGINES = {
    "cpu": {
        "label": "CPU（faster-whisper）",
        "kind": "fw",
        "device": "cpu",
        "compute_type": "int8",
        "builtin": FW_BUILTIN_MODELS,
        "search_hint": "faster-whisper",      # 镜像站搜索默认关键词
    },
    "cuda": {
        "label": "NVIDIA GPU（faster-whisper·CUDA）",
        "kind": "fw",
        "device": "cuda",
        "compute_type": "float16",
        "builtin": FW_BUILTIN_MODELS,
        "search_hint": "faster-whisper",
    },
    "mlx": {
        "label": "Metal GPU（mlx-whisper）",
        "kind": "mlx",
        "device": "metal",                    # mlx 自动使用 Metal，仅作展示
        "compute_type": "float16",
        "builtin": MLX_BUILTIN_MODELS,
        "search_hint": "mlx whisper",
    },
}
