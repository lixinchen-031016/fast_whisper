#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""应用全局配置与路径常量。"""
import os
import sys

# 项目根目录（app/ 的上一级）；兼容 PyInstaller 打包后的只读场景
if getattr(sys, "frozen", False):
    APP_ROOT = os.path.dirname(sys.executable)
else:
    APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# 内置推荐模型（CTranslate2 格式，供 faster-whisper CPU 引擎使用）
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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
