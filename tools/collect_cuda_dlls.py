#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收集 NVIDIA CUDA 运行库 DLL，供 PyInstaller 打包进单文件 exe。

CTranslate2 在 Windows 上推理时按名称加载 cublas64_12.dll、cudnn64_9.dll 等，
它们由 pip 包 nvidia-cublas-cu12 / nvidia-cudnn-cu12 安装到
site-packages/nvidia/<lib>/bin/ 下——Python 不会 import 它们，
PyInstaller 的依赖分析发现不了，必须显式收集。

用法：
    python tools/collect_cuda_dlls.py <输出目录>
之后用 --add-binary "<输出目录>/*.dll;." 嵌入 exe（onefile 解压目录在
DLL 搜索路径中，无需额外配置）。
"""
import glob
import os
import shutil
import sys

# Windows CI 控制台默认 cp1252，打印中文会抛 UnicodeEncodeError，强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)

    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError
        # nvidia 是命名空间包（无 __init__.py），__file__ 为 None，
        # 必须经 find_spec 取 submodule_search_locations
        search_bases = list(spec.submodule_search_locations)
    except ImportError:
        print("未安装 nvidia-* 运行库（nvidia-cublas-cu12 / nvidia-cudnn-cu12），跳过收集")
        return

    patterns = []
    for base in search_bases:
        patterns += [
            os.path.join(base, "*", "bin", "*.dll"),    # Windows
            os.path.join(base, "*", "bin", "*.so*"),    # Linux（预留）
        ]
    copied = 0
    for pattern in patterns:
        for src in glob.glob(pattern):
            # cuDNN 等包较大且存在同名风险，平铺到同一目录即可
            dest = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            copied += 1
    print(f"已收集 {copied} 个 CUDA 运行库文件到 {out_dir}")
    if copied == 0:
        print("警告：nvidia 包存在但未找到任何 DLL/SO，请检查安装完整性")
        sys.exit(1)


if __name__ == "__main__":
    main()
