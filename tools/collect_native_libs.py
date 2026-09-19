#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收集第三方包的原生二进制库，供 PyInstaller 打包进单文件产物。

PyInstaller 的依赖分析只跟踪 import 关系，以下两类库不会被自动发现：
1. nvidia-cublas/cudnn 的 CUDA 运行库（CTranslate2 的 C++ 层按名加载）
2. PyAV 内置的 FFmpeg 共享库（Windows 上 avcodec-*.dll 等与 av.pyd 同级；
   macOS 的 av/.dylibs 由 hooks-contrib 的 av 钩子处理，无需此脚本）

用法：
    python tools/collect_native_libs.py <输出目录>
退出码：0 = 成功；1 = PyAV 库缺失（转写功能不可用，构建应失败）。
"""
import glob
import os
import shutil
import sys

# Windows CI 控制台默认 cp1252，打印中文会抛 UnicodeEncodeError，强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _pkg_dir(module_name: str):
    """返回包目录路径；命名空间包（__file__ 为 None）走 find_spec。"""
    try:
        module = __import__(module_name)
        d = getattr(module, "__file__", None)
        d = os.path.dirname(d) if d else None
        if d:
            return d
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        if spec and spec.submodule_search_locations:
            return list(spec.submodule_search_locations)[0]
    except ImportError:
        return None
    return None


def _collect(src_dir: str, patterns: list, out_dir: str) -> int:
    copied = 0
    for pattern in patterns:
        # recursive=True：支持 ** 深层匹配（如 lib*/**/*.dll）
        for src in glob.glob(os.path.join(src_dir, pattern), recursive=True):
            dest = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            copied += 1
    return copied


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)

    # ---- PyAV 内置 FFmpeg（必需：视频转写的解码引擎） ----
    # Windows：新版轮子（PyAV ≥13，delvewheel 修补）把带哈希后缀的 FFmpeg DLL
    # 放在与 av/ 平级的 av.libs/ 顶层目录；旧版在 av/ 顶层——两处都收集；
    # macOS/Linux：av/.dylibs 由 hooks-contrib 的 av 钩子处理，此处仅探测报告
    is_windows = sys.platform == "win32"
    av_dir = _pkg_dir("av")
    if is_windows:
        av_dlls = 0
        if av_dir:
            av_dlls += _collect(av_dir, ["*.dll"], out_dir)             # 旧版布局
            libs_dir = os.path.join(os.path.dirname(av_dir), "av.libs")
            if os.path.isdir(libs_dir):
                av_dlls += _collect(libs_dir, ["*.dll"], out_dir)       # 新版布局
        print(f"PyAV 内置 FFmpeg：收集 {av_dlls} 个原生库（av/ 与 av.libs/）")
        if av_dir is None or av_dlls == 0:
            print("错误：未找到 PyAV 的 FFmpeg 原生库，视频转写将不可用，构建终止")
            sys.exit(1)
    else:
        probe = os.path.join(av_dir, ".dylibs") if av_dir else ""
        ok = bool(av_dir and glob.glob(os.path.join(probe, "*.dylib")))
        print(f"PyAV 内置 FFmpeg：{'检测正常（.dylibs 由 PyInstaller 钩子处理）' if ok else '未检测到，本脚本在非 Windows 平台不负责收集'}")

    # ---- NVIDIA CUDA 运行库（可选：缺失仅影响 GPU 加速） ----
    nvidia_dir = _pkg_dir("nvidia")
    n_dlls = 0
    if nvidia_dir:
        n_dlls = _collect(nvidia_dir, ["*/bin/*.dll", "*/bin/*.so*"], out_dir)
        print(f"NVIDIA CUDA 运行库：收集 {n_dlls} 个")
    else:
        print("NVIDIA CUDA 运行库：未安装，跳过（产物将不支持 NVIDIA 加速）")

    print(f"全部原生库已收集到 {out_dir}")


if __name__ == "__main__":
    main()
