# FastWhisper · 语音转文字

基于 **PySide6 + faster-whisper** 的本地语音转文字桌面应用，macOS 风格界面。
支持视音频文件导入、模型检索与国内镜像下载、本地离线推理，以及 TXT / SRT / Markdown / Word 四种格式导出。

## 目录结构

```
fast_whisper/
├── main.py                  # 应用入口
├── requirements.txt         # 依赖清单
├── app/
│   ├── __init__.py
│   ├── config.py            # 路径常量、镜像站地址、媒体过滤器、内置模型列表
│   ├── theme.py             # 苹果风格主题（全局 QSS：圆角卡片、系统蓝、留白）
│   ├── model_manager.py     # 模型搜索/下载（hf-mirror.com，流式下载 + 断点续传）
│   ├── workers.py           # QThread 后台线程：下载进度、转写推理
│   ├── exporter.py          # 导出器：TXT / SRT / Markdown / DOCX
│   ├── widgets.py           # 可复用组件：圆角卡片、拖放导入区、模型搜索对话框
│   └── main_window.py       # 主窗口：导入 → 设置 → 转写 → 预览 → 导出
├── models/                  # 下载的模型存放目录（每个模型一个子目录）
├── output/                  # 默认导出目录
└── .venv/                   # 独立虚拟环境（所有依赖封装于此，可整体拷贝）
```

## 快速开始

### macOS

```bash
cd /Users/lixinchen/PycharmProjects/fast_whisper
source .venv/bin/activate
python main.py
```

### Windows

```powershell
cd fast_whisper
.venv\Scripts\activate
python main.py        # 或 pythonw main.py 隐藏控制台窗口
```

如需在其他机器重建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可选：按平台加装 GPU 引擎
pip install -r requirements-macos.txt    # macOS：Metal 引擎（mlx-whisper）
pip install -r requirements-windows.txt  # Windows：CUDA 12 运行库（NVIDIA 引擎）
```

## 偏好设置（持久化）

主窗口右上角「⚙ 偏好设置」：

- **模型存放目录**：默认项目内 `models/`（打包版为用户主目录 `~/FastWhisper/models`），可改为任意磁盘位置；保存后模型列表自动重新扫描；
- **默认导出目录**：默认跟随媒体文件所在目录，可指定固定目录（如 `D:\转写稿`）。

设置通过 QSettings（ini 格式）持久化：macOS 位于
`~/.config/FastWhisper/FastWhisper.ini`，Windows 位于
`%APPDATA%\FastWhisper\FastWhisper.ini`，可随目录整体备份迁移。
注意：修改模型目录后，已有模型文件需手动迁移到新目录。

## 跨平台兼容说明

| 事项 | 处理方式 |
|------|---------|
| 路径分隔符 | 全部经 `os.path.join`/`os.path.normpath` 构造，无硬编码 `/` 或 `\` |
| 字体 | QSS 字体栈按平台自动选择（macOS 苹方 / Windows 微软雅黑 / Linux Noto） |
| 安装目录只读 | 打包态默认模型目录落在用户主目录，不写安装目录 |
| GPU 探测 | 运行时自动探测 Metal / CUDA，均不可用则回退 CPU；CUDA 缺 cuDNN 时加载失败自动回退 |
| 可选依赖 | mlx-whisper 仅 macOS 可安装，缺失时界面显示"未安装"而非报错 |
| 高分屏 | `HighDpiScaleFactorRoundingPolicy.PassThrough`，Retina / Windows 125% 缩放均正常 |

## 自动构建与发布（GitHub Actions）

`.github/workflows/build.yml` 提供 macOS（.app/.dmg）与 Windows（单文件 .exe）
的自动构建流水线：

- **test 门禁**：Ubuntu 上运行 pytest（设置持久化 / 引擎过滤 / offscreen 界面构造）；
- **并行构建**：Windows x86_64 单文件 exe + macOS arm64 .app/.dmg，PyInstaller 打包
  并显式收集 `faster_whisper`/`ctranslate2`/`av`/`tokenizers`/`onnxruntime` 隐藏依赖，
  构建后自动冒烟启动；
- **GPU 引擎捆绑**：
  - macOS 构建安装 `requirements-macos.txt`（mlx-whisper）并以
    `--collect-all mlx mlx_whisper numba` 内嵌 Metal 引擎；
  - Windows 构建安装 `requirements-windows.txt`（nvidia-cublas/cudnn-cu12），
    经 `tools/collect_cuda_dlls.py` 收集 DLL 后以 `--add-binary` 内嵌，
    启动时自动把解压目录加入 DLL 搜索路径；
- **双轨发布**：推送 `v*` 标签 → 正式 Release；推送 main → 滚动 nightly 预发布
  （覆盖上一次，标题带 commit 哈希）；PR 只测试不发布。

## 使用流程

1. **导入媒体**：将视频/音频文件拖入顶部虚线区域，或点击选择
   （支持 mp4 / mov / mkv / mp3 / wav / m4a / flac 等常见格式）。
2. **选择模型**：
   - 下拉框中 `•` 开头为已下载到 `models/` 的本地模型；
   - `○` 开头为内置推荐项，尚未下载；
   - 点击「搜索 / 下载模型…」可在 **hf-mirror.com（HuggingFace 国内镜像）**
     检索任意语音识别模型（优先展示 CTranslate2 格式，即 faster-whisper 可用格式），
     选中后流式下载并显示进度，支持取消与断点续传。
3. **转写设置**：识别语言（自动/中/英…）、输出模式（转写原文字 / 翻译为英语）、
   推理精细度（beam=1 快速 / beam=5 均衡）、VAD 静音过滤。
4. **开始转写**：后台线程推理，逐段实时显示在「分段预览」表格（带时间戳），
   「纯文本」页合并为连续文本。
5. **导出**：TXT / SRT 字幕 / Markdown / Word 文档，
   默认保存到媒体文件同目录（`<文件名>_转写稿.<ext>`），也可自选路径。

## 模型建议

| 模型 | 大小 | 相对速度 | 适用场景 |
|------|------|---------|---------|
| tiny | ~75 MB | 最快 | 快速预览、测试流程 |
| base | ~145 MB | 快 | 日常短音频 |
| small | ~480 MB | 中 | 质量与速度均衡 |
| medium | ~1.5 GB | 较慢 | 中文采访转写（推荐） |
| large-v3 | ~3 GB | 最慢 | 最高精度 |

全部推理在本机 CPU（int8 量化）完成，**不联网、不上传音频**，隐私安全。

## 加速：多架构引擎（自动识别）

「引擎」下拉框默认 **自动识别最佳架构**：Apple 芯片 → Metal，NVIDIA 显卡 → CUDA，否则 CPU。也可手动指定。

| 引擎 | 运行方式 | 模型格式 | 实测（M4，70s 音频） |
|------|---------|---------|--------------------|
| CPU（faster-whisper） | CPU int8，可选批量推理 | CTranslate2（`Systran/faster-whisper-*`） | tiny 25x；批量模式再快约 1.8x |
| NVIDIA GPU（faster-whisper·CUDA） | CUDA float16 | CTranslate2（同上，与 CPU 通用） | 取决于显卡，需 cuDNN |
| Metal GPU（mlx-whisper） | Apple GPU 原生推理 | MLX（`mlx-community/whisper-*`） | large-v3-turbo 5.1x 实时（含加载），识别质量更高 |

- **CUDA**：需要 NVIDIA 显卡 + 系统安装 cuBLAS/cuDNN（CUDA 12）；未检测到显卡时引擎项显示"未检测到"。若加载失败（如缺 cuDNN），程序会**自动回退 CPU** 并提示，不会中断。
- **Metal**：需安装 mlx-whisper（`.venv/bin/pip install mlx-whisper`，requirements.txt 中已注释标注）。
- **模型搜索按架构过滤**：「搜索 / 下载模型」对话框会跟随当前引擎切换架构——CPU/NVIDIA 只搜 CTranslate2 格式（`faster-whisper`/`ct2`），Apple 只搜 MLX 格式（限定 `mlx-community`）；选错架构格式的模型在启动转写时也会被拦截提醒。
- 经验法则：tiny/base 级别模型 CPU 更快；**medium 及以上模型用 GPU 引擎更划算**；
- 批量推理勾选框仅对 CPU / NVIDIA 引擎生效（Metal 本身已是并行推理）。

## 技术要点

- **引擎**：faster-whisper（CTranslate2 推理），int8 量化，VAD 过滤静音，
  `condition_on_previous_text=False` 抑制重复幻觉；
- **镜像站**：搜索走 `hf-mirror.com/api/models`，下载走
  `hf-mirror.com/<repo>/resolve/main/<file>`，支持 Range 断点续传；
- **界面**：QSS 实现苹果 HIG 风格——`#F5F5F7` 背景、白色 12px 圆角卡片、
  `#0A84FF` 系统蓝强调色、PingFang SC 字体、细窄滚动条；
- **线程模型**：转写与下载均在 QThread 中执行，通过信号逐段/逐块回报进度，
  界面不卡顿，支持随时取消。

## 常见问题

- **Q: 模型下载很慢或失败？** 镜像站高峰期可能波动，程序支持断点续传，重试即可继续。
- **Q: 找不到已下载的模型？** 点击「刷新」重新扫描 `models/` 目录；
  只有含 `model.bin` 的目录才会被识别为完整模型。
- **Q: 想用 GPU？** 当前版本固定 CPU + int8（跨平台零依赖）；
  如需 CUDA，可将 `workers.py` 中 `WhisperModel(...)` 的参数改为
  `device="cuda", compute_type="float16"`。
