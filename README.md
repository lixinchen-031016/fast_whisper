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

```bash
cd /Users/lixinchen/PycharmProjects/fast_whisper

# 1. 激活项目自带虚拟环境（已封装全部依赖）
source .venv/bin/activate

# 2. 启动应用
python main.py
```

如需在其他机器重建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

## 加速：Metal GPU（Apple Silicon）

**说明**：faster-whisper 的底层引擎 CTranslate2 不支持 Apple GPU（实测
`device="metal"` 返回 `unsupported device metal`），因此 Mac 上的 GPU 加速
通过第二引擎 **mlx-whisper**（Apple 官方 MLX 框架）实现。

在设置区的「引擎」中切换：

| 引擎 | 运行方式 | 适用模型格式 | 实测（M4，70s 音频） |
|------|---------|-------------|--------------------|
| CPU（faster-whisper） | CPU int8 量化，可选批量推理 | CTranslate2（`Systran/faster-whisper-*`） | tiny 25x；批量模式再快约 1.8x |
| Metal GPU（mlx-whisper） | Apple GPU 原生推理 | MLX（`mlx-community/whisper-*`） | large-v3-turbo 5.1x 实时（含加载），识别质量更高 |

- Metal 引擎需要先安装：`.venv/bin/pip install mlx-whisper`（requirements.txt 中已注释标注）；
- 两个引擎的模型格式不通用，切换引擎后模型下拉列表会自动联动，未下载的
  `mlx-community/*` 模型仍通过「搜索 / 下载模型…」从镜像站拉取；
- 经验法则：tiny/base 级别模型 CPU 更快；**medium 及以上模型用 Metal 引擎更划算**；
- 批量推理勾选框仅对 CPU 引擎生效（Metal 本身已是并行推理）。

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
