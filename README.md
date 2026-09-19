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
│   ├── model_manager.py     # 模型搜索/下载（hf-mirror.com，多文件并行 + 断点续传 + 完整性校验）
│   ├── workers.py           # QThread 后台线程：下载进度、单文件/批量转写；进程内模型缓存
│   ├── exporter.py          # 导出器：TXT / SRT / Markdown / DOCX
│   ├── widgets.py           # 可复用组件：圆角卡片、拖放导入区、模型搜索对话框（搜索走后台线程）
│   └── main_window.py       # 主窗口：导入 → 设置 → 转写（单文件/批量）→ 预览 → 导出；含取消按钮与偏好记忆
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

此外，**引擎 / 语言 / 输出模式 / 精细度 / VAD / 批量 / 术语表** 等界面选项也会自动记住，
下次启动恢复上次选择（关闭窗口时保存）。

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
    经 `tools/collect_native_libs.py` 收集 DLL 后以 `--add-binary` 内嵌，
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
     选中后**多文件并行下载**并显示进度，支持取消与断点续传（下载完成自动校验完整性）。
3. **转写设置**：识别语言（自动/中/英…）、输出模式（转写原文字 / 翻译为英语）、
   推理精细度（beam=1 快速 / beam=5 均衡）、VAD 静音过滤。
4. **开始转写**：后台线程推理，进度条显示**真实百分比**（Metal 引擎为不确定动画），
   逐段实时显示在「分段预览」表格（带时间戳），「纯文本」页合并为连续文本。
   分段结果按批刷新（80ms / 每 40 段），长音频不卡顿；随时可点「取消」中止。
5. **批量转写（可选）**：点击「批量转写文件夹…」选择目录，按文件名顺序转写其中
   所有视音频文件，逐个自动导出为 TXT 到导出目录；**整批共用一次模型加载**。
6. **导出**：TXT / SRT 字幕 / Markdown / Word 文档，
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

### 引擎能力差异（同一组界面选项在三种引擎下的实际含义）

| 界面选项 | CPU / NVIDIA（faster-whisper） | Metal（mlx-whisper） |
|---------|------------------------------|---------------------|
| 精细度 | `beam_size`（原样生效） | **映射为温度回退链**——mlx 尚未实现 beam search（传 `beam_size` 直接抛 `NotImplementedError`），故快速＝只贪心(0.0)、均衡＝默认回退链、精细＝更密的回退链 |
| 过滤静音段（VAD） | 内置 Silero VAD | **由本程序先用 Silero VAD 算出语音区间**，再经 `clip_timestamps` 交给 mlx 只解码这些片段（mlx 自身没有任何 VAD 能力） |
| 术语表 | `initial_prompt` | 同左 |
| 批量推理 | 并行解码（约 1.8x） | 不适用（Metal 本身已是并行推理） |

### 提升识别率：术语表（实测最有效）

真实采访素材（成都工业学院中外合作办学项目访谈，4K 视频）实测：Whisper 对**专有名词**极不稳定，
两个引擎都会把「德国管理应用技术大学」听成「国安利用技术大学」/「管理用技术大学」，
把「生源质量」听成「声援质量」。在界面「术语表」中填入正确写法后，两台引擎都能稳定纠正：

| 素材 | 引擎 | 无术语表 | 有术语表 |
|------|------|---------|---------|
| C8735（68s） | Metal turbo | 「德国管理**用**技术大学」 | 「德国管理**应用**技术大学」 |
| C8735（68s） | CPU large-v2 | 「德国管理**用**技术大学」 | 「德国管理**应用**技术大学」 |

术语表会按语种自动拼接在提示词后（`专业术语（请按此写法输出）：…`），并做长度上限保护
（`TERM_PROMPT_MAX_CHARS`，避免挤掉 Whisper 提示词窗口里的语种标记）。填入一次即自动记住。

## 性能优化

| 优化 | 说明 |
|------|------|
| 模型缓存 | 进程内按 `(模型路径, 设备, 精度)` 缓存已加载模型（上限 1，切换时释放）；**连续转写 / 批量转写时省去每次重复加载权重**（CPU 大模型加载可达 10s+） |
| 分段批量刷新 | 转写分段先入缓冲，按 80ms 定时器或每 40 段批量写入表格，并用 `setUpdatesEnabled` 关闭重绘；长音频（数百段）UI 不再卡顿 |
| 状态文案同步批刷 | 段数文案原先**每段**改一次 `QLabel`（同样触发逐次布局/重绘），现改为随表格一起按批刷新，长音频下这条路径的控件开销降为约 1/40 |
| 并行下载 | 模型仓库多文件用线程池并行下载（默认 4 并发），小文件（config/vocab）不再逐个排队 |
| 批量进度真实化 | 批量转写进度条改为**整批百分比**（把「当前文件内部进度」折算进整批），不再每个文件只跳一格、长文件转写期间长时间静止 |
| 下载进度总体化 | 多文件并行下载时按**所有文件字节总和**显示进度（单调不回退），并同时展示当前文件与总体量，避免进度条在并行文件之间来回摆动 |
| CUDA 失败记忆 | CUDA 探测通过但实际加载失败（缺 cuBLAS/cuDNN）时记住不可用，批量/连续转写不再为每个文件重复付出一次 CUDA 初始化代价 |
| 精简请求 | 去掉每文件一次 `HEAD` 探测，改为从 `GET` 响应头 / `Content-Range` 直接取总大小，减少一轮 RTT |
| 续传健壮性 | 按响应码区分 206（续传）/ 200（服务器忽略 Range → 截断重下，修复进度 >100% 的计数 bug）/ 416（已完成跳过）；下载完成后**校验文件大小**，不符即报错 |
| mlx 时长准确 | Metal 引擎改用 PyAV 探测媒体真实时长，替代原先"末段结束时间"的近似值 |
| 多语种提示词 | `initial_prompt` 按所选语言动态选择（中/英/日/韩/德/法/西），自动检测时沿用中文提示，避免统一中文提示对非中文音频产生偏置 |

## 健壮性与体验优化

| 优化 | 说明 |
|------|------|
| 搜索异步化 | 模型搜索移入后台线程（`SearchWorker`），镜像站响应慢时界面不再冻结 |
| 转写进度 | 非 Metal 引擎按 `已处理时长 / 总时长` 显示**真实百分比**（Metal 因一次性返回，保持不确定动画） |
| 取消按钮 | 转写 / 批量进行中显示「取消」，可随时中止当前任务 |
| 偏好记忆 | 引擎 / 语言 / 输出模式 / 精细度 / VAD / 批量选项自动持久化，下次启动恢复上次选择 |
| 网络重试 | 搜索与元信息请求失败自动重试（指数退避），应对镜像站瞬时抖动 |
| 导出健壮性 | 导出包裹异常处理（磁盘满 / 权限不足 / 缺 python-docx 时弹窗提示而非崩溃）；保存路径自动补扩展名 |
| 空段清洗 | 导出前去除分段首尾空白并丢弃空段，避免 TXT 空行与空字幕 |
| 已下载判定修正 | 「已下载」状态同时识别 CTranslate2（`model.bin`）与 MLX（`*.safetensors`）格式，修复 MLX 模型永远显示未下载的问题 |
| MLX 内置解码 | Metal 引擎改用**内置 PyAV 解码为波形**再送入模型，不再依赖系统 `ffmpeg` 命令行（修复打包后 `No such file or directory: 'ffmpeg'`） |
| 错误提示纠正 | ffmpeg 缺失 / 无音频轨道等场景给出准确可操作的提示，不再统一误报为「路径不存在」 |
| 缓存一致性 | 切换模型目录时清空进程内模型缓存，释放旧目录已加载的模型 |
| 模型路径统一推导 | 新增 `resolve_model_path()`：主窗口的「模型列表刷新」与「转写前预检」**共用同一处**路径推导（内置项 → `models/Systran__faster-whisper-tiny`），修复内置模型已手动放进模型目录却仍被判定为「未下载」的矛盾 |
| 批量模式参数一致 | 批量推理补回 `initial_prompt`（与逐段模式行为对齐），并显式开启 `vad_filter`——该管线依赖 VAD 切块，关闭 VAD 转写 >30s 音频会直接抛 `RuntimeError`（现已不会再发生） |
| 下载失败重试 | 单个文件下载失败（连接中断 / 5xx / 429 / 大小校验不符）自动重试 3 次并**续传**；4xx 等确定性错误不重试，避免把「文件不存在」放大成 3 倍等待 |
| 工作线程保活 | 所有 `QThread` 继承 `RetainedThread`，启动即登记强引用、结束自动释放。修复「转写进行中拖入新文件可再次点开始」导致的**并发启动 + 运行中线程被回收（Qt 直接崩溃）**，并消除关闭搜索框时线程被回收的同类风险 |

## 技术要点

- **引擎**：faster-whisper（CTranslate2 推理），int8 量化，VAD 过滤静音，
  `condition_on_previous_text=False` 抑制重复幻觉；
- **镜像站**：搜索走 `hf-mirror.com/api/models`，下载走
  `hf-mirror.com/<repo>/resolve/main/<file>`，支持 Range 断点续传；
- **界面**：QSS 实现苹果 HIG 风格——`#F5F5F7` 背景、白色 12px 圆角卡片、
  `#0A84FF` 系统蓝强调色、PingFang SC 字体、细窄滚动条；
- **线程模型**：转写与下载均在 QThread 中执行，通过信号逐段/逐块回报进度，
  界面不卡顿，支持随时取消；线程对象统一由 `RetainedThread` 保活，
  窗口/对话框先于线程销毁也不会触发 Qt 的「运行中析构」崩溃。

## 常见问题

- **Q: 模型下载很慢或失败？** 镜像站高峰期可能波动，程序支持断点续传，重试即可继续。
- **Q: 找不到已下载的模型？** 点击「刷新」重新扫描 `models/` 目录；
  只有含 `model.bin` 的目录才会被识别为完整模型。
- **Q: 想用 GPU？** 在「引擎」下拉框选择 NVIDIA GPU（需系统安装 cuBLAS/cuDNN CUDA 12）
  或 Apple Metal（需 `pip install mlx-whisper`）；默认「自动识别」会自动选最佳架构，
  CUDA 加载失败时自动回退 CPU。

## 关于 FFmpeg 的说明（无需手动安装）

视频/音频解码使用 **PyAV 自带的 FFmpeg 共享库**——安装包已内置，**无需用户
单独安装 FFmpeg 或配置环境变量**。双击即可转写视频。

- **两种引擎都用内置解码**：CPU/NVIDIA（faster-whisper）走 PyAV；
  Apple Metal（mlx-whisper）默认会调用系统 `ffmpeg` 命令行，本程序已改为
  先用内置 PyAV 解码为波形再送入模型，**同样无需安装 ffmpeg**；
- 启动转写前程序会自动预检解码器，不可用时弹窗给出明确指引；
- Windows 安装包构建时经 `tools/collect_native_libs.py` 将 FFmpeg DLL
  （以及 CUDA 运行库）嵌入 exe；macOS .app 由 PyInstaller 的 av 钩子打包 dylibs；
- 若出现"内置 FFmpeg 库加载失败"，说明安装包不完整，重新下载即可。
