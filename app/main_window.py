#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主窗口：导入媒体 → 选择模型 → 转写（单文件 / 批量文件夹）→ 预览 → 导出。"""
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QStatusBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    QHeaderView, QAbstractItemView, QCheckBox, QLineEdit,
)

from .config import ENGINES, MEDIA_FILTER
from .exporter import export_docx, export_markdown, export_srt, export_txt
from .theme import TEXT_SECOND
from .widgets import Card, DropArea, ModelSearchDialog, SettingsDialog
from .workers import (
    BatchTranscribeWorker, TranscribeWorker, check_media_decode,
    clear_model_cache, list_local_models, local_model_kind, resolve_model_path,
    scan_media_files,
)

LANGS = [("自动检测", "auto"), ("中文", "zh"), ("英语", "en"), ("日语", "ja"),
         ("韩语", "ko"), ("德语", "de"), ("法语", "fr"), ("西班牙语", "es")]


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastWhisper · 语音转文字")
        self.resize(920, 720)
        self.media_path = ""
        self.segments = []
        self.info = {}
        self.worker = None
        self.batch_worker = None
        self._batch_total = 0          # >0 表示批量模式进行中（用于状态文案）
        # 分段结果批量刷新的缓冲：避免逐段插入表格导致长音频卡顿
        self._pending_rows = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(80)
        self._flush_timer.timeout.connect(self._flush_rows)

        self._build_ui()
        self._restore_prefs()
        self.refresh_models()

    # ================= UI 构建 =================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 12)
        root.setSpacing(14)

        # ---- 标题区 ----
        title_box = QVBoxLayout()
        t = QLabel("FastWhisper")
        t.setObjectName("titleLabel")
        st = QLabel("本地语音转文字 · faster-whisper / mlx 引擎 · 模型经国内镜像站下载")
        st.setObjectName("subtitleLabel")
        title_box.addWidget(t)
        title_box.addWidget(st)

        # 标题行右侧：偏好设置入口
        header = QHBoxLayout()
        header.addLayout(title_box)
        header.addStretch(1)
        self.settings_btn = QPushButton("⚙ 偏好设置")
        self.settings_btn.setToolTip("设置模型存放目录与默认导出目录（持久化保存）")
        self.settings_btn.clicked.connect(self.open_settings)
        header.addWidget(self.settings_btn)
        root.addLayout(header)

        # ---- 1. 导入 ----
        card1 = Card()
        card1.add(self._section_label("① 导入媒体"))
        self.drop_area = DropArea(MEDIA_FILTER)
        self.drop_area.fileChosen.connect(self._on_file_chosen)
        card1.add(self.drop_area)
        root.addWidget(card1)

        # ---- 2. 设置 ----
        card2 = Card()
        card2.add(self._section_label("② 转写设置"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        grid.addWidget(self._field_label("模型"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        grid.addWidget(self.model_combo, 0, 1)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setToolTip("重新扫描 models/ 目录中的本地模型")
        self.refresh_btn.clicked.connect(self.refresh_models)
        grid.addWidget(self.refresh_btn, 0, 2)

        self.search_model_btn = QPushButton("搜索 / 下载模型…")
        self.search_model_btn.setToolTip("从国内镜像站 hf-mirror.com 检索并下载模型到项目 models/ 目录")
        self.search_model_btn.clicked.connect(self.open_model_dialog)
        grid.addWidget(self.search_model_btn, 0, 3)

        grid.addWidget(self._field_label("语言"), 1, 0)
        self.lang_combo = QComboBox()
        for name, code in LANGS:
            self.lang_combo.addItem(name, code)
        grid.addWidget(self.lang_combo, 1, 1)

        grid.addWidget(self._field_label("输出"), 1, 2)
        self.task_combo = QComboBox()
        self.task_combo.addItem("转写（原语言文字）", "transcribe")
        self.task_combo.addItem("翻译（译为英语）", "translate")
        grid.addWidget(self.task_combo, 1, 3)

        grid.addWidget(self._field_label("精细度"), 2, 0)
        self.beam_combo = QComboBox()
        self.beam_combo.addItem("快速（beam=1）", 1)
        self.beam_combo.addItem("均衡（beam=5）", 5)
        self.beam_combo.setCurrentIndex(1)
        grid.addWidget(self.beam_combo, 2, 1)

        self.vad_check = QCheckBox("过滤静音段（VAD）")
        self.vad_check.setChecked(True)
        self.vad_check.setToolTip("过滤静音片段，减少识别幻觉")
        grid.addWidget(self.vad_check, 2, 2, 1, 2)

        # ---- 加速选项行 ----
        grid.addWidget(self._field_label("引擎"), 3, 0)
        self.engine_combo = QComboBox()
        self._fill_engines()
        self.engine_combo.setToolTip(
            "自动：探测本机硬件（Apple 芯片→Metal，NVIDIA 显卡→CUDA，否则 CPU）；"
            "也可手动指定。CUDA 需系统安装 cuDNN")
        self.engine_combo.currentIndexChanged.connect(self.refresh_models)
        grid.addWidget(self.engine_combo, 3, 1)

        self.batched_check = QCheckBox("批量推理（更快，分段较粗）")
        self.batched_check.setToolTip(
            "VAD 切块后并行解码（CPU 实测约 1.8x）；分段粒度会变粗（按 30 秒窗口合并），"
            "仅对 CPU / NVIDIA 引擎生效；该模式必须开启 VAD 才能切块，故「过滤静音段」"
            "在本模式下强制生效")
        grid.addWidget(self.batched_check, 3, 2, 1, 2)

        # ---- 术语表（提升专有名词识别率，见下） ----
        grid.addWidget(self._field_label("术语表"), 4, 0)
        self.terms_edit = QLineEdit()
        self.terms_edit.setPlaceholderText("专有名词用顿号或逗号分隔，如：成都工业学院、德国管理应用技术大学")
        self.terms_edit.setToolTip(
            "填写人名、机构名、专业术语等专有名词，会作为提示词交给模型，"
            "显著降低专名错识别（实测可把「德国管理应用技术大学」从「国安利用技术大学」纠正回来）。\n"
            "首次转写某类素材时填入一次，之后会自动记住。")
        grid.addWidget(self.terms_edit, 4, 1, 1, 3)

        self.homophone_check = QCheckBox("按术语表纠正近音词")
        self.homophone_check.setChecked(True)
        self.homophone_check.setToolTip(
            "提示词是概率手段，同一段音频里可能只修好一部分（实测「德国管理应用技术大学」\n"
            "修好了、「生源质量」仍错成「声源质量」）。勾选后用拼音比对把剩余的同音错别字\n"
            "确定性改回术语表写法（仅当拼音完全一致才替换，不会误伤近形异音的词）。\n"
            "需要上面填写术语表；未安装 pypinyin 时本项自动失效。")
        grid.addWidget(self.homophone_check, 5, 1, 1, 3)

        card2.layout().addLayout(grid)
        root.addWidget(card2)

        # ---- 3. 转写 ----
        card3 = Card()
        action_row = QHBoxLayout()
        self.start_btn = QPushButton("开始转写")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_transcribe)
        action_row.addWidget(self.start_btn, 1)

        self.batch_btn = QPushButton("批量转写文件夹…")
        self.batch_btn.setToolTip(
            "选择一个文件夹，按文件名顺序转写其中所有视音频文件，"
            "逐个自动导出为 TXT 到导出目录；模型只加载一次")
        self.batch_btn.clicked.connect(self.start_batch_transcribe)
        action_row.addWidget(self.batch_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setToolTip("中止当前转写任务")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_job)
        action_row.addWidget(self.cancel_btn)
        card3.layout().addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # 转写阶段为未知总量 → 流水动画
        self.progress.setVisible(False)
        card3.add(self.progress)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("hintLabel")
        card3.add(self.status_label)
        root.addWidget(card3)

        # ---- 4. 结果 ----
        card4 = Card()
        card4.add(self._section_label("④ 转写结果"))

        self.tabs = QTabWidget()
        self.seg_table = QTableWidget(0, 3)
        self.seg_table.setHorizontalHeaderLabels(["开始", "结束", "文字内容"])
        self.seg_table.verticalHeader().setVisible(False)
        self.seg_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.seg_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.seg_table.setAlternatingRowColors(True)
        hh = self.seg_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.seg_table, "分段预览")

        self.plain_text = QPlainTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setPlaceholderText("转写完成后，此处显示合并后的纯文本…")
        self.tabs.addTab(self.plain_text, "纯文本")
        card4.add(self.tabs)

        export_row = QHBoxLayout()
        for text, handler in [
            ("导出 TXT", self.export_txt),
            ("导出 SRT 字幕", self.export_srt),
            ("导出 Markdown", self.export_md),
            ("导出 Word 文档", self.export_docx),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setEnabled(False)
            export_row.addWidget(btn)
            setattr(self, f"_export_btn_{handler.__name__}", btn)
        export_row.addStretch(1)
        card4.layout().addLayout(export_row)
        root.addWidget(card4, 1)

        # ---- 状态栏 ----
        sb = QStatusBar()
        sb.showMessage("提示：模型默认从 hf-mirror.com 镜像站下载，存放于项目 models/ 目录")
        self.setStatusBar(sb)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_SECOND}; font-size: 13px;")
        return lbl

    # ================= 文件导入 =================
    def _on_file_chosen(self, path: str):
        if not os.path.isfile(path):
            return
        self.media_path = path
        self.drop_area.set_file(path)
        # 必须走统一状态更新：不能直接置 True，否则转写进行中拖入新文件会重新
        # 启用「开始转写」，用户再点一次就会并发启动第二个任务并丢弃正在运行的
        # 线程对象（Qt 在 QThread 运行时析构会直接崩溃）。
        self._update_start_state()
        size_mb = os.path.getsize(path) / 1048576
        self.status_label.setText(f"已选择：{os.path.basename(path)}（{size_mb:.1f} MB）")

    # ================= 模型管理 =================
    def _fill_engines(self):
        """按本机硬件探测结果填充引擎下拉框。"""
        from . import hardware
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        self.engine_combo.addItem("自动识别最佳架构（推荐）", "auto")
        self.engine_combo.addItem(ENGINES["cpu"]["label"], "cpu")
        self.engine_combo.addItem(
            ENGINES["cuda"]["label"] if hardware.cuda_available()
            else "NVIDIA GPU（未检测到）",
            "cuda" if hardware.cuda_available() else "cuda-disabled")
        self.engine_combo.addItem(
            ENGINES["mlx"]["label"] if hardware.metal_available()
            else "Metal GPU（未安装 mlx-whisper）",
            "mlx" if hardware.metal_available() else "mlx-disabled")
        self.engine_combo.setCurrentIndex(0)   # 默认自动
        self.engine_combo.blockSignals(False)

    def _current_engine(self) -> str:
        """把下拉框选择解析为具体引擎；"auto" → 硬件探测的最佳架构。"""
        data = self.engine_combo.currentData()
        if data in ("cpu", "cuda", "mlx"):
            return data
        # auto 或不可用项 → 探测最佳引擎
        from . import hardware
        best = hardware.detect_best()
        return best if best in ("cpu", "cuda", "mlx") else "cpu"

    def _engine_kind(self, engine: str) -> str:
        return ENGINES.get(engine, ENGINES["cpu"])["kind"]

    def refresh_models(self):
        """按当前引擎（kind）扫描本地模型 + 附上对应内置推荐项（标注未下载）。"""
        from . import settings
        current = self.model_combo.currentData()
        engine = self._current_engine()
        kind = self._engine_kind(engine)
        models_dir = settings.get_models_dir()
        self.model_combo.clear()
        local = list_local_models(models_dir, kind=kind)
        for path in local:
            name = os.path.basename(path)
            self.model_combo.addItem(f"• {name}", path)
        for repo in ENGINES[engine]["builtin"]:
            path = resolve_model_path(repo, engine)   # 与预检共用同一处路径推导
            if path not in local:
                self.model_combo.addItem(f"○ {repo}（未下载）", repo)
        # 恢复选择
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if self.model_combo.count() and self.model_combo.currentData() is None:
            self.model_combo.setCurrentIndex(0)
        self._update_start_state()

    def open_model_dialog(self):
        # 传给对话框的架构 kind：cpu/cuda 共用 CTranslate2，mlx 用 MLX
        dlg = ModelSearchDialog(engine=self._engine_kind(self._current_engine()), parent=self)
        dlg.downloadFinished.connect(lambda *_: self.refresh_models())
        dlg.exec()

    def open_settings(self):
        """打开偏好设置；保存后刷新模型列表以反映新的模型目录。"""
        dlg = SettingsDialog(self)
        dlg.settingsSaved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self):
        from . import settings
        clear_model_cache()   # 模型目录可能已变，释放旧目录的已加载模型
        self.refresh_models()
        self.status_label.setText(
            f"设置已保存：模型目录 {settings.get_models_dir()}"
            + (f" · 导出目录 {settings.get_export_dir()}" if settings.get_export_dir() else ""))

    def _update_start_state(self):
        busy = ((self.worker is not None and self.worker.isRunning())
                or (self.batch_worker is not None and self.batch_worker.isRunning()))
        self.start_btn.setEnabled(
            bool(self.media_path) and bool(self.model_combo.currentData()) and not busy
        )
        self.batch_btn.setEnabled(bool(self.model_combo.currentData()) and not busy)

    # ================= 偏好持久化 =================
    def _restore_prefs(self):
        """恢复上次使用的引擎 / 语言 / 输出模式 / 精细度 / 选项。"""
        from . import settings

        def _select(combo, key, cast=str):
            raw = settings.get_ui(key, "")
            if not raw:
                return
            try:
                data = cast(raw)
            except (TypeError, ValueError):
                return
            idx = combo.findData(data)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _select(self.engine_combo, "engine")
        _select(self.lang_combo, "language")
        _select(self.task_combo, "task")
        _select(self.beam_combo, "beam", int)
        self.vad_check.setChecked(settings.get_ui("vad", "1") == "1")
        self.batched_check.setChecked(settings.get_ui("batched", "0") == "1")
        self.terms_edit.setText(settings.get_ui("terms", ""))
        self.homophone_check.setChecked(settings.get_ui("homophone", "1") == "1")

    def _save_prefs(self):
        """保存当前界面偏好，下次启动自动恢复。"""
        from . import settings
        settings.set_ui("engine", self.engine_combo.currentData())
        settings.set_ui("language", self.lang_combo.currentData())
        settings.set_ui("task", self.task_combo.currentData())
        settings.set_ui("beam", str(self.beam_combo.currentData()))
        settings.set_ui("vad", "1" if self.vad_check.isChecked() else "0")
        settings.set_ui("batched", "1" if self.batched_check.isChecked() else "0")
        settings.set_ui("terms", self.terms_edit.text().strip())
        settings.set_ui("homophone", "1" if self.homophone_check.isChecked() else "0")

    # ================= 转写 =================
    def _preflight(self):
        """公共预检：解码器可用性、引擎可用性、模型架构匹配。

        通过则返回 (model_sel, engine)；任一环节不满足返回 None（已弹窗提示）。
        """
        decode_problem = check_media_decode()
        if decode_problem:
            QMessageBox.critical(self, "解码器不可用", decode_problem)
            return None

        model_sel = self.model_combo.currentData()
        engine = self._current_engine()
        if not model_sel:
            QMessageBox.warning(self, "提示", "请先选择模型")
            return None

        if self.engine_combo.currentData() == "mlx-disabled":
            QMessageBox.information(
                self, "Metal 引擎未安装",
                "mlx-whisper 未安装到当前虚拟环境。\n"
                "安装命令：\n"
                ".venv/bin/pip install mlx-whisper")
            return None
        if self.engine_combo.currentData() == "cuda-disabled":
            QMessageBox.information(
                self, "未检测到 NVIDIA GPU",
                "未检测到可用的 CUDA 环境。\n"
                "需要 NVIDIA 显卡并安装 cuBLAS/cuDNN（CUDA 12），"
                "或改用 CPU / Metal 引擎。")
            return None

        # 统一解析为本地目录：内置推荐项（形如 Systran/faster-whisper-tiny）映射到
        # 模型目录下的 Systran__faster-whisper-tiny，避免把仓库 id 直接当路径传给
        # 推理引擎（那会让 faster-whisper 去联网下载）。
        want_kind = self._engine_kind(engine)
        path = resolve_model_path(model_sel, engine)
        kind = local_model_kind(path) if os.path.isdir(path) else None

        # 校验模型格式与引擎匹配（防止误用对方架构的模型）
        if kind and kind != want_kind:
            QMessageBox.warning(
                self, "模型架构不匹配",
                f"所选模型是 {kind.upper()} 格式，与当前引擎（{engine}）不匹配。\n"
                "请切换引擎，或在「搜索 / 下载模型」中下载对应架构的模型。")
            return None
        if kind is None:
            # 既不是本地模型目录，也不是内置推荐项（例如手动输入了非法值）
            QMessageBox.information(
                self, "需要先下载模型",
                "当前选中的模型尚未下载。\n请点击「搜索 / 下载模型…」，在列表中选中它并下载。")
            return None

        return path, engine

    def start_transcribe(self):
        if not self.media_path:
            QMessageBox.warning(self, "提示", "请先选择媒体文件")
            return
        pf = self._preflight()
        if not pf:
            return
        model_sel, engine = pf

        self._clear_results()
        self._batch_total = 0
        self._set_busy(True)
        # mlx 一次性返回、无流式进度 → 保持不确定动画；其余显示真实百分比
        if self._engine_kind(engine) == "mlx":
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status_label.setText("准备中…")

        self.worker = TranscribeWorker(
            self.media_path,
            model_sel,
            engine=engine,
            language=self.lang_combo.currentData(),
            task=self.task_combo.currentData(),
            beam_size=self.beam_combo.currentData(),
            use_vad=self.vad_check.isChecked(),
            batched=(self._engine_kind(engine) == "fw"
                     and self.batched_check.isChecked()),
            terms=self.terms_edit.text().strip(),
            fix_homophones=self.homophone_check.isChecked(),
        )
        self.worker.progress_text.connect(self.status_label.setText)
        self.worker.progress_pct.connect(self.progress.setValue)
        self.worker.segment_ready.connect(self._on_segment)
        self.worker.finished_ok.connect(self._on_transcribe_ok)
        self.worker.failed.connect(self._on_transcribe_fail)
        self.worker.start()

    # ================= 批量转写 =================
    def start_batch_transcribe(self):
        """选择文件夹 → 顺序转写其中所有媒体文件 → 逐个自动导出为 TXT。"""
        pf = self._preflight()
        if not pf:
            return
        model_sel, engine = pf

        from . import settings
        start_dir = settings.get_export_dir() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "选择要批量转写的文件夹", start_dir)
        if not folder:
            return
        files = scan_media_files(folder)
        if not files:
            QMessageBox.information(
                self, "未找到媒体文件",
                "该文件夹下没有可转写的视音频文件（仅扫描一级目录）。")
            return
        out_dir = settings.get_export_dir() or folder

        self._clear_results()
        self._batch_total = len(files)
        self._set_busy(True)
        self.progress.setRange(0, 100)   # 整批百分比（含当前文件内部进度）
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status_label.setText(f"批量转写 {len(files)} 个文件 → {out_dir}")

        self.batch_worker = BatchTranscribeWorker(
            files, model_sel, out_dir,
            engine=engine,
            language=self.lang_combo.currentData(),
            task=self.task_combo.currentData(),
            beam_size=self.beam_combo.currentData(),
            use_vad=self.vad_check.isChecked(),
            batched=(self._engine_kind(engine) == "fw"
                     and self.batched_check.isChecked()),
            terms=self.terms_edit.text().strip(),
            fix_homophones=self.homophone_check.isChecked(),
        )
        self.batch_worker.file_started.connect(self._on_batch_file_started)
        self.batch_worker.segment_ready.connect(self._on_segment)
        self.batch_worker.file_finished.connect(self._on_batch_file_finished)
        self.batch_worker.file_failed.connect(self._on_batch_file_failed)
        self.batch_worker.progress_text.connect(self.status_label.setText)
        self.batch_worker.progress_pct.connect(self.progress.setValue)
        self.batch_worker.finished_ok.connect(self._on_batch_done)
        self.batch_worker.start()

    def _on_batch_file_started(self, idx, total, path):
        self._clear_results()
        self.status_label.setText(
            f"[{idx}/{total}] 正在转写：{os.path.basename(path)}")

    def _on_batch_file_finished(self, idx, path, export_path):
        self.status_label.setText(f"✓ 已导出：{os.path.basename(export_path)}")

    def _on_batch_file_failed(self, idx, path, err):
        self.status_label.setText(f"✗ 失败：{os.path.basename(path)} — {err}")

    def _on_batch_done(self, done, total):
        self._flush_rows()
        self._batch_total = 0
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)   # 恢复单文件模式的流水动画
        self._set_busy(False)
        # 表格中保留最后一个文件的分段，允许手动再导出
        self._set_export_enabled(bool(self.segments))
        self.status_label.setText(f"批量转写完成：成功 {done} / {total} 个文件")

    def _clear_results(self):
        self._flush_timer.stop()
        self._pending_rows = []
        self.segments = []
        self.info = {}
        self.seg_table.setRowCount(0)
        self.plain_text.clear()
        self._set_export_enabled(False)

    def _on_segment(self, start: float, end: float, text: str):
        # 先入缓冲，由定时器按批刷新到表格，避免逐段插入造成长音频卡顿
        self.segments.append({"start": start, "end": end, "text": text})
        self._pending_rows.append((start, end, text))
        if len(self._pending_rows) >= 40:
            self._flush_rows()
        elif not self._flush_timer.isActive():
            self._flush_timer.start()
        # 段数文案随 _flush_rows 一起按批更新：逐段改 QLabel 文本会触发同样多次
        # 布局/重绘，长音频（数千段）下这条路径的控件开销比插入表格还高。

    def _flush_rows(self):
        """把缓冲的分段一次性写入表格（关闭重绘 → 批量插入 → 恢复重绘）。"""
        if not self._pending_rows:
            self._flush_timer.stop()
            return
        rows, self._pending_rows = self._pending_rows, []
        table = self.seg_table
        base = table.rowCount()
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(base + len(rows))
            for i, (start, end, text) in enumerate(rows):
                r = base + i
                for col, val in enumerate([_fmt(start), _fmt(end), text]):
                    item = QTableWidgetItem(val)
                    if col < 2:
                        item.setForeground(Qt.GlobalColor.gray)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(r, col, item)
        finally:
            table.setUpdatesEnabled(True)
        table.scrollToBottom()
        # 批量模式下由批量流程统一维护状态文案，避免来回覆盖
        if not self._batch_total:
            self.status_label.setText(f"正在转写… 已完成 {table.rowCount()} 段")

    def _on_transcribe_ok(self, segments, info):
        self._flush_rows()
        self.info = info or {}
        self.progress.setVisible(False)
        self._set_busy(False)
        dur = self.info.get("duration", 0)
        lang = self.info.get("language", "?")
        fixes = int(self.info.get("homophone_fixes", 0) or 0)
        self.status_label.setText(
            f"转写完成：共 {len(segments)} 段 · 音频时长 {dur / 60:.1f} 分钟 · 识别语言 {lang}"
            + (f" · 已按术语表纠正 {fixes} 处近音词" if fixes else ""))
        self.plain_text.setPlainText("\n".join(s["text"] for s in segments))
        self._set_export_enabled(bool(segments))

    def _on_transcribe_fail(self, msg):
        self._flush_rows()
        self.progress.setVisible(False)
        self._set_busy(False)
        self.status_label.setText(f"转写失败：{msg}")
        QMessageBox.critical(self, "转写失败", msg)

    def _set_busy(self, busy: bool):
        """统一切换忙碌态：忙碌时禁用启动按钮并显示取消按钮。"""
        self.cancel_btn.setVisible(busy)
        self.cancel_btn.setEnabled(busy)
        if busy:
            self.start_btn.setEnabled(False)
            self.batch_btn.setEnabled(False)
        else:
            self._update_start_state()

    def _cancel_job(self):
        for w in (self.worker, self.batch_worker):
            if w and w.isRunning():
                w.cancel()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("正在取消…")

    def _set_export_enabled(self, on: bool):
        for name in ("export_txt", "export_srt", "export_md", "export_docx"):
            btn = getattr(self, f"_export_btn_{name}", None)
            if btn:
                btn.setEnabled(on)

    # ================= 导出 =================
    def _save_path(self, ext: str, filter_text: str) -> str:
        # 默认导出位置：用户设置的导出目录优先，否则跟随媒体文件所在目录
        from . import settings
        base = os.path.splitext(os.path.basename(self.media_path))[0]
        default_dir = settings.get_export_dir() or (os.path.dirname(self.media_path) or ".")
        default = os.path.join(default_dir, f"{base}_转写稿.{ext}")
        path, _ = QFileDialog.getSaveFileName(self, "导出", default, filter_text)
        # 用户未填扩展名时按当前格式自动补全
        if path and not os.path.splitext(path)[1]:
            path = f"{path}.{ext}"
        return path

    def _run_export(self, fn):
        """执行导出并统一处理异常（磁盘满 / 权限 / 缺依赖），避免崩溃。"""
        try:
            path = fn()
        except ImportError as e:
            QMessageBox.critical(
                self, "导出失败",
                f"缺少依赖库：{e}\n（导出 Word 文档需要 python-docx）")
            return
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{type(e).__name__}: {e}")
            return
        self._export_done(path)

    def export_txt(self):
        path = self._save_path("txt", "文本文件 (*.txt)")
        if path:
            self._run_export(lambda: export_txt(self.segments, path))

    def export_srt(self):
        path = self._save_path("srt", "SRT 字幕 (*.srt)")
        if path:
            self._run_export(lambda: export_srt(self.segments, path))

    def export_md(self):
        path = self._save_path("md", "Markdown (*.md)")
        if path:
            self._run_export(lambda: export_markdown(self.segments, path, self._title()))

    def export_docx(self):
        path = self._save_path("docx", "Word 文档 (*.docx)")
        if path:
            self._run_export(lambda: export_docx(self.segments, path, self._title()))

    def _title(self) -> str:
        base = os.path.splitext(os.path.basename(self.media_path))[0] if self.media_path else "语音转写稿"
        return f"{base} 语音转写稿"

    def _export_done(self, path: str):
        self.status_label.setText(f"已导出：{path}")
        QMessageBox.information(self, "导出成功", f"文件已保存到：\n{path}")

    # ================= 关闭 =================
    def closeEvent(self, event):
        for w in (self.worker, self.batch_worker):
            if w and w.isRunning():
                w.cancel()
                w.wait(1500)
        self._save_prefs()
        event.accept()
