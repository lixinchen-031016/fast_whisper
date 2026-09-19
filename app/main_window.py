#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主窗口：导入媒体 → 选择模型 → 转写 → 预览 → 导出。"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QStatusBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    QHeaderView, QAbstractItemView, QCheckBox,
)

from .config import FW_BUILTIN_MODELS, MEDIA_FILTER, MLX_BUILTIN_MODELS, MODELS_DIR
from .exporter import (
    default_export_path, export_docx, export_markdown, export_srt, export_txt,
)
from .theme import TEXT_SECOND
from .widgets import Card, DropArea, ModelSearchDialog
from .workers import TranscribeWorker, list_local_models, local_model_kind, mlx_available

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

        self._build_ui()
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
        st = QLabel("本地语音转文字 · faster-whisper 引擎 · 模型经国内镜像站下载")
        st.setObjectName("subtitleLabel")
        title_box.addWidget(t)
        title_box.addWidget(st)
        root.addLayout(title_box)

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
        self.engine_combo.addItem("CPU（faster-whisper）", "fw")
        if mlx_available():
            self.engine_combo.addItem("Metal GPU（mlx-whisper）", "mlx")
        else:
            self.engine_combo.addItem("Metal GPU（mlx-whisper，未安装）", "mlx-disabled")
        self.engine_combo.setToolTip(
            "CPU：通用稳定；Metal：Apple 芯片 GPU 加速，需安装 mlx-whisper 且模型为 MLX 格式")
        self.engine_combo.currentIndexChanged.connect(self.refresh_models)
        grid.addWidget(self.engine_combo, 3, 1)

        self.batched_check = QCheckBox("批量推理（CPU 引擎更快，实测约 1.8x）")
        self.batched_check.setToolTip(
            "VAD 切块后并行解码；分段粒度会变粗（按 30 秒窗口合并），仅对 CPU 引擎生效")
        grid.addWidget(self.batched_check, 3, 2, 1, 2)

        card2.layout().addLayout(grid)
        root.addWidget(card2)

        # ---- 3. 转写 ----
        card3 = Card()
        self.start_btn = QPushButton("开始转写")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_transcribe)
        card3.add(self.start_btn)

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
        self.start_btn.setEnabled(bool(self.model_combo.currentData()))
        size_mb = os.path.getsize(path) / 1048576
        self.status_label.setText(f"已选择：{os.path.basename(path)}（{size_mb:.1f} MB）")

    # ================= 模型管理 =================
    def _current_engine(self) -> str:
        data = self.engine_combo.currentData()
        return data if data in ("fw", "mlx") else "fw"

    def _builtin_list(self, engine: str) -> list:
        return MLX_BUILTIN_MODELS if engine == "mlx" else FW_BUILTIN_MODELS

    def refresh_models(self):
        """按当前引擎扫描本地模型 + 附上对应内置推荐项（标注未下载）。"""
        current = self.model_combo.currentData()
        engine = self._current_engine()
        self.model_combo.clear()
        local = list_local_models(MODELS_DIR, kind=engine)
        for path in local:
            name = os.path.basename(path)
            self.model_combo.addItem(f"• {name}", path)
        for repo in self._builtin_list(engine):
            path = os.path.join(MODELS_DIR, repo.replace("/", "__"))
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
        dlg = ModelSearchDialog(self)
        dlg.downloadFinished.connect(lambda *_: self.refresh_models())
        dlg.exec()

    def _update_start_state(self):
        self.start_btn.setEnabled(
            bool(self.media_path) and bool(self.model_combo.currentData())
            and (self.worker is None or not self.worker.isRunning())
        )

    # ================= 转写 =================
    def start_transcribe(self):
        model_sel = self.model_combo.currentData()
        engine = self._current_engine()
        if not self.media_path or not model_sel:
            QMessageBox.warning(self, "提示", "请先选择媒体文件与模型")
            return

        if self.engine_combo.currentData() == "mlx-disabled":
            QMessageBox.information(
                self, "Metal 引擎未安装",
                "mlx-whisper 未安装到当前虚拟环境。\n"
                "安装命令：\n"
                ".venv/bin/pip install mlx-whisper")
            return

        # 内置未下载模型 → 引导先下载
        if model_sel in self._builtin_list(engine) and not os.path.isdir(model_sel):
            QMessageBox.information(
                self, "需要先下载模型",
                "当前选中的模型尚未下载。\n请点击「搜索 / 下载模型…」，在列表中选中它并下载。")
            return

        self._clear_results()
        self.start_btn.setEnabled(False)
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
            batched=(engine == "fw" and self.batched_check.isChecked()),
        )
        self.worker.model_loading.connect(lambda: None)
        self.worker.progress_text.connect(self.status_label.setText)
        self.worker.segment_ready.connect(self._on_segment)
        self.worker.finished_ok.connect(self._on_transcribe_ok)
        self.worker.failed.connect(self._on_transcribe_fail)
        self.worker.start()

    def _clear_results(self):
        self.segments = []
        self.info = {}
        self.seg_table.setRowCount(0)
        self.plain_text.clear()
        self._set_export_enabled(False)

    def _on_segment(self, start: float, end: float, text: str):
        self.segments.append({"start": start, "end": end, "text": text})
        row = self.seg_table.rowCount()
        self.seg_table.insertRow(row)
        for col, val in enumerate([_fmt(start), _fmt(end), text]):
            item = QTableWidgetItem(val)
            if col < 2:
                item.setForeground(Qt.GlobalColor.gray)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.seg_table.setItem(row, col, item)
        self.seg_table.scrollToBottom()
        self.status_label.setText(f"正在转写… 已完成 {row + 1} 段")

    def _on_transcribe_ok(self, segments, info):
        self.info = info or {}
        self.progress.setVisible(False)
        self._update_start_state()
        dur = self.info.get("duration", 0)
        lang = self.info.get("language", "?")
        self.status_label.setText(
            f"转写完成：共 {len(segments)} 段 · 音频时长 {dur / 60:.1f} 分钟 · 识别语言 {lang}")
        self.plain_text.setPlainText("\n".join(s["text"] for s in segments))
        self._set_export_enabled(bool(segments))

    def _on_transcribe_fail(self, msg):
        self.progress.setVisible(False)
        self._update_start_state()
        self.status_label.setText(f"转写失败：{msg}")
        QMessageBox.critical(self, "转写失败", msg)

    def _set_export_enabled(self, on: bool):
        for name in ("export_txt", "export_srt", "export_md", "export_docx"):
            btn = getattr(self, f"_export_btn_{name}", None)
            if btn:
                btn.setEnabled(on)

    # ================= 导出 =================
    def _save_path(self, ext: str, filter_text: str) -> str:
        default = default_export_path(self.media_path, ext)
        path, _ = QFileDialog.getSaveFileName(self, "导出", default, filter_text)
        return path

    def export_txt(self):
        path = self._save_path("txt", "文本文件 (*.txt)")
        if path:
            export_txt(self.segments, path)
            self._export_done(path)

    def export_srt(self):
        path = self._save_path("srt", "SRT 字幕 (*.srt)")
        if path:
            export_srt(self.segments, path)
            self._export_done(path)

    def export_md(self):
        path = self._save_path("md", "Markdown (*.md)")
        if path:
            export_markdown(self.segments, path, self._title())
            self._export_done(path)

    def export_docx(self):
        path = self._save_path("docx", "Word 文档 (*.docx)")
        if path:
            export_docx(self.segments, path, self._title())
            self._export_done(path)

    def _title(self) -> str:
        base = os.path.splitext(os.path.basename(self.media_path))[0] if self.media_path else "语音转写稿"
        return f"{base} 语音转写稿"

    def _export_done(self, path: str):
        self.status_label.setText(f"已导出：{path}")
        QMessageBox.information(self, "导出成功", f"文件已保存到：\n{path}")

    # ================= 关闭 =================
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1500)
        event.accept()
