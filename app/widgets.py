#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复用界面组件：圆角卡片、拖放导入区、模型搜索对话框。"""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QDialog,
    QHeaderView, QMessageBox, QAbstractItemView, QComboBox,
)

from . import model_manager
from .theme import ACCENT, BORDER, TEXT, TEXT_SECOND


class Card(QFrame):
    """白色圆角卡片容器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(18, 16, 18, 16)
        self._lay.setSpacing(10)

    def layout(self) -> QVBoxLayout:
        return self._lay

    def add(self, widget):
        self._lay.addWidget(widget)
        return widget


class DropArea(QFrame):
    """拖放导入区：支持拖入媒体文件或点击选择。"""
    fileChosen = Signal(str)

    def __init__(self, media_filter: str, parent=None):
        super().__init__(parent)
        self.media_filter = media_filter
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_style(False)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(6)
        self.icon_label = QLabel("⬆︎")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 24px; color: %s;" % ACCENT)
        self.text_label = QLabel("拖入视音频文件，或点击选择")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"font-size: 13px; color: {TEXT_SECOND};")
        self.file_label = QLabel("")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet(
            f"font-size: 13px; color: {TEXT}; font-weight: 600;")
        self.file_label.setWordWrap(True)
        lay.addWidget(self.icon_label)
        lay.addWidget(self.text_label)
        lay.addWidget(self.file_label)

    def _init_style(self, active: bool):
        if active:
            self.setStyleSheet(
                f"QFrame {{ background: #F0F7FF; border: 1.5px dashed {ACCENT};"
                f" border-radius: 10px; }}")
        else:
            self.setStyleSheet(
                f"QFrame {{ background: #FAFAFC; border: 1.5px dashed {BORDER};"
                f" border-radius: 10px; }}")

    def set_file(self, path: str):
        self.file_label.setText(path)
        self.text_label.setText("已选择文件（点击可更换）")

    def clear_file(self):
        self.file_label.setText("")
        self.text_label.setText("拖入视音频文件，或点击选择")

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(self, "选择媒体文件", os.path.expanduser("~"),
                                              self.media_filter)
        if path:
            self.fileChosen.emit(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._init_style(True)

    def dragLeaveEvent(self, event):
        self._init_style(False)

    def dropEvent(self, event):
        self._init_style(False)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.fileChosen.emit(path)
                break


class ModelSearchDialog(QDialog):
    """模型搜索与下载对话框（走国内镜像站 hf-mirror.com）。

    按引擎架构过滤结果：cpu/cuda 只显示 CTranslate2 格式，mlx 只显示 MLX 格式。
    """
    downloadFinished = Signal(str, str)   # repo_id, local_dir

    def __init__(self, engine: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("搜索并下载模型 — hf-mirror.com")
        self.resize(720, 560)
        self._worker = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(12)

        # 架构选择行：决定搜索过滤方向
        arch_row = QHBoxLayout()
        arch_row.addWidget(QLabel("模型架构："))
        self.arch_combo = QComboBox()
        self.arch_combo.addItem("CPU / NVIDIA GPU（CTranslate2 格式）", "cpu")
        self.arch_combo.addItem("Apple Metal（MLX 格式）", "mlx")
        # 按主窗口传入的引擎预选架构
        if engine == "mlx":
            self.arch_combo.setCurrentIndex(1)
        self.arch_combo.currentIndexChanged.connect(self._on_arch_changed)
        arch_row.addWidget(self.arch_combo, 1)
        lay.addLayout(arch_row)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入关键词，如 faster-whisper、whisper-small…")
        self.search_btn = QPushButton("搜索")
        self.search_btn.setObjectName("primaryButton")
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.search_btn)
        lay.addLayout(search_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["模型", "下载量", "收藏", "状态"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)

        self.status_label = QLabel("提示：搜索结果已按所选架构过滤——CPU/NVIDIA 选 CTranslate2 格式，Apple 选 MLX 格式")
        self.status_label.setStyleSheet(f"color: {TEXT_SECOND}; font-size: 12px;")
        lay.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.download_btn = QPushButton("下载选中模型")
        self.download_btn.setObjectName("primaryButton")
        self.cancel_btn = QPushButton("取消下载")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setVisible(False)
        self.close_btn = QPushButton("关闭")
        btn_row.addWidget(self.download_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        lay.addLayout(btn_row)

        self.search_btn.clicked.connect(self.do_search)
        self.search_edit.returnPressed.connect(self.do_search)
        self.download_btn.clicked.connect(self.do_download)
        self.cancel_btn.clicked.connect(self._cancel_download)
        self.close_btn.clicked.connect(self.reject)
        self.search_edit.setText(self._default_query())

    def _arch_kind(self) -> str:
        """对话框内的架构选择映射为引擎 kind：cpu/cuda → cpu 项；mlx → mlx 项。"""
        return self.arch_combo.currentData()

    def _default_query(self) -> str:
        return "mlx whisper" if self._arch_kind() == "mlx" else "faster-whisper"

    def _on_arch_changed(self):
        self.search_edit.setText(self._default_query())
        self.table.setRowCount(0)

    # ----- 搜索 -----
    def do_search(self):
        query = self.search_edit.text().strip() or "whisper"
        arch = self._arch_kind()
        # mlx 架构在搜索 API 层面用 mlx 过滤，cpu 架构沿用原关键词并后置过滤
        api_query = query if arch != "mlx" else f"{query} mlx" if "mlx" not in query.lower() else query
        self.status_label.setText(f"正在搜索“{api_query}”…")
        self.search_btn.setEnabled(False)
        try:
            results = model_manager.search_models(api_query, engine=arch)
        except Exception as e:
            self.status_label.setText(f"搜索失败：{e}（请检查网络）")
            self.search_btn.setEnabled(True)
            return
        self.search_btn.setEnabled(True)
        fmt = "MLX" if arch == "mlx" else "CTranslate2"
        self.status_label.setText(
            f"找到 {len(results)} 个模型（按下载量排序，{fmt} 格式优先）")
        self._fill_table(results)

    def _fill_table(self, results):
        self.table.setRowCount(len(results))
        for i, m in enumerate(results):
            status = "✓ 已下载" if m.is_downloaded else ""
            item_status = QTableWidgetItem(status)
            if m.is_downloaded:
                item_status.setForeground(Qt.GlobalColor.darkGreen)
            self.table.setItem(i, 0, QTableWidgetItem(m.repo_id))
            self.table.setItem(i, 1, QTableWidgetItem(f"{m.downloads:,}"))
            self.table.setItem(i, 2, QTableWidgetItem(str(m.likes)))
            self.table.setItem(i, 3, item_status)

    def _selected_repo(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在列表中选中一个模型")
            return ""
        return self.table.item(row, 0).text()

    # ----- 下载 -----
    def do_download(self):
        repo_id = self._selected_repo()
        if not repo_id:
            return
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.download_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.status_label.setText(f"准备下载 {repo_id} …")

        from .workers import DownloadWorker
        self._worker = DownloadWorker(repo_id)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_download_ok)
        self._worker.failed.connect(self._on_download_fail)
        self._worker.start()

    def _on_progress(self, filename, done, total):
        if total > 0:
            self.progress.setValue(int(done * 100 / total))
        self.status_label.setText(f"下载中：{filename}  {done / 1048576:.1f} / {total / 1048576:.1f} MB")

    def _on_download_ok(self, local_dir):
        repo_id = self._worker.repo_id
        self.progress.setValue(100)
        self.status_label.setText(f"下载完成：{local_dir}")
        self._reset_buttons()
        self.downloadFinished.emit(repo_id, local_dir)
        QMessageBox.information(self, "完成", f"模型已下载到：\n{local_dir}")

    def _on_download_fail(self, msg):
        self.status_label.setText(f"下载失败：{msg}")
        self._reset_buttons()

    def _cancel_download(self):
        if self._worker:
            self._worker.cancel()

    def _reset_buttons(self):
        self.download_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        event.accept()


class SettingsDialog(QDialog):
    """偏好设置对话框：模型存放目录、默认导出目录（持久化保存）。"""

    settingsSaved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import settings
        self.setWindowTitle("偏好设置")
        self.resize(640, 240)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(12)

        hint = QLabel("设置会立即保存，并在本机所有会话中生效。")
        hint.setStyleSheet(f"color: {TEXT_SECOND}; font-size: 12px;")
        lay.addWidget(hint)

        # --- 模型存放目录 ---
        lay.addWidget(QLabel("模型存放目录"))
        row1 = QHBoxLayout()
        self.models_edit = QLineEdit(settings.get_models_dir())
        self.models_edit.setToolTip("faster-whisper 与 MLX 模型统一存放在此目录下，每个模型一个子目录")
        btn1 = QPushButton("浏览…")
        btn1.clicked.connect(lambda: self._browse(self.models_edit))
        row1.addWidget(self.models_edit, 1)
        row1.addWidget(btn1)
        lay.addLayout(row1)

        # --- 默认导出目录 ---
        lay.addWidget(QLabel("默认导出目录（留空 = 跟随媒体文件所在目录）"))
        row2 = QHBoxLayout()
        self.export_edit = QLineEdit(settings.get_export_dir())
        btn2 = QPushButton("浏览…")
        btn2.clicked.connect(lambda: self._browse(self.export_edit))
        row2.addWidget(self.export_edit, 1)
        row2.addWidget(btn2)
        lay.addLayout(row2)

        lay.addStretch(1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset)
        save_btn = QPushButton("保存")
        save_btn.setObjectName("primaryButton")
        cancel_btn = QPushButton("取消")
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)

    def _browse(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or os.path.expanduser("~"))
        if path:
            edit.setText(path)

    def _save(self):
        from . import settings
        settings.set_models_dir(self.models_edit.text().strip())
        settings.set_export_dir(self.export_edit.text().strip())
        self.settingsSaved.emit()
        self.accept()

    def _reset(self):
        from . import settings
        self.models_edit.setText(settings.default_models_dir())
        self.export_edit.setText("")
