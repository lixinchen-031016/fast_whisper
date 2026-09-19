#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复用界面组件：圆角卡片、拖放导入区、模型搜索对话框。"""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QDialog,
    QHeaderView, QMessageBox, QAbstractItemView,
)

from . import model_manager
from .theme import ACCENT, BORDER, CARD, DANGER, TEXT, TEXT_SECOND


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
    """模型搜索与下载对话框（走国内镜像站 hf-mirror.com）。"""
    downloadFinished = Signal(str, str)   # repo_id, local_dir

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("搜索并下载模型 — hf-mirror.com")
        self.resize(720, 520)
        self._worker = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(12)

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

        self.status_label = QLabel("提示：优先选择名称含 faster-whisper 的 CTranslate2 格式模型")
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
        self.search_edit.setText("faster-whisper")

    # ----- 搜索 -----
    def do_search(self):
        query = self.search_edit.text().strip() or "whisper"
        self.status_label.setText(f"正在搜索“{query}”…")
        self.search_btn.setEnabled(False)
        try:
            results = model_manager.search_models(query)
        except Exception as e:
            self.status_label.setText(f"搜索失败：{e}（请检查网络）")
            self.search_btn.setEnabled(True)
            return
        self.search_btn.setEnabled(True)
        self.status_label.setText(f"找到 {len(results)} 个模型（按下载量排序，优先 CTranslate2 格式）")
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
