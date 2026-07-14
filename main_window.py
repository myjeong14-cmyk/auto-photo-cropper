"""증명사진 자동 편집 프로그램 메인 윈도우"""
import os

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QFileDialog, QMessageBox, QStatusBar, QGroupBox
)

from auto_processor import auto_process, AutoProcessError
from crop_editor import CropEditor
from settings import AppSettings
from utils import generate_output_filename, is_supported_image, ensure_dir


class AutoProcessWorker(QObject):
    """백그라운드 스레드에서 자동 처리(auto_process)를 실행하는 워커"""

    progress = Signal(int, int, str)              # current, total, filename
    item_done = Signal(str, object, object, str)  # path, before_img, after_img, method
    item_failed = Signal(str, str)                # path, error message
    finished = Signal()

    def __init__(self, paths):
        super().__init__()
        self._paths = paths

    def run(self):
        total = len(self._paths)
        for i, path in enumerate(self._paths, start=1):
            self.progress.emit(i, total, os.path.basename(path))
            try:
                before_img, after_img, method = auto_process(path)
                self.item_done.emit(path, before_img, after_img, method or "")
            except AutoProcessError as e:
                self.item_failed.emit(path, str(e))
            except Exception as e:
                self.item_failed.emit(path, f"알 수 없는 오류: {e}")
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("증명사진 자동 편집 프로그램")
        self.resize(1150, 720)
        self.setAcceptDrops(True)

        self.settings = AppSettings()
        self.save_folder = self.settings.get_last_save_folder()

        self._images = {}          # path -> {"before": PIL.Image, "after": PIL.Image}
        self._current_path = None
        self._current_mode = "after"   # "before" 또는 "after"

        self._thread = None
        self._worker = None

        self._build_ui()
        self._setup_shortcuts()

    # ---------------- UI 구성 ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # 왼쪽: 파일 목록 패널
        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)

        self.add_btn = QPushButton("이미지 추가 (Ctrl+N)")
        self.add_btn.clicked.connect(self.on_add_images)
        left_layout.addWidget(self.add_btn)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)

        drop_hint = QLabel("이미지를 이 창에 드래그하여\n추가할 수도 있습니다.")
        drop_hint.setAlignment(Qt.AlignCenter)
        drop_hint.setStyleSheet("color: gray;")
        left_layout.addWidget(drop_hint)

        # 가운데: 편집 영역
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        mode_box = QGroupBox("보기 모드")
        mode_layout = QHBoxLayout(mode_box)
        self.before_btn = QPushButton("이전 (자동 편집 전) (Ctrl+←)")
        self.after_btn = QPushButton("다음 (자동 편집 후) (Ctrl+→)")
        self.before_btn.clicked.connect(lambda: self.switch_mode("before"))
        self.after_btn.clicked.connect(lambda: self.switch_mode("after"))
        mode_layout.addWidget(self.before_btn)
        mode_layout.addWidget(self.after_btn)
        center_layout.addWidget(mode_box)

        self.editor = CropEditor()
        center_layout.addWidget(self.editor, stretch=1)

        edit_box = QGroupBox("편집")
        edit_layout = QHBoxLayout(edit_box)
        self.rotate_left_btn = QPushButton("왼쪽 90도 회전 (Ctrl+Shift+←)")
        self.rotate_right_btn = QPushButton("오른쪽 90도 회전 (Ctrl+Shift+→)")
        self.clear_sel_btn = QPushButton("선택 영역 초기화 (Ctrl+Z)")
        self.rotate_left_btn.clicked.connect(self.editor.rotate_left)
        self.rotate_right_btn.clicked.connect(self.editor.rotate_right)
        self.clear_sel_btn.clicked.connect(self.editor.clear_selection)
        edit_layout.addWidget(self.rotate_left_btn)
        edit_layout.addWidget(self.rotate_right_btn)
        edit_layout.addWidget(self.clear_sel_btn)
        center_layout.addWidget(edit_box)

        # 오른쪽: 저장 설정 패널
        right_panel = QWidget()
        right_panel.setFixedWidth(260)
        right_layout = QVBoxLayout(right_panel)

        save_box = QGroupBox("저장 설정")
        save_layout = QVBoxLayout(save_box)
        self.folder_label = QLabel(self.save_folder if self.save_folder else "(저장 폴더 미지정)")
        self.folder_label.setWordWrap(True)
        self.folder_btn = QPushButton("저장 폴더 선택 (Ctrl+E)")
        self.folder_btn.clicked.connect(self.on_select_folder)
        self.apply_btn = QPushButton("적용 (저장) (Ctrl+S)")
        self.apply_btn.setStyleSheet("font-weight: bold;")
        self.apply_btn.clicked.connect(self.on_apply)
        self.save_as_btn = QPushButton("다른 이름으로 저장 (Ctrl+Shift+S)")
        self.save_as_btn.clicked.connect(self.on_save_as)
        save_layout.addWidget(self.folder_label)
        save_layout.addWidget(self.folder_btn)
        save_layout.addWidget(self.apply_btn)
        save_layout.addWidget(self.save_as_btn)
        right_layout.addWidget(save_box)
        right_layout.addStretch(1)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(center_panel, stretch=1)
        root_layout.addWidget(right_panel)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("준비됨")

    # ---------------- 단축키 ----------------
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self.on_add_images)
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=lambda: self.switch_mode("before"))
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=lambda: self.switch_mode("after"))
        QShortcut(QKeySequence("Ctrl+Shift+Left"), self, activated=self.editor.rotate_left)
        QShortcut(QKeySequence("Ctrl+Shift+Right"), self, activated=self.editor.rotate_right)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.editor.clear_selection)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.on_apply)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.on_select_folder)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.on_save_as)

    # ---------------- 드래그 앤 드롭 ----------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and is_supported_image(path):
                paths.append(path)
        if paths:
            self._start_auto_process(paths)
        else:
            QMessageBox.warning(self, "알림", "지원하는 JPEG 파일이 없습니다.")

    # ---------------- 이미지 추가 ----------------
    def on_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "이미지 선택", "", "JPEG 이미지 (*.jpg *.jpeg)"
        )
        if paths:
            self._start_auto_process(paths)

    def _start_auto_process(self, paths):
        self.add_btn.setEnabled(False)
        self.status_bar.showMessage("자동 처리 중...")

        self._thread = QThread()
        self._worker = AutoProcessWorker(paths)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.item_done.connect(self.on_item_done)
        self._worker.item_failed.connect(self.on_item_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_all_done)

        self._thread.start()

    def on_progress(self, current, total, filename):
        self.status_bar.showMessage(f"자동 처리 중... ({current}/{total}) {filename}")

    def on_item_done(self, path, before_img, after_img, method):
        base_method, _, detail = (method or "").partition("|")
        self._images[path] = {"before": before_img, "after": after_img, "method": base_method}
        label = os.path.basename(path) + self._method_tag(base_method)
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, path)
        if detail:
            item.setToolTip(detail)
        self.file_list.addItem(item)
        if self._current_path is None:
            self.file_list.setCurrentItem(item)
            self.on_file_selected(item)
        status_msg = f"{os.path.basename(path)} - {self._method_label(base_method)}"
        if detail:
            status_msg += f"  |  {detail}"
        self.status_bar.showMessage(status_msg)

    @staticmethod
    def _method_label(method):
        return {
            "yunet": "얼굴인식(YuNet) 크롭 완료",
            "haar": "얼굴인식(Haar, 저정확도) 크롭 완료",
            "border": "얼굴 미인식 - 여백/테두리 크롭 사용",
            "raw": "자동 크롭 실패 - 원본 유지",
        }.get(method, "처리 완료")

    @staticmethod
    def _method_tag(method):
        return {
            "yunet": "  [얼굴인식]",
            "haar": "  [얼굴인식-저정확도]",
            "border": "  [여백인식]",
            "raw": "  [실패]",
        }.get(method, "")

    def on_item_failed(self, path, message):
        QMessageBox.warning(
            self, "처리 오류",
            f"'{os.path.basename(path)}' 처리 중 오류가 발생했습니다.\n{message}"
        )

    def _on_all_done(self):
        self.add_btn.setEnabled(True)
        self.status_bar.showMessage("준비됨")

    # ---------------- 파일 선택 / 모드 전환 ----------------
    def on_file_selected(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        self._current_path = path
        self._current_mode = "after"
        self._load_current_into_editor()

    def switch_mode(self, mode):
        if self._current_path is None:
            return
        self._current_mode = mode
        self._load_current_into_editor()

    def _load_current_into_editor(self):
        data = self._images.get(self._current_path)
        if data is None:
            return
        img = data["before"] if self._current_mode == "before" else data["after"]
        self.editor.load_image(img)

    # ---------------- 저장 폴더 ----------------
    def on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", self.save_folder or "")
        if folder:
            self.save_folder = folder
            self.folder_label.setText(folder)
            self.settings.set_last_save_folder(folder)

    # ---------------- 적용(저장) ----------------
    def on_apply(self):
        if self._current_path is None:
            QMessageBox.warning(self, "알림", "저장할 이미지가 없습니다.")
            return
        if not self.save_folder:
            QMessageBox.warning(self, "알림", "먼저 저장 폴더를 선택하세요.")
            return

        result_img = self.editor.get_result_image()
        if result_img is None:
            QMessageBox.warning(self, "알림", "저장할 이미지가 없습니다.")
            return

        try:
            ensure_dir(self.save_folder)
            filename = generate_output_filename(self._current_path)
            out_path = os.path.join(self.save_folder, filename)
            result_img.convert("RGB").save(out_path, "JPEG", quality=95)
            self.status_bar.showMessage(f"저장 완료: {out_path}")
            QMessageBox.information(self, "저장 완료", f"저장되었습니다:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"저장 중 오류가 발생했습니다.\n{e}")

    def on_save_as(self):
        """다른 이름으로 저장: 이번 저장에 한해 파일명/위치를 직접 지정 (기본 저장 폴더 설정은 변경하지 않음)"""
        if self._current_path is None:
            QMessageBox.warning(self, "알림", "저장할 이미지가 없습니다.")
            return

        result_img = self.editor.get_result_image()
        if result_img is None:
            QMessageBox.warning(self, "알림", "저장할 이미지가 없습니다.")
            return

        default_name = generate_output_filename(self._current_path)
        default_dir = self.save_folder or ""
        default_path = os.path.join(default_dir, default_name) if default_dir else default_name

        out_path, _ = QFileDialog.getSaveFileName(
            self, "다른 이름으로 저장", default_path, "JPEG 이미지 (*.jpg *.jpeg)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith((".jpg", ".jpeg")):
            out_path += ".jpg"

        try:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                ensure_dir(out_dir)
            result_img.convert("RGB").save(out_path, "JPEG", quality=95)
            self.status_bar.showMessage(f"저장 완료: {out_path}")
            QMessageBox.information(self, "저장 완료", f"저장되었습니다:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"저장 중 오류가 발생했습니다.\n{e}")
