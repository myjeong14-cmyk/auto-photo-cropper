"""수동 Crop / 회전 편집 위젯"""
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, QRect, QPoint, QSize
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QRubberBand, QVBoxLayout, QSizePolicy

from utils import pil_to_qpixmap


class ImageCanvas(QLabel):
    """이미지를 표시하고 마우스 드래그로 Crop 영역을 선택하는 QLabel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #2b2b2b;")

        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = QPoint()
        self._selection = QRect()
        self._pixmap: Optional[QPixmap] = None
        self._display_rect = QRect()

    def set_pixmap_fitted(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._selection = QRect()
        self._rubber_band.hide()
        self._update_display()

    def _update_display(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return
        self._origin = event.position().toPoint()
        self._rubber_band.setGeometry(QRect(self._origin, QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pixmap is None or not self._rubber_band.isVisible():
            return
        rect = QRect(self._origin, event.position().toPoint()).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return
        rect = self._rubber_band.geometry()
        if rect.width() < 5 or rect.height() < 5:
            self._selection = QRect()
            self._rubber_band.hide()
        else:
            self._selection = rect

    def get_selection_in_image_coords(self) -> Optional[QRect]:
        """화면 좌표의 선택 영역을 원본 이미지 픽셀 좌표로 변환"""
        if self._pixmap is None or self._selection.isNull() or self._display_rect.isNull():
            return None

        sel = self._selection.intersected(self._display_rect)
        if sel.width() < 5 or sel.height() < 5:
            return None

        scale_x = self._pixmap.width() / self._display_rect.width()
        scale_y = self._pixmap.height() / self._display_rect.height()

        x = int((sel.x() - self._display_rect.x()) * scale_x)
        y = int((sel.y() - self._display_rect.y()) * scale_y)
        w = int(sel.width() * scale_x)
        h = int(sel.height() * scale_y)

        x = max(0, x)
        y = max(0, y)
        w = min(self._pixmap.width() - x, w)
        h = min(self._pixmap.height() - y, h)

        if w < 5 or h < 5:
            return None

        return QRect(x, y, w, h)

    def clear_selection(self):
        self._selection = QRect()
        self._rubber_band.hide()


class CropEditor(QWidget):
    """이미지 회전 및 Crop 편집을 담당하는 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[Image.Image] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)

    def load_image(self, image: Image.Image):
        self._image = image.copy()
        self._refresh()

    def _refresh(self):
        if self._image is None:
            return
        pixmap = pil_to_qpixmap(self._image)
        self.canvas.set_pixmap_fitted(pixmap)

    def rotate_left(self):
        if self._image is None:
            return
        self._image = self._image.rotate(90, expand=True)
        self._refresh()

    def rotate_right(self):
        if self._image is None:
            return
        self._image = self._image.rotate(-90, expand=True)
        self._refresh()

    def get_result_image(self) -> Optional[Image.Image]:
        """현재 회전 상태 + (선택 영역이 있으면) Crop 이 적용된 최종 이미지 반환"""
        if self._image is None:
            return None
        sel = self.canvas.get_selection_in_image_coords()
        if sel is None:
            return self._image.copy()
        box = (sel.x(), sel.y(), sel.x() + sel.width(), sel.y() + sel.height())
        return self._image.crop(box)

    def clear_selection(self):
        self.canvas.clear_selection()

    def get_current_image(self) -> Optional[Image.Image]:
        return self._image
