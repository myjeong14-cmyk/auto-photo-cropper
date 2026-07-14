"""공용 유틸리티 함수 모음"""
import os
import datetime

from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """PIL Image -> QPixmap 변환"""
    rgb_img = img.convert("RGB")
    data = rgb_img.tobytes("raw", "RGB")
    qimg = QImage(data, rgb_img.width, rgb_img.height, rgb_img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def generate_output_filename(original_path: str, index: int = 0) -> str:
    """원본 파일명 + 타임스탬프 기반 자동 파일명 생성 (.jpg 고정)"""
    base = os.path.splitext(os.path.basename(original_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base = "".join(c for c in base if c.isalnum() or c in ("_", "-"))
    if not safe_base:
        safe_base = "photo"
    return f"{safe_base}_{timestamp}_{index:03d}.jpg"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_supported_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg"))
