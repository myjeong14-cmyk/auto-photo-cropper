"""
증명사진 자동 편집 처리 모듈

처리 순서:
1. EXIF 정보 기반 자동 회전
2. 밝기 Threshold + Canny Edge + 배경 차분(대형 블러) 마스크를 결합한
   Hybrid Foreground Mask 생성 (흰색/회색 배경, 스캐너 그림자, 약한 테두리 대응)
3. Contour/minAreaRect 로 회전각 추정 후 이미지 전체를 회전 보정 (deskew)
4. 얼굴 인식(YuNet DNN, 실패 시 Haar Cascade)으로 얼굴 위치를 찾아,
   증명사진 규격(3.5 x 4.5) 비율에 맞춰 얼굴 중심 기준으로 정밀 Crop
   - 얼굴 인식에 모두 실패한 경우 기존 방식(좌/우/상/하 Projection 분석 Crop)으로 대체
"""
import os
import sys
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps

# 증명사진 규격 (가로:세로 = 3.5cm : 4.5cm)
ID_PHOTO_RATIO_W = 3.5
ID_PHOTO_RATIO_H = 4.5

# 얼굴(턱~정수리 추정 높이)이 최종 사진 세로 길이에서 차지하는 비율
FACE_HEIGHT_RATIO = 0.72
# 정수리 위 여백 비율 (최종 사진 세로 길이 대비)
TOP_MARGIN_RATIO = 0.12

YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


class AutoProcessError(Exception):
    """자동 처리 중 발생하는 예외"""
    pass


def _is_ascii(path: str) -> bool:
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _to_ascii_safe_path(path: str) -> str:
    """
    Windows 에서 OpenCV(cv2.CascadeClassifier, cv2.FaceDetectorYN 등)가 내부적으로
    fopen() 계열 함수를 사용해 파일을 읽기 때문에, 사용자 계정 폴더명 등에
    한글(비ASCII) 경로가 섞여 있으면 파일을 열지 못하는 문제가 있다.

    1) 경로가 이미 ASCII 이면 그대로 반환
    2) Windows 8.3 짧은 경로(예: '자격시험부' -> 'JAGYEO~1')로 변환 시도
    3) 그래도 안 되면 ASCII 경로가 보장되는 위치(C:\\ProgramData)로 파일을 복사 후 그 경로 반환
    """
    if not os.path.exists(path):
        return path

    if _is_ascii(path):
        return path

    if sys.platform.startswith("win"):
        try:
            import ctypes
            get_short = ctypes.windll.kernel32.GetShortPathNameW
            buf = ctypes.create_unicode_buffer(260)
            result = get_short(path, buf, 260)
            if result and _is_ascii(buf.value):
                return buf.value
        except Exception:
            pass

    try:
        safe_dir = os.path.join(
            os.environ.get("PROGRAMDATA", "C:\\ProgramData"),
            "IDPhotoEditor_resources",
        )
        os.makedirs(safe_dir, exist_ok=True)
        dest = os.path.join(safe_dir, os.path.basename(path))
        if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(path):
            shutil.copyfile(path, dest)
        if _is_ascii(dest):
            return dest
    except Exception:
        pass

    return path


def load_image_with_exif(path: str) -> Image.Image:
    """EXIF Orientation 정보를 반영하여 이미지를 로드"""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        raise AutoProcessError(f"이미지를 열 수 없습니다: {e}")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(mat: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# 얼굴 인식 1순위: YuNet (OpenCV 공식 경량 딥러닝 얼굴 검출기, ONNX)
#   - Haar Cascade 보다 훨씬 정확 (측면 얼굴, 약한 조명, 안경 등에도 강함)
#   - models/face_detection_yunet_2023mar.onnx 모델 파일 필요
# 얼굴 인식 2순위: Haar Cascade (YuNet 모델 파일이 없을 때 대체)
# ---------------------------------------------------------------------------
def _get_model_dir() -> str:
    """개발 환경(스크립트 실행)과 PyInstaller 빌드(exe 실행) 양쪽에서
    올바른 리소스 경로를 찾는다."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models")


_yunet_detector = None
_yunet_load_attempted = False
_yunet_status = "not_attempted"   # 진단용: not_attempted | ok | file_missing | load_error | detect_error
_yunet_status_detail = ""         # 진단용: 에러 메시지 등 상세 정보


def get_yunet_status():
    """YuNet 로드 상태를 진단용으로 반환: (status, detail, model_path)"""
    return _yunet_status, _yunet_status_detail, os.path.join(_get_model_dir(), YUNET_MODEL_FILENAME)


def _get_yunet_detector(img_w: int, img_h: int):
    """YuNet 검출기를 1회 로드하여 재사용. 실패 시 사유를 _yunet_status 에 기록한다."""
    global _yunet_detector, _yunet_load_attempted, _yunet_status, _yunet_status_detail

    model_path = os.path.join(_get_model_dir(), YUNET_MODEL_FILENAME)

    if _yunet_detector is None:
        if _yunet_load_attempted:
            return None
        _yunet_load_attempted = True
        if not os.path.exists(model_path):
            _yunet_status = "file_missing"
            _yunet_status_detail = f"모델 파일 없음: {model_path}"
            return None
        safe_model_path = _to_ascii_safe_path(model_path)
        try:
            _yunet_detector = cv2.FaceDetectorYN.create(
                safe_model_path, "", (320, 320),
                score_threshold=0.7, nms_threshold=0.3, top_k=5000,
            )
            _yunet_status = "ok"
            _yunet_status_detail = ""
        except Exception as e:
            _yunet_detector = None
            _yunet_status = "load_error"
            _yunet_status_detail = f"{type(e).__name__}: {e}"
            return None

    try:
        _yunet_detector.setInputSize((img_w, img_h))
    except Exception as e:
        _yunet_status = "load_error"
        _yunet_status_detail = f"setInputSize 실패 - {type(e).__name__}: {e}"
        return None

    return _yunet_detector


def _detect_largest_face_yunet(image_bgr: np.ndarray):
    global _yunet_status, _yunet_status_detail

    h, w = image_bgr.shape[:2]
    detector = _get_yunet_detector(w, h)
    if detector is None:
        return None

    try:
        _, faces = detector.detect(image_bgr)
    except Exception as e:
        _yunet_status = "detect_error"
        _yunet_status_detail = f"{type(e).__name__}: {e}"
        return None

    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4]
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    fw = int(round(fw))
    fh = int(round(fh))
    if fw <= 0 or fh <= 0:
        return None
    return x, y, fw, fh


_face_cascade = None
_face_cascade_load_attempted = False
_haar_status_ref = ["not_attempted", ""]  # [status, detail] - 진단용


def _get_face_cascade():
    """OpenCV 내장 Haar Cascade 얼굴 검출기를 1회 로드하여 재사용 (YuNet 실패 시 대체).
    파일이 없거나 로드 실패 시 None 을 반환하여 얼굴 인식 기능만 조용히 비활성화한다."""
    global _face_cascade, _face_cascade_load_attempted
    if _face_cascade_load_attempted:
        return _face_cascade
    _face_cascade_load_attempted = True
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        safe_cascade_path = _to_ascii_safe_path(cascade_path)
        clf = cv2.CascadeClassifier(safe_cascade_path)
        if clf.empty():
            _face_cascade = None
            _haar_status_ref[0] = "load_error"
            _haar_status_ref[1] = f"Cascade 로드 실패 (파일 없음 또는 손상): {cascade_path}"
        else:
            _face_cascade = clf
            _haar_status_ref[0] = "ok"
            _haar_status_ref[1] = ""
    except Exception as e:
        _face_cascade = None
        _haar_status_ref[0] = "load_error"
        _haar_status_ref[1] = f"{type(e).__name__}: {e}"
    return _face_cascade


def get_haar_status():
    """Haar Cascade 로드 상태를 진단용으로 반환: (status, detail)"""
    return _haar_status_ref[0], _haar_status_ref[1]


def _detect_largest_face_haar(image_bgr: np.ndarray):
    clf = _get_face_cascade()
    if clf is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    min_dim = min(image_bgr.shape[:2])
    min_size = max(40, min_dim // 12)

    faces = clf.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def detect_largest_face(image_bgr: np.ndarray):
    """이미지에서 가장 큰 얼굴의 바운딩 박스 (x, y, w, h) 와 사용된 엔진 이름을 반환.
    YuNet(딥러닝) 우선 시도 후, 실패하면 Haar Cascade 로 대체한다.
    반환값: (box, engine) - box 는 없으면 None, engine 은 "yunet"/"haar"/None"""
    box = _detect_largest_face_yunet(image_bgr)
    if box is not None:
        return box, "yunet"
    box = _detect_largest_face_haar(image_bgr)
    if box is not None:
        return box, "haar"
    return None, None


def get_face_detection_diagnosis() -> str:
    """얼굴 인식이 왜 실패했는지 사람이 읽을 수 있는 진단 메시지를 반환 (UI 표시용)"""
    y_status, y_detail, y_path = get_yunet_status()
    h_status, h_detail = get_haar_status()

    if y_status == "file_missing":
        y_msg = f"YuNet: 모델 파일 없음 ({y_path})"
    elif y_status == "load_error":
        y_msg = f"YuNet: 로드 실패 - {y_detail}"
    elif y_status == "detect_error":
        y_msg = f"YuNet: 인식 중 오류 - {y_detail}"
    elif y_status == "ok":
        y_msg = "YuNet: 로드는 성공했으나 얼굴을 찾지 못함"
    else:
        y_msg = "YuNet: 시도 안 함"

    if h_status == "load_error":
        h_msg = f"Haar: 로드 실패 - {h_detail}"
    elif h_status == "ok":
        h_msg = "Haar: 로드는 성공했으나 얼굴을 찾지 못함"
    else:
        h_msg = "Haar: 시도 안 함"

    return f"{y_msg} / {h_msg}"


def yunet_model_available() -> bool:
    """YuNet 모델 파일이 정상적으로 존재/로드 가능한지 확인 (진단용)"""
    model_path = os.path.join(_get_model_dir(), YUNET_MODEL_FILENAME)
    return os.path.exists(model_path)


def _fit_rect_into_bounds(left, top, right, bottom, bound_x0, bound_y0, bound_x1, bound_y1):
    """계산된 crop 사각형이 주어진 경계(bound_x0..bound_y1, 예: 실제 사진 테두리)를
    벗어나지 않도록 보정한다.

    - 경계보다 사각형이 크면: 비율을 유지한 채 축소(scale down)하여 경계 안에 맞춘다.
      (얼굴 기준 이상적인 프레임이 실제 사진보다 클 경우, 배경/여백을 침범하지
      않도록 실제 사진 크기에 맞춰 줄인다.)
    - 경계 안에서 위치만 벗어난 경우: 위치를 이동시켜 같은 크기를 유지한 채
      경계 안으로 들어오도록 한다. (얼굴이 사진 가장자리에 가까운 경우)
    """
    w = right - left
    h = bottom - top
    bound_w = bound_x1 - bound_x0
    bound_h = bound_y1 - bound_y0

    # 1) 이상적인 crop 크기가 실제 경계보다 크면 비율 유지한 채 축소
    scale = 1.0
    if w > 0 and bound_w > 0:
        scale = min(scale, bound_w / w)
    if h > 0 and bound_h > 0:
        scale = min(scale, bound_h / h)
    if scale < 1.0:
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        w *= scale
        h *= scale
        left = cx - w / 2.0
        right = cx + w / 2.0
        top = cy - h / 2.0
        bottom = cy + h / 2.0

    # 2) 경계 밖으로 위치가 벗어나면 같은 크기를 유지한 채 안으로 이동
    if left < bound_x0:
        shift = bound_x0 - left
        left += shift
        right += shift
    if right > bound_x1:
        shift = right - bound_x1
        left -= shift
        right -= shift
    if top < bound_y0:
        shift = bound_y0 - top
        top += shift
        bottom += shift
    if bottom > bound_y1:
        shift = bottom - bound_y1
        top -= shift
        bottom -= shift

    # 3) 안전을 위한 최종 clamp
    left = max(bound_x0, min(left, bound_x1))
    top = max(bound_y0, min(top, bound_y1))
    right = max(bound_x0, min(right, bound_x1))
    bottom = max(bound_y0, min(bottom, bound_y1))
    return int(left), int(top), int(right), int(bottom)


def crop_to_id_photo_by_face(image: np.ndarray, face_box, bounds=None):
    """
    얼굴 바운딩 박스를 기준으로 증명사진 규격(3.5:4.5) 비율로 Crop.

    Haar Cascade/YuNet 의 얼굴 박스는 대략 이마~턱 부근을 감지하므로,
    정수리(머리 위)와 턱 끝 위치를 근사적으로 확장 추정한 뒤,
    실무 증명사진 규격 비율(얼굴 길이 비율, 상단 여백 비율)에 맞춰
    최종 crop 영역을 계산한다.

    bounds: (left, top, right, bottom) - 실제 스캔된 사진의 테두리(여백 제외) 좌표.
            스캔마다 사진 크기가 다를 수 있으므로, 얼굴 기준으로 계산한 이상적인
            crop 영역이 이 경계를 벗어나지 않도록 강제한다 (배경/여백 침범 방지).
            None 이면 이미지 전체를 경계로 사용.
    """
    img_h, img_w = image.shape[:2]
    fx, fy, fw, fh = face_box

    if bounds is None:
        bound_x0, bound_y0, bound_x1, bound_y1 = 0, 0, img_w, img_h
    else:
        bound_x0, bound_y0, bound_x1, bound_y1 = bounds

    # 정수리(머리 위)와 턱 끝 위치 근사 추정
    head_top = fy - fh * 0.35
    chin_bottom = fy + fh * 1.0
    head_height = chin_bottom - head_top
    if head_height <= 0:
        head_height = fh * 1.35
        head_top = fy - fh * 0.35

    crop_height = head_height / FACE_HEIGHT_RATIO
    crop_width = crop_height * (ID_PHOTO_RATIO_W / ID_PHOTO_RATIO_H)

    center_x = fx + fw / 2.0
    top = head_top - crop_height * TOP_MARGIN_RATIO
    bottom = top + crop_height
    left = center_x - crop_width / 2.0
    right = left + crop_width

    left, top, right, bottom = _fit_rect_into_bounds(
        left, top, right, bottom, bound_x0, bound_y0, bound_x1, bound_y1
    )

    if right - left < 10 or bottom - top < 10:
        return None

    return image[top:bottom, left:right]


# ---------------------------------------------------------------------------
# 얼굴 인식 실패 시 fallback: 여백/테두리 인식 기반 Crop
# ---------------------------------------------------------------------------
def _build_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """
    밝기 Threshold + Canny Edge + 배경 차분(대형 블러) 3가지 신호를 결합한
    Hybrid Foreground Mask.
    """
    h, w = gray.shape[:2]

    k = max(21, (min(h, w) // 6) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    diff = cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, mask_bgsub = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(blur))
    lower = int(max(0, 0.5 * v))
    upper = int(min(255, 1.5 * v))
    edges = cv2.Canny(blur, lower, upper)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)

    _, mask_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(mask_bgsub, mask_edges)
    combined = cv2.bitwise_or(combined, mask_bright)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return combined


def _find_candidate_box(mask: np.ndarray, img_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05 or area > img_area * 0.98:
            continue
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        return None

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    return box, rect[-1]


def _deskew_angle_from_rect(angle: float) -> float:
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _refine_bbox_by_projection(mask: np.ndarray, seed_box=None, pad_ratio: float = 0.08):
    h, w = mask.shape[:2]

    if seed_box is not None:
        x, y, bw, bh = cv2.boundingRect(seed_box.astype(np.int32))
    else:
        x, y, bw, bh = 0, 0, w, h

    pad_x = int(bw * pad_ratio) + 15
    pad_y = int(bh * pad_ratio) + 15
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return x0, y0, x1, y1

    col_sum = roi.sum(axis=0) / 255.0
    row_sum = roi.sum(axis=1) / 255.0

    col_thresh = max(1.0, roi.shape[0] * 0.03)
    row_thresh = max(1.0, roi.shape[1] * 0.03)

    cols_fg = np.where(col_sum > col_thresh)[0]
    rows_fg = np.where(row_sum > row_thresh)[0]

    if len(cols_fg) == 0 or len(rows_fg) == 0:
        return x0, y0, x1, y1

    left = x0 + int(cols_fg[0])
    right = x0 + int(cols_fg[-1]) + 1
    top = y0 + int(rows_fg[0])
    bottom = y0 + int(rows_fg[-1]) + 1

    return left, top, right, bottom


def _detect_photo_bounds(cv_img: np.ndarray):
    """
    스캔 이미지 안에서 '실제 인화된 사진'의 테두리(좌/상/우/하)를 찾는다.
    회전 보정은 이미 끝난 이미지를 입력으로 받는다고 가정.
    반환값: (left, top, right, bottom) - 찾지 못하면 None
    """
    h, w = cv_img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mask = _build_foreground_mask(gray)
    candidate = _find_candidate_box(mask, img_area)
    seed_box = candidate[0] if candidate is not None else None

    left, top, right, bottom = _refine_bbox_by_projection(mask, seed_box)

    if right - left < 10 or bottom - top < 10:
        return None

    return left, top, right, bottom


def auto_process(path: str):
    """
    증명사진 자동 편집 실행.

    처리 순서:
      1) 스캔 전체에서 기울기 보정 (deskew)
      2) 여백/테두리 인식으로 '실제 인화된 사진' 영역의 경계를 찾음
      3) 그 경계 안에서만 얼굴 인식 + 3.5:4.5 규격 크롭 수행
         (경계를 절대 벗어나지 않도록 제한 -> 스캔 여백이 섞여 들어가지 않음)
      4) 얼굴 인식 실패 시, 경계 크롭 결과를 그대로 사용

    반환값: (before_image, after_image, method)
      - before_image: EXIF 회전만 적용된 원본 (수동 편집 시 "이전" 상태)
      - after_image : 위 로직으로 처리된 결과
      - method      : "yunet" | "haar" | "border" | "raw" 중 실제 사용된 방식
                      (디버그/상태 표시용)
    """
    before_pil = load_image_with_exif(path)
    cv_img = pil_to_cv(before_pil)

    # 1) 기울기 보정 (전체 캔버스 기준)
    h, w = cv_img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mask = _build_foreground_mask(gray)
    candidate = _find_candidate_box(mask, img_area)

    working_img = cv_img
    if candidate is not None:
        _, raw_angle = candidate
        angle = _deskew_angle_from_rect(float(raw_angle))
        if 0.3 <= abs(angle) <= 20:
            working_img = _rotate_image(cv_img, angle)

    # 2) 실제 인화된 사진의 테두리 경계를 먼저 찾는다
    bounds = _detect_photo_bounds(working_img)
    if bounds is not None:
        b_left, b_top, b_right, b_bottom = bounds
        photo_img = working_img[b_top:b_bottom, b_left:b_right]
    else:
        photo_img = working_img

    if photo_img is None or photo_img.size == 0:
        photo_img = working_img

    # 3) 사진 경계 '안에서만' 얼굴 인식 + 규격 크롭 시도
    #    (crop_to_id_photo_by_face 는 입력 이미지 크기 안으로만 결과를 clamp 하므로,
    #     photo_img 를 넘겨주면 결과가 실제 사진 테두리를 절대 벗어나지 않는다)
    face_box, engine = detect_largest_face(photo_img)
    cropped = None
    method = None
    if face_box is not None:
        cropped = crop_to_id_photo_by_face(photo_img, face_box)
        if cropped is not None:
            method = engine  # "yunet" 또는 "haar"

    # 4) 얼굴 인식 실패 시, 테두리 경계로 자른 결과를 그대로 사용
    if cropped is None:
        cropped = photo_img
        method = "border|" + get_face_detection_diagnosis()

    if cropped is None or cropped.size == 0:
        after_pil = before_pil.copy()
        method = "raw"
    else:
        after_pil = cv_to_pil(cropped)

    return before_pil, after_pil, method
