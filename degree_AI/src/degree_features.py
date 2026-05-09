import numpy as np


def _safe_point(points: np.ndarray, idx: int) -> np.ndarray | None:
    if idx < 0 or idx >= len(points):
        return None
    p = points[idx]
    if not np.all(np.isfinite(p)):
        return None
    return p.astype(np.float32)


def _dist(points: np.ndarray, a: int, b: int) -> float:
    pa = _safe_point(points, a)
    pb = _safe_point(points, b)
    if pa is None or pb is None:
        return 0.0
    return float(np.linalg.norm(pa - pb))


def _mean_y(points: np.ndarray, indices: list[int]) -> float:
    ys = []
    for idx in indices:
        p = _safe_point(points, idx)
        if p is not None:
            ys.append(float(p[1]))
    if not ys:
        return 0.0
    return float(np.mean(ys))


def _mean_x(points: np.ndarray, indices: list[int]) -> float:
    xs = []
    for idx in indices:
        p = _safe_point(points, idx)
        if p is not None:
            xs.append(float(p[0]))
    if not xs:
        return 0.0
    return float(np.mean(xs))


def normalize_face_points(points: np.ndarray) -> np.ndarray:
    """
    points: (N, 2)
    얼굴 중심과 얼굴 크기 기준으로 정규화.
    DISFA 66점, AIHub/OpenPose 70점 모두 처리 가능.
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be shape (N, 2), got {points.shape}")

    valid = np.isfinite(points).all(axis=1)
    if valid.sum() < 10:
        return np.zeros_like(points, dtype=np.float32)

    valid_points = points[valid]
    center = valid_points.mean(axis=0, keepdims=True)

    min_xy = valid_points.min(axis=0)
    max_xy = valid_points.max(axis=0)

    scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    if scale < 1e-6:
        scale = 1.0

    return ((points - center) / scale).astype(np.float32)


def extract_degree_features_from_points(points: np.ndarray) -> np.ndarray:
    """
    얼굴 landmark 좌표를 공통 feature로 변환.
    DISFA 66점과 AIHub/OpenPose 70점 모두 비슷하게 적용하기 위한 요약 feature.

    반환: 16차원 feature
    """
    pts = normalize_face_points(points)

    # OpenPose/DLIB 계열 공통 인덱스에 가까운 구간 사용
    # 얼굴 윤곽: 0~16
    # 눈썹: 17~26
    # 눈: 36~47
    # 입: 48~67 계열
    face_width = _dist(pts, 0, 16)
    if face_width < 1e-6:
        face_width = 1.0

    face_height = _dist(pts, 8, 27) if len(pts) > 27 else 1.0
    if face_height < 1e-6:
        face_height = 1.0

    # 눈 크기
    left_eye_width = _dist(pts, 36, 39)
    left_eye_open = (_dist(pts, 37, 41) + _dist(pts, 38, 40)) / 2.0

    right_eye_width = _dist(pts, 42, 45)
    right_eye_open = (_dist(pts, 43, 47) + _dist(pts, 44, 46)) / 2.0

    left_eye_ratio = left_eye_open / (left_eye_width + 1e-6)
    right_eye_ratio = right_eye_open / (right_eye_width + 1e-6)

    # 눈썹-눈 거리
    left_brow_y = _mean_y(pts, [17, 18, 19, 20, 21])
    left_eye_y = _mean_y(pts, [36, 37, 38, 39, 40, 41])
    right_brow_y = _mean_y(pts, [22, 23, 24, 25, 26])
    right_eye_y = _mean_y(pts, [42, 43, 44, 45, 46, 47])

    brow_eye_left = abs(left_eye_y - left_brow_y)
    brow_eye_right = abs(right_eye_y - right_brow_y)

    # 입 모양
    mouth_width = _dist(pts, 48, 54)

    # OpenPose 70이면 62/66, 63/65 사용 가능
    # DISFA 66이면 66번이 없을 수 있어 대체 인덱스 사용
    if len(pts) > 66:
        mouth_open = (_dist(pts, 62, 66) + _dist(pts, 63, 65)) / 2.0
    else:
        mouth_open = (_dist(pts, 61, 64) + _dist(pts, 62, 63)) / 2.0

    mouth_open_ratio = mouth_open / (mouth_width + 1e-6)

    # 입꼬리/입 중심 위치
    mouth_center_x = _mean_x(pts, [48, 54])
    mouth_center_y = _mean_y(pts, [51, 57]) if len(pts) > 57 else _mean_y(pts, [51, 55])

    # 얼굴 전체 압축/비율
    x_span = float(np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0]))
    y_span = float(np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1]))
    aspect = y_span / (x_span + 1e-6)

    # 좌우 비대칭
    eye_asym = abs(left_eye_ratio - right_eye_ratio)
    brow_asym = abs(brow_eye_left - brow_eye_right)

    features = np.array([
        left_eye_ratio,
        right_eye_ratio,
        eye_asym,
        brow_eye_left,
        brow_eye_right,
        brow_asym,
        mouth_width,
        mouth_open,
        mouth_open_ratio,
        mouth_center_x,
        mouth_center_y,
        face_width,
        face_height,
        aspect,
        x_span,
        y_span,
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features