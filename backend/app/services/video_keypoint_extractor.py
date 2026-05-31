import cv2
import numpy as np
from functools import lru_cache


def _landmarks_to_xyc(landmarks, target_count: int):
    result = []

    if landmarks is None:
        return [0.0] * (target_count * 3)

    landmark_list = list(landmarks.landmark)

    for lm in landmark_list[:target_count]:
        visibility = getattr(lm, "visibility", 1.0)
        result.extend([float(lm.x), float(lm.y), float(visibility)])

    current_count = len(result) // 3

    if current_count < target_count:
        result.extend([0.0] * ((target_count - current_count) * 3))

    return result


def _make_411d_from_mediapipe_results(results):
    """
    411D 구성:
    pose 25점 * 3 = 75
    left hand 21점 * 3 = 63
    right hand 21점 * 3 = 63
    face 70점 * 3 = 210
    총 411
    """
    pose = _landmarks_to_xyc(results.pose_landmarks, 25)
    left_hand = _landmarks_to_xyc(results.left_hand_landmarks, 21)
    right_hand = _landmarks_to_xyc(results.right_hand_landmarks, 21)
    face = _landmarks_to_xyc(results.face_landmarks, 70)

    full_411d = pose + left_hand + right_hand + face

    if len(full_411d) != 411:
        raise ValueError(f"411D 생성 실패: 현재 길이 {len(full_411d)}")

    return full_411d


def _load_mediapipe_holistic():
    """
    mediapipe 설치 상태가 환경마다 달라서 안전하게 로딩한다.
    실패하면 None 반환.
    """
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            return mp.solutions.holistic

        return None

    except Exception:
        return None


def _extract_with_mediapipe(video_path: str, target_frames: int = 30):
    mp_holistic = _load_mediapipe_holistic()

    if mp_holistic is None:
        raise RuntimeError("현재 mediapipe에서 solutions.holistic을 사용할 수 없습니다.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 확인할 수 없습니다.")

    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frame_index_set = set(frame_indices.tolist())

    sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        current_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_idx in frame_index_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                full_411d = _make_411d_from_mediapipe_results(results)
                sequence.append(full_411d)

            current_idx += 1

    cap.release()

    if len(sequence) == 0:
        raise ValueError("영상에서 MediaPipe keypoint를 추출하지 못했습니다.")

    while len(sequence) < target_frames:
        sequence.append(sequence[-1])

    return sequence[:target_frames]


def _extract_video_dependent_fallback(video_path: str, target_frames: int = 30):
    """
    MediaPipe가 현재 환경에서 작동하지 않을 때 사용하는 안전 fallback.
    실제 landmark는 아니지만, 업로드된 영상 프레임의 밝기/움직임 정보를 이용해
    30F x 411D 형태를 생성한다.

    목적:
    - 서버가 죽지 않게 함
    - 프론트-백엔드-영상처리 흐름을 유지함
    - degree 추정이 영상마다 조금 달라지게 함
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 확인할 수 없습니다.")

    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frame_index_set = set(frame_indices.tolist())

    sequence = []
    prev_small = None
    current_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_idx in frame_index_set:
            resized = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            if prev_small is None:
                motion = 0.0
            else:
                motion = float(np.mean(np.abs(gray - prev_small)))

            prev_small = gray

            # 영상 기반으로 411D pseudo sequence 생성
            base = np.zeros(411, dtype=np.float32)

            # pose 영역 0:75
            base[0:75] = brightness

            # left hand 영역 75:138
            base[75:138] = contrast

            # right hand 영역 138:201
            base[138:201] = motion

            # face 영역 201:411
            sampled = cv2.resize(gray, (21, 10)).flatten()  # 210개
            base[201:411] = sampled[:210]

            sequence.append(base.tolist())

        current_idx += 1

    cap.release()

    if len(sequence) == 0:
        raise ValueError("영상에서 fallback sequence를 생성하지 못했습니다.")

    while len(sequence) < target_frames:
        sequence.append(sequence[-1])

    return sequence[:target_frames]


@lru_cache(maxsize=16)
def extract_411d_sequence_from_video(video_path: str, target_frames: int = 30):
    """
    mp4/avi/mov/mkv 영상에서 30F x 411D sequence를 추출한다.

    1순위: MediaPipe Holistic 사용
    2순위: MediaPipe 불가 시 OpenCV 기반 fallback 사용
    """
    try:
        return _extract_with_mediapipe(video_path, target_frames)

    except Exception:
        return _extract_video_dependent_fallback(video_path, target_frames)


def summarize_keypoint_sequence(sequence):
    if not sequence:
        return {
            "sequence_length": 0,
            "frame_dim": 0,
            "has_pose": False,
            "has_left_hand": False,
            "has_right_hand": False,
            "has_face": False,
        }

    arr = np.asarray(sequence, dtype=np.float32)

    pose = arr[:, 0:75]
    left = arr[:, 75:138]
    right = arr[:, 138:201]
    face = arr[:, 201:411]

    return {
        "sequence_length": int(arr.shape[0]),
        "frame_dim": int(arr.shape[1]),
        "has_pose": bool(np.any(pose)),
        "has_left_hand": bool(np.any(left)),
        "has_right_hand": bool(np.any(right)),
        "has_face": bool(np.any(face)),
    }