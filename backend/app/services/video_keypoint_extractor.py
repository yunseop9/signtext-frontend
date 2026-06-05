import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[2]

POSE_POINT_COUNT = 25
HAND_POINT_COUNT = 21
FACE_POINT_COUNT = 70

POSE_LEN = POSE_POINT_COUNT * 3
HAND_LEN = HAND_POINT_COUNT * 3
FACE_LEN = FACE_POINT_COUNT * 3
WORD_FEATURE_DIM = POSE_LEN + HAND_LEN + HAND_LEN + FACE_LEN
SENTENCE_FEATURE_DIM = 120
DEGREE_FEATURE_DIM = 280


def _openpose_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("OPENPOSE_DEMO_PATH")

    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            candidates.extend(
                [
                    configured_path / "OpenPoseDemo.exe",
                    configured_path / "bin" / "OpenPoseDemo.exe",
                    configured_path / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
                ]
            )
        else:
            candidates.append(configured_path)

    candidates.extend(
        [
            BACKEND_ROOT / "openpose" / "bin" / "OpenPoseDemo.exe",
            BACKEND_ROOT / "openpose" / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
            BACKEND_ROOT / "tools" / "openpose" / "bin" / "OpenPoseDemo.exe",
            BACKEND_ROOT / "tools" / "openpose" / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
        ]
    )

    for command_name in ("OpenPoseDemo.exe", "OpenPoseDemo", "openpose"):
        resolved = shutil.which(command_name)
        if resolved:
            candidates.append(Path(resolved))

    return candidates


def _resolve_openpose_executable() -> Path:
    for candidate in _openpose_candidates():
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "OpenPoseDemo executable was not found. Set OPENPOSE_DEMO_PATH to "
        "OpenPoseDemo.exe, or place OpenPose under backend/openpose."
    )


def _resolve_openpose_model_folder(executable: Path) -> Path | None:
    configured = os.environ.get("OPENPOSE_MODEL_FOLDER")
    if configured:
        model_folder = Path(configured).expanduser()
        if not model_folder.is_dir():
            raise FileNotFoundError(
                f"OPENPOSE_MODEL_FOLDER does not exist: {model_folder}"
            )
        return model_folder.resolve()

    candidates = [
        BACKEND_ROOT / "openpose" / "models",
        BACKEND_ROOT / "tools" / "openpose" / "models",
    ]
    candidates.extend(parent / "models" for parent in executable.parents)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return None


def _extract_frame_number(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else -1


def _run_openpose(video_path: str) -> tuple[dict[str, Any], ...]:
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video}")

    executable = _resolve_openpose_executable()
    model_folder = _resolve_openpose_model_folder(executable)
    timeout_seconds = int(os.environ.get("OPENPOSE_TIMEOUT_SECONDS", "300"))

    with tempfile.TemporaryDirectory(prefix="signtext_openpose_") as output_dir:
        output_path = Path(output_dir).resolve()
        command = [
            str(executable),
            "--video",
            str(video),
            "--write_json",
            str(output_path),
            "--display",
            "0",
            "--render_pose",
            "0",
            "--model_pose",
            "BODY_25",
            "--hand",
            "--face",
        ]

        if model_folder is not None:
            command.extend(["--model_folder", str(model_folder)])

        net_resolution = os.environ.get("OPENPOSE_NET_RESOLUTION")
        if net_resolution:
            command.extend(["--net_resolution", net_resolution])

        working_directory = model_folder.parent if model_folder else executable.parent

        try:
            completed = subprocess.run(
                command,
                cwd=str(working_directory),
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"OpenPose timed out after {timeout_seconds} seconds."
            ) from error
        except OSError as error:
            raise RuntimeError(f"Failed to start OpenPose: {error}") from error

        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            if len(details) > 2000:
                details = details[-2000:]
            raise RuntimeError(
                f"OpenPose failed with exit code {completed.returncode}: {details}"
            )

        json_files = sorted(output_path.glob("*.json"), key=_extract_frame_number)
        if not json_files:
            raise RuntimeError("OpenPose completed without producing keypoint JSON files.")

        frames = []
        for json_path in json_files:
            with json_path.open("r", encoding="utf-8") as file:
                frame = json.load(file)
            if isinstance(frame, dict):
                frames.append(frame)

    if not frames:
        raise RuntimeError("No valid OpenPose keypoint frames were produced.")

    return tuple(frames)


@lru_cache(maxsize=16)
def extract_openpose_frames_from_video(video_path: str) -> tuple[dict[str, Any], ...]:
    return _run_openpose(video_path)


def _get_people(data: dict[str, Any]) -> list[dict[str, Any]]:
    people = data.get("people")
    if isinstance(people, list):
        return [person for person in people if isinstance(person, dict)]
    if isinstance(people, dict):
        return [people]
    return []


def _word_person(data: dict[str, Any]) -> dict[str, Any]:
    people = data.get("people", {})
    if isinstance(people, list):
        if not people:
            return {}
        return people[0] if isinstance(people[0], dict) else {}
    if isinstance(people, dict):
        return people
    return {}


def _fix_length(values: Any, target_len: int) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * target_len
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))


def _reshape_fixed_keypoints(values: Any, target_len: int) -> np.ndarray:
    return np.asarray(_fix_length(values, target_len), dtype=np.float32).reshape(-1, 3)


def _word_origin_and_scale(pose: np.ndarray) -> tuple[np.ndarray, float]:
    origin = np.array([0.0, 0.0], dtype=np.float32)
    scale = 1.0

    try:
        neck = pose[1]
        right_shoulder = pose[2]
        left_shoulder = pose[5]

        if right_shoulder[2] > 0 and left_shoulder[2] > 0:
            origin = (right_shoulder[:2] + left_shoulder[:2]) / 2.0
            shoulder_width = float(
                np.linalg.norm(right_shoulder[:2] - left_shoulder[:2])
            )
            if shoulder_width > 1e-6:
                scale = shoulder_width
        elif neck[2] > 0:
            origin = neck[:2].astype(np.float32)
    except Exception:
        pass

    return origin, scale


def _normalize_word_points(
    points: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> np.ndarray:
    normalized = points.copy()
    confident = normalized[:, 2] > 0
    normalized[~confident, 0:2] = 0.0
    normalized[confident, 0] = (normalized[confident, 0] - origin[0]) / scale
    normalized[confident, 1] = (normalized[confident, 1] - origin[1]) / scale
    return normalized


def _word_frame_feature(data: dict[str, Any]) -> np.ndarray:
    person = _word_person(data)
    pose = _reshape_fixed_keypoints(person.get("pose_keypoints_2d", []), POSE_LEN)
    left = _reshape_fixed_keypoints(
        person.get("hand_left_keypoints_2d", []), HAND_LEN
    )
    right = _reshape_fixed_keypoints(
        person.get("hand_right_keypoints_2d", []), HAND_LEN
    )
    face = _reshape_fixed_keypoints(person.get("face_keypoints_2d", []), FACE_LEN)

    origin, scale = _word_origin_and_scale(pose)
    features = np.concatenate(
        [
            _normalize_word_points(pose, origin, scale).ravel(),
            _normalize_word_points(left, origin, scale).ravel(),
            _normalize_word_points(right, origin, scale).ravel(),
            _normalize_word_points(face, origin, scale).ravel(),
        ]
    ).astype(np.float32)

    if features.shape != (WORD_FEATURE_DIM,):
        raise ValueError(
            f"Word frame shape mismatch: {features.shape}, "
            f"expected ({WORD_FEATURE_DIM},)"
        )

    return features


def _resize_word_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    current_len = len(sequence)
    if current_len == 0:
        raise ValueError("OpenPose sequence is empty.")
    if current_len == target_frames:
        return sequence.astype(np.float32)
    if current_len > target_frames:
        indices = np.linspace(0, current_len - 1, target_frames).astype(int)
        return sequence[indices].astype(np.float32)

    padding = np.zeros(
        (target_frames - current_len, sequence.shape[1]), dtype=np.float32
    )
    return np.vstack([sequence, padding]).astype(np.float32)


def preprocess_word_openpose_frames(
    frames: Sequence[dict[str, Any]],
    target_frames: int = 30,
) -> np.ndarray:
    raw = np.stack([_word_frame_feature(frame) for frame in frames], axis=0)
    return _resize_word_sequence(raw, target_frames)


def extract_word_411d_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    frames = extract_openpose_frames_from_video(video_path)
    return preprocess_word_openpose_frames(frames, target_frames)


def extract_411d_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    return extract_word_411d_sequence_from_video(video_path, target_frames)


def extract_hands_126_from_411d(sequence: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != WORD_FEATURE_DIM:
        raise ValueError(f"411D sequence shape mismatch: {array.shape}")

    hands = array[:, 75:201].astype(np.float32)
    if hands.shape != (30, 126):
        raise ValueError(
            f"Word model input shape mismatch: {hands.shape}, expected (30, 126)"
        )
    return hands


def _mean_keypoint_confidence(values: Any) -> float | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0 or len(array) % 3 != 0:
        return None
    return float(np.mean(array.reshape(-1, 3)[:, 2]))


def _sentence_person(data: dict[str, Any]) -> dict[str, Any]:
    people = data.get("people")
    if isinstance(people, list) and people:
        best = people[0] if isinstance(people[0], dict) else {}
        best_score = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue

            scores = []
            for key in (
                "hand_left_keypoints_2d",
                "hand_right_keypoints_2d",
                "pose_keypoints_2d",
            ):
                score = _mean_keypoint_confidence(person.get(key))
                if score is not None:
                    scores.append(score)

            person_score = float(np.mean(scores)) if scores else 0.0
            if person_score > best_score:
                best_score = person_score
                best = person

        return best

    if isinstance(people, dict):
        return people

    return data


def _recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def _sentence_keypoints_to_xy(
    values: Any,
    num_points: int,
    width: float = 1920.0,
    height: float = 1080.0,
    confidence_threshold: float = 0.05,
) -> np.ndarray:
    output = np.zeros((num_points, 2), dtype=np.float32)
    if values is None:
        return output

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        return output

    if len(array) % 3 == 0:
        array_3d = array.reshape(-1, 3)
        xy = array_3d[:, :2]
        confidence = array_3d[:, 2]
    elif len(array) % 2 == 0:
        xy = array.reshape(-1, 2)
        confidence = np.ones((len(xy),), dtype=np.float32)
    else:
        return output

    count = min(num_points, len(xy))
    xy = xy[:count].astype(np.float32)
    confidence = confidence[:count].astype(np.float32)

    if np.nanmax(xy) > 2.0:
        xy[:, 0] = xy[:, 0] / width
        xy[:, 1] = xy[:, 1] / height

    xy[confidence < confidence_threshold] = 0.0
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    xy = np.clip(xy, 0.0, 1.0)
    output[:count] = xy
    return output


def _sentence_frame_feature(data: dict[str, Any]) -> np.ndarray:
    person = _sentence_person(data)

    left_values = person.get("hand_left_keypoints_2d")
    right_values = person.get("hand_right_keypoints_2d")
    pose_values = person.get("pose_keypoints_2d")

    if left_values is None:
        left_values = _recursive_find_key(data, "hand_left_keypoints_2d")
    if right_values is None:
        right_values = _recursive_find_key(data, "hand_right_keypoints_2d")
    if pose_values is None:
        pose_values = _recursive_find_key(data, "pose_keypoints_2d")

    left_xy = _sentence_keypoints_to_xy(left_values, HAND_POINT_COUNT)
    right_xy = _sentence_keypoints_to_xy(right_values, HAND_POINT_COUNT)
    pose_xy = _sentence_keypoints_to_xy(pose_values, POSE_POINT_COUNT)

    feature = np.concatenate(
        [left_xy.flatten(), right_xy.flatten(), pose_xy[:18].flatten()]
    ).astype(np.float32)

    if feature.shape != (SENTENCE_FEATURE_DIM,):
        raise ValueError(
            f"Sentence frame shape mismatch: {feature.shape}, "
            f"expected ({SENTENCE_FEATURE_DIM},)"
        )
    return feature


def _resize_sentence_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    current_len = len(sequence)
    if current_len == 0:
        return np.zeros((target_frames, SENTENCE_FEATURE_DIM), dtype=np.float32)
    if current_len == target_frames:
        return sequence.astype(np.float32)
    if current_len > target_frames:
        indices = np.linspace(0, current_len - 1, target_frames).astype(int)
        return sequence[indices].astype(np.float32)

    padding = np.repeat(sequence[-1:], target_frames - current_len, axis=0)
    return np.concatenate([sequence, padding], axis=0).astype(np.float32)


def preprocess_sentence_openpose_frames(
    frames: Sequence[dict[str, Any]],
    target_frames: int = 30,
) -> np.ndarray:
    if not frames:
        return np.zeros((target_frames, SENTENCE_FEATURE_DIM), dtype=np.float32)

    raw = np.stack([_sentence_frame_feature(frame) for frame in frames], axis=0)
    return _resize_sentence_sequence(raw, target_frames)


def extract_sentence_120_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    frames = extract_openpose_frames_from_video(video_path)
    return preprocess_sentence_openpose_frames(frames, target_frames)


def _safe_point(points: np.ndarray, index: int) -> np.ndarray | None:
    if index < 0 or index >= len(points):
        return None
    point = points[index]
    if not np.all(np.isfinite(point)):
        return None
    return point.astype(np.float32)


def _distance(points: np.ndarray, first: int, second: int) -> float:
    first_point = _safe_point(points, first)
    second_point = _safe_point(points, second)
    if first_point is None or second_point is None:
        return 0.0
    return float(np.linalg.norm(first_point - second_point))


def _mean_axis(points: np.ndarray, indices: list[int], axis: int) -> float:
    values = []
    for index in indices:
        point = _safe_point(points, index)
        if point is not None:
            values.append(float(point[axis]))
    return float(np.mean(values)) if values else 0.0


def _normalize_face_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Face points shape mismatch: {points.shape}")

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


def _extract_degree_summary(points: np.ndarray) -> np.ndarray:
    points = _normalize_face_points(points)

    left_eye_width = _distance(points, 36, 39)
    left_eye_open = (
        _distance(points, 37, 41) + _distance(points, 38, 40)
    ) / 2.0
    right_eye_width = _distance(points, 42, 45)
    right_eye_open = (
        _distance(points, 43, 47) + _distance(points, 44, 46)
    ) / 2.0

    left_eye_ratio = left_eye_open / (left_eye_width + 1e-6)
    right_eye_ratio = right_eye_open / (right_eye_width + 1e-6)
    left_brow_y = _mean_axis(points, [17, 18, 19, 20, 21], 1)
    left_eye_y = _mean_axis(points, [36, 37, 38, 39, 40, 41], 1)
    right_brow_y = _mean_axis(points, [22, 23, 24, 25, 26], 1)
    right_eye_y = _mean_axis(points, [42, 43, 44, 45, 46, 47], 1)
    brow_eye_left = abs(left_eye_y - left_brow_y)
    brow_eye_right = abs(right_eye_y - right_brow_y)
    mouth_width = _distance(points, 48, 54)
    face_width = _distance(points, 0, 16)
    face_height = _distance(points, 8, 27)

    if face_width < 1e-6:
        face_width = 1.0
    if face_height < 1e-6:
        face_height = 1.0

    if len(points) > 66:
        mouth_open = (
            _distance(points, 62, 66) + _distance(points, 63, 65)
        ) / 2.0
    else:
        mouth_open = (
            _distance(points, 61, 64) + _distance(points, 62, 63)
        ) / 2.0

    x_span = float(np.nanmax(points[:, 0]) - np.nanmin(points[:, 0]))
    y_span = float(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))

    summary = np.array(
        [
            left_eye_ratio,
            right_eye_ratio,
            abs(left_eye_ratio - right_eye_ratio),
            brow_eye_left,
            brow_eye_right,
            abs(brow_eye_left - brow_eye_right),
            mouth_width,
            mouth_open,
            mouth_open / (mouth_width + 1e-6),
            _mean_axis(points, [48, 54], 0),
            _mean_axis(points, [51, 57], 1),
            face_width,
            face_height,
            y_span / (x_span + 1e-6),
            x_span,
            y_span,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def _degree_face_values(data: dict[str, Any]) -> Any:
    if "face_keypoints_2d" in data:
        return data["face_keypoints_2d"]

    people = data.get("people")
    if isinstance(people, list) and people:
        best_face = None
        best_confidence = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue
            face = person.get("face_keypoints_2d")
            if face is None:
                continue

            array = np.asarray(face, dtype=np.float32).flatten()
            if len(array) > 0 and len(array) % 3 == 0:
                confidence = float(np.mean(array.reshape(-1, 3)[:, 2]))
            else:
                confidence = 1.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_face = face

        if best_face is not None:
            return best_face

    if isinstance(people, dict) and people.get("face_keypoints_2d") is not None:
        return people["face_keypoints_2d"]

    return _recursive_find_key(data, "face_keypoints_2d")


def _face_points_and_confidence(values: Any) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        raise ValueError("face_keypoints_2d is empty.")

    if len(array) % 3 == 0:
        array_3d = array.reshape(-1, 3)
        return array_3d[:, :2].astype(np.float32), float(np.mean(array_3d[:, 2]))
    if len(array) % 2 == 0:
        return array.reshape(-1, 2).astype(np.float32), 1.0

    raise ValueError(
        "face_keypoints_2d length must be divisible by 3 or 2, "
        f"got {len(array)}"
    )


def _degree_frame_feature(
    points: np.ndarray,
    previous_normalized: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    summary = _extract_degree_summary(points)
    normalized = _normalize_face_points(points).flatten().astype(np.float32)

    if len(normalized) > 132:
        normalized = normalized[:132]
    if len(normalized) < 132:
        padded = np.zeros(132, dtype=np.float32)
        padded[: len(normalized)] = normalized
        normalized = padded

    delta = (
        np.zeros_like(normalized, dtype=np.float32)
        if previous_normalized is None
        else normalized - previous_normalized
    )
    feature = np.concatenate([summary, normalized, delta]).astype(np.float32)

    if feature.shape != (DEGREE_FEATURE_DIM,):
        raise ValueError(
            f"Degree frame shape mismatch: {feature.shape}, "
            f"expected ({DEGREE_FEATURE_DIM},)"
        )
    return feature, normalized


def preprocess_degree_openpose_frames(
    frames: Sequence[dict[str, Any]],
    min_face_confidence: float = 0.05,
) -> tuple[np.ndarray, bool]:
    features = []
    previous_normalized = None

    for frame in frames:
        face_values = _degree_face_values(frame)
        if face_values is None:
            continue

        try:
            points, face_confidence = _face_points_and_confidence(face_values)
            if face_confidence < min_face_confidence:
                continue

            feature, previous_normalized = _degree_frame_feature(
                points, previous_normalized
            )
            features.append(feature)
        except (TypeError, ValueError):
            continue

    if not features:
        return np.zeros((0, DEGREE_FEATURE_DIM), dtype=np.float32), False

    return np.stack(features, axis=0).astype(np.float32), True


def extract_degree_280_sequence_from_video(
    video_path: str,
    min_face_confidence: float = 0.05,
) -> tuple[np.ndarray, bool]:
    frames = extract_openpose_frames_from_video(video_path)
    return preprocess_degree_openpose_frames(frames, min_face_confidence)


def _keypoints_detected(values: Any) -> bool:
    if values is None:
        return False

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        return False
    if len(array) % 3 == 0:
        return bool(np.any(array.reshape(-1, 3)[:, 2] > 0))
    return bool(np.any(array))


def summarize_openpose_frames(
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    has_pose = False
    has_left_hand = False
    has_right_hand = False
    has_face = False

    for frame in frames:
        people = _get_people(frame)
        if not people and isinstance(frame, dict):
            people = [frame]

        for person in people:
            has_pose = has_pose or _keypoints_detected(
                person.get("pose_keypoints_2d")
            )
            has_left_hand = has_left_hand or _keypoints_detected(
                person.get("hand_left_keypoints_2d")
            )
            has_right_hand = has_right_hand or _keypoints_detected(
                person.get("hand_right_keypoints_2d")
            )
            has_face = has_face or _keypoints_detected(
                person.get("face_keypoints_2d")
            )

    return {
        "extractor": "openpose",
        "sequence_length": int(len(frames)),
        "frame_dim": WORD_FEATURE_DIM,
        "has_pose": has_pose,
        "has_left_hand": has_left_hand,
        "has_right_hand": has_right_hand,
        "has_face": has_face,
    }
