import numpy as np

from app.services.video_keypoint_extractor import extract_411d_sequence_from_video


def _movement_score(values: np.ndarray) -> float:
    """
    프레임 간 변화량 평균을 계산한다.
    """
    if values.ndim != 2 or len(values) < 2:
        return 0.0

    diff = np.abs(np.diff(values, axis=0))
    return float(np.mean(diff))


def predict_degree(video_path: str) -> dict:
    """
    실제 degree_AI 모델 파일이 백엔드에 연결되기 전,
    업로드 영상에서 추출한 face keypoint 변화량을 기반으로 weak/normal/strong을 추정한다.

    이건 완전한 학습 모델은 아니지만, mock이 아니라 실제 영상 keypoint 변화량을 사용한다.
    """
    try:
        sequence = extract_411d_sequence_from_video(video_path, target_frames=30)
        arr = np.asarray(sequence, dtype=np.float32)

        # 411D 구성: pose 0:75, left 75:138, right 138:201, face 201:411
        face = arr[:, 201:411]
        hands = arr[:, 75:201]

        face_score = _movement_score(face)
        hand_score = _movement_score(hands)

        # 얼굴 움직임을 우선 사용하되 손동작 변화도 약간 반영
        score = face_score * 0.75 + hand_score * 0.25

        if score < 0.006:
            degree = "weak"
            degree_ko = "약함"
            prob_weak, prob_normal, prob_strong = 0.72, 0.22, 0.06
        elif score < 0.018:
            degree = "normal"
            degree_ko = "보통"
            prob_weak, prob_normal, prob_strong = 0.18, 0.68, 0.14
        else:
            degree = "strong"
            degree_ko = "강함"
            prob_weak, prob_normal, prob_strong = 0.08, 0.22, 0.70

        confidence = max(prob_weak, prob_normal, prob_strong)

        return {
            "degree": degree,
            "degree_ko": degree_ko,
            "confidence": float(confidence),
            "prob_weak": float(prob_weak),
            "prob_normal": float(prob_normal),
            "prob_strong": float(prob_strong),
            "score": float(score),
            "model_status": "keypoint_rule_based",
            "message": "영상에서 추출한 face/hand keypoint 변화량 기반으로 표현 강도를 추정했습니다.",
        }

    except Exception as e:
        return {
            "degree": "normal",
            "degree_ko": "보통",
            "confidence": 0.0,
            "prob_weak": 0.0,
            "prob_normal": 1.0,
            "prob_strong": 0.0,
            "score": 0.0,
            "model_status": "degree_error_fallback",
            "message": str(e),
        }