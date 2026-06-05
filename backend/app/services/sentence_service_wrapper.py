from pathlib import Path

import numpy as np

from app.services.video_keypoint_extractor import (
    extract_mediapipe_frames_from_video,
    extract_sentence_120_sequence_from_video,
    summarize_mediapipe_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SENTENCE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "gru_augmented_20260522-061239.keras",
    PROJECT_ROOT / "sen_AI" / "gru" / "gru_best.keras",
]

CLASSES_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "data" / "validation" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "data" / "processed" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "classes.npy",
]


_sentence_model = None
_sentence_model_path = None
_sentence_classes = None


def _find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_sentence_model():
    global _sentence_model, _sentence_model_path

    if _sentence_model is not None:
        return _sentence_model

    model_path = _find_existing_path(SENTENCE_MODEL_CANDIDATES)

    if model_path is None:
        searched = "\n".join(str(p) for p in SENTENCE_MODEL_CANDIDATES)
        raise FileNotFoundError(
            "sentence_AI 모델 파일을 찾을 수 없습니다.\n"
            "필요한 파일 후보:\n"
            f"{searched}"
        )

    import tensorflow as tf

    _sentence_model = tf.keras.models.load_model(str(model_path), compile=False)
    _sentence_model_path = model_path

    return _sentence_model


def get_sentence_classes():
    global _sentence_classes

    if _sentence_classes is not None:
        return _sentence_classes

    classes_path = _find_existing_path(CLASSES_CANDIDATES)

    if classes_path is None:
        searched = "\n".join(str(p) for p in CLASSES_CANDIDATES)
        raise FileNotFoundError(
            "sentence_AI classes.npy 파일을 찾을 수 없습니다.\n"
            "필요한 파일 후보:\n"
            f"{searched}"
        )

    _sentence_classes = np.load(classes_path, allow_pickle=True)

    model = get_sentence_model()
    output_dim = int(model.output_shape[-1])
    if len(_sentence_classes) != output_dim:
        raise ValueError(
            f"sentence classes 개수({len(_sentence_classes)})와 "
            f"모델 출력 차원({output_dim})이 다릅니다."
        )

    return _sentence_classes


def _predict_sentence(sequence_120):
    model = get_sentence_model()
    classes = get_sentence_classes()

    x = np.expand_dims(sequence_120, axis=0).astype(np.float32)
    probs = model.predict(x, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:3]

    top_k = []
    for idx in top_indices:
        idx_int = int(idx)
        text = str(classes[idx_int]) if idx_int < len(classes) else f"class_{idx_int}"

        top_k.append({
            "label": idx_int,
            "text": text,
            "confidence": float(probs[idx_int]),
        })

    best = top_k[0]

    return {
        "text": best["text"],
        "confidence": best["confidence"],
        "label": best["label"],
        "top_k": top_k,
    }


def predict_sentence(video_path: str) -> dict:
    """
    Run the sentence GRU with MediaPipe keypoints in the training-compatible shape.
    """
    try:
        sequence_120 = extract_sentence_120_sequence_from_video(
            video_path, target_frames=30
        )
        summary = summarize_mediapipe_frames(extract_mediapipe_frames_from_video(video_path))

        result = _predict_sentence(sequence_120)

        return {
            "text": result["text"],
            "confidence": float(result["confidence"]),
            "label": result["label"],
            "status": "success",
            "top_k": result["top_k"],
            "model_status": "sentence_ai_model_connected",
            "model_path": str(_sentence_model_path),
            "keypoint_extractor": summary["extractor"],
            "model_input_shape": list(sequence_120.shape),
            "model_input_type": "30F×120D sentence sequence",
            "source_keypoint_shape": [summary["sequence_length"], summary["frame_dim"]],
            "keypoint_summary": summary,
            "message": "MediaPipe keypoints were preprocessed with the sentence-training pipeline before GRU inference.",
        }

    except Exception as e:
        return {
            "text": "문장 인식 실패",
            "confidence": 0.0,
            "label": None,
            "status": "error",
            "top_k": [],
            "model_status": "sentence_ai_error",
            "message": str(e),
        }
