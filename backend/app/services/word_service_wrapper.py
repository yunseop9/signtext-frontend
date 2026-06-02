import sys
from pathlib import Path

from app.services.video_keypoint_extractor import (
    extract_411d_sequence_from_video,
    summarize_keypoint_sequence,
    extract_hands_126_from_411d,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

WORD_SERVICE_DIR = (
    PROJECT_ROOT
    / "word_AI"
    / "Final_Model_GRU"
    / "services"
)

WORD_MODEL_DIR = (
    PROJECT_ROOT
    / "word_AI"
    / "Final_Model_GRU"
    / "artifacts"
    / "Final_GRU_HANDS_126D"
    / "models"
)

if str(WORD_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(WORD_SERVICE_DIR))

try:
    from word_service import GRUWordInference
except Exception as e:
    GRUWordInference = None
    WORD_IMPORT_ERROR = e
else:
    WORD_IMPORT_ERROR = None


_word_engine = None


def get_word_engine():
    """
    word_AI GRU 모델을 한 번만 로딩해서 재사용한다.
    """
    global _word_engine

    if _word_engine is not None:
        return _word_engine

    if GRUWordInference is None:
        raise RuntimeError(f"word_service import 실패: {WORD_IMPORT_ERROR}")

    model_path = WORD_MODEL_DIR / "best_model.keras"
    label_path = WORD_MODEL_DIR / "label_map.json"
    mapping_path = WORD_MODEL_DIR / "word_label_mapping.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"word 모델 파일을 찾을 수 없습니다: {model_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"label_map 파일을 찾을 수 없습니다: {label_path}")

    if not mapping_path.exists():
        raise FileNotFoundError(f"word_label_mapping 파일을 찾을 수 없습니다: {mapping_path}")

    _word_engine = GRUWordInference(
        str(model_path),
        str(label_path),
        str(mapping_path),
        threshold=0.4,
    )

    return _word_engine


def _predict_with_gru(sequence):
    """
    30F x 411D sequence를 GRUWordInference에 넣어 단어를 예측한다.
    """
    engine = get_word_engine()
    engine.clear_buffer()

    for frame in sequence:
        engine.add_frame(frame)

    result = engine.predict()

    if result is None:
        return {
            "text": "인식 대기",
            "confidence": 0.0,
            "status": "processing",
            "top_k": [],
        }

    return result


def predict_word(video_path: str) -> dict:
    """
    업로드된 mp4 영상에서 keypoint를 추출하고,
    실제 word_AI GRU 모델로 단어 예측을 시도한다.

    모델 연결에 실패해도 keypoint 추출 결과는 반환하여
    백엔드 시연 흐름이 끊기지 않게 한다.
    """
    try:
        sequence = extract_411d_sequence_from_video(video_path, target_frames=30)
        summary = summarize_keypoint_sequence(sequence)
        hands_126 = extract_hands_126_from_411d(sequence)
        try:
            word_result = _predict_with_gru(sequence)

            top_k = word_result.get("top3", word_result.get("top_k", []))

            return {
                "text": word_result.get("text", "인식불가"),
                "confidence": float(word_result.get("confidence", 0.0)),
                "word_id": word_result.get("word_id"),
                "status": word_result.get("status", "success"),
                "top_k": top_k,
                "model_status": "word_ai_gru_connected",
                "model_input_shape": list(hands_126.shape),
                "model_input_type": "30F×126D hands keypoint",
                "source_keypoint_shape": [summary["sequence_length"], summary["frame_dim"]],
                "keypoint_summary": summary,
                "message": "영상에서 30F x 411D keypoint를 추출하고 word_AI GRU 모델 추론을 수행했습니다.",
            }

        except Exception as model_error:
            return {
                "text": "keypoint 추출 성공",
                "confidence": 1.0,
                "word_id": None,
                "status": "keypoint_only",
                "top_k": [],
                "model_status": "keypoint_extracted_model_error",
                "keypoint_summary": summary,
                "message": f"keypoint 추출은 성공했지만 word_AI 모델 연결 중 오류가 발생했습니다: {model_error}",
            }

    except Exception as e:
        return {
            "text": "영상 분석 실패",
            "confidence": 0.0,
            "word_id": None,
            "status": "error",
            "top_k": [],
            "model_status": "video_keypoint_error",
            "message": str(e),
        }