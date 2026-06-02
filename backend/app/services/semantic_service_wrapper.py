from app.services.word_service_wrapper import predict_word


def make_sentence_from_word(word: str) -> str:
    """
    sentence_AI 실제 추론 모델 연결 전,
    word_AI 결과를 문장형 출력으로 변환하는 fallback.
    """
    if not word or word in ["인식불가", "인식 대기", "영상 분석 실패", "keypoint 추출 성공"]:
        return word

    sentence_map = {
        "화나다": "나는 화났다",
        "아프다": "몸이 아픕니다",
        "감사합니다": "정말 감사합니다",
        "도와주세요": "도움이 필요합니다",
        "싫다": "나는 싫습니다",
        "좋다": "나는 좋습니다",
        "미안합니다": "미안합니다",
        "안녕하세요": "안녕하세요",
    }

    return sentence_map.get(word, word)


def predict_sentence(video_path: str) -> dict:
    """
    문장 수어 분석 서비스.
    현재 sentence_AI는 부분구현 범위이므로, word_AI 예측 결과를 기반으로 문장형 결과를 구성한다.
    추후 sentence_AI 실제 모델이 확정되면 이 함수 내부만 교체하면 된다.
    """
    word_result = predict_word(video_path)

    word_text = word_result.get("text", "인식불가")
    sentence_text = make_sentence_from_word(word_text)
    confidence = float(word_result.get("confidence", 0.0))

    return {
        "text": sentence_text,
        "confidence": confidence,
        "status": word_result.get("status", "success"),
        "source_word": word_text,
        "word_id": word_result.get("word_id"),
        "top_k": word_result.get("top_k", []),
        "keypoint_summary": word_result.get("keypoint_summary", {}),
        "model_status": "sentence_fallback_from_word_ai",
        "message": "sentence_AI 실제 모델 연결 전, word_AI 결과를 기반으로 문장형 결과를 구성했습니다."
    }