from app.services.word_service_wrapper import predict_word


def predict_sentence(video_path: str) -> dict:
    """
    sentence_AI는 현재 부분구현 범위로 두고,
    업로드 영상에서 word_AI 결과를 기반으로 문장형 결과를 구성한다.
    """
    word_result = predict_word(video_path)

    text = word_result.get("text", "인식 결과 없음")
    confidence = float(word_result.get("confidence", 0.0))

    if text in ["영상 분석 실패", "keypoint 추출 성공", "인식 대기", "인식불가"]:
        sentence = text
    else:
        sentence = text

    return {
        "text": sentence,
        "confidence": confidence,
        "status": word_result.get("status", "success"),
        "model_status": "sentence_fallback_from_word",
        "message": "sentence_AI 실제 연결 전 word_AI 결과를 문장 결과로 사용했습니다.",
    }