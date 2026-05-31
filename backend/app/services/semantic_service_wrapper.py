def apply_semantic_postprocess(
    mode: str,
    text: str,
    degree: str,
    degree_ko: str
) -> dict:
    """
    단어/문장 결과와 표현 강도를 결합하는 규칙 기반 후처리.
    실제 Ollama 연결 전에도 최종 결과 문장을 안정적으로 만들어준다.
    """

    if not text:
        text = "인식 결과 없음"

    # keypoint 확인용 결과는 문장 보정하지 않음
    if "keypoint" in text or "실패" in text or "대기" in text:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "모델 추론 결과가 아닌 상태 메시지이므로 원문을 유지했습니다.",
            "processor_status": "rule_fallback",
        }

    emotion_keywords = [
        "화나다",
        "아프다",
        "힘들다",
        "슬프다",
        "감사합니다",
        "감사",
        "도와주세요",
        "필요",
        "싫다",
        "좋다",
    ]

    is_emotion_or_state = any(keyword in text for keyword in emotion_keywords)

    if not is_emotion_or_state:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "일반 단어 또는 일반 문장으로 판단되어 표현 강도를 직접 반영하지 않았습니다.",
            "processor_status": "rule_fallback",
        }

    if degree == "strong":
        modifier = "매우"
        final_text = f"{modifier} {text}"

        return {
            "apply_degree": True,
            "final_text": final_text,
            "target_expression": text,
            "modifier": modifier,
            "reason": "감정 또는 상태 표현으로 판단되어 strong 강도를 반영했습니다.",
            "processor_status": "rule_fallback",
        }

    if degree == "weak":
        modifier = "조금"
        final_text = f"{modifier} {text}"

        return {
            "apply_degree": True,
            "final_text": final_text,
            "target_expression": text,
            "modifier": modifier,
            "reason": "감정 또는 상태 표현으로 판단되어 weak 강도를 반영했습니다.",
            "processor_status": "rule_fallback",
        }

    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": text,
        "modifier": "",
        "reason": "normal 강도이므로 원문을 유지했습니다.",
        "processor_status": "rule_fallback",
    }