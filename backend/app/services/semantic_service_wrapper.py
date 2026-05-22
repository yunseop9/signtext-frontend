def apply_semantic_postprocess(
    mode: str,
    text: str,
    degree: str,
    degree_ko: str
) -> dict:
    emotion_keywords = ["화나다", "아프다", "슬프다", "힘들다", "감사합니다", "좋다"]

    is_emotion_text = any(keyword in text for keyword in emotion_keywords)

    if not is_emotion_text:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "일반 행동 또는 사실 문장으로 판단되어 원문을 유지함"
        }

    if degree == "strong":
        return {
            "apply_degree": True,
            "final_text": f"매우 {text}",
            "target_expression": text,
            "modifier": "매우",
            "reason": "감정 표현에 strong 강도를 반영함"
        }

    if degree == "weak":
        return {
            "apply_degree": True,
            "final_text": f"조금 {text}",
            "target_expression": text,
            "modifier": "조금",
            "reason": "감정 표현에 weak 강도를 반영함"
        }

    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": text,
        "modifier": "",
        "reason": "normal 강도이므로 원문을 유지함"
    }