EMOTION_STATE_TERMS = [
    "화났다",
    "화나다",
    "슬프다",
    "무섭다",
    "행복하다",
    "기쁘다",
    "아프다",
    "피곤하다",
    "걱정된다",
    "불안하다",
    "좋다",
    "싫다",
    "힘들다",
    "우울하다",
    "짜증난다",
    "놀랐다",
]


def _modifier_for_degree(degree):
    if degree == "weak":
        return "조금"
    if degree == "strong":
        return "매우"
    return ""


def apply_semantic_postprocess(mode: str, text: str, degree: str, degree_ko: str) -> dict:
    text = str(text or "").strip()
    modifier = _modifier_for_degree(str(degree).lower())

    if not text or not modifier:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "텍스트가 비어 있거나 표현 정도가 보통이므로 원문을 유지합니다.",
            "processor_status": "rule",
        }

    if mode == "sentence":
        for term in sorted(EMOTION_STATE_TERMS, key=len, reverse=True):
            if term in text:
                return {
                    "apply_degree": True,
                    "final_text": text.replace(term, f"{modifier} {term}", 1),
                    "target_expression": term,
                    "modifier": modifier,
                    "reason": f"문장 내 감정·상태 표현에 {degree_ko} 정도를 반영했습니다.",
                    "processor_status": "rule_guard",
                }

        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "일반 문장으로 판단하여 원문을 유지합니다.",
            "processor_status": "rule_guard",
        }

    return {
        "apply_degree": True,
        "final_text": f"{modifier} {text}",
        "target_expression": text,
        "modifier": modifier,
        "reason": f"단어 결과에 {degree_ko} 정도를 반영했습니다.",
        "processor_status": "rule",
    }
