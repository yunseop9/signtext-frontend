from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


DEGREE_MAP = {
    "weak": "weak",
    "normal": "normal",
    "strong": "strong",
    "약함": "weak",
    "보통": "normal",
    "강함": "strong",
}


CATEGORY_RULES = {
    "emotion": {
        "weak": "조금 {text}",
        "normal": "{text}",
        "strong": "매우 {text}",
    },
    "pain": {
        "weak": "조금 {text}",
        "normal": "{text}",
        "strong": "많이 {text}",
    },
    "positive": {
        "weak": "약간 {text}",
        "normal": "{text}",
        "strong": "정말 {text}",
    },
    "sensation": {
        "weak": "조금 {text}",
        "normal": "{text}",
        "strong": "너무 {text}",
    },
    "request": {
        "weak": "{text}",
        "normal": "{text}",
        "strong": "꼭 {text}",
    },
}


# 자연스럽게 바꿔야 하는 핵심 표현만 예외 처리
OVERRIDES = {
    "화난다": {
        "weak": "조금 화났어요",
        "normal": "화났어요",
        "strong": "매우 화났어요",
    },
    "아프다": {
        "weak": "조금 아파요",
        "normal": "아파요",
        "strong": "많이 아파요",
    },
    "좋다": {
        "weak": "괜찮아요",
        "normal": "좋아요",
        "strong": "정말 좋아요",
    },
    "싫다": {
        "weak": "조금 싫어요",
        "normal": "싫어요",
        "strong": "정말 싫어요",
    },
    "춥다": {
        "weak": "조금 추워요",
        "normal": "추워요",
        "strong": "너무 추워요",
    },
    "덥다": {
        "weak": "조금 더워요",
        "normal": "더워요",
        "strong": "너무 더워요",
    },
}


def normalize_degree(degree: str) -> str:
    degree = str(degree).strip()
    return DEGREE_MAP.get(degree, "normal")


def load_label_meta(csv_path: str | Path) -> dict[str, dict]:
    """
    label_meta.csv 형식:
    label,text,category,degree_applicable
    WORD00001,화난다,emotion,true
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    result = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row.get("label", "").strip()
            if not label:
                continue

            text = row.get("text", "").strip()
            category = row.get("category", "none").strip()

            degree_applicable_raw = row.get("degree_applicable", "false").strip().lower()
            degree_applicable = degree_applicable_raw in {"true", "1", "yes", "y"}

            result[label] = {
                "text": text,
                "category": category,
                "degree_applicable": degree_applicable,
            }

    return result


def infer_category(text: str) -> str:
    """
    label_meta.csv가 부족할 때 쓰는 간단한 fallback.
    정확도는 낮으므로, 최종 제출용은 label_meta.csv 권장.
    """
    emotion_keywords = ["화", "짜증", "슬프", "무섭", "걱정", "불안"]
    pain_keywords = ["아프", "힘들", "피곤"]
    positive_keywords = ["좋", "기쁘", "행복", "즐겁"]
    sensation_keywords = ["춥", "덥", "배고", "목마"]

    if any(k in text for k in emotion_keywords):
        return "emotion"
    if any(k in text for k in pain_keywords):
        return "pain"
    if any(k in text for k in positive_keywords):
        return "positive"
    if any(k in text for k in sensation_keywords):
        return "sensation"

    return "none"


def apply_degree_to_text(
    text: str,
    degree: str,
    category: Optional[str] = None,
    degree_applicable: bool = True,
) -> str:
    degree = normalize_degree(degree)
    text = str(text).strip()

    if not text:
        return text

    if not degree_applicable:
        return text

    if degree == "normal":
        # normal은 보통 원문 그대로
        if text in OVERRIDES:
            return OVERRIDES[text]["normal"]
        return text

    # 1순위: 예외 처리
    if text in OVERRIDES:
        return OVERRIDES[text].get(degree, text)

    # 2순위: category 기반 처리
    if category is None or category == "":
        category = infer_category(text)

    if category not in CATEGORY_RULES:
        return text

    template = CATEGORY_RULES[category].get(degree, "{text}")
    return template.format(text=text)


def postprocess_prediction(
    sign_result: dict,
    degree_result: dict,
    label_meta: Optional[dict[str, dict]] = None,
) -> dict:
    """
    sign_result 예시:
    {
        "label": "WORD00001",
        "text": "화난다"
    }

    degree_result 예시:
    {
        "degree": "strong",
        "confidence": 0.82
    }
    """
    label = sign_result.get("label", "")
    text = sign_result.get("text", "")
    degree = degree_result.get("degree", "normal")

    category = None
    degree_applicable = True

    if label_meta is not None and label in label_meta:
        meta = label_meta[label]
        text = meta.get("text", text)
        category = meta.get("category", None)
        degree_applicable = bool(meta.get("degree_applicable", True))

    final_text = apply_degree_to_text(
        text=text,
        degree=degree,
        category=category,
        degree_applicable=degree_applicable,
    )

    return {
        "label": label,
        "base_text": text,
        "degree": normalize_degree(degree),
        "degree_confidence": degree_result.get("confidence", None),
        "category": category,
        "degree_applicable": degree_applicable,
        "final_text": final_text,
    }


if __name__ == "__main__":
    examples = [
        (
            {"label": "WORD00001", "text": "화난다"},
            {"degree": "weak", "confidence": 0.71},
        ),
        (
            {"label": "WORD00001", "text": "화난다"},
            {"degree": "strong", "confidence": 0.82},
        ),
        (
            {"label": "WORD00002", "text": "아프다"},
            {"degree": "strong", "confidence": 0.79},
        ),
        (
            {"label": "WORD99999", "text": "학교"},
            {"degree": "strong", "confidence": 0.90},
        ),
    ]

    for sign_result, degree_result in examples:
        print(postprocess_prediction(sign_result, degree_result))