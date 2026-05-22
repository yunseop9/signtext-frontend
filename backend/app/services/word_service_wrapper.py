def predict_word(video_path: str) -> dict:
    return {
        "text": "화나다",
        "confidence": 0.87,
        "top_k": [
            {"text": "화나다", "confidence": 0.87},
            {"text": "싫다", "confidence": 0.08},
            {"text": "아프다", "confidence": 0.05}
        ]
    }