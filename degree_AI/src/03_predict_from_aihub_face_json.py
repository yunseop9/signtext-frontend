import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from degree_features import extract_degree_features_from_points


def load_aihub_face_points(json_path: Path) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", {})
    face = people.get("face_keypoints_2d", None)

    if face is None:
        raise KeyError("face_keypoints_2d not found in AIHub JSON")

    arr = np.asarray(face, dtype=np.float32)

    if len(arr) % 3 != 0:
        raise ValueError(f"face_keypoints_2d length must be divisible by 3, got {len(arr)}")

    arr = arr.reshape(-1, 3)

    # x, y만 사용. confidence는 현재 feature에는 사용하지 않음.
    points = arr[:, :2]

    return points


def predict_degree(model_path: Path, json_path: Path) -> dict:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_names = bundle["label_names"]

    points = load_aihub_face_points(json_path)
    features = extract_degree_features_from_points(points).reshape(1, -1)

    pred_idx = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probs_arr = model.predict_proba(features)[0]
    else:
        probs_arr = np.zeros(len(label_names), dtype=np.float32)
        probs_arr[pred_idx] = 1.0

    probs = {
        label_names[i]: float(probs_arr[i])
        for i in range(len(label_names))
    }

    result = {
        "degree": label_names[pred_idx],
        "confidence": float(probs_arr[pred_idx]),
        "probs": probs
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/degree_anger_rf.joblib")
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    result = predict_degree(
        model_path=Path(args.model),
        json_path=Path(args.json)
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()