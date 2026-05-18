"""
integrated_sentence_degree_eval.py

sentence_AI + degree_AI 통합 end-to-end 성능검증 코드.

기존 코드와 역할 구분:
1. degree_manual_eval/evaluate_manual_degree.py
   - degree_AI 단독 성능검증
   - true_degree vs pred_degree 비교

2. sen_AI+degree_AI/sentence_degree_eval.py
   - 문장 + degree 후처리 로직 검증
   - raw_text + degree -> expected_final_text 비교

3. 이 파일
   - sentence_AI 예측 + degree_AI 예측을 합친 최종 성능검증
   - true_sentence, pred_sentence, true_degree, pred_degree, true_final_text, pred_final_text 비교

입력 CSV 기본 위치:
    sen_AI+degree_AI/inputs/integrated_sentence_degree_eval_input.csv

필수 컬럼:
    sample_id,true_sentence,pred_sentence,true_degree,pred_degree

선택 컬럼:
    true_final_text,pred_final_text,pred_degree_ko

실행:
    python integrated_sentence_degree_eval.py

또는:
    python integrated_sentence_degree_eval.py --input inputs/my_eval.csv

pred_final_text가 비어 있을 때 semantic_sentence_postprocessor로 자동 생성:
    python integrated_sentence_degree_eval.py --generate-final-text

Ollama까지 사용해서 후처리:
    python integrated_sentence_degree_eval.py --generate-final-text --use-llm
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = BASE_DIR / "inputs" / "integrated_sentence_degree_eval_input.csv"
DEFAULT_OUT_DIR = BASE_DIR / "outputs" / "integrated_sentence_degree_eval"

LABEL_ORDER = ["weak", "normal", "strong"]
VALID_DEGREES = set(LABEL_ORDER)

DEGREE_KO = {
    "weak": "약함",
    "normal": "보통",
    "strong": "강함",
}

COLUMN_ALIASES = {
    "sample_id": [
        "sample_id",
        "sequence_name",
        "sequence",
        "video_uid",
        "name",
        "id",
    ],
    "true_sentence": [
        "true_sentence",
        "label_sentence",
        "manual_sentence",
        "ground_truth_sentence",
        "sentence_true",
        "expected_sentence",
    ],
    "pred_sentence": [
        "pred_sentence",
        "sentence_pred",
        "prediction_sentence",
        "ai_sentence",
        "raw_text",
        "pred_text",
    ],
    "true_degree": [
        "true_degree",
        "manual_degree",
        "label_degree",
        "ground_truth_degree",
        "degree_true",
        "expected_degree",
    ],
    "pred_degree": [
        "pred_degree",
        "degree_pred",
        "prediction_degree",
        "ai_degree",
        "degree",
    ],
    "true_final_text": [
        "true_final_text",
        "expected_final_text",
        "label_final_text",
        "ground_truth_final_text",
    ],
    "pred_final_text": [
        "pred_final_text",
        "actual_final_text",
        "ai_final_text",
        "final_text",
    ],
    "pred_degree_ko": [
        "pred_degree_ko",
        "degree_ko",
    ],
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_degree(value: Any) -> str:
    value = normalize_text(value).lower()

    mapping = {
        "약함": "weak",
        "약": "weak",
        "weak": "weak",
        "w": "weak",
        "0": "weak",

        "보통": "normal",
        "중간": "normal",
        "normal": "normal",
        "n": "normal",
        "1": "normal",

        "강함": "strong",
        "강": "strong",
        "strong": "strong",
        "s": "strong",
        "2": "strong",
    }

    return mapping.get(value, value)


def find_column(
    fieldnames: Iterable[str],
    aliases: List[str],
    required: bool = True,
) -> Optional[str]:
    columns = list(fieldnames)
    lower_map = {col.lower().strip(): col for col in columns}

    for alias in aliases:
        key = alias.lower().strip()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise ValueError(
            f"필수 컬럼을 찾지 못했습니다. 후보={aliases}, 실제 컬럼={columns}"
        )

    return None


def get_optional(row: Dict[str, Any], col: Optional[str]) -> str:
    if col is None:
        return ""
    return normalize_text(row.get(col, ""))


def generate_final_text(
    sentence: str,
    degree: str,
    degree_ko: str,
    use_llm: bool,
) -> str:
    """
    pred_final_text가 입력 CSV에 없을 때만 사용한다.
    같은 폴더의 semantic_sentence_postprocessor.py를 호출한다.
    """
    if not sentence:
        return ""

    try:
        from semantic_sentence_postprocessor import semantic_sentence_postprocess
    except ImportError as e:
        raise ImportError(
            "pred_final_text가 없어서 자동 생성이 필요하지만 "
            "semantic_sentence_postprocessor.py를 import할 수 없습니다. "
            "같은 폴더에 파일이 있는지 확인하세요."
        ) from e

    result = semantic_sentence_postprocess(
        sentence_result={
            "text": sentence,
            "confidence": 1.0,
        },
        degree_result={
            "degree": degree,
            "degree_ko": degree_ko or DEGREE_KO.get(degree, ""),
            "confidence": 1.0,
        },
        use_llm=use_llm,
    )

    return normalize_text(result.get("final_text", ""))


def load_eval_rows(
    input_csv: Path,
    generate_missing_final_text: bool,
    use_llm: bool,
) -> List[Dict[str, Any]]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"입력 CSV가 없습니다: {input_csv}\n"
            "필수 컬럼: sample_id,true_sentence,pred_sentence,true_degree,pred_degree"
        )

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        cols = {
            key: find_column(
                fieldnames,
                aliases,
                required=key not in {
                    "true_final_text",
                    "pred_final_text",
                    "pred_degree_ko",
                },
            )
            for key, aliases in COLUMN_ALIASES.items()
        }

        rows: List[Dict[str, Any]] = []

        for raw in reader:
            sample_id = normalize_text(raw.get(cols["sample_id"], ""))
            true_sentence = normalize_text(raw.get(cols["true_sentence"], ""))
            pred_sentence = normalize_text(raw.get(cols["pred_sentence"], ""))

            true_degree = normalize_degree(raw.get(cols["true_degree"], ""))
            pred_degree = normalize_degree(raw.get(cols["pred_degree"], ""))

            pred_degree_ko = (
                get_optional(raw, cols.get("pred_degree_ko"))
                or DEGREE_KO.get(pred_degree, "")
            )

            true_final_text = get_optional(raw, cols.get("true_final_text"))
            pred_final_text = get_optional(raw, cols.get("pred_final_text"))

            if true_degree not in VALID_DEGREES:
                raise ValueError(
                    f"잘못된 true_degree: sample_id={sample_id}, value={true_degree}"
                )

            if pred_degree not in VALID_DEGREES:
                raise ValueError(
                    f"잘못된 pred_degree: sample_id={sample_id}, value={pred_degree}"
                )

            if not pred_final_text and generate_missing_final_text:
                pred_final_text = generate_final_text(
                    sentence=pred_sentence,
                    degree=pred_degree,
                    degree_ko=pred_degree_ko,
                    use_llm=use_llm,
                )

            rows.append(
                {
                    "sample_id": sample_id,
                    "true_sentence": true_sentence,
                    "pred_sentence": pred_sentence,
                    "true_degree": true_degree,
                    "pred_degree": pred_degree,
                    "true_final_text": true_final_text,
                    "pred_final_text": pred_final_text,
                }
            )

    if not rows:
        raise ValueError("입력 CSV에 평가 행이 없습니다.")

    return rows


def accuracy(flags: List[bool]) -> Optional[float]:
    if not flags:
        return None
    return round(sum(1 for flag in flags if flag) / len(flags), 4)


def build_degree_confusion_matrix(
    rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, int]]:
    matrix = {
        true_label: {pred_label: 0 for pred_label in LABEL_ORDER}
        for true_label in LABEL_ORDER
    }

    for row in rows:
        matrix[row["true_degree"]][row["pred_degree"]] += 1

    return matrix


def evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated_rows = []

    sentence_flags: List[bool] = []
    degree_flags: List[bool] = []
    joint_flags: List[bool] = []
    final_text_flags: List[bool] = []

    for row in rows:
        sentence_correct = row["true_sentence"] == row["pred_sentence"]
        degree_correct = row["true_degree"] == row["pred_degree"]
        joint_correct = sentence_correct and degree_correct

        has_final_target = bool(row["true_final_text"] and row["pred_final_text"])

        if has_final_target:
            final_text_correct = row["true_final_text"] == row["pred_final_text"]
        else:
            final_text_correct = None

        sentence_flags.append(sentence_correct)
        degree_flags.append(degree_correct)
        joint_flags.append(joint_correct)

        if final_text_correct is not None:
            final_text_flags.append(final_text_correct)

        evaluated_rows.append(
            {
                **row,
                "sentence_correct": int(sentence_correct),
                "degree_correct": int(degree_correct),
                "joint_correct": int(joint_correct),
                "final_text_correct": (
                    "" if final_text_correct is None else int(final_text_correct)
                ),
            }
        )

    metrics = {
        "total": len(evaluated_rows),

        "sentence_correct": int(sum(sentence_flags)),
        "degree_correct": int(sum(degree_flags)),
        "joint_correct": int(sum(joint_flags)),

        "sentence_accuracy": accuracy(sentence_flags),
        "degree_accuracy": accuracy(degree_flags),
        "joint_accuracy": accuracy(joint_flags),

        "final_text_accuracy": accuracy(final_text_flags),
        "final_text_evaluable_count": len(final_text_flags),

        "degree_confusion_matrix": build_degree_confusion_matrix(rows),
    }

    return {
        "rows": evaluated_rows,
        "metrics": metrics,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_degree_confusion_matrix_csv(
    path: Path,
    matrix: Dict[str, Dict[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *LABEL_ORDER])

        for true_label in LABEL_ORDER:
            writer.writerow(
                [
                    true_label,
                    *[matrix[true_label][pred_label] for pred_label in LABEL_ORDER],
                ]
            )


def build_report(metrics: Dict[str, Any], out_dir: Path) -> str:
    report = f"""
sentence_AI + degree_AI 통합 성능검증 리포트
===========================================

검증 목적:
- sentence_AI가 수어 문장 의미를 맞혔는가
- degree_AI가 얼굴표현 정도를 맞혔는가
- 문장과 degree가 동시에 맞았는가
- 최종 텍스트가 정답 최종 텍스트와 같은가

샘플 수:
- 전체 샘플 수: {metrics["total"]}
- 문장 인식 정답 수: {metrics["sentence_correct"]}
- degree 정답 수: {metrics["degree_correct"]}
- 문장+degree 동시 정답 수: {metrics["joint_correct"]}

정확도:
- sentence_accuracy: {metrics["sentence_accuracy"]}
- degree_accuracy: {metrics["degree_accuracy"]}
- joint_accuracy: {metrics["joint_accuracy"]}
- final_text_accuracy: {metrics["final_text_accuracy"]}
- final_text 평가 가능 샘플 수: {metrics["final_text_evaluable_count"]}

해석:
- sentence_accuracy = true_sentence == pred_sentence
- degree_accuracy = true_degree == pred_degree
- joint_accuracy = sentence_correct AND degree_correct
- final_text_accuracy = true_final_text == pred_final_text

생성 파일:
- {out_dir / "integrated_sentence_degree_eval_results.csv"}
- {out_dir / "integrated_sentence_degree_eval_metrics.json"}
- {out_dir / "integrated_sentence_degree_fail_cases.csv"}
- {out_dir / "integrated_sentence_degree_confusion_matrix.csv"}
- {out_dir / "integrated_sentence_degree_eval_report.txt"}
""".strip()

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help="통합 평가 입력 CSV 경로",
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="결과 저장 폴더",
    )

    parser.add_argument(
        "--generate-final-text",
        action="store_true",
        help="pred_final_text가 비어 있을 때 semantic_sentence_postprocessor로 자동 생성",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="--generate-final-text 사용 시 Ollama 기반 LLM 후처리 사용",
    )

    args = parser.parse_args()

    input_csv = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(
        input_csv=input_csv,
        generate_missing_final_text=args.generate_final_text,
        use_llm=args.use_llm,
    )

    result = evaluate_rows(rows)
    evaluated_rows = result["rows"]
    metrics = result["metrics"]

    result_csv = out_dir / "integrated_sentence_degree_eval_results.csv"
    metrics_json = out_dir / "integrated_sentence_degree_eval_metrics.json"
    fail_csv = out_dir / "integrated_sentence_degree_fail_cases.csv"
    confusion_csv = out_dir / "integrated_sentence_degree_confusion_matrix.csv"
    report_txt = out_dir / "integrated_sentence_degree_eval_report.txt"

    write_csv(result_csv, evaluated_rows)

    fail_rows = [
        row
        for row in evaluated_rows
        if row["joint_correct"] == 0 or row.get("final_text_correct") == 0
    ]
    write_csv(fail_csv, fail_rows)

    write_degree_confusion_matrix_csv(
        confusion_csv,
        metrics["degree_confusion_matrix"],
    )

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = build_report(metrics, out_dir)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()