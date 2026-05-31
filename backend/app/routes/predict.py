import os
import uuid
import json
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.word_service_wrapper import predict_word
from app.services.sentence_service_wrapper import predict_sentence
from app.services.degree_service_wrapper import predict_degree
from app.services.semantic_service_wrapper import apply_semantic_postprocess

router = APIRouter(prefix="/api/predict", tags=["predict"])

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@router.post("/upload")
async def predict_upload(
    file: UploadFile = File(...),
    mode: str = Form("word")
):
    try:
        if mode not in ["word", "sentence"]:
            raise HTTPException(
                status_code=400,
                detail="mode는 word 또는 sentence만 가능합니다."
            )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="업로드된 파일이 없습니다."
            )

        allowed_ext = [".mp4", ".avi", ".mov", ".mkv"]
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다. mp4, avi, mov, mkv 파일만 업로드할 수 있습니다."
            )

        saved_filename = f"{uuid.uuid4()}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)

        content = await file.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="파일 내용이 비어 있습니다."
            )

        with open(saved_path, "wb") as f:
            f.write(content)

        if mode == "word":
            raw_ai_result = predict_word(saved_path)
        else:
            raw_ai_result = predict_sentence(saved_path)

        degree_result = predict_degree(saved_path)

        semantic_llm_result = apply_semantic_postprocess(
            mode=mode,
            text=raw_ai_result["text"],
            degree=degree_result["degree"],
            degree_ko=degree_result["degree_ko"]
        )

        final_result = {
            "text": semantic_llm_result["final_text"],
            "modified": semantic_llm_result["apply_degree"]
        }

        response = {
            "status": "success",
            "mode": mode,
            "input_type": "upload",
            "file": {
                "original_name": file.filename,
                "saved_name": saved_filename,
                "saved_path": saved_path
            },
            "raw_ai_result": raw_ai_result,
            "degree_result": degree_result,
            "semantic_llm_result": semantic_llm_result,
            "final_result": final_result
        }

        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(RESULT_DIR, result_filename)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        response["result_path"] = result_path

        return response

    except HTTPException:
        raise

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/word")
async def predict_word_only(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    saved_filename = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    result = predict_word(saved_path)

    return {
        "status": "success",
        "mode": "word",
        "raw_ai_result": result
    }


@router.post("/sentence")
async def predict_sentence_only(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    saved_filename = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    result = predict_sentence(saved_path)

    return {
        "status": "success",
        "mode": "sentence",
        "raw_ai_result": result
    }

@router.post("/webcam-frame")
async def predict_webcam_frame(
    file: UploadFile = File(...),
    mode: str = Form("word")
):
    try:
        if mode not in ["word", "sentence"]:
            raise HTTPException(
                status_code=400,
                detail="mode는 word 또는 sentence만 가능합니다."
            )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="웹캠 프레임 파일이 없습니다."
            )

        allowed_ext = [".jpg", ".jpeg", ".png", ".webp"]
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 이미지 형식입니다. jpg, jpeg, png, webp 파일만 전송할 수 있습니다."
            )

        saved_filename = f"webcam_{uuid.uuid4()}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)

        content = await file.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="웹캠 프레임 내용이 비어 있습니다."
            )

        with open(saved_path, "wb") as f:
            f.write(content)

        if mode == "word":
            raw_ai_result = predict_word(saved_path)
        else:
            raw_ai_result = predict_sentence(saved_path)

        degree_result = predict_degree(saved_path)

        semantic_llm_result = apply_semantic_postprocess(
            mode=mode,
            text=raw_ai_result["text"],
            degree=degree_result["degree"],
            degree_ko=degree_result["degree_ko"]
        )

        final_result = {
            "text": semantic_llm_result["final_text"],
            "modified": semantic_llm_result["apply_degree"]
        }

        response = {
            "status": "success",
            "mode": mode,
            "input_type": "webcam",
            "file": {
                "original_name": file.filename,
                "saved_name": saved_filename,
                "saved_path": saved_path
            },
            "raw_ai_result": raw_ai_result,
            "degree_result": degree_result,
            "semantic_llm_result": semantic_llm_result,
            "final_result": final_result
        }

        result_filename = f"webcam_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(RESULT_DIR, result_filename)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        response["result_path"] = result_path

        return response

    except HTTPException:
        raise

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }