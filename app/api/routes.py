from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ocr_service import OcrService

router = APIRouter()
ocr_service = OcrService()


@router.post("/ocr/english")
async def ocr_english(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    result = ocr_service.run(image)
    return {
        "route": result.route,
        "confidence": result.confidence,
        "text": result.text,
    }
