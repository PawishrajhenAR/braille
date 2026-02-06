# English Handwritten and Printed OCR System (Offline)

## System Architecture (English OCR Only)

**Flow**
1. Client uploads an image (JPG/PNG) to `POST /ocr/english`.
2. Image is decoded with OpenCV.
3. Handwritten-vs-printed detector scores the image.
4. OCR routing:
   - Handwritten → TrOCR (`microsoft/trocr-base-handwritten`).
   - Printed → PaddleOCR (`lang="en"`).
   - Ambiguous → run both and select higher confidence.
5. JSON response returns route, confidence, and extracted text.

**Modules**
- `app/services/handwriting_detector.py`: Offline heuristics to estimate handwritten likelihood.
- `app/services/ocr_service.py`: Model initialization, OCR inference, routing logic.
- `app/api/routes.py`: FastAPI route `POST /ocr/english`.
- `app/main.py`: FastAPI app entry point.

## Exact Python Imports and Classes

**TrOCR (handwritten)**
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
```
**PaddleOCR (printed)**
```python
from paddleocr import PaddleOCR
```
**GPU detection**
```python
import torch
from paddle import is_compiled_with_cuda
```

## OCR Routing Logic (Pseudocode)

```text
image = decode_upload()
score = handwriting_detector.estimate_handwritten_score(image)

if score >= 0.55:
    route = "handwritten"
    text, conf = trocr(image)
elif score <= 0.45:
    route = "printed"
    text, conf = paddleocr(image)
else:
    trocr_text, trocr_conf = trocr(image)
    paddle_text, paddle_conf = paddleocr(image)
    if trocr_conf >= paddle_conf:
        route = "handwritten"
        text = trocr_text
    else:
        route = "printed"
        text = paddle_text

return {"route": route, "confidence": conf, "text": text}
```

## FastAPI Endpoint Code

```python
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
```

## Folder Structure

```
app/
  api/
    routes.py
  core/
    config.py
  services/
    handwriting_detector.py
    ocr_service.py
  main.py
requirements.txt
README.md
```

## Setup Instructions

### 1) Create Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) GPU Acceleration Notes (RTX 4050)
- Ensure CUDA drivers are installed.
- If your CUDA version differs, install matching `torch` and `paddlepaddle-gpu` wheels per vendor instructions.

### 3) Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) Example Request
```bash
curl -X POST "http://localhost:8000/ocr/english" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.png"
```

## Requirements
See `requirements.txt` for exact versions.
