from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
from paddle import is_compiled_with_cuda
from paddleocr import PaddleOCR
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.services.handwriting_detector import HandwritingDetector


@dataclass
class OcrResult:
    text: str
    confidence: float
    route: str


class OcrService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(self.device)
        self.trocr_model.eval()

        self.paddle_ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=is_compiled_with_cuda(),
        )
        self.detector = HandwritingDetector()

    def _trocr_infer(self, image: np.ndarray) -> Tuple[str, float]:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.trocr_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            generated = self.trocr_model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = self.trocr_processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0]

        scores = generated.scores
        if not scores:
            return text, 0.0

        log_probs = []
        for score, token_id in zip(scores, generated.sequences[0, 1:]):
            probs = torch.log_softmax(score[0], dim=-1)
            log_probs.append(probs[token_id].item())
        avg_log_prob = float(np.mean(log_probs)) if log_probs else 0.0
        return text, avg_log_prob

    def _paddle_infer(self, image: np.ndarray) -> Tuple[str, float]:
        result = self.paddle_ocr.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0

        lines = []
        confidences = []
        for line in result[0]:
            text = line[1][0]
            conf = float(line[1][1])
            lines.append(text)
            confidences.append(conf)
        return "\n".join(lines), float(np.mean(confidences)) if confidences else 0.0

    def run(self, image: np.ndarray) -> OcrResult:
        handwritten_score = self.detector.estimate_handwritten_score(image)
        if handwritten_score >= 0.55:
            text, conf = self._trocr_infer(image)
            return OcrResult(text=text, confidence=conf, route="handwritten")
        if handwritten_score <= 0.45:
            text, conf = self._paddle_infer(image)
            return OcrResult(text=text, confidence=conf, route="printed")

        trocr_text, trocr_conf = self._trocr_infer(image)
        paddle_text, paddle_conf = self._paddle_infer(image)

        if trocr_conf >= paddle_conf:
            return OcrResult(text=trocr_text, confidence=trocr_conf, route="handwritten")
        return OcrResult(text=paddle_text, confidence=paddle_conf, route="printed")
