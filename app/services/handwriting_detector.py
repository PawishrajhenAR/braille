from __future__ import annotations

import cv2
import numpy as np


class HandwritingDetector:
    def estimate_handwritten_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = float(np.mean(edges > 0))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5

        irregularities = []
        for contour in contours[:200]:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            irregularities.append(1 - circularity)
        if not irregularities:
            return 0.5

        irregularity_score = float(np.clip(np.mean(irregularities), 0.0, 1.0))
        score = 0.6 * edge_density + 0.4 * irregularity_score
        return float(np.clip(score, 0.0, 1.0))

    def is_handwritten(self, image: np.ndarray, threshold: float = 0.55) -> bool:
        return self.estimate_handwritten_score(image) >= threshold
