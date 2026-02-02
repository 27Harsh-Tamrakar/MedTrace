# cavity_analysis.py

import cv2
import numpy as np


def analyze_cavities(image):
    """
    Detect blister cavities and analyze their geometric integrity.

    Returns:
        cavity_score (0–100): Higher means more deformation
        annotated image saved as cavity_analysis.jpg
    """

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detect edges of cavities
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours (possible cavities)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    deformation_scores = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore tiny noise
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        # Circularity measure
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Ideal cavity ≈ circular → circularity ~ 0.8–1.0
        # Deformed cavity → irregular → lower circularity

        deformation = (1 - circularity) * 100
        deformation_scores.append(deformation)

        # Draw contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Average deformation
    cavity_score = (
        np.mean(deformation_scores) if deformation_scores else 0
    )

    cv2.imwrite("cavity_analysis.jpg", output)

    return float(min(100, cavity_score))