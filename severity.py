import cv2
import numpy as np


def calculate_severity(image_path):
    """Returns (percent_diseased, level) for a leaf image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image: " + image_path)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Healthy leaf area = green pixels
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Whole leaf = anything that is not near-white background
    leaf_mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
    leaf_pixels = cv2.countNonZero(leaf_mask)
    green_pixels = cv2.countNonZero(green_mask)
    if leaf_pixels == 0:
        return 0.0, "low"
    # Diseased area = leaf pixels that are not green
    diseased_pixels = max(leaf_pixels - green_pixels, 0)
    percent = diseased_pixels / leaf_pixels * 100
    if percent < 15:
        level = "low"
    elif percent < 40:
        level = "medium"
    else:
        level = "high"
    return round(percent, 2), level
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    percent, level = calculate_severity(path)
    print(f"Diseased area: {percent}%  |  Severity: {level}")
