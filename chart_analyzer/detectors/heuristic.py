# chart_analyzer/detectors/heuristic.py
import cv2
import numpy as np


def detect_right_edge_regions(img, fraction: float = 0.16):
    h, w = img.shape[:2]
    right = img[:, int(w * (1 - fraction)) :]
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if 18 <= hh <= 48 and ww >= 40:  # typical TradingView price pill
            pad = 6
            y0 = max(y - pad, 0)
            y1 = min(y + hh + pad, right.shape[0])
            x0 = max(x - pad, 0)
            x1 = min(x + ww + pad, right.shape[1])
            crops.append(right[y0:y1, x0:x1])
    # fall back to the full right strip if nothing found
    return crops or [right]


def center_band(img, h_frac: float = 0.5):
    h, w = img.shape[:2]
    band_h = int(h * h_frac)
    y0 = (h - band_h) // 2
    return img[y0 : y0 + band_h, :]
