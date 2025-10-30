# chart_analyzer/pipeline.py
from .detectors.heuristic import center_band, detect_right_edge_regions
from .loaders import load_image
from .ocr.paddle import run_ocr
from .parsers.generic import parse_generic
from .parsers.tradingview import parse_tradingview
from .validators import normalize_and_validate


def analyze(path: str):
    img = load_image(path)
    ocr_full = run_ocr(img)

    # Heuristic crops to boost price read
    right_crops = detect_right_edge_regions(img, fraction=0.16)
    ocr_right = [run_ocr(crop) for crop in right_crops]

    # Watermark/center band (TradingView has huge faded ticker in center)
    wm_crop = center_band(img, h_frac=0.4)
    ocr_wm = run_ocr(wm_crop)

    # Try TradingView-specific rules first, then generic fallback
    info = parse_tradingview(ocr_full, ocr_right, ocr_wm, img.shape)
    if not info.symbol and not info.price:
        info = parse_generic(ocr_full, ocr_right, img.shape)

    return normalize_and_validate(info)


if __name__ == "__main__":
    print(analyze("img/chart.png"))
