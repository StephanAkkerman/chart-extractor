# chart_analyzer/ocr/paddle.py
from paddleocr import PaddleOCR

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr


def run_ocr(img):
    """Return list of dicts: {'text','conf','box','cx','cy'}."""
    ocr = _get_ocr()
    res = ocr.ocr(img, cls=True)
    out = []
    for block in res:
        for poly, (text, conf) in block:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            out.append(
                {
                    "text": text,
                    "conf": float(conf),
                    "box": poly,
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                }
            )
    return out
