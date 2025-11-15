import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------- Config ----------
# Class IDs (must match your training config)
CLS_SYMBOL_TITLE = 0
CLS_LAST_PRICE_PILL = 1

# HF model repo (change if you version differently)
HF_MODEL_REPO = "StephanAkkerman/chart-info-detector-yolo12n"
HF_MODEL_FILE = "weights/best.pt"  # path inside the model repo

# OCR model name (PaddleOCR)
_OCR = None  # lazy global to avoid heavy init on import


# ---------- Types ----------
@dataclass
class DetBox:
    cls: int
    conf: float
    xyxy: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class ExtractResult:
    symbol: str | None
    exchange: str | None
    timeframe: str | None
    price: float | None
    session: str | None  # "regular" | "pre" | "post" | None
    raw_title_text: str
    raw_pill_text: str
    det_title_box: tuple[int, int, int, int] | None
    det_pill_box: tuple[int, int, int, int] | None


# ---------- Helpers ----------
def _download_weights_if_needed(
    repo_id: str = HF_MODEL_REPO, filename: str = HF_MODEL_FILE
) -> str:
    """Download YOLO weights from Hugging Face if not cached, return local path."""
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


def _ensure_ocr():
    global _OCR
    if _OCR is None:
        # Lazy import to keep import-time light
        from paddleocr import PaddleOCR

        # Use English by default; add 'lang="en"' to reduce alphabet
        _OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _OCR


def _read_image(img: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    p = Path(img)
    if not p.exists():
        raise FileNotFoundError(p)
    im = cv2.imread(str(p))
    if im is None:
        raise ValueError(f"Failed to read image: {p}")
    return im


def _crop(im: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    h, w = im.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return im[y1:y2, x1:x2]


def _ocr_text(im_bgr: np.ndarray) -> str:
    """Run PaddleOCR and return a single concatenated string (highest-conf first)."""
    if im_bgr.size == 0:
        return ""
    ocr = _ensure_ocr()
    # Convert to RGB for better OCR contrast
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    res = ocr.ocr(im_rgb, cls=True)
    if not res or not res[0]:
        return ""
    # Sort by confidence desc, join
    lines = sorted(res[0], key=lambda x: x[1][1], reverse=True)
    return " ".join([x[1][0] for x in lines]).strip()


# ---------- Parsing ----------
_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9.\-]{0,9})\b")  # e.g., SPY, BRK.B, RY-UN.TO
_EXCHANGE_HINTS = re.compile(
    r"\b(NYSE|NASDAQ|CBOE|ARCA|TSX|LSE|FWB|XETRA|CME|CBOT|HKEX)\b", re.I
)
_TIMEFRAME_RE = re.compile(r"\b(1m|3m|5m|15m|30m|45m|1h|2h|4h|D|1D|W|1W|M|1M)\b", re.I)
# Price: 1,234.56 or 1234.56; optional currency prefix/suffix
_PRICE_RE = re.compile(
    r"(?<![A-Za-z])(?:\$|€|£)?\s*([0-9]{1,3}(?:[, ]?[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?:USD|EUR|GBP)?(?![A-Za-z])"
)


def _parse_title(text: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse ticker, exchange, timeframe from title text.
    Heuristic: first strong ticker-like token is the symbol.
    """
    if not text:
        return None, None, None
    # timeframe is often separate; pick it but don't let it be the symbol
    timeframe = None
    tfm = _TIMEFRAME_RE.search(text)
    if tfm:
        timeframe = tfm.group(1).upper()

    # pick first ticker-looking token
    sym = None
    for m in _TICKER_RE.finditer(text):
        token = m.group(1)
        # filter out pure numbers
        if not token.isdigit():
            sym = token
            break

    exch = None
    em = _EXCHANGE_HINTS.search(text)
    if em:
        exch = em.group(1).upper()

    return sym, exch, timeframe


def _parse_pill(text: str) -> tuple[float | None, str | None]:
    """
    Parse price and session hint from pill.
    Detect 'Pre'/'Post' tokens as session markers.
    """
    if not text:
        return None, None
    sess = None
    if re.search(r"\bpost\b", text, re.I):
        sess = "post"
    elif re.search(r"\bpre\b", text, re.I):
        sess = "pre"
    # take the first plausible price
    m = _PRICE_RE.search(text.replace(",", ""))
    if not m:
        # allow comma thousand sep as fallback
        m = _PRICE_RE.search(text)
    price = None
    if m:
        try:
            price = float(m.group(1).replace(",", "").replace(" ", ""))
        except Exception:
            price = None
    return price, ("regular" if sess is None else sess)


# ---------- Core pipeline ----------
class ChartExtractor:
    """
    Detects chart widgets and extracts structured info via OCR.

    Parameters
    ----------
    weights : str | None
        Local path to YOLO weights. If None, downloads from HF.
    imgsz : int
        Inference size for detection.
    conf : float
        Confidence threshold.
    iou : float
        IoU threshold for NMS.
    """

    def __init__(
        self,
        weights: str | None = None,
        imgsz: int = 1536,
        conf: float = 0.25,
        iou: float = 0.5,
    ):
        if weights is None:
            weights = _download_weights_if_needed()
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _detect(self, img: np.ndarray) -> list[DetBox]:
        res = self.model.predict(
            source=img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]
        out: list[DetBox] = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append(DetBox(cls=cls_id, conf=conf, xyxy=(x1, y1, x2, y2)))
        return out

    @staticmethod
    def _pick_best_per_class(boxes: list[DetBox]) -> dict[int, DetBox]:
        picks: dict[int, DetBox] = {}
        for b in sorted(boxes, key=lambda z: z.conf, reverse=True):
            if b.cls == CLS_LAST_PRICE_PILL:
                # special rule: prefer the rightmost pill if confidences tie
                prev = picks.get(CLS_LAST_PRICE_PILL)
                if (
                    prev is None
                    or (b.conf > prev.conf)
                    or (b.conf == prev.conf and b.xyxy[2] > prev.xyxy[2])
                ):
                    picks[CLS_LAST_PRICE_PILL] = b
            else:
                picks.setdefault(b.cls, b)
        return picks

    def analyze(self, image: str | Path | np.ndarray) -> ExtractResult:
        """
        Run detection → OCR → parsing on a chart screenshot.

        Parameters
        ----------
        image : str | Path | np.ndarray
            Input image path or loaded BGR array.

        Returns
        -------
        ExtractResult
            Structured extraction with raw OCR text for debugging.
        """
        im = _read_image(image)
        boxes = self._detect(im)
        picks = self._pick_best_per_class(boxes)

        # Title OCR (detected box or fallback to top band)
        raw_title = ""
        title_box = None
        if CLS_SYMBOL_TITLE in picks:
            title_box = picks[CLS_SYMBOL_TITLE].xyxy
            raw_title = _ocr_text(_crop(im, title_box))
        if not raw_title:
            h, w = im.shape[:2]
            title_box = (0, 0, w, int(0.18 * h))
            raw_title = _ocr_text(im[: title_box[3], :])

        # Price OCR (if any pill detected)
        raw_pill = ""
        pill_box = None
        if CLS_LAST_PRICE_PILL in picks:
            pill_box = picks[CLS_LAST_PRICE_PILL].xyxy
            raw_pill = _ocr_text(_crop(im, pill_box))

        # Parse
        symbol, exchange, timeframe = _parse_title(raw_title)
        price, session = _parse_pill(raw_pill)

        return ExtractResult(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            price=price,
            session=session,
            raw_title_text=raw_title,
            raw_pill_text=raw_pill,
            det_title_box=title_box,
            det_pill_box=pill_box,
        )
