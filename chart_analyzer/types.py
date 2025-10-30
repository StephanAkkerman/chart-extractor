# chart_analyzer/types.py
from dataclasses import dataclass
from typing import Any


@dataclass
class ChartInfo:
    """Structured output for a single chart image.

    Attributes
    ----------
    symbol : str | None
        Best-guess short symbol (e.g., 'SPY').
    name : str | None
        Long/marketing name if available (e.g., 'SPDR S&P 500 ETF Trust').
    price : float | None
        Last/close price extracted from the image.
    timeframe : str | None
        E.g., '1D', '4H' when detectable.
    exchange : str | None
        E.g., 'NYSE Arca', 'NASDAQ'.
    raw_ocr : list[dict]
        Raw OCR lines with boxes/confidence for debugging.
    meta : dict[str, Any]
        Extra facts (e.g., method used, detection boxes).
    """

    symbol: str | None
    name: str | None
    price: float | None
    timeframe: str | None
    exchange: str | None
    raw_ocr: list[dict]
    meta: dict[str, Any]
