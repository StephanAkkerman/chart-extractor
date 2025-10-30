# chart_analyzer/parsers/tradingview.py
import re
from dataclasses import asdict

from ..types import ChartInfo

TICKER = re.compile(r"\b[A-Z][A-Z.\-]{0,5}\b")
PRICE = re.compile(r"\d{2,5}\.\d{2}")
TIMEFRAME = re.compile(r"\b([1-9][0-9]?|[1-9])([smhdw]|[MH]|\s?D)\b", re.I)


def _top_band(lines, h, top_ratio=0.18):
    return [l for l in lines if l["cy"] / h < top_ratio]


def _join_text(lines):
    return "  ".join([l["text"] for l in lines])


def parse_tradingview(ocr_full, ocr_right_list, shape):
    h, w, *_ = shape
    top = _top_band(ocr_full, h)
    joined_top = _join_text(top)

    symbol = None
    name = None
    exchange = None
    timeframe = None
    price = None

    # 1) symbol/name/exchange/timeframe from the top band
    tickers = TICKER.findall(joined_top)
    if tickers:
        symbol = tickers[0]
    # long-ish name
    candidates = [l["text"] for l in top if len(l["text"].split()) >= 3]
    if candidates:
        name = max(candidates, key=len)
    # timeframe + exchange (best-effort)
    m_tf = TIMEFRAME.search(joined_top)
    if m_tf:
        timeframe = m_tf.group(0).replace(" ", "").upper()
    for tok in ("NYSE", "NASDAQ", "ARCA", "CBOE"):
        if tok in joined_top.upper():
            exchange = tok if tok != "ARCA" else "NYSE Arca"

    # 2) price from right-edge OCR (prefer highest-conf match)
    best = None
    for ocr_right in ocr_right_list:
        for l in ocr_right:
            m = PRICE.search(l["text"])
            if m:
                c = l["conf"]
                txt = m.group(0)
                if not best or c > best[1]:
                    best = (txt, c)
    if best:
        price = float(best[0])

    # 3) Fallback: OHLC line (often in top band; use 'C' or last number)
    if price is None:
        m_close = re.search(r"\bC[: ]?(\d{2,5}\.\d{2})\b", joined_top)
        if m_close:
            price = float(m_close.group(1))
        else:
            nums = PRICE.findall(joined_top)
            if nums:
                price = float(nums[-1])

    return ChartInfo(
        symbol=symbol,
        name=name,
        price=price,
        timeframe=timeframe,
        exchange=exchange,
        raw_ocr=ocr_full,
        meta={"right_ocr": ocr_right_list[:1]},  # keep small
    )
