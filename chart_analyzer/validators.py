# chart_analyzer/validators.py
from .types import ChartInfo

NAME_TO_TICKER = {"SPDR S&P 500 ETF TRUST": "SPY"}


def normalize_and_validate(info: ChartInfo) -> ChartInfo:
    if not info.symbol and info.name:
        key = info.name.upper()
        if key in NAME_TO_TICKER:
            info.symbol = NAME_TO_TICKER[key]
    # coarse price sanity (optional)
    if info.price is not None and not (0.01 <= info.price <= 100000):
        info.price = None
    return info
