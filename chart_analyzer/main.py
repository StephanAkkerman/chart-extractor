from chart_extractor import ChartExtractor

extractor = ChartExtractor(weights=None, imgsz=1792, conf=0.25, iou=0.5)
res = extractor.analyze("img/chart.png")

print(res)
# ExtractResult(symbol='SPY', exchange='ARCA', timeframe='1D', price=500.12, session='regular', ...)
