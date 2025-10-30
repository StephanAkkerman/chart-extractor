# cli.py
import json

import typer

from chart_analyzer.pipeline import analyze

app = typer.Typer()


@app.command()
def chartinfo(path: str):
    info = analyze(path)
    print(json.dumps(info.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
