"""A small desk that files one Kinyarwanda article into a news section."""

from __future__ import annotations

import json
import os

from flask import Flask, render_template, request

from inkuru.labels import KIN, LABELS, load_kin
from inkuru.paths import DATA, MODELS, RESULTS
from inkuru.predict import Router


def create_app(router=None) -> Flask:
    app = Flask(__name__)
    holder = {"router": router}

    def current():
        if holder["router"] is None:
            holder["router"] = Router()
        return holder["router"]

    def examples():
        path = DATA / "ui_examples.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def metrics():
        path = RESULTS / "metrics.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def sections() -> list[str]:
        load_kin(DATA / "kin_labels.json")
        return [KIN[label] for label in LABELS]

    def reads() -> str:
        path = MODELS / "deployed.json"
        if path.exists() and json.loads(path.read_text()).get("kind") == "transformer":
            return "The model reads the headline and the opening. It does not publish anything."
        if path.exists():
            return "The model reads the whole paste, up to 6,000 characters. It does not publish anything."
        return "It does not publish anything."

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            examples=examples(),
            text=request.args.get("text", "")[:6000],
            result=None,
            error="",
            system=None,
            reads=reads(),
            sections=sections(),
        )

    @app.post("/")
    def classify():
        text = request.form.get("text", "")[:6000]
        try:
            result = current().route(text)
            error = ""
            system = result["system"]
        except ValueError as exc:
            result = None
            error = str(exc)
            system = None
        return render_template(
            "index.html",
            examples=examples(),
            text=text,
            result=result,
            error=error,
            system=system,
            reads=reads(),
            sections=sections(),
        )

    @app.get("/lab")
    def lab():
        return render_template("lab.html", metrics=metrics())

    @app.get("/health")
    def health():
        ready = (MODELS / "deployed.json").exists()
        return {"ok": True, "model_loaded": holder["router"] is not None, "deployed": ready}

    return app


def main() -> None:
    # Local default. The Colab notebook sets HOST and PORT for its tunnel.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    create_app(Router()).run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
