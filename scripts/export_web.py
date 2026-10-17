"""Build the static Hugging Face Space in web/ from the deployed model.

It writes the deployed linear model to web/model/ in the layout web/ngram.js reads, and renders
the Flask desk's two pages as static HTML. Run it from the repository root whenever notebook 04
or 05 changes models/deployed.json, results/metrics.json, or data/ui_examples.json:

    PYTHONPATH=src python scripts/export_web.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import joblib
import numpy as np

from inkuru.labels import BLURB, KIN, LABELS, load_kin
from inkuru.paths import BASELINES, DATA, MODELS, ROOT

sys.path.insert(0, str(ROOT))
from app.app import create_app  # noqa: E402

WEB = ROOT / "web"


def export_model() -> None:
    """Vocabulary, IDF weights, coefficients, and intercepts, with the classes in LABELS order."""
    load_kin(DATA / "kin_labels.json")
    deployed = json.loads((MODELS / "deployed.json").read_text())
    if deployed["kind"] != "baseline" or deployed["system"] == "majority":
        raise SystemExit(f"web/ runs a TF-IDF linear model in JavaScript, not {deployed['system']}")
    pipe = joblib.load(BASELINES / f"{deployed['system']}.joblib")
    vectorizer = pipe.named_steps["tfidf"]
    classifier = pipe.named_steps["clf"]
    # web/ngram.js implements exactly these settings, so anything else must fail here.
    expected = {"analyzer": "char_wb", "lowercase": True, "sublinear_tf": True, "norm": "l2", "use_idf": True}
    for name, value in expected.items():
        if getattr(vectorizer, name) != value:
            raise SystemExit(f"web/ngram.js expects {name}={value!r}, the model has {getattr(vectorizer, name)!r}")
    rows = [list(classifier.classes_).index(label) for label in LABELS]
    names = vectorizer.get_feature_names_out().tolist()
    model = WEB / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "vocab.json").write_text(json.dumps(names, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    np.asarray(vectorizer.idf_, dtype="<f8").tofile(model / "idf.f64")
    np.ascontiguousarray(classifier.coef_[rows], dtype="<f8").tofile(model / "coef.f64")
    meta = {
        "system": deployed["system"],
        "threshold": deployed["threshold"],
        "labels": LABELS,
        "kin": {label: KIN[label] for label in LABELS},
        "blurb": BLURB,
        "ngram_range": list(vectorizer.ngram_range),
        "intercept": classifier.intercept_[rows].tolist(),
    }
    (model / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"web/model: {deployed['system']}, {len(names):,} features, cutoff {deployed['threshold']}")


def version(path) -> str:
    """Cache-busting query value: the first 8 hex digits of the file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def static_page(html: str, scripts: tuple[str, ...] = ()) -> str:
    """Point the Flask routes at the static files and add the scripts that replace the server."""
    for old, new in (
        ('href="/static/styles.css', 'href="styles.css'),
        ('href="/lab"', 'href="lab.html"'),
        ('href="/?text=', 'href="./?text='),
        ('href="/"', 'href="./"'),
        ('action="/"', 'action="./"'),
    ):
        html = html.replace(old, new)
    tags = "".join(f'    <script src="{name}?v={version(WEB / name)}"></script>\n' for name in scripts)
    return html.replace("  </body>", tags + "  </body>")


def export_pages() -> None:
    client = create_app().test_client()
    shutil.copyfile(ROOT / "app" / "static" / "styles.css", WEB / "styles.css")
    (WEB / "index.html").write_text(static_page(client.get("/").get_data(as_text=True), ("ngram.js", "desk.js")),
                                    encoding="utf-8")
    (WEB / "lab.html").write_text(static_page(client.get("/lab").get_data(as_text=True)), encoding="utf-8")
    print("web/index.html, web/lab.html, web/styles.css")


if __name__ == "__main__":
    export_model()
    export_pages()
