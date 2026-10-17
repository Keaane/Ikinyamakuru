"""Read and write the prepared splits and the per-article prediction records."""

from __future__ import annotations

import json
import re

import numpy as np

from inkuru.labels import LABELS
from inkuru.paths import RAW

WHITESPACE = re.compile(r"\s+")
# The raw CSV writes the first class as "politic". classes.txt and the paper say "politics".
ALIASES = {"politic": "politics"}


def normalize(text: str) -> str:
    """The deduplication key: lowercased text with whitespace collapsed."""
    return WHITESPACE.sub(" ", text).strip().lower()


def canonical(label: str) -> str:
    return ALIASES.get(label, label)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(name: str) -> list[dict]:
    if name not in {"train", "dev", "test"}:
        raise ValueError(name)
    return read_jsonl(RAW / f"{name}.jsonl")


def align_proba(pipe, texts) -> np.ndarray:
    """predict_proba with its columns in LABELS order (classes_ is sorted alphabetically)."""
    proba = pipe.predict_proba(texts)
    classes = list(pipe.named_steps["clf"].classes_)
    aligned = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    for column, label in enumerate(classes):
        aligned[:, LABELS.index(label)] = proba[:, column]
    return aligned


def records(rows, proba) -> list[dict]:
    """One prediction record per article: the gold section, the top section, its score, and the top three."""
    out = []
    for row, scores in zip(rows, proba):
        order = np.argsort(-scores)[:3]
        pred = int(order[0])
        out.append({
            "id": row["id"],
            "gold": row["label"],
            "pred": LABELS[pred],
            "confidence": float(scores[pred]),
            "top": [{"label": LABELS[int(i)], "p": float(scores[int(i)])} for i in order],
        })
    return out
