"""Classification scores. Macro averages are unweighted across the 14 sections."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from inkuru.labels import LABELS


def classification_scores(gold: list[str] | np.ndarray, pred: list[str] | np.ndarray) -> dict:
    accuracy = float(accuracy_score(gold, pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        gold, pred, labels=LABELS, average="macro", zero_division=0
    )
    per_class = {}
    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "n": int(len(gold)),
        "per_class": per_class,
    }


def coverage_stats(gold: list[str], pred: list[str], confidence: np.ndarray, threshold: float) -> dict:
    keep = confidence >= threshold
    kept = int(keep.sum())
    n = int(len(gold))
    if kept == 0:
        accuracy = 0.0
    else:
        accuracy = float(accuracy_score(np.asarray(gold)[keep], np.asarray(pred)[keep]))
    return {
        "threshold": float(threshold),
        "coverage": kept / n if n else 0.0,
        "n_kept": kept,
        "accuracy_kept": accuracy,
    }


def choose_threshold(gold: list[str], pred: list[str], confidence: np.ndarray, min_coverage: float = 0.80) -> dict:
    """Highest accuracy among filed articles, subject to filing at least min_coverage of development."""
    candidates = sorted(set(np.round(confidence, 4).tolist()))
    best = None
    for threshold in candidates:
        stats = coverage_stats(gold, pred, confidence, float(threshold))
        if stats["coverage"] + 1e-12 < min_coverage:
            continue
        if best is None or stats["accuracy_kept"] > best["accuracy_kept"] + 1e-12:
            best = stats
        elif best and abs(stats["accuracy_kept"] - best["accuracy_kept"]) <= 1e-12:
            if stats["coverage"] > best["coverage"]:
                best = stats
    if best is None:
        best = coverage_stats(gold, pred, confidence, 0.0)
    return best
