"""Load the development-selected model and file one article."""

from __future__ import annotations

import json
import re

import joblib
import numpy as np

from inkuru.labels import BLURB, KIN, LABELS, load_kin
from inkuru.paths import BASELINES, DATA, MAX_LENGTH, MODELS, RUNS


class Router:
    def __init__(self):
        load_kin(DATA / "kin_labels.json")
        deployed = json.loads((MODELS / "deployed.json").read_text())
        self.system = deployed["system"]
        self.kind = deployed["kind"]
        self.threshold = float(deployed["threshold"])
        self.max_length = int(deployed.get("max_length", MAX_LENGTH))
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.majority = None
        if self.kind == "transformer":
            # torch and transformers are only needed for an encoder checkpoint.
            # The desk installs app/requirements.txt alone, so they load here.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            path = RUNS / self.system
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
        elif self.system == "majority":
            self.majority = json.loads((BASELINES / "majority.json").read_text())["label"]
        else:
            self.pipe = joblib.load(BASELINES / f"{self.system}.joblib")

    def route(self, text: str) -> dict:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Paste a Kinyarwanda headline or article first.")
        scores = self._scores(text.strip())
        order = np.argsort(-scores)
        top = []
        for index in order[:3]:
            label = LABELS[int(index)]
            top.append({
                "label": label,
                "kin": KIN[label],
                "blurb": BLURB[label],
                "p": float(scores[int(index)]),
            })
        best = top[0]
        return {
            "system": self.system,
            "label": best["label"],
            "kin": best["kin"],
            "blurb": best["blurb"],
            "confidence": best["p"],
            "filed": best["p"] >= self.threshold,
            "threshold": self.threshold,
            "top": top,
            "reason": self._reason(text.strip(), scores),
        }

    def _reason(self, text: str, scores: np.ndarray) -> dict:
        winner = int(np.argmax(scores))
        runner = int(np.argsort(-scores)[1])
        label = LABELS[winner]
        kin = KIN[label]
        plain = {"summary": "", "parts": [{"text": text, "role": "plain"}], "marked": False, "rival_kin": None}
        if self.majority is not None:
            plain["summary"] = (
                f"The majority system always names {kin} ({label}), "
                "the most common training section. It does not read the paste."
            )
            return plain
        if self.pipe is None:
            plain["summary"] = (
                f"The encoder turns the opening into one score for each section. "
                f"The highest score is {kin} ({label}). The paste is shown without marked words."
            )
            return plain
        return self._linear_reason(text, winner, runner)

    def _linear_reason(self, text: str, winner: int, runner: int) -> dict:
        vectorizer = self.pipe.named_steps["tfidf"]
        classifier = self.pipe.named_steps["clf"]
        names = vectorizer.get_feature_names_out()
        row = vectorizer.transform([text])
        classes = list(classifier.classes_)
        winner_label = LABELS[winner]
        runner_label = LABELS[runner]
        coef = classifier.coef_
        support_cores = _cores(names, row, coef[classes.index(winner_label)], limit=25)
        rival_weights = coef[classes.index(runner_label)] - coef[classes.index(winner_label)]
        rival_cores = _cores(
            names, row, rival_weights, limit=15, skip={core for _, core in support_cores}
        )
        parts = _marked_passage(text, support_cores, rival_cores)
        marked = any(part["role"] != "plain" for part in parts)
        rival_kin = KIN[runner_label] if any(part["role"] == "rival" for part in parts) else None
        kin = KIN[winner_label]
        if marked:
            summary = f"Words marked in the paste raised {kin} ({winner_label})."
        else:
            summary = (
                f"No word in this paste carried a positive weight for {kin} ({winner_label}). "
                "The score is the class weight, not a phrase in the text."
            )
        return {"summary": summary, "parts": parts, "marked": marked, "rival_kin": rival_kin}

    def _scores(self, text: str) -> np.ndarray:
        if self.majority is not None:
            scores = np.zeros(len(LABELS), dtype=np.float64)
            scores[LABELS.index(self.majority)] = 1.0
            return scores
        if self.pipe is not None:
            proba = self.pipe.predict_proba([text])[0]
            classes = list(self.pipe.named_steps["clf"].classes_)
            aligned = np.zeros(len(LABELS), dtype=np.float64)
            for column, label in enumerate(classes):
                aligned[LABELS.index(label)] = proba[column]
            return aligned
        import torch

        encoded = self.tokenizer(
            text, truncation=True, padding=False, max_length=self.max_length, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            return torch.softmax(logits, dim=-1)[0].cpu().numpy()


_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def _cores(names, row, weights, limit: int, skip: set[str] | None = None) -> list[tuple[float, str]]:
    """Strongest character fragments, reduced to the letters inside them."""
    skip = skip or set()
    ranked = []
    for index, value in zip(row.indices, row.data):
        core = str(names[int(index)]).replace(" ", "").lower()
        if core in skip or sum(character.isalpha() for character in core) < 3:
            continue
        pull = float(weights[int(index)] * value)
        if pull > 0:
            ranked.append((pull, core))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = []
    seen = set()
    for pull, core in ranked:
        if core in seen:
            continue
        seen.add(core)
        chosen.append((pull, core))
        if len(chosen) >= limit:
            break
    return chosen


def _marked_passage(text: str, support_cores, rival_cores) -> list[dict]:
    """Rebuild the paste as one paragraph, marking whole words rather than fragments."""
    matches = list(_WORD.finditer(text))
    scored = []
    for match in matches:
        word = match.group().lower()
        if sum(character.isalpha() for character in word) < 3:
            scored.append((match, 0.0, 0.0))
            continue
        support = sum(pull for pull, core in support_cores if core in word)
        rival = sum(pull for pull, core in rival_cores if core in word)
        scored.append((match, support, rival))
    best_support = {}
    best_rival = {}
    for match, support, rival in scored:
        key = match.group().lower()
        if support > best_support.get(key, 0.0):
            best_support[key] = support
        if rival > best_rival.get(key, 0.0):
            best_rival[key] = rival
    max_support = max(best_support.values(), default=0.0)
    max_rival = max(best_rival.values(), default=0.0)
    support_keys = [
        key for key, score in best_support.items()
        if max_support > 0 and score >= 0.45 * max_support
    ]
    support_keys.sort(key=lambda key: (-best_support[key], key))
    support_keys = set(support_keys[:6])
    rival_keys = [
        key for key, score in best_rival.items()
        if key not in support_keys and max_rival > 0 and score >= 0.45 * max_rival
    ]
    rival_keys.sort(key=lambda key: (-best_rival[key], key))
    rival_keys = set(rival_keys[:4])
    parts = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            parts.append({"text": text[cursor:match.start()], "role": "plain"})
        key = match.group().lower()
        if key in support_keys:
            role = "support"
        elif key in rival_keys:
            role = "rival"
        else:
            role = "plain"
        parts.append({"text": match.group(), "role": role})
        cursor = match.end()
    if cursor < len(text):
        parts.append({"text": text[cursor:], "role": "plain"})
    if not parts:
        parts = [{"text": text, "role": "plain"}]
    return parts
