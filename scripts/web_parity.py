"""Check that the browser copy answers like the Flask desk, and write results/web_parity.json.

Every development and test article and a set of edge cases go through the deployed Router and
through web/ngram.js (with Node.js), and the two must agree on the section, the filing decision,
the top three, the marked words, and the empty-input error. Run it from the repository root after
scripts/export_web.py:

    PYTHONPATH=src python scripts/web_parity.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile

from inkuru.data import load_split
from inkuru.paths import RESULTS, ROOT
from inkuru.predict import Router

# Inputs where Python and JavaScript treat text differently unless ngram.js copies Python's rules.
EDGE_CASES = [
    "",
    " \n\t ",
    "　 \x85",  # Unicode whitespace only
    "\x1c\x1d\x1e\x1f",  # whitespace to Python's str.split, not to JavaScript's \s
    "﻿",  # whitespace to JavaScript's \s, not to Python
    "Amavubi\x1cyatsinze\x1fUganda\x85ibitego bibiri",
    "Inkuru\xa0y’ubuzima\xa0mu\xa0bitaro bya Kigali",
    "ΟΔΥΣΣΕΥΣ ΟΔΟΣ ΚΑΙ ΝΟΣΟΚΟΜΕΙΟ",  # Greek final sigma
    "Café résumé i Kigali",  # combining accents
    "İSTANBUL na Kigali",  # dotted capital I lowercases to two code points
    "ẞ STRASSE ya Kigali",
    "Amavubi ⚽️🇷🇼 yatsinze 👨‍👩‍👧 Uganda",  # emoji, flags, and joiners
    "Kigali 北京 Москва ubukungu",
    "12345 67890",
    "!!! ??? ...",
    "a",
    "ubuzima " * 800,
    "Amavubi yatsinze Uganda ibitego bibiri ku busa mu mukino wa gicuti i Kigali",
]


def python_answer(router: Router, text: str) -> dict:
    try:
        result = router.route(text)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "label": result["label"],
        "filed": result["filed"],
        "top": [item["label"] for item in result["top"]],
        "scores": router._scores(text.strip()).tolist(),
        "marked": [[part["role"], part["text"]] for part in result["reason"]["parts"] if part["role"] != "plain"],
    }


def main() -> None:
    articles = [row["text"] for split in ("dev", "test") for row in load_split(split)]
    inputs = articles + EDGE_CASES
    router = Router()
    expected = [python_answer(router, text) for text in inputs]
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(inputs, handle)
        handle.flush()
        run = subprocess.run(["node", str(ROOT / "scripts" / "web_parity.js"), handle.name],
                             capture_output=True, text=True, check=True)
    actual = json.loads(run.stdout)

    counts = {key: 0 for key in ("same_section", "same_filing_decision", "same_top3",
                                 "same_marked_words", "same_empty_input_error")}
    largest = 0.0
    mismatches = []
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        both_answered = "error" not in want and "error" not in got
        checks = {
            "same_empty_input_error": want.get("error") == got.get("error"),
            "same_section": want.get("label") == got.get("label"),
            "same_filing_decision": want.get("filed") == got.get("filed"),
            "same_top3": want.get("top") == got.get("top"),
            "same_marked_words": want.get("marked") == got.get("marked"),
        }
        for key, same in checks.items():
            counts[key] += same
        if both_answered:
            largest = max(largest, max(abs(a - b) for a, b in zip(want["scores"], got["scores"], strict=True)))
        failed = [key for key, same in checks.items() if not same]
        if failed:
            mismatches.append({"input": index, "failed": failed, "text": inputs[index][:80]})

    report = {
        "checked": len(inputs),
        "articles": len(articles),
        "edge_cases": len(EDGE_CASES),
        **counts,
        "max_probability_difference": largest,
        "mismatches": mismatches,
    }
    (RESULTS / "web_parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "mismatches"}, indent=2))
    if mismatches:
        raise SystemExit(f"{len(mismatches)} inputs differ; see results/web_parity.json")


if __name__ == "__main__":
    main()
