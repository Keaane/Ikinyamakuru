"""Desk behaviour that does not need a trained model, plus split invariants."""

import json
import re
import unittest

import numpy as np

from app.app import create_app
from inkuru.metrics import choose_threshold
from inkuru.data import canonical, normalize


class AliasTest(unittest.TestCase):
    def test_politic_is_politics(self):
        self.assertEqual(canonical("politic"), "politics")
        self.assertEqual(canonical("sport"), "sport")

    def test_normalize_collapses_space(self):
        self.assertEqual(normalize("A  b\n c"), "a b c")


class FakeRouter:
    system = "fake"

    def route(self, text: str) -> dict:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Paste a Kinyarwanda headline or article first.")
        return {
            "system": "fake",
            "label": "health",
            "kin": "ubuzima",
            "blurb": "Hospitals, disease, and public health.",
            "confidence": 0.91,
            "filed": True,
            "threshold": 0.4,
            "top": [{
                "label": "health",
                "kin": "ubuzima",
                "blurb": "Hospitals, disease, and public health.",
                "p": 0.91,
            }],
        }


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app(FakeRouter()).test_client()

    def test_home_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"File a Kinyarwanda article.", response.data)

    def test_empty_submit(self):
        response = self.client.post("/", data={"text": "   "})
        page = response.data.decode()
        self.assertIn("Paste a Kinyarwanda headline", page)
        answer = page.split('class="answer', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("ubuzima", answer)
        self.assertNotIn("Filed", answer)

    def test_classify(self):
        response = self.client.post("/", data={"text": "Ubuzima bwiza"})
        self.assertIn(b"ubuzima", response.data)
        self.assertIn(b"Filed", response.data)

    def test_example_query_fills_without_filing(self):
        response = self.client.get("/?text=inkuru+y%27ubuzima")
        page = response.data.decode()
        self.assertIn("inkuru y&#39;ubuzima", page)
        self.assertNotIn("Filed", page)
        examples_at = page.find('class="examples"')
        if examples_at != -1:
            self.assertNotIn('method="post"', page[examples_at:])


class ThresholdTest(unittest.TestCase):
    def test_cutoff_keeps_coverage_and_prefers_accuracy(self):
        gold = ["a"] * 100
        pred = ["a"] * 90 + ["b"] * 10
        confidence = np.array([0.95] * 90 + [0.10] * 10)
        chosen = choose_threshold(gold, pred, confidence, min_coverage=0.80)
        self.assertGreaterEqual(chosen["coverage"], 0.80)
        self.assertAlmostEqual(chosen["accuracy_kept"], 1.0)
        self.assertGreater(chosen["threshold"], 0.10)


class RealDeskTest(unittest.TestCase):
    """Drive the shipped Flask handlers and the deployed Router."""

    def setUp(self):
        from inkuru.paths import DATA, MODELS
        if not (MODELS / "deployed.json").exists():
            self.skipTest("model not deployed")
        self.client = create_app().test_client()
        self.kin = json.loads((DATA / "kin_labels.json").read_text())

    def test_empty_and_whitespace_do_not_file(self):
        for text in ("", " \n\t "):
            response = self.client.post("/", data={"text": text})
            page = response.data.decode()
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paste a Kinyarwanda headline or article first.", page)
            self.assertNotIn("Filed", page)
            self.assertNotIn("Left for an editor", page)

    def test_nonempty_submit_shows_a_section(self):
        text = "Inkuru y'ubuzima mu bitaro bya Kigali."
        response = self.client.post("/", data={"text": text})
        page = response.data.decode()
        self.assertEqual(response.status_code, 200)
        answer = page.split('class="answer', 1)[1].split("</section>", 1)[0]
        self.assertTrue("Filed" in answer or "Left for an editor" in answer)
        self.assertTrue(any(name in answer for name in self.kin.values()))
        self.assertIn("Why this section", answer)
        self.assertIn('class="passage"', answer)
        passage = answer.split('class="passage"', 1)[1]
        marks = re.findall(r'class="support">(.*?)</mark>', passage)
        self.assertTrue(marks)
        folded = " ".join(text.lower().split())
        for mark in marks:
            word = mark.replace("&#39;", "'").replace("’", "'").strip().lower()
            self.assertIn(word, folded)
            self.assertNotIn(word, {"taro", "ubuz", "buzi", "uzim", "zima"})
        self.assertTrue(any("bitaro" in mark.lower() or "ubuzima" in mark.lower() for mark in marks))
        compact = " ".join(passage.split())
        self.assertIn(
            "Inkuru <mark class=\"support\">y&#39;ubuzima</mark> mu <mark class=\"support\">bitaro</mark>",
            compact,
        )

    def test_prefill_does_not_file(self):
        response = self.client.get("/?text=inkuru+y%27ubuzima")
        page = response.data.decode()
        self.assertIn("inkuru y&#39;ubuzima", page)
        self.assertNotIn("Filed", page)
        self.assertNotIn("Left for an editor", page)
        examples_at = page.find('class="examples"')
        self.assertGreater(examples_at, -1)
        self.assertNotIn('method="post"', page[examples_at:])

    def test_repeat_classify_is_stable(self):
        text = "Inkuru y'ubuzima mu bitaro bya Kigali."
        first = self.client.post("/", data={"text": text}).data.decode()
        second = self.client.post("/", data={"text": text}).data.decode()
        self.assertEqual(first, second)

    def test_lab_lists_the_scored_systems(self):
        from inkuru.paths import RESULTS
        metrics = json.loads((RESULTS / "metrics.json").read_text())
        response = self.client.get("/lab")
        page = response.data.decode()
        self.assertEqual(response.status_code, 200)
        for name in metrics["systems"]:
            self.assertIn(f"<td>{name}</td>", page)
        chosen = page.split('class="chosen"', 1)
        self.assertEqual(len(chosen), 2)
        self.assertIn(f"<td>{metrics['selected']}</td>", chosen[1][:500])


class PreparedSplitTest(unittest.TestCase):
    def test_no_text_leakage_if_splits_exist(self):
        from inkuru.paths import RAW
        if not (RAW / "train.jsonl").exists():
            self.skipTest("splits not prepared")
        seen = {}
        for name in ("train", "dev", "test"):
            with (RAW / f"{name}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = normalize(row["text"])
                    self.assertNotIn(key, seen, f"{row['id']} repeats {seen.get(key)}")
                    seen[key] = row["id"]
                    self.assertEqual(row["split"], name)


class WebExportTest(unittest.TestCase):
    """web/ must hold the deployed model."""

    def test_browser_copy_matches_deployed_model(self):
        from inkuru.paths import MODELS, ROOT
        model = ROOT / "web" / "model"
        if not (model / "meta.json").exists() or not (MODELS / "deployed.json").exists():
            self.skipTest("browser copy not exported")
        meta = json.loads((model / "meta.json").read_text())
        deployed = json.loads((MODELS / "deployed.json").read_text())
        self.assertEqual(meta["system"], deployed["system"])
        self.assertEqual(meta["threshold"], deployed["threshold"])
        self.assertEqual(meta["labels"], deployed["labels"])
        names = json.loads((model / "vocab.json").read_text())
        self.assertEqual((model / "idf.f64").stat().st_size, 8 * len(names))
        self.assertEqual((model / "coef.f64").stat().st_size, 8 * len(names) * len(meta["labels"]))
