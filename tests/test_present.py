import unittest

import numpy as np

from mmrag.extractors import Chunk
from mmrag.index import VectorIndex
from mmrag.present import (
    best_per_source,
    chips_from,
    clean_query,
    compose_answer,
    present,
    subtitle_for,
    to_percent,
)

HYDRAULIK = (
    "Sollwert 180 bar. Die Anleitung prüft vier Stellen, das Video zeigt den Ablauf "
    "am Hydraulikkreis Stopfen. Relevant sind Druckspeicher, Pumpe A2, Ventil RV-12 "
    "und der Rücklauffilter."
)


class PresentTests(unittest.TestCase):
    def test_percent_and_quotes(self):
        self.assertEqual(to_percent(0.94), 94)
        self.assertEqual(to_percent(0.944), 94)
        self.assertEqual(to_percent(-0.2), 0)
        self.assertEqual(to_percent(float("nan")), 0)
        self.assertEqual(clean_query("„Druckverlust“"), "Druckverlust")
        self.assertEqual(clean_query("  ohne  "), "ohne")

    def test_chips_follow_the_checklist(self):
        self.assertEqual(
            chips_from([HYDRAULIK]),
            ["Druckspeicher", "Pumpe A2", "Ventil RV-12", "Rücklauffilter"],
        )

    def test_answer_keeps_the_lead_sentences(self):
        answer = compose_answer(HYDRAULIK)
        self.assertTrue(answer.startswith("Sollwert 180 bar."))
        self.assertIn("Hydraulikkreis Stopfen", answer)
        self.assertLessEqual(len(answer), 280)

    def test_pdf_subtitle_uses_chapter_and_page(self):
        chunk = Chunk(
            text="Siehe Kapitel 7.4 für den Druckspeicher.",
            source="Wartungsanleitung GB-204.pdf",
            modality="pdf",
            kind="text",
            page=214,
        )
        self.assertEqual(subtitle_for(chunk, [chunk]), "Kapitel 7.4, Seite 214")

    def test_view_ranks_files_and_keeps_two_evidence_rows(self):
        chunks = [
            Chunk(text=HYDRAULIK.replace("Die Anleitung", "Kapitel 7.4: die Anleitung"), source="Wartungsanleitung GB-204.pdf", modality="pdf", kind="text", page=214),
            Chunk(text="Druckabfall am Stopfen im Video.", source="SV-204-HYD-03.mp4", modality="video", kind="speech", start=80, end=110),
            Chunk(text="Schicht ohne Befund.", source="Schichtbericht Gleis 4.pdf", modality="pdf", kind="text", page=2),
        ]
        index = VectorIndex(dim=2)
        vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.2, 0.8]], np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add(chunks, vectors)
        query = np.array([1.0, 0.0], np.float32)
        ranked = best_per_source(index, query)
        view = present(
            "Druck",
            ranked,
            chunks,
            durations={"SV-204-HYD-03.mp4": 252},
            thumbs={"SV-204-HYD-03.mp4"},
            media={"Wartungsanleitung GB-204.pdf", "SV-204-HYD-03.mp4"},
        )
        self.assertEqual(view["answer"][:12], "Sollwert 180")
        self.assertEqual([item["source"] for item in view["evidence"]], [
            "Wartungsanleitung GB-204.pdf",
            "SV-204-HYD-03.mp4",
        ])
        self.assertEqual(view["evidence"][0]["subtitle"], "Kapitel 7.4, Seite 214")
        self.assertEqual(view["evidence"][1]["duration_label"], "4:12")
        self.assertTrue(view["evidence"][1]["thumb_url"].endswith("SV-204-HYD-03.mp4"))
        self.assertEqual(view["files"][0]["percent"] >= view["files"][-1]["percent"], True)
        self.assertEqual(view["files"][0]["title"], "Wartungsanleitung GB-204.pdf")
        self.assertIn("Rücklauffilter", view["chips"])


if __name__ == "__main__":
    unittest.main()
