import os
import time
import unittest

os.environ["MMRAG_EMBEDDER"] = "hash"

from fastapi.testclient import TestClient

import server


def pdf_bytes(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 48 740 Td ({safe}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class ApiTests(unittest.TestCase):
    def setUp(self):
        server.reset_state()
        self.client = TestClient(server.app)

    def test_beispiel_matches_the_reference_fields(self):
        data = self.client.get("/api/beispiel").json()
        self.assertIn("GB-204", data["query"])
        self.assertEqual(data["chips"], ["Druckspeicher", "Pumpe A2", "Ventil RV-12", "Rücklauffilter"])
        self.assertEqual(data["evidence"][0]["subtitle"], "Kapitel 7.4, Seite 214")
        self.assertEqual(data["evidence"][1]["duration_label"], "4:12")
        self.assertEqual([item["percent"] for item in data["files"]], [94, 91, 74, 58, 31])

    def test_page_shell(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Top Files", page.text)
        self.assertEqual(self.client.get("/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/assets/hydraulik-thumb.jpg").status_code, 200)

    def test_index_and_search_roundtrip(self):
        text = (
            "Sollwert 180 bar. Kapitel 7.4 prueft den Druckspeicher, die Pumpe A2, "
            "das Ventil RV-12 und den Ruecklauffilter am Hydraulikkreis."
        )
        upload = self.client.post(
            "/api/files",
            files=[("files", ("Wartungsanleitung GB-204.pdf", pdf_bytes(text), "application/pdf"))],
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        started = self.client.post("/api/index", json={})
        self.assertEqual(started.status_code, 200, started.text)
        status = {}
        for _ in range(50):
            status = self.client.get("/api/status").json()
            if status["state"] in {"ready", "error"}:
                break
            time.sleep(0.05)
        self.assertEqual(status["state"], "ready", status)
        result = self.client.post("/api/search", json={"query": "„Druckspeicher Sollwert“"}).json()
        self.assertIn("Sollwert 180 bar", result["answer"])
        self.assertEqual(result["evidence"][0]["title"], "Wartungsanleitung GB-204")
        self.assertIn("Kapitel 7.4", result["evidence"][0]["subtitle"])
        self.assertEqual(result["files"][0]["title"], "Wartungsanleitung GB-204.pdf")
        self.assertGreater(result["files"][0]["percent"], 0)
        self.assertIn("Druckspeicher", result["chips"])

    def test_rejects_a_fourth_pdf(self):
        blob = pdf_bytes("Kurztext zum Index.")
        for i in range(3):
            res = self.client.post("/api/files", files=[("files", (f"a{i}.pdf", blob, "application/pdf"))])
            self.assertEqual(res.status_code, 200, res.text)
        extra = self.client.post("/api/files", files=[("files", ("a3.pdf", blob, "application/pdf"))])
        self.assertEqual(extra.status_code, 400)
        self.assertIn("3", extra.json()["detail"])

    def test_search_before_index(self):
        res = self.client.post("/api/search", json={"query": "Druck"})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
