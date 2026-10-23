"""Build the Gleisbau demo corpus: one file set per stakeholder use case.

Writes PDFs, one recorded conversation and two narrated videos into ``demo/gleisbau``.
Requires ``ffmpeg`` on PATH, plus ``fpdf2``, ``edge-tts`` and Pillow.

    python scripts/make_gleisbau_demo.py

Speech is synthesised with a neural German voice so Whisper can transcribe it.
The documents are fictional demo material. They are labelled as such and do not
authorise real maintenance work.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "gleisbau"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DISCLAIMER = (
    "Demo-Unterlage. Sie gibt keinen Reparaturauftrag frei und ersetzt keine "
    "freigegebene Herstelleranweisung. Vor einer Arbeit Variante, Revision und "
    "Dokumentenstand prüfen."
)


VOICE_LENA = "de-DE-KatjaNeural"
VOICE_KOWALCZYK = "de-DE-ConradNeural"


def tts(text: str, wav: Path, *, voice: str = VOICE_LENA) -> None:
    import asyncio

    import edge_tts

    wav.parent.mkdir(parents=True, exist_ok=True)
    mp3 = wav.with_suffix(".mp3")

    async def _save() -> None:
        await edge_tts.Communicate(text, voice, rate="-8%").save(str(mp3))

    asyncio.run(_save())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "22050", "-ac", "1", str(wav)],
        check=True,
    )
    mp3.unlink(missing_ok=True)


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


def silence(path: Path, seconds: float = 0.35) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", f"{seconds:.2f}", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )


def concat_wavs(parts: list[Path], dest: Path) -> None:
    listing = dest.with_suffix(".concat.txt")
    listing.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )
    listing.unlink()


class Manual(FPDF):
    def __init__(self, doc_id: str, revision: str, status: str, variant: str):
        super().__init__(format="A4")
        self.doc_id = doc_id
        self.revision = revision
        self.status = status
        self.variant = variant
        self.add_font("DejaVu", "", FONT)
        self.add_font("DejaVu", "B", FONT_B)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title(doc_id)

    def header(self):
        self.set_fill_color(12, 48, 84)
        self.rect(0, 0, 210, 14, style="F")
        self.set_xy(12, 3.5)
        self.set_text_color(255, 255, 255)
        self.set_font("DejaVu", "B", 8)
        self.cell(0, 7, f"DEMO  ·  {self.doc_id}  ·  Rev. {self.revision}  ·  Status: {self.status}  ·  {self.variant}")
        self.set_text_color(20, 20, 20)
        self.ln(16)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(90, 90, 90)
        self.cell(160, 5, "Demo-Unterlage · keine Reparaturfreigabe · nur den genannten Dokumentenstand zitieren")
        self.cell(30, 5, f"Seite {self.page_no()}", align="R")
        self.set_text_color(20, 20, 20)

    def h1(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 16)
        self.multi_cell(0, 8, text, align="LEFT", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str) -> None:
        self.ln(1)
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "B", 13)
        self.multi_cell(0, 7, text, align="LEFT", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def p(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 5.6, text, align="LEFT", new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bullets(self, items: list[str]) -> None:
        self.set_font("DejaVu", "", 11)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.6, f"- {item}", align="LEFT", new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def meta_table(self, rows: list[tuple[str, str]]) -> None:
        for key, value in rows:
            self.set_font("DejaVu", "B", 11)
            self.cell(48, 8, key, border=1)
            self.set_font("DejaVu", "", 11)
            self.cell(132, 8, value, border=1)
            self.ln(8)
        self.ln(3)


def write_manual(path: Path) -> None:
    pdf = Manual("WA-GB-204-HYD", "C", "freigegeben", "Variante GB-204")
    pdf.add_page()
    pdf.h1("Wartungsanleitung Hydraulik")
    pdf.p("Maschine GB-204. Dieses Dokument ist die freigegebene Wartungsanleitung für die Hydraulikanlage.")
    pdf.meta_table([
        ("Dokument", "WA-GB-204-HYD"),
        ("Revision", "C"),
        ("Status", "freigegeben"),
        ("Stand", "15.01.2026"),
        ("Gültig für", "ausschließlich Maschine GB-204"),
        ("Verweis Video", "Servicevideo SV-GB204-HYD-01"),
    ])
    pdf.p(DISCLAIMER)

    pdf.add_page()
    pdf.h1("Kapitel 2 · Fahrwerk schmieren")
    pdf.p(
        "Die Laufräder der Maschine werden an Schmierstelle SL-4 mit Fett LG-2 versorgt. "
        "Intervall: 50 Betriebsstunden. Spurkranz und Radsatzlager gehören in das Fahrwerkprotokoll, "
        "nicht in die Hydraulikprüfung. Für die Beleuchtung ist Scheinwerfermodul LM-9 zuständig. "
        "Dieses Kapitel enthält keine Druckprüfung und keine Filterarbeit."
    )

    pdf.add_page()
    pdf.h1("Kapitel 4 · Hydraulik verliert Druck")
    pdf.h2("Prüfhinweise, wenn die Hydraulik an Maschine GB-204 Druck verliert")
    pdf.p(
        "Symptom: Die Hydraulik an Maschine GB-204 verliert Druck, erkennbar am Absinken der Anzeige "
        "während die Stopfeinheit im Leerlauf steht. Nur die folgenden freigegebenen Prüfhinweise "
        "aus dieser Wartungsanleitung verwenden. Dieselbe Prüfung zeigt das Servicevideo SV-GB204-HYD-01."
    )
    pdf.bullets([
        "Prüfhinweis 1. Am Messpunkt MP-H3 den Druck messen. Sollwert 180 bar, Toleranz plus minus 5 bar. Liegt der Wert darunter, nicht weiterarbeiten, sondern die nächsten Hinweise abarbeiten.",
        "Prüfhinweis 2. Die Schlauchleitung H12 auf äußere Leckage und scheuernde Schellen prüfen. Eine nasse Kupplung ist zu sichern und zu melden.",
        "Prüfhinweis 3. Die Anzeige des Rücklauffilters ablesen und das Proportionalventil Y14 auf Sitz und Stecker prüfen.",
        "Prüfhinweis 4. Ergebnis, Messwert und Uhrzeit in das Maschinenprotokoll der GB-204 eintragen. Eine Reparaturfreigabe erteilt diese Anleitung nicht.",
    ])
    pdf.p("Zugehöriges Servicevideo: SV-GB204-HYD-01, Datei servicevideo-gb204-hydraulikdruck.")

    pdf.add_page()
    pdf.h1("Kapitel 7 · Beleuchtung")
    pdf.p(
        "Das LED-Modul LM-9 sitzt am Vorbau. Tausch nur bei kaltem Gehäuse. Dieses Kapitel betrifft "
        "weder die Hydraulik noch die Stopfeinheit."
    )
    pdf.output(path)


def write_parts(path: Path) -> None:
    pdf = Manual("ET-STF-4418", "B", "freigegeben", "Maschinenvariante B")
    pdf.add_page()
    pdf.h1("Ersatzteilkatalog Stopfeinheit")
    pdf.p("Zeichnungssatz zur Stopfeinheit. Die Blattzahl unten nennt die gültige Variante.")
    pdf.meta_table([
        ("Dokument", "ET-STF-4418"),
        ("Revision", "B"),
        ("Status", "freigegeben"),
        ("Stand", "02.03.2026"),
        ("Gültig für", "nur Maschinenvariante B"),
        ("Zeichnung", "EZ-4418 Blatt 3"),
    ])
    pdf.p(DISCLAIMER)

    pdf.add_page()
    pdf.h1("Blatt 2 · Stopfpickel")
    pdf.p(
        "Verschleißplatte des Stopfpickels, Position 4, Sachnummer 61-2200-04. "
        "Die Platte wird getauscht, sobald die Kante abgeschliffen ist. "
        "Anzug der Befestigung mit 120 Newtonmeter. Dieses Blatt beschreibt nur den Pickel."
    )

    pdf.add_page()
    pdf.h1("Blatt 3 · Ersatzteilzeichnung Hydraulikzylinder")
    pdf.h2("Dichtung für den Hydraulikzylinder der Stopfeinheit, Maschinenvariante B")
    pdf.p(
        "Gesucht wird die Dichtung für den Hydraulikzylinder der Stopfeinheit. "
        "Die passende Ersatzteilzeichnung ist EZ-4418 Blatt 3. Sie gilt nur für Maschinenvariante B. "
        "Maschinenvariante A verwendet andere Sachnummern und darf mit diesem Blatt nicht bedient werden."
    )
    draw_cylinder(pdf)
    pdf.ln(2)
    pdf.h2("Stückliste zu Zeichnung EZ-4418, nur Variante B")
    pdf.meta_table([
        ("Pos. 12", "Stangendichtung  ·  Teilenummer 73-4418-B-12"),
        ("Pos. 13", "Kolbendichtung  ·  Teilenummer 73-4418-B-13"),
        ("Nicht verwenden", "Variante A: Stangendichtung 73-4418-A-12"),
        ("Einbauort", "Hydraulikzylinder der Stopfeinheit, Variante B"),
    ])
    pdf.p(
        "Die Teilenummer der gesuchten Stangendichtung für Maschinenvariante B lautet 73-4418-B-12. "
        "Die Kolbendichtung derselben Zeichnung lautet 73-4418-B-13. "
        "Vor der Entnahme Variante B am Typenschild prüfen."
    )

    pdf.add_page()
    pdf.h1("Blatt 5 · Schmierstoffe Fahrantrieb")
    pdf.p("Getriebeöl des Fahrantriebs: Sorte CLP 220. Wechselintervall 1000 Betriebsstunden. Dieses Blatt beschreibt nur den Fahrantrieb.")
    pdf.output(path)


def draw_cylinder(pdf: Manual) -> None:
    """Simple section of a hydraulic cylinder with seal callouts 12 and 13."""
    x, y, w, h = 20, pdf.get_y(), 170, 70
    pdf.set_draw_color(20, 40, 70)
    pdf.set_line_width(0.4)
    pdf.rect(x, y, w, h)
    pdf.set_line_width(0.35)
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_xy(x + 48, y + 5)
    pdf.cell(55, 5, "Pos. 13 Kolbendichtung")
    pdf.set_xy(x + 112, y + 5)
    pdf.cell(52, 5, "Pos. 12 Stangendichtung")
    # barrel and rod
    pdf.rect(x + 22, y + 28, 78, 22)
    pdf.set_fill_color(214, 214, 214)
    pdf.rect(x + 62, y + 30, 9, 18, style="FD")
    pdf.rect(x + 108, y + 34, 42, 10)
    # piston seal and its leader, staying left of the rod
    pdf.set_fill_color(176, 42, 42)
    pdf.ellipse(x + 60, y + 30, 13, 5, style="F")
    pdf.line(x + 66, y + 30, x + 66, y + 12)
    # rod seal at the gland, leader up to its own label
    pdf.ellipse(x + 100, y + 32, 8, 14, style="F")
    pdf.line(x + 128, y + 12, x + 128, y + 34)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_xy(x + 4, y + 48)
    pdf.multi_cell(40, 4, "EZ-4418\nStopfeinheit\nVariante B", align="LEFT")
    pdf.set_y(y + h + 3)


def write_report(path: Path) -> None:
    pdf = Manual("SB-2024-0842", "1", "freigegeben, archiviert", "Maschine GB-187")
    pdf.add_page()
    pdf.h1("Servicebericht")
    pdf.p("Archivierter Bericht über einen abgeschlossenen Einsatz an einer Gleisbaumaschine.")
    pdf.meta_table([
        ("Bericht", "SB-2024-0842"),
        ("Datum", "14.03.2024"),
        ("Status", "freigegeben, archiviert"),
        ("Maschine", "Gleisbaumaschine GB-187"),
        ("Verfasser", "Meister Kowalczyk"),
        ("Gespräch dazu", "18.03.2024, Lena Vogt"),
    ])
    pdf.p(DISCLAIMER)

    pdf.add_page()
    pdf.h1("Einsatz vom 14.03.2024")
    pdf.h2("Vergleichbarer Fehler an der Gleisbaumaschine")
    pdf.p(
        "An der Gleisbaumaschine GB-187 gab es einen vergleichbaren Fehler: die Hydraulik an der "
        "Stopfeinheit hat während des Leerlaufs Druck verloren. Der Fall liegt zwei Jahre vor der "
        "aktuellen Anfrage und ist in diesem Servicebericht dokumentiert."
    )
    pdf.h2("Was damals festgestellt wurde")
    pdf.p(
        "Festgestellt wurde eine gealterte Stangendichtung an Position 12 des Hydraulikzylinders "
        "der Stopfeinheit. Die verbaute Teilenummer war 73-4418-B-12, passend zu Maschinenvariante B. "
        "Äußere Leckage an der Schlauchleitung wurde nicht festgestellt. Der Messpunkt MP-H3 zeigte "
        "vor dem Tausch 140 bar."
    )
    pdf.h2("Welche Maßnahmen dokumentiert wurden")
    pdf.bullets([
        "Stangendichtung Position 12 gegen Teilenummer 73-4418-B-12 getauscht.",
        "Hydrauliksystem gespült und danach am Messpunkt MP-H3 erneut gemessen.",
        "Nach der Maßnahme wurden wieder 180 bar erreicht und im Protokoll notiert.",
        "Keine weitere Freigabe über diesen Bericht hinaus. Der Bericht beschreibt nur den damaligen, abgeschlossenen Fall an GB-187.",
    ])
    pdf.p(
        "Meister Kowalczyk hat die Feststellung und die Maßnahmen am 14.03.2024 freigegeben. "
        "Im Gespräch vom 18.03.2024 mit Lena Vogt wurde derselbe Verlauf noch einmal mündlich bestätigt."
    )

    pdf.add_page()
    pdf.h1("Anderer Einsatz · Bremssohle")
    pdf.p(
        "Am 02.02.2024 wurde an einer anderen Maschine nur eine Bremssohle gewechselt. "
        "Dieser Abschnitt ist kein Hydraulikfall und enthält keine Feststellung zum Druckverlust."
    )
    pdf.output(path)


def write_training(path: Path) -> None:
    pdf = Manual("SU-HYD-FLT-03", "3", "freigegeben", "Hydraulikeinheit, alle GB-Maschinen")
    pdf.add_page()
    pdf.h1("Schulungsunterlage Filterwechsel")
    pdf.p("Unterlage zum Wechsel des Filters an der Hydraulikeinheit. Das Schulungsvideo zeigt denselben Ablauf.")
    pdf.meta_table([
        ("Dokument", "SU-HYD-FLT-03"),
        ("Revision", "3"),
        ("Status", "freigegeben"),
        ("Stand", "20.09.2026"),
        ("Gültig für", "Hydraulikeinheit der GB-Maschinen"),
        ("Schulungsvideo", "SV-FLT-03"),
    ])
    pdf.p(DISCLAIMER)

    pdf.add_page()
    pdf.h1("Tageskontrolle Beleuchtung")
    pdf.p(
        "Vor Schichtbeginn die Vorbau-Leuchten einschalten und auf Ausfälle prüfen. "
        "Ein defektes Modul wird als LM-9 gemeldet. Dieses Blatt beschreibt nur die Leuchten."
    )

    pdf.add_page()
    pdf.h1("Wechsel des Filters an der Hydraulikeinheit")
    pdf.p(
        "In dieser Schulungsunterlage wird der Wechsel des Filters an der Hydraulikeinheit gezeigt. "
        "Dieselben vier Schritte spricht das Schulungsvideo SV-FLT-03, Datei schulung-filterwechsel-hydraulikeinheit. "
        "Die Unterlage ist freigegeben und erteilt keine eigene Reparaturfreigabe."
    )
    pdf.bullets([
        "Schritt 1. Die Hydraulikeinheit drucklos machen und den Absperrhahn H-F1 schließen.",
        "Schritt 2. Die Auffangwanne unter das Filtergehäuse stellen.",
        "Schritt 3. Das Filtergehäuse öffnen und das Filterelement 44-HYD-FLT entnehmen.",
        "Schritt 4. Das neue Filterelement 44-HYD-FLT einsetzen, die Gehäusedichtung mit dem freigegebenen Öl benetzen, das Gehäuse schließen, entlüften und den Stand am Schauglas prüfen.",
    ])
    pdf.p("Nur Filterelement 44-HYD-FLT aus dem freigegebenen Schulungsstand Revision 3 verwenden.")

    pdf.add_page()
    pdf.h1("Anhang · Werkzeugliste allgemein")
    pdf.p("Maulschlüssel 19 und Auffangwanne gehören zum Satz. Kein weiterer Arbeitsablauf auf diesem Blatt.")
    pdf.output(path)


def render_card(path: Path, *, kicker: str, title: str, lines: list[str], accent: tuple[int, int, int], kind: str) -> None:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1280, 96], fill=accent)
    kicker_font = ImageFont.truetype(FONT_B, 22)
    title_font = ImageFont.truetype(FONT_B, 40)
    body_font = ImageFont.truetype(FONT, 32)
    small = ImageFont.truetype(FONT, 20)
    draw.text((40, 32), kicker, font=kicker_font, fill="white")
    draw.text((40, 130), title, font=title_font, fill=accent)
    if kind == "gauge":
        draw.ellipse([960, 250, 1180, 470], outline=accent, width=8)
        draw.polygon([(1070, 360), (1125, 300), (1078, 360)], fill=accent)
        draw.text((990, 490), "MP-H3", font=body_font, fill=accent)
    elif kind == "filter":
        draw.rounded_rectangle([980, 240, 1160, 520], radius=30, outline=accent, width=8)
        draw.rectangle([1020, 200, 1120, 250], outline=accent, width=6)
        draw.text((990, 540), "44-HYD-FLT", font=small, fill=accent)
    y = 230
    for line in lines:
        draw.text((40, y), line, font=body_font, fill=(30, 30, 30))
        y += 58
    draw.rectangle([0, 670, 1280, 720], fill=(236, 236, 236))
    draw.text((40, 684), "Demo-Video · keine Reparaturfreigabe · freigegebenen Dokumentenstand beachten", font=small, fill=(70, 70, 70))
    image.save(path)


def make_video(path: Path, segments: list[dict], accent: tuple[int, int, int]) -> None:
    work = path.parent / f"_{path.stem}_build"
    work.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for index, segment in enumerate(segments):
        frame = work / f"f{index}.png"
        audio = work / f"a{index}.wav"
        clip = work / f"c{index}.mp4"
        render_card(
            frame,
            kicker=segment["kicker"],
            title=segment["title"],
            lines=segment["lines"],
            accent=accent,
            kind=segment.get("kind", ""),
        )
        tts(segment["speech"], audio, voice=segment.get("voice", VOICE_LENA))
        duration = max(wav_duration(audio) + 0.35, 2.5)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "12", "-t", f"{duration:.2f}",
             "-i", str(frame), "-i", str(audio), "-c:v", "libx264", "-tune", "stillimage",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)],
            check=True,
        )
        clips.append(clip)
    listing = work / "clips.txt"
    listing.write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(path)],
        check=True,
    )
    for child in work.iterdir():
        child.unlink()
    work.rmdir()


def make_conversation(path: Path) -> None:
    turns = [
        (VOICE_LENA, "Lena Vogt hier. Ich frage nach einem alten Fall an unseren Gleisbaumaschinen."),
        (VOICE_KOWALCZYK, "Meister Kowalczyk. Ja. Vor zwei Jahren gab es einen vergleichbaren Fehler."),
        (VOICE_LENA, "Was wurde damals festgestellt?"),
        (VOICE_KOWALCZYK, "Festgestellt wurde eine gealterte Stangendichtung. Die Hydraulik der Stopfeinheit hat Druck verloren."),
        (VOICE_LENA, "Welche Maßnahmen wurden dokumentiert?"),
        (VOICE_KOWALCZYK, "Die Maßnahmen stehen im Servicebericht. Die Stangendichtung wurde getauscht, das System gespült, und der Druck lag wieder bei 180 Bar."),
    ]
    work = path.parent / "_gespraech_build"
    work.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for index, (voice, text) in enumerate(turns):
        spoken = work / f"t{index}.wav"
        gap = work / f"g{index}.wav"
        tts(text, spoken, voice=voice)
        silence(gap, 0.45)
        parts.extend([spoken, gap])
    concat_wavs(parts, path)
    for child in work.iterdir():
        child.unlink()
    work.rmdir()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_manual(OUT / "WA-GB204-Hydraulik-RevC.pdf")
    write_parts(OUT / "ET-Stopfeinheit-Variante-B.pdf")
    write_report(OUT / "SB-2024-0842-Servicebericht.pdf")
    write_training(OUT / "SU-HYD-FLT-03-Schulung-Filterwechsel.pdf")
    make_conversation(OUT / "gespraech-kowalczyk-vogt-2024.wav")
    make_video(
        OUT / "servicevideo-gb204-hydraulikdruck.mp4",
        [
            {
                "kicker": "Servicevideo SV-GB204-HYD-01  ·  freigegeben  ·  Rev. C",
                "title": "Hydraulik GB-204 verliert Druck",
                "lines": ["Wartungsanleitung WA-GB-204-HYD", "Nur diese Prüfhinweise verwenden"],
                "speech": "Servicevideo zur Wartungsanleitung der Maschine GB 204. Die Hydraulik verliert Druck. Es gelten die Hinweise der freigegebenen Wartungsanleitung.",
                "kind": "gauge",
            },
            {
                "kicker": "Prüfhinweis 1  ·  Messpunkt MP-H3",
                "title": "Druck am Messpunkt messen",
                "lines": ["Sollwert 180 bar", "Toleranz plus minus 5 bar"],
                "speech": "Hinweis eins. Am Messpunkt den Druck messen. Der Sollwert beträgt 180 Bar.",
                "kind": "gauge",
            },
            {
                "kicker": "Prüfhinweis 2  ·  Schlauchleitung H12",
                "title": "Leitung auf Leckage prüfen",
                "lines": ["Schlauchleitung H12", "Nasse Kupplung sichern und melden"],
                "speech": "Hinweis zwei. Die Schlauchleitung auf äußere Leckage prüfen.",
                "kind": "gauge",
            },
            {
                "kicker": "Prüfhinweis 3  ·  Ventil Y14",
                "title": "Filteranzeige und Ventil",
                "lines": ["Rücklauffilter ablesen", "Proportionalventil Y14 prüfen"],
                "speech": "Hinweis drei. Die Anzeige des Rücklauffilters und das Ventil prüfen. Die Schritte stehen in der Wartungsanleitung.",
                "kind": "gauge",
            },
        ],
        accent=(12, 48, 84),
    )
    make_video(
        OUT / "schulung-filterwechsel-hydraulikeinheit.mp4",
        [
            {
                "kicker": "Schulungsvideo SV-FLT-03  ·  freigegeben  ·  Rev. 3",
                "title": "Wechsel des Filters",
                "lines": ["Hydraulikeinheit", "Schulungsunterlage SU-HYD-FLT-03"],
                "speech": "Schulungsvideo. Hier wird der Wechsel des Filters an der Hydraulikeinheit gezeigt. Die zugehörige Anleitung ist die Schulungsunterlage.",
                "kind": "filter",
            },
            {
                "kicker": "Schritt 1",
                "title": "Anlage drucklos machen",
                "lines": ["Absperrhahn H-F1 schließen"],
                "speech": "Schritt eins. Die Hydraulikeinheit drucklos machen und den Absperrhahn schließen.",
                "kind": "filter",
            },
            {
                "kicker": "Schritt 2 und 3",
                "title": "Filterelement entnehmen",
                "lines": ["Auffangwanne darunter stellen", "Filterelement 44-HYD-FLT herausnehmen"],
                "speech": "Schritt zwei. Die Auffangwanne unter das Filtergehäuse stellen. Schritt drei. Das Filtergehäuse öffnen und das Filterelement entnehmen.",
                "kind": "filter",
            },
            {
                "kicker": "Schritt 4",
                "title": "Neues Element einsetzen",
                "lines": ["Dichtung benetzen, Gehäuse zu", "Entlüften und Schauglas prüfen"],
                "speech": "Schritt vier. Das neue Filterelement einsetzen, die Gehäusedichtung benetzen, das Gehäuse schließen, entlüften und den Stand prüfen.",
                "kind": "filter",
            },
        ],
        accent=(14, 90, 62),
    )
    manifest = {
        "disclaimer": DISCLAIMER,
        "cases": [
            {
                "id": "A",
                "name": "The Machine Whisperer",
                "blurb": "Prüfhinweise aus der Wartungsanleitung und das passende Servicevideo.",
                "prompt": "Die Hydraulik an Maschine GB-204 verliert Druck. Welche relevanten Prüfhinweise stehen in der Wartungsanleitung, und gibt es dazu ein Servicevideo?",
                "files": ["WA-GB204-Hydraulik-RevC.pdf", "servicevideo-gb204-hydraulikdruck.mp4"],
            },
            {
                "id": "B",
                "name": "Find That Part",
                "blurb": "Ersatzteilzeichnung und Teilenummer, nur für Maschinenvariante B.",
                "prompt": "Ich brauche die Dichtung für den Hydraulikzylinder der Stopfeinheit. Finde die passende Ersatzteilzeichnung und die Teilenummer für Maschinenvariante B.",
                "files": ["ET-Stopfeinheit-Variante-B.pdf"],
            },
            {
                "id": "C",
                "name": "Learn from the Last Repair",
                "blurb": "Archivierter Servicebericht und das aufgezeichnete Gespräch dazu.",
                "prompt": "Gab es bei unseren Gleisbaumaschinen schon einmal einen vergleichbaren Fehler? Was wurde damals festgestellt und welche Maßnahmen wurden dokumentiert?",
                "files": ["SB-2024-0842-Servicebericht.pdf", "gespraech-kowalczyk-vogt-2024.wav"],
            },
            {
                "id": "D",
                "name": "Show Me How",
                "blurb": "Schulungsvideo und die zugehörige Anleitung zum Filterwechsel.",
                "prompt": "Wo wird in unseren Schulungsunterlagen der Wechsel des Filters an der Hydraulikeinheit gezeigt? Zeige mir das Video und die zugehörige Anleitung.",
                "files": ["SU-HYD-FLT-03-Schulung-Filterwechsel.pdf", "schulung-filterwechsel-hydraulikeinheit.mp4"],
            },
        ],
        "documents": [
            {
                "file": "WA-GB204-Hydraulik-RevC.pdf",
                "modality": "pdf",
                "case": "A",
                "title": "Wartungsanleitung Hydraulik GB-204",
                "doc_id": "WA-GB-204-HYD",
                "revision": "C",
                "status": "freigegeben",
                "variant": "Maschine GB-204",
                "issued": "15.01.2026",
            },
            {
                "file": "servicevideo-gb204-hydraulikdruck.mp4",
                "modality": "video",
                "case": "A",
                "title": "Servicevideo Hydraulikdruck GB-204",
                "doc_id": "SV-GB204-HYD-01",
                "revision": "C",
                "status": "freigegeben",
                "variant": "Maschine GB-204",
                "issued": "15.01.2026",
            },
            {
                "file": "ET-Stopfeinheit-Variante-B.pdf",
                "modality": "pdf",
                "case": "B",
                "title": "Ersatzteilzeichnung Stopfeinheit Variante B",
                "doc_id": "ET-STF-4418",
                "revision": "B",
                "status": "freigegeben",
                "variant": "Maschinenvariante B",
                "issued": "02.03.2026",
            },
            {
                "file": "SB-2024-0842-Servicebericht.pdf",
                "modality": "pdf",
                "case": "C",
                "title": "Servicebericht SB-2024-0842",
                "doc_id": "SB-2024-0842",
                "revision": "1",
                "status": "freigegeben, archiviert",
                "variant": "Maschine GB-187",
                "issued": "14.03.2024",
            },
            {
                "file": "gespraech-kowalczyk-vogt-2024.wav",
                "modality": "audio",
                "case": "C",
                "title": "Gespräch Kowalczyk / Vogt, 18.03.2024",
                "doc_id": "AUF-2024-0318",
                "revision": "Originalaufnahme",
                "status": "archiviert",
                "variant": "bezieht sich auf GB-187",
                "issued": "18.03.2024",
            },
            {
                "file": "SU-HYD-FLT-03-Schulung-Filterwechsel.pdf",
                "modality": "pdf",
                "case": "D",
                "title": "Schulungsunterlage Filterwechsel",
                "doc_id": "SU-HYD-FLT-03",
                "revision": "3",
                "status": "freigegeben",
                "variant": "Hydraulikeinheit der GB-Maschinen",
                "issued": "20.09.2026",
            },
            {
                "file": "schulung-filterwechsel-hydraulikeinheit.mp4",
                "modality": "video",
                "case": "D",
                "title": "Schulungsvideo Filterwechsel",
                "doc_id": "SV-FLT-03",
                "revision": "3",
                "status": "freigegeben",
                "variant": "Hydraulikeinheit der GB-Maschinen",
                "issued": "20.09.2026",
            },
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Gleisbau demo to {OUT}")


if __name__ == "__main__":
    main()
