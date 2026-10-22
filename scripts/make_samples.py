"""Generate 3 PDFs, 3 audio clips and 3 videos on distinct topics for a quick demo.

Requires ``ffmpeg`` and ``espeak-ng`` on PATH.  Usage:

    python scripts/make_samples.py  # writes into ./samples
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"

PDFS = {
    "photosynthesis.pdf": (
        "Photosynthesis",
        "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide "
        "to produce glucose and oxygen. It takes place in the chloroplasts, where chlorophyll absorbs "
        "mostly red and blue light. The light dependent reactions split water molecules and release "
        "oxygen, while the Calvin cycle fixes carbon dioxide into sugars.",
    ),
    "rome_history.pdf": (
        "The Roman Empire",
        "The Roman Empire was founded in 27 BC when Augustus became the first emperor. At its height "
        "under Trajan it stretched from Britain to Mesopotamia. Roman engineers built aqueducts, roads "
        "and the Colosseum. The western empire fell in 476 AD when Odoacer deposed Romulus Augustulus.",
    ),
    "espresso_guide.pdf": (
        "Brewing Espresso",
        "A good espresso uses about eighteen grams of finely ground coffee and extracts roughly thirty "
        "six grams of liquid in twenty five to thirty seconds at nine bars of pressure. Grind size is the "
        "main lever: if the shot runs too fast and tastes sour, grind finer; if it is bitter, grind coarser.",
    ),
}

AUDIO = {
    "podcast_marathon.wav": "Today we talk about marathon training. Build your weekly mileage slowly, "
    "include one long run every weekend, and taper for two weeks before race day.",
    "lecture_black_holes.wav": "A black hole forms when a massive star collapses. Its event horizon "
    "marks the boundary beyond which not even light can escape the gravity.",
    "recipe_pancakes.wav": "To make pancakes, whisk flour, milk, two eggs and a pinch of salt. "
    "Cook each pancake on a hot buttered pan until bubbles appear, then flip it.",
}

VIDEOS = {
    "ocean_waves.mp4": ("blue", "Ocean waves crash on the beach. The tide rises in the evening and surfers love the swell."),
    "volcano_eruption.mp4": ("red", "This volcano erupts lava and ash. Magma rises from the mantle through the crust."),
    "forest_hike.mp4": ("green", "We are hiking through a green pine forest, following a trail to a waterfall."),
}


def write_pdf(path: Path, title: str, body: str) -> None:
    """Write a minimal single-page text PDF without third-party dependencies."""
    lines = [title, ""] + textwrap.wrap(body, 80)
    esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 12 Tf 15 TL 72 740 Td " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


def tts(text: str, wav: Path) -> None:
    subprocess.run(["espeak-ng", "-s", "150", "-w", str(wav), text], check=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, (title, body) in PDFS.items():
        write_pdf(OUT / name, title, body)
    for name, text in AUDIO.items():
        tts(text, OUT / name)
    for name, (color, text) in VIDEOS.items():
        wav = OUT / (Path(name).stem + "_tmp.wav")
        tts(text, wav)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=24",
             "-i", str(wav), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(OUT / name)],
            check=True,
        )
        wav.unlink()
    print(f"Wrote samples to {OUT}")


if __name__ == "__main__":
    main()
