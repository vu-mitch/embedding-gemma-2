"""Turn PDFs, audio and video into timestamped / paged text chunks.

Everything is projected into text so a single text-embedding model
(EmbeddingGemma) can index all modalities in one vector space:

* PDF   -> page text (pypdf), split into overlapping chunks
* Audio -> speech transcript (Whisper via ``transformers``), grouped into time windows
* Video -> speech transcript of the audio track + captions of sampled keyframes (BLIP)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

ProgressFn = Callable[[str], None]


@dataclass
class Chunk:
    text: str
    source: str
    modality: str  # "pdf" | "audio" | "video"
    kind: str  # "text" | "speech" | "visual"
    page: int | None = None
    start: float | None = None
    end: float | None = None
    meta: dict = field(default_factory=dict)

    def locator(self) -> str:
        if self.page is not None:
            return f"page {self.page}"
        if self.start is not None:
            return f"{fmt_ts(self.start)}-{fmt_ts(self.end or self.start)}"
        return ""

    def to_dict(self) -> dict:
        return asdict(self)


def fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def split_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words) - overlap), step)]


# --------------------------------------------------------------------------- PDF


def extract_pdf(path: str | Path, source: str, chunk_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    from pypdf import PdfReader

    chunks: list[Chunk] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        for piece in split_text(text, chunk_words, overlap_words):
            chunks.append(Chunk(text=piece, source=source, modality="pdf", kind="text", page=page_no))
    return chunks


# ------------------------------------------------------------------------- Audio


class Transcriber:
    """Whisper ASR through the ``transformers`` pipeline API."""

    def __init__(self, model_id: str = "openai/whisper-base", device: str | int | None = None):
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device if device is not None else -1,
            chunk_length_s=30,
        )

    def __call__(self, audio_path: str | Path, language: str | None = None) -> list[dict]:
        wav = _to_wav16k(audio_path)
        try:
            kwargs: dict = {"return_timestamps": True, "batch_size": 4}
            if language and language != "auto":
                kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}
            out = self.pipe(wav, **kwargs)
        finally:
            Path(wav).unlink(missing_ok=True)
        segments = []
        for c in out.get("chunks", []):
            start, end = c.get("timestamp", (None, None))
            text = (c.get("text") or "").strip()
            if text:
                segments.append({"start": float(start or 0.0), "end": float(end if end is not None else start or 0.0), "text": text})
        if not segments and out.get("text", "").strip():
            segments.append({"start": 0.0, "end": 0.0, "text": out["text"].strip()})
        return segments


def group_segments(segments: list[dict], window_s: float) -> list[dict]:
    """Merge Whisper segments into windows of roughly ``window_s`` seconds."""
    groups: list[dict] = []
    for seg in segments:
        if groups and seg["end"] - groups[-1]["start"] <= window_s:
            groups[-1]["end"] = seg["end"]
            groups[-1]["text"] += " " + seg["text"]
        else:
            groups.append(dict(seg))
    return groups


def extract_audio(
    path: str | Path,
    source: str,
    transcriber: Transcriber,
    window_s: float = 30.0,
    modality: str = "audio",
    language: str | None = None,
) -> list[Chunk]:
    return [
        Chunk(text=g["text"], source=source, modality=modality, kind="speech", start=g["start"], end=g["end"])
        for g in group_segments(transcriber(path, language=language), window_s)
    ]


# ------------------------------------------------------------------------- Video


class Captioner:
    """BLIP image captioning through ``transformers``."""

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).eval()

    def __call__(self, images: list) -> list[str]:
        import torch

        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=30)
        return [c.strip() for c in self.processor.batch_decode(ids, skip_special_tokens=True)]


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout).get("format", {}).get("duration", 0.0) or 0.0)


def has_audio_stream(path: str | Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def _to_wav16k(path: str | Path) -> str:
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", wav],
        check=True,
    )
    return wav


def sample_frames(path: str | Path, every_s: float, max_frames: int) -> list[tuple[float, float, "Image.Image"]]:  # noqa: F821
    """Grab the middle frame of each ``every_s`` interval: (start, end, image)."""
    from PIL import Image

    duration = probe_duration(path)
    if duration <= 0:
        return []
    every_s = max(every_s, duration / max_frames)
    intervals = [(t, min(t + every_s, duration)) for t in _frange(0.0, duration, every_s)][:max_frames]
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (t0, t1) in enumerate(intervals):
            out = Path(tmp) / f"f{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{(t0 + t1) / 2:.2f}", "-i", str(path),
                 "-frames:v", "1", "-vf", "scale=384:-2", str(out)],
                check=False,
            )
            if out.exists():
                frames.append((t0, t1, Image.open(out).convert("RGB")))
    return frames


def _frange(start: float, stop: float, step: float):
    while start < stop:
        yield start
        start += step


def extract_video(
    path: str | Path,
    source: str,
    transcriber: Transcriber | None,
    captioner: Captioner | None,
    window_s: float = 30.0,
    frame_every_s: float = 10.0,
    max_frames: int = 12,
    progress: ProgressFn | None = None,
    language: str | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    if transcriber is not None and has_audio_stream(path):
        if progress:
            progress(f"Transcribing audio track of {source}")
        chunks += extract_audio(path, source, transcriber, window_s=window_s, modality="video", language=language)

    if captioner is not None:
        if progress:
            progress(f"Captioning keyframes of {source}")
        frames = sample_frames(path, frame_every_s, max_frames)
        captions = captioner([img for _, _, img in frames])
        for (t0, t1, img), caption in zip(frames, captions):
            chunks.append(
                Chunk(
                    text=f"Video frame showing {caption}",
                    source=source, modality="video", kind="visual",
                    start=t0, end=t1,
                    meta={"thumb": _thumb_bytes(img)},
                )
            )
    return chunks


def _thumb_bytes(img) -> bytes:
    import io

    buf = io.BytesIO()
    img.copy().resize((192, int(192 * img.height / img.width))).save(buf, format="JPEG", quality=80)
    return buf.getvalue()
