"""Web UI for multimodal retrieval.

Streamlit cannot reproduce the result layout (quoted question, answer card,
evidence rows, file meters), so the demo serves a small static frontend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from mmrag import (
    GGUF_MODELS,
    MRL_DIMS,
    Captioner,
    GGUFEmbedder,
    HFEmbedder,
    Transcriber,
    VectorIndex,
    extract_audio,
    extract_pdf,
    extract_video,
)
from mmrag.extractors import probe_duration
from mmrag.present import best_per_source, clean_query, present

WEB = Path(__file__).resolve().parent / "web"
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "mmrag_uploads"
MAX_PER_TYPE = 3
MAX_BYTES = 200 * 1024 * 1024
BACKENDS = ("GGUF (llama.cpp)", "Transformers (HF)")
WHISPER_MODELS = ("openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small")
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

BEISPIEL = {
    "query": (
        "„Die Hydraulik an Maschine GB-204 verliert Druck. Welche relevanten "
        "Prüfinweise stehen in der Wartungsanleitung, und gibt es dazu ein Servicevideo?“"
    ),
    "answer": (
        "Sollwert 180 bar. Die Anleitung prüft vier Stellen, das Video zeigt den "
        "Ablauf am Hydraulikkreis Stopfen."
    ),
    "chips": ["Druckspeicher", "Pumpe A2", "Ventil RV-12", "Rücklauffilter"],
    "evidence": [
        {
            "source": "Wartungsanleitung GB-204.pdf",
            "title": "Wartungsanleitung GB-204",
            "subtitle": "Kapitel 7.4, Seite 214",
            "modality": "pdf",
            "icon": "pdf",
            "percent": 94,
            "text": (
                "Sollwert 180 bar. Kapitel 7.4 prüft vier Stellen am Hydraulikkreis Stopfen: "
                "Druckspeicher, Pumpe A2, Ventil RV-12 und Rücklauffilter."
            ),
            "page": 214,
            "start": None,
            "end": None,
            "kind": "text",
            "duration_label": None,
            "thumb_url": None,
            "media_url": None,
        },
        {
            "source": "SV-204-HYD-03.mp4",
            "title": "Druckabfall am Hydraulikkreis Stopfen",
            "subtitle": "Servicevideo SV-204-HYD-03",
            "modality": "video",
            "icon": "video",
            "percent": 91,
            "text": (
                "Servicevideo SV-204-HYD-03, Minute 2:10: Druckabfall am Hydraulikkreis Stopfen. "
                "Der Ablauf zeigt die Prüfung von Druckspeicher, Pumpe A2, Ventil RV-12 und Rücklauffilter."
            ),
            "page": None,
            "start": 130,
            "end": 160,
            "kind": "speech",
            "duration_label": "4:12",
            "thumb_url": "/assets/hydraulik-thumb.jpg?v=3",
            "media_url": None,
        },
    ],
    "files": [
        {"source": "Wartungsanleitung GB-204.pdf", "title": "Wartungsanleitung GB-204.pdf", "modality": "pdf", "icon": "pdf", "percent": 94, "text": "Sollwert 180 bar. Kapitel 7.4, Seite 214.", "subtitle": "Kapitel 7.4, Seite 214", "page": 214, "start": None, "end": None, "kind": "text", "duration_label": None, "thumb_url": None, "media_url": None},
        {"source": "SV-204-HYD-03.mp4", "title": "SV-204-HYD-03.mp4", "modality": "video", "icon": "video", "percent": 91, "text": "Druckabfall am Hydraulikkreis Stopfen.", "subtitle": "Servicevideo SV-204-HYD-03", "page": None, "start": 130, "end": 160, "kind": "speech", "duration_label": "4:12", "thumb_url": "/assets/hydraulik-thumb.jpg?v=3", "media_url": None},
        {"source": "Hydraulikschema GB-204.pdf", "title": "Hydraulikschema GB-204.pdf", "modality": "pdf", "icon": "image", "percent": 74, "text": "Hydraulikschema des Kreises Stopfen an Maschine GB-204.", "subtitle": "Schema", "page": 1, "start": None, "end": None, "kind": "text", "duration_label": None, "thumb_url": None, "media_url": None},
        {"source": "Prüfprotokoll Pumpe A2.xlsx", "title": "Prüfprotokoll Pumpe A2.xlsx", "modality": "sheet", "icon": "sheet", "percent": 58, "text": "Prüfprotokoll Pumpe A2, letzter Lauf unter Sollwert.", "subtitle": "Tabelle", "page": None, "start": None, "end": None, "kind": "text", "duration_label": None, "thumb_url": None, "media_url": None},
        {"source": "Schichtbericht Gleis 4.pdf", "title": "Schichtbericht Gleis 4.pdf", "modality": "pdf", "icon": "pdf", "percent": 31, "text": "Schichtbericht Gleis 4 ohne Befund zur Hydraulik.", "subtitle": "Seite 2", "page": 2, "start": None, "end": None, "kind": "text", "duration_label": None, "thumb_url": None, "media_url": None},
    ],
    "fixture": True,
}


class IndexIn(BaseModel):
    backend: str = "GGUF (llama.cpp)"
    variant: str = "Q8_0 (ggml-org, 330 MB)"
    dim: int = 768
    whisper_id: str = "openai/whisper-base"
    use_captions: bool = True
    chunk_words: int = 180
    window_s: int = 30
    frame_every: int = 10
    max_frames: int = 12


class SearchIn(BaseModel):
    query: str = ""
    modalities: list[str] = Field(default_factory=list)


@dataclass
class Stored:
    name: str
    modality: str
    path: Path
    mime: str
    duration: float | None = None
    thumb: bytes | None = None


@dataclass
class AppState:
    library: dict[str, Stored] = field(default_factory=dict)
    media: dict[str, Stored] = field(default_factory=dict)
    index: VectorIndex | None = None
    embedder: object | None = None
    settings: dict | None = None
    status: str = "idle"
    message: str = ""
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    stale: bool = False
    extract_cache: dict = field(default_factory=dict)


state = AppState()
lock = threading.Lock()
_embedders: dict[tuple, object] = {}
_transcribers: dict[str, Transcriber] = {}
_captioners: dict[str, Captioner] = {}


class HashEmbedder:
    """Lexical stand-in used when MMRAG_EMBEDDER=hash (no model download)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = "hash"

    def _vec(self, text: str) -> np.ndarray:
        import zlib

        vec = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[0-9A-Za-zÄÖÜäöüß]+", text.casefold()):
            vec[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    def embed_documents(self, texts, titles=None):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._vec(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def reset_state() -> None:
    with lock:
        state.library.clear()
        state.media.clear()
        state.index = None
        state.embedder = None
        state.settings = None
        state.status = "idle"
        state.message = ""
        state.progress = 0.0
        state.log = []
        state.stale = False
        state.extract_cache.clear()


def _check_index(body: IndexIn) -> IndexIn:
    if body.backend not in BACKENDS:
        raise HTTPException(400, "Unbekanntes Embedding-Backend.")
    if body.variant not in GGUF_MODELS:
        raise HTTPException(400, "Unbekannte GGUF-Variante.")
    if body.dim not in MRL_DIMS:
        raise HTTPException(400, "Diese Embedding-Dimension wird nicht unterstützt.")
    if body.whisper_id not in WHISPER_MODELS:
        raise HTTPException(400, "Unbekanntes Whisper-Modell.")
    if not 60 <= body.chunk_words <= 400:
        raise HTTPException(400, "PDF-Abschnittsgröße liegt außerhalb des zulässigen Bereichs.")
    if not 10 <= body.window_s <= 120 or not 2 <= body.frame_every <= 60 or not 2 <= body.max_frames <= 30:
        raise HTTPException(400, "Chunking-Werte liegen außerhalb des zulässigen Bereichs.")
    return body


def make_embedder(body: IndexIn):
    if os.getenv("MMRAG_EMBEDDER") == "hash":
        return HashEmbedder()
    if body.backend == "Transformers (HF)":
        key = ("hf", body.dim)
        if key not in _embedders:
            _embedders[key] = HFEmbedder(dim=body.dim)
        return _embedders[key]
    key = ("gguf", body.variant, body.dim)
    if key not in _embedders:
        _embedders[key] = GGUFEmbedder(variant=body.variant, dim=body.dim)
    return _embedders[key]


def _modality(name: str, mime: str) -> str | None:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in AUDIO_EXT or mime.startswith("audio/"):
        return "audio"
    if ext in VIDEO_EXT or mime.startswith("video/"):
        return "video"
    if ext == ".webm":
        return "audio" if mime.startswith("audio/") else "video"
    return None


def _counts() -> dict[str, int]:
    counts = {"pdf": 0, "audio": 0, "video": 0}
    for item in state.library.values():
        counts[item.modality] = counts.get(item.modality, 0) + 1
    return counts


def _log(message: str) -> None:
    with lock:
        state.log.append(message)
        state.message = message


def _poster(path: Path) -> bytes | None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(path), "-frames:v", "1", "-vf", "scale=320:-2", tmp],
            check=False,
        )
        data = Path(tmp).read_bytes() if Path(tmp).exists() else b""
        return data or None
    finally:
        Path(tmp).unlink(missing_ok=True)


def _build(items: list[Stored], body: IndexIn, library_sig: set[tuple[str, str]]) -> None:
    try:
        embedder = make_embedder(body)
        needs_asr = any(item.modality in {"audio", "video"} for item in items)
        transcriber = None
        if needs_asr:
            _log("Lade Whisper …")
            transcriber = _transcribers.get(body.whisper_id)
            if transcriber is None:
                transcriber = Transcriber(body.whisper_id)
                _transcribers[body.whisper_id] = transcriber
        captioner = None
        if body.use_captions and any(item.modality == "video" for item in items):
            _log("Lade Bildunterschriften …")
            caption_id = "Salesforce/blip-image-captioning-base"
            captioner = _captioners.get(caption_id)
            if captioner is None:
                captioner = Captioner(caption_id)
                _captioners[caption_id] = captioner

        index = VectorIndex(dim=embedder.dim)
        media: dict[str, Stored] = {}
        for i, item in enumerate(items):
            _log(f"{item.name}: extrahiere")
            with lock:
                state.progress = i / max(1, len(items))
            data = item.path.read_bytes()
            digest = hashlib.sha1(data).hexdigest()[:16]
            key = (
                digest, item.modality, body.chunk_words, body.window_s, body.whisper_id,
                body.use_captions, body.frame_every, body.max_frames,
            )
            with lock:
                cached = state.extract_cache.get(key)
            if cached is not None:
                chunks = cached
                _log(f"{item.name}: {len(chunks)} gespeicherte Abschnitte")
            else:
                try:
                    if item.modality == "pdf":
                        chunks = extract_pdf(item.path, item.name, chunk_words=body.chunk_words, overlap_words=body.chunk_words // 5)
                    elif item.modality == "audio":
                        chunks = extract_audio(item.path, item.name, transcriber, window_s=body.window_s)
                    else:
                        chunks = extract_video(
                            item.path, item.name, transcriber, captioner,
                            window_s=body.window_s, frame_every_s=float(body.frame_every), max_frames=body.max_frames,
                            progress=_log,
                        )
                except Exception as exc:
                    _log(f"{item.name}: Extraktion fehlgeschlagen ({exc})")
                    chunks = []
                with lock:
                    state.extract_cache[key] = chunks

            thumb = next((c.meta.get("thumb") for c in chunks if c.meta.get("thumb")), None)
            duration = None
            if item.modality in {"audio", "video"}:
                try:
                    duration = probe_duration(item.path)
                except Exception:
                    duration = None
                if thumb is None:
                    thumb = _poster(item.path)
            media[item.name] = Stored(item.name, item.modality, item.path, item.mime, duration, thumb)

            if chunks:
                _log(f"{item.name}: bette {len(chunks)} Abschnitte ein")
                vectors = embedder.embed_documents(
                    [c.text for c in chunks],
                    titles=[Path(c.source).stem for c in chunks],
                )
                index.add(chunks, vectors)
            else:
                _log(f"{item.name}: kein Text extrahiert")
            with lock:
                state.progress = (i + 1) / len(items)

        with lock:
            state.index = index
            state.media = media
            state.embedder = embedder
            state.settings = body.model_dump()
            state.progress = 1
            state.stale = {(item.name, str(item.path)) for item in state.library.values()} != library_sig
            if len(index) == 0:
                state.status = "error"
                state.message = "Kein Text konnte indexiert werden."
            else:
                state.status = "ready"
                state.message = f"{len(index)} Abschnitte aus {len(items)} Dateien indexiert."
                state.log.append(state.message)
    except Exception as exc:
        with lock:
            state.status = "error"
            state.message = f"Index fehlgeschlagen: {exc}"
            state.log.append(state.message)


def _public_files() -> list[dict]:
    return [
        {"name": item.name, "modality": item.modality, "mime": item.mime}
        for item in state.library.values()
    ]


def _status() -> dict:
    with lock:
        indexed = len(state.index) if state.index is not None else 0
        return {
            "state": state.status,
            "message": state.message,
            "progress": state.progress,
            "log": list(state.log[-12:]),
            "files": _public_files(),
            "ready": state.status == "ready" and indexed > 0,
            "stale": state.stale,
            "chunks": indexed,
            "settings": state.settings,
            "hf_token": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
        }


app = FastAPI(title="Multimodale Suche")


@app.get("/api/config")
def config():
    return {
        "backends": list(BACKENDS),
        "variants": list(GGUF_MODELS),
        "dims": list(MRL_DIMS),
        "whisper": list(WHISPER_MODELS),
        "max_per_type": MAX_PER_TYPE,
        "hf_token": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
    }


@app.get("/api/status")
def status():
    return _status()


@app.get("/api/beispiel")
def beispiel():
    return BEISPIEL


@app.post("/api/files")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for upload in files:
        name = Path(upload.filename or "datei").name
        mime = upload.content_type or ""
        modality = _modality(name, mime)
        if modality is None:
            raise HTTPException(400, f"{name}: dieser Dateityp wird nicht unterstützt.")
        data = await upload.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{name} ist größer als 200 MB.")
        with lock:
            replacing = name in state.library
            if not replacing and _counts().get(modality, 0) >= MAX_PER_TYPE:
                label = {"pdf": "PDFs", "audio": "Audiodateien", "video": "Videos"}[modality]
                raise HTTPException(400, f"Höchstens {MAX_PER_TYPE} {label}.")
            UPLOAD_ROOT.mkdir(exist_ok=True)
            digest = hashlib.sha1(data).hexdigest()[:16]
            path = UPLOAD_ROOT / f"{digest}_{name}"
            if not path.exists():
                path.write_bytes(data)
            state.library[name] = Stored(name, modality, path, mime)
            state.stale = state.index is not None
        saved.append(name)
    return {"ok": True, "saved": saved, "files": _public_files()}


@app.delete("/api/files/{name:path}")
def delete_file(name: str):
    with lock:
        if name not in state.library:
            raise HTTPException(404, "Datei nicht gefunden.")
        state.library.pop(name)
        state.stale = state.index is not None
    return {"ok": True, "files": _public_files()}


@app.post("/api/clear")
def clear_index():
    with lock:
        state.index = None
        state.media = {}
        state.embedder = None
        state.settings = None
        state.status = "idle"
        state.message = ""
        state.progress = 0
        state.log = []
        state.stale = False
    return _status()


@app.post("/api/index")
def start_index(body: IndexIn):
    body = _check_index(body)
    with lock:
        if state.status == "running":
            raise HTTPException(409, "Der Index wird bereits gebaut.")
        if not state.library:
            raise HTTPException(400, "Keine Dateien zum Indexieren.")
        items = list(state.library.values())
        library_sig = {(item.name, str(item.path)) for item in items}
        state.status = "running"
        state.progress = 0
        state.log = []
        state.message = "Index wird gebaut …"
        state.stale = False
    threading.Thread(target=_build, args=(items, body, library_sig), daemon=True).start()
    return {"ok": True}


@app.post("/api/search")
def search(body: SearchIn):
    query = clean_query(body.query)
    if not query:
        raise HTTPException(400, "Bitte eine Frage eingeben.")
    if len(query) > 2000:
        raise HTTPException(400, "Die Frage ist zu lang.")
    mods = set(body.modalities)
    if mods and not mods <= {"pdf", "audio", "video"}:
        raise HTTPException(400, "Unbekannte Modalität.")
    with lock:
        index = state.index
        embedder = state.embedder
        media = dict(state.media)
        stale = state.stale
        ready = state.status == "ready" and index is not None and len(index) > 0
    if not ready or embedder is None or index is None:
        raise HTTPException(400, "Zuerst Quellen hochladen und den Index bauen.")
    qvec = embedder.embed_query(query)
    ranked = best_per_source(index, qvec, mods or None)
    durations = {name: item.duration for name, item in media.items()}
    thumbs = {name for name, item in media.items() if item.thumb}
    view = present(
        query,
        ranked,
        list(index.chunks),
        durations=durations,
        thumbs=thumbs,
        media=set(media),
    )
    view["stale"] = stale
    view["fixture"] = False
    return view


@app.get("/api/media/{name:path}")
def media(name: str):
    item = state.media.get(name)
    if item is None or not item.path.exists():
        raise HTTPException(404, "Datei nicht gefunden.")
    return FileResponse(item.path, media_type=item.mime or "application/octet-stream", filename=item.name)


@app.get("/api/thumb/{name:path}")
def thumb(name: str):
    item = state.media.get(name)
    if item is None or not item.thumb:
        raise HTTPException(404, "Kein Vorschaubild.")
    return Response(content=item.thumb, media_type="image/jpeg")


@app.get("/")
def index_page(beispiel: int = 0):
    html = (WEB / "index.html").read_text(encoding="utf-8")
    if beispiel:
        blob = json.dumps(BEISPIEL, ensure_ascii=False).replace("<", "\\u003c")
        html = html.replace(
            '<script src="/app.js?v=3"></script>',
            f'<script>window.__BEISPIEL = {blob};</script>\n  <script src="/app.js?v=3"></script>',
        )
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/styles.css")
def styles():
    return FileResponse(WEB / "styles.css", media_type="text/css", headers={"Cache-Control": "no-cache"})


@app.get("/app.js")
def script():
    return FileResponse(WEB / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/assets/{name}")
def asset(name: str):
    path = (WEB / "assets" / name).resolve()
    if not str(path).startswith(str((WEB / "assets").resolve())) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
