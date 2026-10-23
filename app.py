"""Streamlit demo: multimodal retrieval (PDF / audio / video) with EmbeddingGemma.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from mmrag import (
    GGUF_MODELS, HF_MODEL_ID, MRL_DIMS, Captioner, GGUFEmbedder, HFEmbedder,
    Transcriber, VectorIndex, extract_audio, extract_pdf, extract_video,
)
from mmrag.extractors import Chunk
from mmrag.index import Hit

MAX_PER_TYPE = 6
PDF_TYPES = ["pdf"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
MODALITY_LABEL = {"pdf": "PDF", "audio": "Audio", "video": "Video"}
KIND_LABEL = {"text": "page text", "speech": "transcript", "visual": "keyframe caption"}
DEMO_DIR = Path(__file__).resolve().parent / "demo" / "gleisbau"
MIME = {"pdf": "application/pdf", "audio": "audio/wav", "video": "video/mp4"}

st.set_page_config(page_title="Gleisbau-Demo · EmbeddingGemma", layout="wide")


def load_manifest() -> dict:
    path = DEMO_DIR / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def use_prompt(prompt: str) -> None:
    st.session_state.query = prompt
    st.session_state.evidence_open = False
    st.session_state.pop("evidence_pick_box", None)


def pdf_page_image(data: bytes, page: int) -> bytes:
    """Rasterise one PDF page. The optional streamlit-pdf extra is not required."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(data)
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "140", "-f", str(page), "-l", str(page), str(src), str(prefix)],
            check=True, capture_output=True,
        )
        pngs = sorted(Path(tmp).glob("page*.png"))
        if not pngs:
            raise RuntimeError("pdftoppm produced no page image")
        return pngs[0].read_bytes()


def file_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def save_upload(data: bytes, name: str) -> Path:
    root = Path(tempfile.gettempdir()) / "mmrag_uploads"
    root.mkdir(exist_ok=True)
    path = root / f"{file_digest(data)}_{Path(name).name}"
    if not path.exists():
        path.write_bytes(data)
    return path


# ----------------------------------------------------------------- cached models


@st.cache_resource(show_spinner="Loading EmbeddingGemma ...")
def load_embedder(backend: str, variant: str, dim: int):
    if backend == "GGUF (llama.cpp)":
        return GGUFEmbedder(variant=variant, dim=dim)
    return HFEmbedder(model_id=HF_MODEL_ID, dim=dim)


@st.cache_resource(show_spinner="Loading Whisper ...")
def load_transcriber(model_id: str):
    return Transcriber(model_id)


@st.cache_resource(show_spinner="Loading BLIP captioner ...")
def load_captioner(model_id: str):
    return Captioner(model_id)


# ----------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Models")
    backend = st.radio(
        "Embedding backend",
        ["GGUF (llama.cpp)", "Transformers (HF)"],
        help="GGUF runs a quantised EmbeddingGemma via llama.cpp. "
        f"Transformers loads `{HF_MODEL_ID}` (gated, needs HF_TOKEN).",
    )
    variant = st.selectbox("GGUF variant", list(GGUF_MODELS), disabled=backend != "GGUF (llama.cpp)")
    dim = st.select_slider("Embedding dimension (Matryoshka)", options=list(reversed(MRL_DIMS)), value=768)
    if backend == "Transformers (HF)" and not (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")):
        st.warning("No HF_TOKEN found; the gated google/embeddinggemma-300m download will fail.")

    whisper_id = st.selectbox(
        "Speech-to-text (Whisper)",
        ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
        index=1,
    )
    asr_language = st.selectbox(
        "Speech language",
        ["auto", "german", "english"],
        help="The Gleisbau demo is transcribed in German even when this stays on auto.",
    )
    use_captions = st.toggle("Caption video keyframes (BLIP)", value=True)
    caption_id = "Salesforce/blip-image-captioning-base"

    st.header("Chunking")
    chunk_words = st.slider("PDF chunk size (words)", 60, 400, 180, 20)
    window_s = st.slider("Transcript window (seconds)", 10, 120, 30, 5)
    frame_every = st.slider("Keyframe every (seconds)", 2, 60, 10, 1)
    max_frames = st.slider("Max keyframes per video", 2, 30, 12, 1)

    if st.button("Clear index", use_container_width=True):
        for key in ("index", "media", "index_sig", "evidence_open", "corpus_label"):
            st.session_state.pop(key, None)
        st.session_state.pop("evidence_pick_box", None)
        st.rerun()


manifest = load_manifest()
documents = {doc["file"]: doc for doc in manifest.get("documents", [])}
cases = manifest.get("cases", [])
settings_sig = (backend, variant, dim, whisper_id, use_captions, chunk_words, window_s, frame_every, max_frames, asr_language)
if "extract_cache" not in st.session_state:
    st.session_state.extract_cache = {}


def build_index(
    items: list[tuple[str, str, bytes, str]],
    *,
    asr_language: str | None,
    index_sig: tuple,
    label: str,
    caption: bool | None = None,
    speech_window: float | None = None,
) -> None:
    """Index ``items`` of (filename, modality, bytes, mime)."""
    embedder = load_embedder(backend, variant, dim)
    do_caption = use_captions if caption is None else caption
    used_window = window_s if speech_window is None else speech_window
    needs_asr = any(modality in ("audio", "video") for _, modality, _, _ in items)
    needs_caption = do_caption and any(modality == "video" for _, modality, _, _ in items)
    transcriber = load_transcriber(whisper_id) if needs_asr else None
    captioner = load_captioner(caption_id) if needs_caption else None

    index = VectorIndex(dim=dim)
    media: dict[str, dict] = {}
    cache = st.session_state.extract_cache
    with st.status("Indexing ...", expanded=True) as status:
        progress = st.progress(0.0)
        for i, (name, modality, data, mime) in enumerate(items):
            path = save_upload(data, name)
            media[name] = {"modality": modality, "bytes": data, "mime": mime}
            key = (file_digest(data), modality, chunk_words, used_window, whisper_id, do_caption, frame_every, max_frames, asr_language or "auto")
            if key in cache:
                chunks = cache[key]
                st.write(f"{name}: reusing {len(chunks)} cached chunks")
            else:
                st.write(f"{name}: extracting ({MODALITY_LABEL[modality]})")
                try:
                    if modality == "pdf":
                        chunks = extract_pdf(path, name, chunk_words=chunk_words, overlap_words=chunk_words // 5)
                    elif modality == "audio":
                        chunks = extract_audio(path, name, transcriber, window_s=used_window, language=asr_language)
                    else:
                        chunks = extract_video(
                            path, name, transcriber, captioner,
                            window_s=used_window, frame_every_s=frame_every, max_frames=max_frames,
                            progress=st.write, language=asr_language,
                        )
                except Exception as exc:  # keep indexing the other files
                    st.error(f"{name}: extraction failed: {exc}")
                    chunks = []
                else:
                    cache[key] = chunks
            if chunks:
                st.write(f"{name}: embedding {len(chunks)} chunks")
                vecs = embedder.embed_documents([c.text for c in chunks], titles=[Path(name).stem for c in chunks])
                index.add(chunks, vecs)
            else:
                st.warning(f"{name}: no text could be extracted")
            progress.progress((i + 1) / len(items))
        status.update(label=f"Indexed {len(index)} chunks from {len(items)} files with {embedder.name}", state="complete")
    st.session_state.index = index
    st.session_state.media = media
    st.session_state.index_sig = index_sig
    st.session_state.corpus_label = label


# ------------------------------------------------------------------------ header

st.title("Multimodal retrieval with EmbeddingGemma")
st.caption(
    "PDFs, audio and video are turned into text, embedded with EmbeddingGemma into one vector space, "
    "and opened again at the original page or timestamp."
)

st.subheader("Gleisbau-Demo")
if not documents:
    st.error("Demo files are missing. Run `python scripts/make_gleisbau_demo.py`.")
else:
    st.caption(manifest.get("disclaimer", ""))
    case_cols = st.columns(4)
    for col, case in zip(case_cols, cases):
        with col:
            st.markdown(f"**{case['id']} · {case['name']}**")
            st.caption(case["blurb"])
    demo_sig = ("demo", backend, variant, dim, whisper_id, chunk_words)
    if st.button("Load Gleisbau demo", type="primary"):
        items = []
        for doc in manifest["documents"]:
            data = (DEMO_DIR / doc["file"]).read_bytes()
            items.append((doc["file"], doc["modality"], data, MIME[doc["modality"]]))
        build_index(
            items, asr_language="german", index_sig=demo_sig, label="Gleisbau-Demo",
            caption=False, speech_window=90,
        )
    stored = st.session_state.get("index_sig")
    if stored and stored[0] == "demo" and stored != demo_sig:
        st.info("Settings changed since the demo index was built. Click **Load Gleisbau demo** again.")

st.divider()
col_pdf, col_audio, col_video = st.columns(3)
with col_pdf:
    pdfs = st.file_uploader("PDFs", type=PDF_TYPES, accept_multiple_files=True)
with col_audio:
    audios = st.file_uploader("Audio", type=AUDIO_TYPES, accept_multiple_files=True)
with col_video:
    videos = st.file_uploader("Videos", type=VIDEO_TYPES, accept_multiple_files=True)

too_many = [name for name, files in (("PDFs", pdfs), ("audio files", audios), ("videos", videos)) if len(files or []) > MAX_PER_TYPE]
if too_many:
    st.error(f"Please upload at most {MAX_PER_TYPE} {', '.join(too_many)}. Only the first {MAX_PER_TYPE} will be used.")
pdfs, audios, videos = (pdfs or [])[:MAX_PER_TYPE], (audios or [])[:MAX_PER_TYPE], (videos or [])[:MAX_PER_TYPE]

uploads = [(f, "pdf") for f in pdfs] + [(f, "audio") for f in audios] + [(f, "video") for f in videos]
upload_sig = tuple(sorted((f.name, f.size) for f, _ in uploads))
upload_index_sig = ("upload", settings_sig, upload_sig)

build = st.button("Build index", type="secondary", disabled=not uploads)
stored = st.session_state.get("index_sig")
if stored and stored[0] == "upload" and stored != upload_index_sig and uploads:
    st.info("Uploads or settings changed since the last build. Click **Build index** to refresh.")

if build:
    items = [(up.name, modality, up.getvalue(), up.type or MIME[modality]) for up, modality in uploads]
    build_index(items, asr_language=None if asr_language == "auto" else asr_language, index_sig=upload_index_sig, label="Upload")

# ------------------------------------------------------------------------ search

def render_evidence(hits: list[Hit], media: dict, documents: dict, cases: list[dict]) -> None:
    cases_by_id = {case["id"]: case for case in cases}
    with st.container(border=True):
        head, close = st.columns([8, 1])
        head.markdown("### Evidence")
        head.caption(
            f"{len(hits)} Originalstellen, die diese Antwort verwendet hat. "
            "Jede Vorschau öffnet die Datei auf der gefundenen Seite oder am Zeitstempel."
        )
        if close.button("Close", key="close_evidence"):
            st.session_state.evidence_open = False
            st.rerun()
        for rank, hit in enumerate(hits, start=1):
            if rank > 1:
                st.divider()
            show_hit(hit, media.get(hit.chunk.source), documents.get(hit.chunk.source, {}), cases_by_id, rank)


def show_hit(hit: Hit, media: dict | None, doc: dict, cases_by_id: dict, rank: int) -> None:
    chunk: Chunk = hit.chunk
    case = cases_by_id.get(doc.get("case", ""))
    if case:
        st.markdown(f"**Use Case {case['id']} — {case['name']}**")
    if doc:
        st.markdown(
            f"**{doc.get('title', chunk.source)}**  ·  {doc.get('doc_id', '')}  ·  "
            f"Revision {doc.get('revision', '—')}  ·  Status **{doc.get('status', '—')}**  ·  "
            f"{doc.get('variant', '—')}  ·  Stand {doc.get('issued', '—')}"
        )
    else:
        st.markdown(f"**{chunk.source}**")
        st.caption("Für diese Datei ist kein Freigabestatus hinterlegt. Gezeigt wird nur die indexierte Originalstelle.")

    quote, viewer = st.columns([2, 3], vertical_alignment="top")
    with quote:
        st.markdown(f"**Abgerufene Stelle** · {chunk.locator()} · _{KIND_LABEL[chunk.kind]}_ · score {hit.score:.3f}")
        st.markdown(
            "<div style='border-left:4px solid #0c3054;padding:0.7rem 0.9rem;background:#f4f7fb;white-space:pre-wrap'>"
            + html.escape(chunk.text.strip())
            + "</div>",
            unsafe_allow_html=True,
        )
    with viewer:
        if media is None:
            st.warning("Originaldatei ist im Index nicht mehr vorhanden.")
            return
        if chunk.modality == "pdf":
            page = int(chunk.page or 1)
            try:
                st.image(pdf_page_image(media["bytes"], page), caption=f"{chunk.source} · Seite {page}", width="stretch")
            except Exception as exc:
                st.warning(f"Seitenvorschau nicht möglich: {exc}")
            st.download_button("Download PDF", media["bytes"], file_name=chunk.source, mime="application/pdf", key=f"dl-{rank}-{chunk.source}")
        else:
            start = int(chunk.start or 0)
            if chunk.modality == "audio":
                st.audio(media["bytes"], format=media.get("mime") or "audio/wav", start_time=start)
            else:
                if chunk.meta.get("thumb"):
                    st.image(chunk.meta["thumb"], caption=f"frame @ {chunk.locator()}")
                st.video(media["bytes"], start_time=start)
            st.caption(f"Wiedergabe ab {chunk.locator() or '00:00'}.")


index: VectorIndex | None = st.session_state.get("index")
stored_sig = st.session_state.get("index_sig")
if not (isinstance(stored_sig, tuple) and len(stored_sig) >= 2 and stored_sig[0] in {"demo", "upload"}):
    index = None
if index is None or len(index) == 0:
    st.stop()

with st.expander(f"Index contents: {len(index)} chunks", expanded=False):
    rows = {}
    for chunk in index.chunks:
        row = rows.setdefault(chunk.source, {"file": chunk.source, "modality": MODALITY_LABEL[chunk.modality], "chunks": 0, "transcript": 0, "captions": 0, "pages": set()})
        row["chunks"] += 1
        row["transcript"] += chunk.kind == "speech"
        row["captions"] += chunk.kind == "visual"
        if chunk.page:
            row["pages"].add(chunk.page)
    st.dataframe(
        [{**row, "pages": len(row["pages"]) or None} for row in rows.values()],
        width="stretch", hide_index=True,
    )

st.subheader("Ask")
if cases:
    prompt_cols = st.columns(4)
    for col, case in zip(prompt_cols, cases):
        col.button(
            f"{case['id']} · {case['name']}",
            key=f"case-{case['id']}",
            help=case["prompt"],
            on_click=use_prompt,
            args=(case["prompt"],),
            use_container_width=True,
        )
query = st.text_input(
    "Prompt",
    key="query",
    placeholder="Die Hydraulik an Maschine GB-204 verliert Druck. Welche Prüfhinweise stehen in der Wartungsanleitung?",
)
k_col, m_col = st.columns([1, 2])
with k_col:
    top_k = st.number_input("Top-k", 1, 50, 4)
with m_col:
    selected = st.multiselect(
        "Modalities", ["pdf", "audio", "video"], default=["pdf", "audio", "video"],
        format_func=MODALITY_LABEL.get,
    )

if not query:
    st.stop()

evidence_scope = (query, int(top_k), tuple(selected))
if st.session_state.get("evidence_scope") != evidence_scope:
    st.session_state.evidence_open = False
    st.session_state.pop("evidence_pick_box", None)
    st.session_state.evidence_scope = evidence_scope

sig = st.session_state.index_sig
if sig[0] == "demo":
    embedder = load_embedder(sig[1], sig[2], sig[3])
else:
    embedder = load_embedder(*sig[1][:3])
qvec = embedder.embed_query(query)
mods = set(selected) or None
ranking = index.rank_sources(qvec, mods)
hits = index.search(qvec, top_k=int(top_k), modalities=mods)
if hits:
    # Keep the passages that actually support the answer. A much weaker hit
    # is usually a neighbour page, not evidence the question used.
    cutoff = hits[0].score * 0.75
    hits = [hit for hit in hits if hit.score >= cutoff]

if ranking:
    st.markdown("**Best matching files**")
    cols = st.columns(min(len(ranking), 6))
    for col, (src, modality, score) in zip(cols, ranking):
        doc = documents.get(src, {})
        col.metric(doc.get("title", src), f"{score:.3f}", help=f"{MODALITY_LABEL[modality]} · {src}")

st.markdown("**Answer from the retrieved passages**")
st.caption(
    "Nur Zitate aus den indexierten Unterlagen. Variante, Revision und Freigabestatus stehen am Dokument. "
    "Es wird keine eigenständige Reparaturfreigabe erteilt."
)
if not hits:
    st.warning("Keine Stelle in den gewählten Modalitäten gefunden.")
    st.stop()

for rank, hit in enumerate(hits, start=1):
    chunk = hit.chunk
    doc = documents.get(chunk.source, {})
    title = doc.get("title", chunk.source)
    st.markdown(f"**#{rank} · {MODALITY_LABEL[chunk.modality]} · {title}** — {chunk.locator()} · score {hit.score:.3f}")
    preview = chunk.text.strip()
    if len(preview) > 420:
        preview = preview[:420].rsplit(" ", 1)[0] + " …"
    st.markdown(f"> {preview}")

if st.button("Show the Evidence", type="primary"):
    st.session_state.evidence_open = True

if st.session_state.get("evidence_open"):
    render_evidence(hits, st.session_state.media, documents, cases)
