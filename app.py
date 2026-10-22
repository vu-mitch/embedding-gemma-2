"""Streamlit demo: multimodal retrieval (PDF / audio / video) with EmbeddingGemma.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from mmrag import (
    GGUF_MODELS, HF_MODEL_ID, MRL_DIMS, Captioner, GGUFEmbedder, HFEmbedder,
    Transcriber, VectorIndex, extract_audio, extract_pdf, extract_video,
)

MAX_PER_TYPE = 3
PDF_TYPES = ["pdf"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
MODALITY_LABEL = {"pdf": "PDF", "audio": "Audio", "video": "Video"}
KIND_LABEL = {"text": "page text", "speech": "transcript", "visual": "keyframe caption"}

st.set_page_config(page_title="EmbeddingGemma Multimodal RAG", layout="wide")


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


def file_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def save_upload(data: bytes, name: str) -> Path:
    root = Path(tempfile.gettempdir()) / "mmrag_uploads"
    root.mkdir(exist_ok=True)
    path = root / f"{file_digest(data)}_{Path(name).name}"
    if not path.exists():
        path.write_bytes(data)
    return path


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
    use_captions = st.toggle("Caption video keyframes (BLIP)", value=True)
    caption_id = "Salesforce/blip-image-captioning-base"

    st.header("Chunking")
    chunk_words = st.slider("PDF chunk size (words)", 60, 400, 180, 20)
    window_s = st.slider("Transcript window (seconds)", 10, 120, 30, 5)
    frame_every = st.slider("Keyframe every (seconds)", 2, 60, 10, 1)
    max_frames = st.slider("Max keyframes per video", 2, 30, 12, 1)

    if st.button("Clear index", use_container_width=True):
        for k in ("index", "media", "index_sig"):
            st.session_state.pop(k, None)
        st.rerun()

# ------------------------------------------------------------------------ header

st.title("Multimodal retrieval with EmbeddingGemma")
st.caption(
    "Upload up to 3 PDFs, 3 audio files and 3 videos. Each item is converted to text "
    "(PDF text, Whisper transcripts, BLIP keyframe captions), embedded with EmbeddingGemma "
    "into one vector space, and searched with your prompt."
)

col_pdf, col_audio, col_video = st.columns(3)
with col_pdf:
    pdfs = st.file_uploader("PDFs", type=PDF_TYPES, accept_multiple_files=True)
with col_audio:
    audios = st.file_uploader("Audio", type=AUDIO_TYPES, accept_multiple_files=True)
with col_video:
    videos = st.file_uploader("Videos", type=VIDEO_TYPES, accept_multiple_files=True)

too_many = [n for n, f in (("PDFs", pdfs), ("audio files", audios), ("videos", videos)) if len(f or []) > MAX_PER_TYPE]
if too_many:
    st.error(f"Please upload at most {MAX_PER_TYPE} {', '.join(too_many)}. Only the first {MAX_PER_TYPE} will be used.")
pdfs, audios, videos = (pdfs or [])[:MAX_PER_TYPE], (audios or [])[:MAX_PER_TYPE], (videos or [])[:MAX_PER_TYPE]

uploads = [(f, "pdf") for f in pdfs] + [(f, "audio") for f in audios] + [(f, "video") for f in videos]
settings_sig = (backend, variant, dim, whisper_id, use_captions, chunk_words, window_s, frame_every, max_frames)
upload_sig = tuple(sorted((f.name, f.size) for f, _ in uploads))
index_sig = (settings_sig, upload_sig)

if "extract_cache" not in st.session_state:
    st.session_state.extract_cache = {}

build = st.button("Build index", type="primary", disabled=not uploads)
if st.session_state.get("index_sig") not in (None, index_sig) and uploads:
    st.info("Uploads or settings changed since the last build. Click **Build index** to refresh.")

# --------------------------------------------------------------------- indexing

if build:
    embedder = load_embedder(backend, variant, dim)
    needs_asr = bool(audios or videos)
    transcriber = load_transcriber(whisper_id) if needs_asr else None
    captioner = load_captioner(caption_id) if (videos and use_captions) else None

    index = VectorIndex(dim=dim)
    media: dict[str, dict] = {}
    cache = st.session_state.extract_cache

    with st.status("Indexing ...", expanded=True) as status:
        progress = st.progress(0.0)
        for i, (up, modality) in enumerate(uploads):
            data = up.getvalue()
            path = save_upload(data, up.name)
            media[up.name] = {"modality": modality, "bytes": data, "mime": up.type}
            key = (file_digest(data), modality, chunk_words, window_s, whisper_id, use_captions, frame_every, max_frames)

            if key in cache:
                chunks = cache[key]
                st.write(f"{up.name}: reusing {len(chunks)} cached chunks")
            else:
                st.write(f"{up.name}: extracting ({MODALITY_LABEL[modality]})")
                try:
                    if modality == "pdf":
                        chunks = extract_pdf(path, up.name, chunk_words=chunk_words, overlap_words=chunk_words // 5)
                    elif modality == "audio":
                        chunks = extract_audio(path, up.name, transcriber, window_s=window_s)
                    else:
                        chunks = extract_video(
                            path, up.name, transcriber, captioner,
                            window_s=window_s, frame_every_s=frame_every, max_frames=max_frames,
                            progress=st.write,
                        )
                except Exception as exc:  # keep indexing the other files
                    st.error(f"{up.name}: extraction failed: {exc}")
                    chunks = []
                cache[key] = chunks

            if chunks:
                st.write(f"{up.name}: embedding {len(chunks)} chunks")
                vecs = embedder.embed_documents([c.text for c in chunks], titles=[Path(c.source).stem for c in chunks])
                index.add(chunks, vecs)
            else:
                st.warning(f"{up.name}: no text could be extracted")
            progress.progress((i + 1) / len(uploads))
        status.update(label=f"Indexed {len(index)} chunks from {len(uploads)} files with {embedder.name}", state="complete")

    st.session_state.index = index
    st.session_state.media = media
    st.session_state.index_sig = index_sig

# ------------------------------------------------------------------------ search

index: VectorIndex | None = st.session_state.get("index")
if index is None or len(index) == 0:
    st.stop()

with st.expander(f"Index contents: {len(index)} chunks", expanded=False):
    rows = {}
    for c in index.chunks:
        r = rows.setdefault(c.source, {"file": c.source, "modality": MODALITY_LABEL[c.modality], "chunks": 0, "transcript": 0, "captions": 0, "pages": set()})
        r["chunks"] += 1
        r["transcript"] += c.kind == "speech"
        r["captions"] += c.kind == "visual"
        if c.page:
            r["pages"].add(c.page)
    st.dataframe(
        [{**r, "pages": len(r["pages"]) or None} for r in rows.values()],
        use_container_width=True, hide_index=True,
    )

st.subheader("Ask")
q_col, k_col, m_col = st.columns([4, 1, 2])
with q_col:
    query = st.text_input("Prompt", placeholder="e.g. Where does the speaker explain gradient descent?")
with k_col:
    top_k = st.number_input("Top-k", 1, 50, 6)
with m_col:
    selected = st.multiselect(
        "Modalities", ["pdf", "audio", "video"], default=["pdf", "audio", "video"],
        format_func=MODALITY_LABEL.get,
    )

if not query:
    st.stop()

embedder = load_embedder(*st.session_state.index_sig[0][:3])
qvec = embedder.embed_query(query)
mods = set(selected) or None

ranking = index.rank_sources(qvec, mods)
if ranking:
    st.markdown("**Best matching files**")
    cols = st.columns(min(len(ranking), 9))
    for col, (src, modality, score) in zip(cols, ranking):
        col.metric(f"{MODALITY_LABEL[modality]} · {src}", f"{score:.3f}")

st.markdown("**Top passages**")
media = st.session_state.media
for rank, hit in enumerate(index.search(qvec, top_k=int(top_k), modalities=mods), start=1):
    c = hit.chunk
    with st.container(border=True):
        head, score_col = st.columns([5, 1])
        head.markdown(f"**#{rank} · {MODALITY_LABEL[c.modality]} · {c.source}** — {c.locator()} · _{KIND_LABEL[c.kind]}_")
        score_col.markdown(f"score **{hit.score:.3f}**")

        body, preview = st.columns([3, 2])
        body.write(c.text)
        m = media.get(c.source)
        if m is None:
            continue
        start = int(c.start or 0)
        with preview:
            if c.modality == "audio":
                st.audio(m["bytes"], format=m["mime"] or "audio/wav", start_time=start)
            elif c.modality == "video":
                if c.meta.get("thumb"):
                    st.image(c.meta["thumb"], caption=f"frame @ {c.locator()}")
                st.video(m["bytes"], start_time=start)
            else:
                st.download_button("Download PDF", m["bytes"], file_name=c.source, mime="application/pdf", key=f"dl{rank}")
