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
    GGUF_MODELS, HF_MODEL_ID, MRL_DIMS, MANUFACTURING_QUERY, PRESETS,
    Captioner, GGUFEmbedder, HFEmbedder, LexicalEmbedder, Transcriber, VectorIndex,
    collect_events, demo_bundle, extract_audio, extract_pdf, extract_video,
    parse_constraints, render_timeline, run_benchmark, solve,
)
from mmrag.extractors import probe_duration

MAX_PER_TYPE = 3
PDF_TYPES = ["pdf"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
MODALITY_LABEL = {"pdf": "PDF", "audio": "Audio", "video": "Video"}
KIND_LABEL = {"text": "page text", "speech": "transcript", "visual": "keyframe caption"}

st.set_page_config(page_title="EmbeddingGemma Multimodal RAG", layout="wide")

# Demo / preset state has to be applied before the widgets that read it.
st.session_state.setdefault("query", "")
st.session_state.setdefault("temporal_query", "")
st.session_state.setdefault("dim", 768)
want_demo = st.session_state.pop("load_demo", False)
pending_temporal = st.session_state.pop("pending_temporal", None)
if want_demo:
    st.session_state.backend = "Lexical (offline)"
    st.session_state.dim = 256
    st.session_state.query = "abnormal vibration reported on the line"
    st.session_state.temporal_query = MANUFACTURING_QUERY
if pending_temporal is not None:
    st.session_state.temporal_query = pending_temporal


# ----------------------------------------------------------------- cached models


@st.cache_resource(show_spinner="Loading EmbeddingGemma ...")
def load_embedder(backend: str, variant: str, dim: int):
    if backend == "Lexical (offline)":
        return LexicalEmbedder(dim=dim)
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


def install_demo(dim: int) -> None:
    chunks, events, durations, titles = demo_bundle()
    embedder = LexicalEmbedder(dim=dim)
    index = VectorIndex(dim=dim)
    if chunks:
        vecs = embedder.embed_documents([c.text for c in chunks], titles=[Path(c.source).stem for c in chunks])
        index.add(chunks, vecs)
    st.session_state.index = index
    st.session_state.events = events
    st.session_state.durations = durations
    st.session_state.titles = titles
    st.session_state.media = {}
    st.session_state.demo = True


# ----------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Models")
    backend = st.radio(
        "Embedding backend",
        ["GGUF (llama.cpp)", "Transformers (HF)", "Lexical (offline)"],
        key="backend",
        help="GGUF runs a quantised EmbeddingGemma via llama.cpp. "
        f"Transformers loads `{HF_MODEL_ID}` (gated, needs HF_TOKEN). "
        "Lexical is a hashed bag-of-words for the manufacturing demo, with no download.",
    )
    variant = st.selectbox("GGUF variant", list(GGUF_MODELS), disabled=backend != "GGUF (llama.cpp)")
    dim = st.select_slider(
        "Embedding dimension (Matryoshka)", options=list(reversed(MRL_DIMS)), key="dim",
    )
    if backend == "Transformers (HF)" and not (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")):
        st.warning("No HF_TOKEN found; the gated google/embeddinggemma-300m download will fail.")

    whisper_id = st.selectbox(
        "Speech-to-text (Whisper)",
        ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
        index=1,
    )
    use_captions = st.toggle("Caption video keyframes (BLIP)", value=True)
    caption_id = "Salesforce/blip-image-captioning-base"
    detect_signals = st.toggle(
        "Localize short events (motion and loudness)",
        value=True,
        help="Finds vibration and unusual-sound spans from frame difference and RMS, "
        "instead of leaving them inside a transcript window.",
    )

    st.header("Chunking")
    chunk_words = st.slider("PDF chunk size (words)", 60, 400, 180, 20)
    window_s = st.slider("Transcript window (seconds)", 10, 120, 30, 5)
    frame_every = st.slider("Keyframe every (seconds)", 2, 60, 10, 1)
    max_frames = st.slider("Max keyframes per video", 2, 30, 12, 1)

    if st.button("Clear index", width="stretch"):
        for k in ("index", "media", "index_sig", "events", "durations", "titles", "demo"):
            st.session_state.pop(k, None)
        st.rerun()

if want_demo:
    install_demo(int(dim))

# ------------------------------------------------------------------------ header

st.title("Multimodal retrieval with EmbeddingGemma")
st.caption(
    "Upload up to 3 PDFs, 3 audio files and 3 videos. Text, transcripts and keyframe "
    "captions share one EmbeddingGemma space. Temporal constraints then test order and "
    "timing across visual, acoustic and spoken events — for example that a vibration "
    "begins within 15 seconds before the operator report, with an unusual sound during it."
)

demo_col, eval_col = st.columns(2)
if demo_col.button("Load manufacturing example", type="primary"):
    st.session_state.load_demo = True
    st.rerun()
if eval_col.button("Run temporal evaluation"):
    st.session_state.show_eval = True

if st.session_state.get("show_eval"):
    report = run_benchmark()
    st.subheader("Manufacturing benchmark")
    st.dataframe(report.as_records(), width="stretch", hide_index=True)
    st.markdown(
        f"Query: `{report.query}`\n\n"
        f"Temporal answer: **{report.temporal_answer}**. "
        f"Semantic answer: **{report.semantic_answer}**. "
        f"Gold: **{report.gold_answer}**."
    )
    st.caption(report.notes)

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
settings_sig = (
    backend, variant, dim, whisper_id, use_captions, chunk_words, window_s, frame_every, max_frames, detect_signals,
)
upload_sig = tuple(sorted((f.name, f.size) for f, _ in uploads))
index_sig = (settings_sig, upload_sig)
if want_demo:
    st.session_state.index_sig = (settings_sig, ("manufacturing-demo",))
    index_sig = st.session_state.index_sig

if "extract_cache" not in st.session_state:
    st.session_state.extract_cache = {}

build = st.button("Build index", type="primary", disabled=not uploads)
if st.session_state.get("index_sig") not in (None, index_sig) and uploads:
    st.info("Uploads or settings changed since the last build. Click **Build index** to refresh.")

# --------------------------------------------------------------------- indexing

if build:
    embedder = load_embedder(backend, variant, dim)
    event_embedder = None if backend == "Lexical (offline)" else embedder
    needs_asr = bool(audios or videos)
    transcriber = load_transcriber(whisper_id) if needs_asr else None
    captioner = load_captioner(caption_id) if (videos and use_captions) else None

    index = VectorIndex(dim=dim)
    media: dict[str, dict] = {}
    events = []
    durations: dict[str, float] = {}
    titles: dict[str, str] = {}
    cache = st.session_state.extract_cache

    with st.status("Indexing ...", expanded=True) as status:
        progress = st.progress(0.0)
        for i, (up, modality) in enumerate(uploads):
            data = up.getvalue()
            path = save_upload(data, up.name)
            media[up.name] = {"modality": modality, "bytes": data, "mime": up.type}
            titles[up.name] = up.name
            key = (
                file_digest(data), modality, chunk_words, window_s, whisper_id,
                use_captions, frame_every, max_frames,
            )

            if key in cache:
                chunks, segments = cache[key]
                st.write(f"{up.name}: reusing {len(chunks)} cached chunks")
            else:
                st.write(f"{up.name}: extracting ({MODALITY_LABEL[modality]})")
                segments = []
                try:
                    if modality == "pdf":
                        chunks = extract_pdf(path, up.name, chunk_words=chunk_words, overlap_words=chunk_words // 5)
                    elif modality == "audio":
                        chunks = extract_audio(
                            path, up.name, transcriber, window_s=window_s, speech_segments=segments,
                        )
                    else:
                        chunks = extract_video(
                            path, up.name, transcriber, captioner,
                            window_s=window_s, frame_every_s=frame_every, max_frames=max_frames,
                            progress=st.write, speech_segments=segments,
                        )
                except Exception as exc:  # keep indexing the other files
                    st.error(f"{up.name}: extraction failed: {exc}")
                    chunks = []
                cache[key] = (chunks, segments)

            if modality != "pdf":
                try:
                    durations[up.name] = probe_duration(path)
                except Exception:
                    durations[up.name] = 0.0
            file_events = collect_events(
                path, up.name, modality, chunks, segments,
                embedder=event_embedder, enable_signals=detect_signals,
            )
            events.extend(file_events)
            if file_events:
                st.write(f"{up.name}: {len(file_events)} localized events")

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
    st.session_state.events = events
    st.session_state.durations = durations
    st.session_state.titles = titles
    st.session_state.demo = False
    st.session_state.index_sig = index_sig

# ------------------------------------------------------------------------ search

index: VectorIndex | None = st.session_state.get("index")
if index is None or len(index) == 0:
    st.stop()

events = st.session_state.get("events") or []
durations = st.session_state.get("durations") or {}
titles = st.session_state.get("titles") or {}

with st.expander(f"Index contents: {len(index)} chunks · {len(events)} events", expanded=False):
    rows = {}
    for c in index.chunks:
        r = rows.setdefault(c.source, {"file": c.source, "modality": MODALITY_LABEL[c.modality], "chunks": 0, "transcript": 0, "captions": 0, "pages": set(), "events": 0})
        r["chunks"] += 1
        r["transcript"] += c.kind == "speech"
        r["captions"] += c.kind == "visual"
        if c.page:
            r["pages"].add(c.page)
    for event in events:
        r = rows.setdefault(event.source, {"file": event.source, "modality": MODALITY_LABEL.get(event.modality, event.modality), "chunks": 0, "transcript": 0, "captions": 0, "pages": set(), "events": 0})
        r["events"] += 1
    st.dataframe(
        [{**r, "pages": len(r["pages"]) or None} for r in rows.values()],
        width="stretch", hide_index=True,
    )

if events:
    st.subheader("Localized events")
    st.caption("Short spans from motion, loudness and transcript cues. Each bar is evidence with a start and an end.")
    by_source: dict[str, list] = {}
    for event in events:
        by_source.setdefault(event.source, []).append(event)
    for source, group in by_source.items():
        duration = durations.get(source) or max(event.end for event in group)
        st.markdown(render_timeline(group, duration, titles.get(source, source)), unsafe_allow_html=True)

st.subheader("Ask")
q_col, k_col, m_col = st.columns([4, 1, 2])
with q_col:
    query = st.text_input("Prompt", placeholder="e.g. Where does the speaker explain gradient descent?", key="query")
with k_col:
    top_k = st.number_input("Top-k", 1, 50, 6)
with m_col:
    selected = st.multiselect(
        "Modalities", ["pdf", "audio", "video"], default=["pdf", "audio", "video"],
        format_func=MODALITY_LABEL.get,
    )

if query:
    embedder = load_embedder(*st.session_state.index_sig[0][:3])
    qvec = embedder.embed_query(query)
    mods = set(selected) or None

    ranking = index.rank_sources(qvec, mods)
    if ranking:
        st.markdown("**Best matching files** — semantic similarity")
        cols = st.columns(min(len(ranking), 9))
        for col, (src, modality, score) in zip(cols, ranking):
            col.metric(f"{MODALITY_LABEL[modality]} · {src}", f"{score:.3f}")

    st.markdown("**Top passages** — semantic similarity")
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

st.subheader("Temporal constraints")
st.caption(
    "Retrieval by explicit timing, not only by similar words. "
    "`t_vibration < t_report` is the figure's ordering. The 15 second window is the test "
    "a semantic hit can still fail."
)
preset_cols = st.columns(len(PRESETS))
for col, (name, expression) in zip(preset_cols, PRESETS.items()):
    if col.button(name, key=f"preset-{name}", width="stretch"):
        st.session_state.pending_temporal = expression
        st.rerun()

temporal_query = st.text_input(
    "Constraint",
    key="temporal_query",
    placeholder="vibration within 15 seconds before the report, while an acoustic anomaly occurs during the vibration",
)
parsed = parse_constraints(temporal_query)
for error in parsed.errors:
    st.warning(error)
if parsed.constraints:
    st.markdown(" ".join(f"`{c.formula()}`" for c in parsed.constraints))
    solutions = solve(events, parsed.constraints)
    if not solutions:
        st.info("No events to test. Load the manufacturing example or index a recording.")
    passing = [s for s in solutions if s.holds]
    st.markdown(
        f"**{len(passing)}** of **{len({s.source for s in solutions})}** recordings satisfy every constraint."
    )
    for solution in solutions:
        with st.container(border=True):
            status = "holds" if solution.holds else "does not hold"
            st.markdown(f"**{titles.get(solution.source, solution.source)}** — {status}")
            for check in solution.checks:
                mark = "✓" if check.passed else "✗"
                st.write(f"{mark} {check.detail}")
            if solution.bindings:
                st.markdown("**Evidence**")
                for event in sorted(solution.bindings.values(), key=lambda e: e.start):
                    st.write(
                        f"{event.label} · {event.channel} · {event.locator()} · "
                        f"{event.origin} · score {event.score:.2f}"
                    )
                    st.caption(event.evidence)
