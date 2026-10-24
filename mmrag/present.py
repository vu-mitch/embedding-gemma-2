"""Shape retrieval hits into the result view: answer, evidence rows, top files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from .extractors import Chunk
from .index import VectorIndex

_STOP = frozenset(
    """
    der die das den dem des ein eine einer einem einen eines und oder aber mit von vom am im an auf
    für fur zu zum zur ist sind war wird werden nicht auch nur bei nach vor über ueber aus als wie
    was welche welcher welches gibt dazu stehen steht prüft prueft vier drei zwei fünf funf stellen
    stelle anleitung video ablauf relevant relevante kapitel seite maschine sollwert dokument
    transkript keyframe servicevideo text seiten abschnitt abschnitte datei dateien frage fragen
    the a an of to for in on with without from by at as is are was be this that these those
    und der die das
    """.split()
)

_PHRASE = re.compile(
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,})\s+([A-Z]{1,4}-?\d[\w-]*)"
)
_NOUN = re.compile(r"(?<![A-Za-zÄÖÜäöüß])([A-ZÄÖÜ][a-zäöüß]{4,})")
_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,4}-?\d[\w-]*)")
_CHAPTER = re.compile(r"(?i)kapitel\s+(\d+(?:\.\d+)*)")


@dataclass
class _Cand:
    text: str
    start: int
    end: int
    kind: str


def clean_query(query: str) -> str:
    text = (query or "").strip()
    if len(text) >= 2 and text[0] in "„\"'«" and text[-1] in "“\"'»":
        text = text[1:-1].strip()
    return text


def to_percent(score: float) -> int:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0
    if value != value or value <= 0:
        return 0
    if value > 1:
        value = 1.0
    return int(value * 100 + 0.5)


def fmt_duration(seconds: float | None) -> str | None:
    if seconds is None or seconds <= 0:
        return None
    total = int(round(float(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def fmt_span(start: float | None, end: float | None) -> str | None:
    if start is None:
        return None
    a = fmt_duration(start if start > 0 else 0.0) or "0:00"
    b = fmt_duration(end if end and end > 0 else start)
    if b is None or a == b:
        return a
    return f"{a}–{b}"


def pretty_stem(name: str) -> str:
    stem = name.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem.replace("_", " ").strip() or name


def icon_for(name: str, modality: str) -> str:
    lower = name.lower()
    if modality == "audio" or lower.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        return "audio"
    if modality == "video" or lower.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        return "video"
    if lower.endswith((".xlsx", ".xls", ".csv")):
        return "sheet"
    if any(key in lower for key in ("schema", "zeichnung", "plan", "diagramm")):
        return "image"
    if modality == "pdf" or lower.endswith(".pdf"):
        return "pdf"
    return "file"


def compose_answer(text: str, limit: int = 280) -> str:
    compact = " ".join((text or "").split())
    if not compact:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p.strip()]
    chosen: list[str] = []
    total = 0
    for part in parts:
        if chosen and total + 1 + len(part) > limit:
            break
        chosen.append(part)
        total += len(part)
        if len(chosen) >= 3 or (len(chosen) >= 2 and total >= 160):
            break
    answer = " ".join(chosen).strip()
    if len(answer) > limit:
        cut = answer[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;")
        answer = (cut or answer[:limit].rstrip()) + "…"
    return answer


def _list_parts(text: str) -> list[str]:
    parts: list[str] = []
    for sentence in re.split(r"[.!?]", text):
        bits = [bit.strip() for bit in re.split(r",|\bund\b", sentence) if bit.strip()]
        if len(bits) >= 3:
            parts.extend(bits)
    return parts


def chips_from(texts: list[str], limit: int = 4) -> list[str]:
    blob = "\n".join(t for t in texts if t)
    if not blob.strip():
        return []
    found: list[_Cand] = []
    for pattern, kind in ((_PHRASE, "phrase"), (_NOUN, "noun"), (_CODE, "code")):
        for match in pattern.finditer(blob):
            found.append(_Cand(match.group(0).strip(), match.start(), match.end(), kind))
    kept: list[_Cand] = []
    for cand in found:
        covered = any(
            other is not cand
            and other.start <= cand.start
            and other.end >= cand.end
            and (other.end - other.start) > (cand.end - cand.start)
            for other in found
        )
        if not covered:
            kept.append(cand)
    parts = _list_parts(blob)
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for cand in kept:
        key = cand.text.casefold()
        if key in seen:
            continue
        if cand.kind == "phrase":
            base = 5
        elif cand.kind == "code":
            base = 4
        else:
            if key in _STOP:
                continue
            length = len(cand.text)
            base = 3 if length >= 12 else 2 if length >= 8 else 1
        if any(key in part.casefold() for part in parts):
            base += 2
        if base < 2:
            continue
        seen.add(key)
        scored.append((base, cand.start, cand.text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    top = scored[:limit]
    top.sort(key=lambda item: item[1])
    return [text for _, _, text in top]


def _chapter(chunks: list[Chunk], source: str, page: int | None) -> str | None:
    fallback = None
    for chunk in chunks:
        if chunk.source != source:
            continue
        match = _CHAPTER.search(chunk.text or "")
        if not match:
            continue
        label = f"Kapitel {match.group(1)}"
        if page is not None and chunk.page == page:
            return label
        fallback = fallback or label
    return fallback


def subtitle_for(chunk: Chunk, chunks: list[Chunk]) -> str:
    if chunk.modality == "pdf":
        bits = []
        chapter = _chapter(chunks, chunk.source, chunk.page)
        if chapter:
            bits.append(chapter)
        if chunk.page:
            bits.append(f"Seite {chunk.page}")
        return ", ".join(bits) if bits else "Dokument"
    label = "Video" if chunk.modality == "video" else "Audio"
    span = fmt_span(chunk.start, chunk.end)
    return f"{label} · {span}" if span else label


def best_per_source(index: VectorIndex, query_vec, modalities: set[str] | None = None) -> list[tuple[Chunk, float]]:
    if index.vectors is None or not index.chunks:
        return []
    scores = index.vectors @ query_vec
    best: dict[str, tuple[int, float]] = {}
    for i, chunk in enumerate(index.chunks):
        if modalities is not None and chunk.modality not in modalities:
            continue
        score = float(scores[i])
        if chunk.source not in best or score > best[chunk.source][1]:
            best[chunk.source] = (i, score)
    ranked = [(index.chunks[i], score) for i, score in best.values()]
    ranked.sort(key=lambda item: -item[1])
    return ranked


def _pick_evidence(ranked: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    picked: list[tuple[Chunk, float]] = []
    seen: set[str] = set()
    for chunk, score in ranked:
        if chunk.modality in seen:
            continue
        seen.add(chunk.modality)
        picked.append((chunk, score))
    if len(picked) < 2:
        have = {chunk.source for chunk, _ in picked}
        for chunk, score in ranked:
            if chunk.source in have:
                continue
            picked.append((chunk, score))
            have.add(chunk.source)
            if len(picked) == 2:
                break
    picked.sort(key=lambda item: -item[1])
    return picked[:2]


def _answer_text(ranked: list[tuple[Chunk, float]]) -> str:
    for chunk, _score in ranked[:8]:
        if chunk.kind != "visual" and (chunk.text or "").strip():
            return compose_answer(chunk.text)
    if ranked:
        return compose_answer(ranked[0][0].text)
    return ""


def _item(
    chunk: Chunk,
    score: float,
    chunks: list[Chunk],
    durations: dict[str, float | None],
    thumbs: set[str],
    media: set[str],
    *,
    as_file: bool,
) -> dict:
    duration = durations.get(chunk.source)
    show_duration = chunk.modality in {"audio", "video"}
    return {
        "source": chunk.source,
        "title": chunk.source if as_file else pretty_stem(chunk.source),
        "subtitle": subtitle_for(chunk, chunks),
        "modality": chunk.modality,
        "icon": icon_for(chunk.source, chunk.modality),
        "percent": to_percent(score),
        "text": chunk.text,
        "page": chunk.page,
        "start": chunk.start,
        "end": chunk.end,
        "kind": chunk.kind,
        "duration_label": fmt_duration(duration) if show_duration else None,
        "thumb_url": f"/api/thumb/{quote(chunk.source)}" if chunk.source in thumbs else None,
        "media_url": f"/api/media/{quote(chunk.source)}" if chunk.source in media else None,
    }


def present(
    query: str,
    ranked: list[tuple[Chunk, float]],
    chunks: list[Chunk],
    durations: dict[str, float | None] | None = None,
    thumbs: set[str] | None = None,
    media: set[str] | None = None,
) -> dict:
    durations = durations or {}
    thumbs = thumbs or set()
    media = media or set()
    evidence = [
        _item(chunk, score, chunks, durations, thumbs, media, as_file=False)
        for chunk, score in _pick_evidence(ranked)
    ]
    files = [
        _item(chunk, score, chunks, durations, thumbs, media, as_file=True)
        for chunk, score in ranked
    ]
    chip_source = [_answer_text(ranked)] + [chunk.text for chunk, _ in ranked[:4]]
    return {
        "query": query,
        "answer": _answer_text(ranked),
        "chips": chips_from(chip_source),
        "evidence": evidence,
        "files": files,
    }
