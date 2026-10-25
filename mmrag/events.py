"""Domain events on a shared timeline.

EmbeddingGemma retrieves text. This layer names the things a manufacturing
recording is about — normal operation, abnormal vibration, an unusual sound,
an operator report — and keeps a tight start and end for each one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .extractors import Chunk


@dataclass(frozen=True)
class EventType:
    id: str
    label: str
    channel: str  # "visual" | "acoustic" | "spoken"
    cues: tuple[str, ...]


# Channels that may supply a text cue for each event channel.
_TEXT_KIND = {
    "visual": frozenset({"visual"}),
    "acoustic": frozenset({"speech", "acoustic"}),
    "spoken": frozenset({"speech"}),
}

MANUFACTURING_EVENTS: tuple[EventType, ...] = (
    EventType(
        "normal_operation",
        "Normal operation",
        "visual",
        ("normal operation", "running normally", "steady state", "nominal speed"),
    ),
    EventType(
        "abnormal_vibration",
        "Abnormal vibration",
        "visual",
        ("abnormal vibration", "vibration", "vibrating", "shaking", "oscillat"),
    ),
    EventType(
        "acoustic_anomaly",
        "Unusual sound",
        "acoustic",
        ("unusual sound", "acoustic anomaly", "grinding", "screech", "abnormal noise", "knocking"),
    ),
    EventType(
        "operator_report",
        "Operator report",
        "spoken",
        ("operator report", "operator", "reporting", "maintenance report"),
    ),
)

_BY_ID = {et.id: et for et in MANUFACTURING_EVENTS}

# Longest phrase first so "abnormal vibration" wins over "vibration".
ALIASES: tuple[tuple[str, str], ...] = tuple(sorted((
    ("normal operation", "normal_operation"),
    ("abnormal vibration", "abnormal_vibration"),
    ("unusual sound", "acoustic_anomaly"),
    ("acoustic anomaly", "acoustic_anomaly"),
    ("operator report", "operator_report"),
    ("vibration", "abnormal_vibration"),
    ("sound", "acoustic_anomaly"),
    ("report", "operator_report"),
    ("normal_operation", "normal_operation"),
    ("abnormal_vibration", "abnormal_vibration"),
    ("acoustic_anomaly", "acoustic_anomaly"),
    ("operator_report", "operator_report"),
), key=lambda item: len(item[0]), reverse=True))

_FILLERS = frozenset({
    "a", "an", "the", "begins", "begin", "occurs", "occur", "happens", "happen",
    "starts", "start", "is", "was", "interval", "event", "events", "of", "on",
})


@dataclass
class Event:
    type_id: str
    source: str
    channel: str
    start: float
    end: float
    score: float
    evidence: str
    modality: str = "video"
    origin: str = "text"  # "text" | "motion" | "acoustic" | "embedding"

    @property
    def label(self) -> str:
        return _BY_ID[self.type_id].label if self.type_id in _BY_ID else self.type_id

    def locator(self) -> str:
        return f"{fmt_clock(self.start)}–{fmt_clock(self.end)}"


def fmt_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - 60 * minutes
    return f"{minutes:02d}:{rest:04.1f}"


def event_type(type_id: str) -> EventType:
    return _BY_ID[type_id]


def resolve_event_id(phrase: str) -> str | None:
    """Map a free-text event mention onto a catalog id."""
    cleaned = " ".join(w for w in phrase.lower().strip(" .,:;").split() if w not in _FILLERS)
    if not cleaned:
        return None
    for alias, type_id in ALIASES:
        if cleaned == alias or alias in cleaned:
            return type_id
    return None


def events_from_text(chunks: list[Chunk], catalog: tuple[EventType, ...] = MANUFACTURING_EVENTS) -> list[Event]:
    """Label chunks whose text contains a domain cue. Bounds stay on the chunk."""
    found: list[Event] = []
    for chunk in chunks:
        if chunk.start is None:
            continue
        haystack = chunk.text.lower()
        for spec in catalog:
            if chunk.kind not in _TEXT_KIND.get(spec.channel, frozenset()):
                continue
            hits = [cue for cue in spec.cues if cue in haystack]
            if not hits:
                continue
            found.append(Event(
                type_id=spec.id,
                source=chunk.source,
                channel=spec.channel,
                start=float(chunk.start),
                end=float(chunk.end if chunk.end is not None else chunk.start),
                score=min(1.0, 0.45 + 0.2 * len(hits)),
                evidence=chunk.text.strip()[:400],
                modality=chunk.modality,
                origin="text",
            ))
    return merge_events(found)


def events_from_embeddings(
    chunks: list[Chunk],
    embedder,
    catalog: tuple[EventType, ...] = MANUFACTURING_EVENTS,
    threshold: float = 0.55,
    margin: float = 0.05,
) -> list[Event]:
    """Assign chunks to event prototypes in embedding space.

    Keyword cues miss paraphrases. A prototype sentence per event type gives
    the same catalog a domain-specific second opinion from EmbeddingGemma.
    """
    timed = [c for c in chunks if c.start is not None and c.text.strip()]
    if not timed:
        return []
    prototypes = [f"{spec.label}. " + ", ".join(spec.cues) for spec in catalog]
    proto = embedder.embed_documents(prototypes, titles=["event"] * len(prototypes))
    vecs = embedder.embed_documents([c.text for c in timed], titles=[c.source for c in timed])
    found: list[Event] = []
    for chunk, vec in zip(timed, vecs):
        scores = proto @ vec
        order = scores.argsort()[::-1]
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        if float(scores[best]) < threshold:
            continue
        if float(scores[best] - scores[second]) < margin and best != second:
            continue
        spec = catalog[best]
        if chunk.kind not in _TEXT_KIND.get(spec.channel, frozenset()):
            continue
        found.append(Event(
            type_id=spec.id,
            source=chunk.source,
            channel=spec.channel,
            start=float(chunk.start),
            end=float(chunk.end if chunk.end is not None else chunk.start),
            score=float(scores[best]),
            evidence=chunk.text.strip()[:400],
            modality=chunk.modality,
            origin="embedding",
        ))
    return merge_events(found)


def merge_events(events: list[Event], max_gap: float = 0.75) -> list[Event]:
    """Fold touching events of the same type.

    A motion spike is the time authority. A caption that only confirms the
    label must not stretch that spike out to the keyframe window.
    """
    merged: list[Event] = []
    for event in sorted(events, key=lambda e: (e.source, e.type_id, e.start, e.end)):
        if merged and _can_merge(merged[-1], event, max_gap):
            merged[-1] = _combine(merged[-1], event)
        else:
            merged.append(event)
    return merged


def _can_merge(left: Event, right: Event, max_gap: float) -> bool:
    if left.source != right.source or left.type_id != right.type_id:
        return False
    return left.start <= right.end + max_gap and right.start <= left.end + max_gap


def _combine(left: Event, right: Event) -> Event:
    signal = {"motion", "acoustic"}
    if left.origin in signal and right.origin not in signal:
        start, end, origin = left.start, left.end, left.origin
    elif right.origin in signal and left.origin not in signal:
        start, end, origin = right.start, right.end, right.origin
    else:
        start, end = min(left.start, right.start), max(left.end, right.end)
        origin = left.origin if left.origin in signal else right.origin if right.origin in signal else left.origin
    evidence = left.evidence
    if right.evidence and right.evidence not in evidence:
        evidence = f"{evidence} | {right.evidence}"[:400]
    return replace(
        left,
        start=start,
        end=end,
        score=max(left.score, right.score),
        evidence=evidence,
        origin=origin,
    )
