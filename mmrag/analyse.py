"""Attach domain events to extracted chunks.

Speech segments keep the operator's words on their real timestamps. Motion and
loudness localize the vibration and the unusual sound inside long recordings.
"""

from __future__ import annotations

from pathlib import Path

from .events import Event, events_from_embeddings, events_from_text, merge_events
from .extractors import Chunk
from .signals import acoustic_series, events_from_series, motion_series


def segments_to_chunks(segments: list[dict], source: str, modality: str) -> list[Chunk]:
    chunks = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") if segment.get("end") is not None else start)
        chunks.append(Chunk(text=text, source=source, modality=modality, kind="speech", start=start, end=end))
    return chunks


def collect_events(
    path: str | Path,
    source: str,
    modality: str,
    chunks: list[Chunk],
    speech_segments: list[dict] | None = None,
    embedder=None,
    enable_signals: bool = True,
) -> list[Event]:
    """Events for one file: transcript cues, optional prototypes, signal spans."""
    if speech_segments:
        speech = segments_to_chunks(speech_segments, source, modality)
    else:
        speech = [c for c in chunks if c.kind == "speech"]
    visual = [c for c in chunks if c.kind == "visual"]
    events = events_from_text(speech) + events_from_text(visual)
    if embedder is not None and (speech or visual):
        events += events_from_embeddings(speech + visual, embedder)
    if enable_signals and modality in {"audio", "video"}:
        acoustic = _acoustic_events(path, source, modality)
        events += omit_voiced_anomalies(acoustic, speech)
    if enable_signals and modality == "video":
        events += _motion_events(path, source)
    return merge_events(events)


def omit_voiced_anomalies(events: list[Event], speech: list[Chunk]) -> list[Event]:
    """Drop loudness spikes whose middle sits inside a speech segment.

    The operator report is spoken. An unusual sound is the loud stretch that
    is not that speech.
    """
    spans = [(float(c.start), float(c.end if c.end is not None else c.start)) for c in speech if c.start is not None]
    kept: list[Event] = []
    for event in events:
        if event.origin != "acoustic":
            kept.append(event)
            continue
        mid = (event.start + event.end) / 2.0
        if any(start <= mid <= end for start, end in spans):
            continue
        kept.append(event)
    return kept


def _acoustic_events(path: str | Path, source: str, modality: str) -> list[Event]:
    try:
        times, energy, bin_s = acoustic_series(path)
    except Exception:
        return []
    if times.size == 0:
        return []
    return events_from_series(
        times, energy, source=source, bin_s=bin_s,
        high_type="acoustic_anomaly", high_channel="acoustic", high_origin="acoustic",
        modality=modality,
    )


def _motion_events(path: str | Path, source: str) -> list[Event]:
    try:
        times, energy, bin_s = motion_series(path)
    except Exception:
        return []
    if times.size == 0:
        return []
    return events_from_series(
        times, energy, source=source, bin_s=bin_s,
        high_type="abnormal_vibration", high_channel="visual", high_origin="motion",
        low_type="normal_operation", modality="video",
    )
