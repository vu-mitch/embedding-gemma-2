"""Manufacturing timeline benchmark.

The figure in the research note has four events on one recording:

* normal operation, then abnormal vibration (visual)
* an unusual sound inside the vibration (acoustic)
* an operator report after the vibration (spoken)

``t_vibration < t_report`` holds there, but the vibration does **not** begin
within 15 seconds of the report. Two further recordings do meet that window,
one of them with a two-second vibration inside a five-minute take.

The temporal pipeline is scored against a semantic ablation: coarse windows
ranked by lexical overlap, with no constraint check and no short-span
localization. Both sides run without EmbeddingGemma so the numbers are
reproducible. In the app, real retrieval still uses EmbeddingGemma; this
harness isolates the temporal layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import Event, MANUFACTURING_EVENTS, events_from_text
from .extractors import Chunk
from .metrics import (
    answer_correctness,
    event_ordering_accuracy,
    mean_best_iou,
    recall_at_k,
    tokenize,
)
from .signals import events_from_series
from .temporal import (
    PRESETS,
    Solution,
    best_order,
    holds_answer,
    parse_constraints,
    solve,
)

MANUFACTURING_QUERY = PRESETS["Full manufacturing constraint"]
GOLD_HOLDS = "holds on line_b and line_c"
GOLD_WINDOW = "no"
REPORT_TEXT = "operator report of the vibration and an acoustic anomaly seconds before"
VIBRATION_TEXT = "abnormal vibration on the spindle housing"
K = 5


@dataclass
class Recording:
    source: str
    title: str
    duration: float
    gold: list[Event]
    motion_times: np.ndarray
    motion_energy: np.ndarray
    acoustic_times: np.ndarray
    acoustic_energy: np.ndarray
    bin_s: float
    speech_chunks: list[Chunk]
    coarse_chunks: list[Chunk] = field(default_factory=list)


@dataclass
class MetricRow:
    metric: str
    temporal: float
    semantic: float


@dataclass
class BenchmarkReport:
    rows: list[MetricRow]
    temporal_answer: str
    semantic_answer: str
    gold_answer: str
    query: str
    notes: str
    k: int = K

    def as_records(self) -> list[dict]:
        return [
            {"metric": row.metric, "temporal pipeline": round(row.temporal, 3), "semantic baseline": round(row.semantic, 3)}
            for row in self.rows
        ]


def manufacturing_corpus() -> list[Recording]:
    """Three labeled lines. Line A is the figure. B and C meet the 15s window."""
    return [
        _line(
            "line_a",
            "Line A · figure timeline (vibration begins 30s before the report)",
            duration=50.0,
            vibration=(10.0, 30.0),
            sound=(15.0, 30.0),
            report=(40.0, 50.0),
            normal=(0.0, 10.0),
            coarse=(0.0, 50.0),
        ),
        _line(
            "line_b",
            "Line B · vibration begins 12s before the report",
            duration=50.0,
            vibration=(28.0, 36.0),
            sound=(30.0, 34.0),
            report=(40.0, 48.0),
            coarse=(0.0, 50.0),
        ),
        _line(
            "line_c",
            "Line C · a 2.5s vibration inside a 5 min recording",
            duration=300.0,
            vibration=(200.0, 202.5),
            sound=(200.5, 202.0),
            report=(210.0, 214.0),
            coarse=(180.0, 240.0),
        ),
    ]


def detect_recording(rec: Recording) -> list[Event]:
    """Localize events from the energy series and the transcript spans."""
    events = events_from_series(
        rec.motion_times, rec.motion_energy,
        source=rec.source, bin_s=rec.bin_s,
        high_type="abnormal_vibration", high_channel="visual", high_origin="motion",
        low_type="normal_operation", modality="video",
    )
    events += events_from_series(
        rec.acoustic_times, rec.acoustic_energy,
        source=rec.source, bin_s=rec.bin_s,
        high_type="acoustic_anomaly", high_channel="acoustic", high_origin="acoustic",
        modality="video",
    )
    events += events_from_text(rec.speech_chunks)
    return events


def demo_bundle() -> tuple[list[Chunk], list[Event], dict[str, float], dict[str, str]]:
    """Chunks, detected events, durations and titles for the offline demo."""
    chunks: list[Chunk] = []
    events: list[Event] = []
    durations: dict[str, float] = {}
    titles: dict[str, str] = {}
    for rec in manufacturing_corpus():
        chunks.extend(rec.coarse_chunks)
        events.extend(detect_recording(rec))
        durations[rec.source] = rec.duration
        titles[rec.source] = rec.title
    return chunks, events, durations, titles


def run_benchmark(query: str = MANUFACTURING_QUERY, k: int = K) -> BenchmarkReport:
    corpus = manufacturing_corpus()
    parsed = parse_constraints(query)
    if parsed.errors or not parsed.constraints:
        raise ValueError("; ".join(parsed.errors) or "no constraints")

    detected: list[Event] = []
    gold_events: list[Event] = []
    coarse: list[Chunk] = []
    for rec in corpus:
        detected.extend(detect_recording(rec))
        gold_events.extend(rec.gold)
        coarse.extend(rec.coarse_chunks)

    temporal_solutions = solve(detected, parsed.constraints)
    gold_solutions = solve(gold_events, parsed.constraints)
    gold_order = best_order(gold_solutions)
    gold_spans = _vibration_spans(gold_solutions)
    temporal_spans = _vibration_spans(temporal_solutions)

    ranked = sorted(coarse, key=lambda c: (-_overlap(query, c.text), c.source, c.text))
    semantic_spans = [(c.start or 0.0, c.end or 0.0) for c in ranked[:k]]
    semantic_order = _reading_order(ranked[0].text) if ranked else []

    temporal_iou = _localization(corpus, detected, same_type=True)
    semantic_iou = _localization(corpus, None, same_type=False)

    temporal_holds = holds_answer(temporal_solutions)
    semantic_holds = "matches " + ", ".join(_unique_sources(ranked))
    window_temporal = _window_word(temporal_solutions, "line_a")
    window_semantic = "yes"

    rows = [
        MetricRow("Temporal IoU", temporal_iou, semantic_iou),
        MetricRow(f"Recall@{k}", recall_at_k(temporal_spans, gold_spans, k), recall_at_k(semantic_spans, gold_spans, k)),
        MetricRow(
            "Event-ordering accuracy",
            event_ordering_accuracy(best_order(temporal_solutions), gold_order),
            event_ordering_accuracy(semantic_order, gold_order),
        ),
        MetricRow(
            "Answer correctness",
            _mean_f1([(temporal_holds, GOLD_HOLDS), (window_temporal, GOLD_WINDOW)]),
            _mean_f1([(semantic_holds, GOLD_HOLDS), (window_semantic, GOLD_WINDOW)]),
        ),
    ]
    notes = (
        "Line A matches the figure: the vibration begins 30s before the operator report, "
        "so 'within 15 seconds' fails, while the unusual sound still falls inside the vibration. "
        "Line B (12s) and line C (10s, 2.5s long inside 5 min) pass both constraints. "
        "The semantic baseline ranks coarse windows by lexical overlap and does not test the window. "
        "Temporal IoU compares each gold event with the tightest same-type detection, "
        "against the coarse window that contains it."
    )
    return BenchmarkReport(
        rows=rows,
        temporal_answer=temporal_holds,
        semantic_answer=semantic_holds,
        gold_answer=GOLD_HOLDS,
        query=query,
        notes=notes,
        k=k,
    )


def _line(
    source: str,
    title: str,
    duration: float,
    vibration: tuple[float, float],
    sound: tuple[float, float],
    report: tuple[float, float],
    coarse: tuple[float, float],
    normal: tuple[float, float] | None = None,
    bin_s: float = 0.5,
) -> Recording:
    motion_t, motion_e = _energy(duration, bin_s, [vibration])
    acoustic_t, acoustic_e = _energy(duration, bin_s, [sound], high=7.0, low=0.05)
    speech = Chunk(
        "This is the operator report. Abnormal vibration was observed on the spindle.",
        source, "video", "speech", start=report[0], end=report[1],
    )
    gold = [
        _gold("abnormal_vibration", source, "visual", *vibration),
        _gold("acoustic_anomaly", source, "acoustic", *sound),
        _gold("operator_report", source, "spoken", *report),
    ]
    if normal is not None:
        gold.insert(0, _gold("normal_operation", source, "visual", *normal))
    coarse_chunks = [
        Chunk(REPORT_TEXT, source, "video", "speech", start=coarse[0], end=coarse[1]),
        Chunk(VIBRATION_TEXT, source, "video", "visual", start=coarse[0], end=coarse[1]),
    ]
    return Recording(
        source, title, duration, gold,
        motion_t, motion_e, acoustic_t, acoustic_e, bin_s,
        [speech], coarse_chunks,
    )


def _energy(
    duration: float,
    bin_s: float,
    spans: list[tuple[float, float]],
    high: float = 8.0,
    low: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(duration / bin_s))
    times = np.arange(n, dtype=np.float64) * bin_s
    energy = np.full(n, low, dtype=np.float64)
    for start, end in spans:
        energy[(times >= start) & (times < end)] = high
    return times, energy


def _gold(type_id: str, source: str, channel: str, start: float, end: float) -> Event:
    return Event(type_id, source, channel, start, end, 1.0, "gold", origin="gold")


def _vibration_spans(solutions: list[Solution]) -> list[tuple[float, float]]:
    spans = []
    seen: set[str] = set()
    for solution in solutions:
        if not solution.holds or solution.source in seen:
            continue
        span = solution.target_span("abnormal_vibration")
        if span is None:
            continue
        seen.add(solution.source)
        spans.append(span)
    return spans


def _localization(corpus: list[Recording], detected: list[Event] | None, same_type: bool) -> float:
    scores: list[float] = []
    for rec in corpus:
        if same_type:
            pool = [e for e in (detected or []) if e.source == rec.source]
            for gold in rec.gold:
                spans = [(e.start, e.end) for e in pool if e.type_id == gold.type_id]
                scores.append(mean_best_iou(spans, [(gold.start, gold.end)]))
        else:
            spans = [(c.start or 0.0, c.end or 0.0) for c in rec.coarse_chunks]
            for gold in rec.gold:
                scores.append(mean_best_iou(spans, [(gold.start, gold.end)]))
    return sum(scores) / len(scores) if scores else 0.0


def _overlap(query: str, text: str) -> int:
    wanted = set(tokenize(query))
    return sum(1 for token in tokenize(text) if token in wanted)


def _reading_order(text: str) -> list[str]:
    haystack = text.lower()
    hits: list[tuple[int, str]] = []
    for spec in MANUFACTURING_EVENTS:
        indexes = [haystack.find(cue) for cue in spec.cues if cue in haystack]
        if indexes:
            hits.append((min(indexes), spec.id))
    hits.sort()
    order: list[str] = []
    for _, type_id in hits:
        if type_id not in order:
            order.append(type_id)
    return order


def _unique_sources(chunks: list[Chunk]) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        if chunk.source not in names:
            names.append(chunk.source)
    return names


def _window_word(solutions: list[Solution], source: str) -> str:
    for solution in solutions:
        if solution.source == source:
            return "yes" if solution.holds else "no"
    return "no"


def _mean_f1(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(answer_correctness(pred, gold) for pred, gold in pairs) / len(pairs)


def main() -> None:
    report = run_benchmark()
    for row in report.rows:
        print(f"{row.metric:28} temporal {row.temporal:.3f}   semantic {row.semantic:.3f}")
    print("temporal answer:", report.temporal_answer)
    print("semantic answer:", report.semantic_answer)
    print("gold answer:    ", report.gold_answer)


if __name__ == "__main__":
    main()
