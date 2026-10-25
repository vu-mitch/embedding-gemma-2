"""Temporal constraints, short-event localization, and the manufacturing benchmark."""

from __future__ import annotations

import numpy as np

from mmrag.benchmark import (
    GOLD_HOLDS,
    MANUFACTURING_QUERY,
    detect_recording,
    manufacturing_corpus,
    run_benchmark,
)
from mmrag.analyse import omit_voiced_anomalies
from mmrag.events import Event, events_from_embeddings, events_from_text, merge_events
from mmrag.extractors import Chunk
from mmrag.metrics import (
    answer_correctness,
    event_ordering_accuracy,
    recall_at_k,
    temporal_iou,
)
from mmrag.signals import abnormal_mask, events_from_series
from mmrag.temporal import parse_constraints, solve
from mmrag.timeline import render_timeline


def test_temporal_iou_partial_and_exact():
    assert temporal_iou((10, 30), (10, 30)) == 1
    assert temporal_iou((0, 30), (10, 30)) == 20 / 30
    assert temporal_iou((0, 10), (20, 30)) == 0


def test_recall_at_k_respects_iou_and_rank():
    gold = [(10.0, 12.0), (40.0, 42.0)]
    wide = [(0.0, 50.0), (0.0, 50.0)]
    tight = [(40.0, 42.0), (10.0, 12.0)]
    assert recall_at_k(wide, gold, k=5, iou_threshold=0.5) == 0
    assert recall_at_k(tight, gold, k=1, iou_threshold=0.5) == 0.5
    assert recall_at_k(tight, gold, k=2, iou_threshold=0.5) == 1


def test_ordering_and_answer_metrics():
    gold = ["abnormal_vibration", "acoustic_anomaly", "operator_report"]
    swapped = ["operator_report", "abnormal_vibration", "acoustic_anomaly"]
    assert event_ordering_accuracy(gold, gold) == 1
    assert event_ordering_accuracy(swapped, gold) == 1 / 3
    assert event_ordering_accuracy(["operator_report"], gold) == 0
    assert answer_correctness("holds on line_b and line_c", GOLD_HOLDS) == 1
    assert answer_correctness("yes", "no") == 0
    assert answer_correctness("no", "no") == 1
    # Identifiers stay intact, so line_a does not count as a match for line_b.
    partial = answer_correctness("matches line_a, line_b, line_c", GOLD_HOLDS)
    assert 0 < partial < 0.6


def test_parser_reads_the_manufacturing_sentence():
    text = (
        "vibration within 15 seconds before the report, "
        "while an acoustic anomaly occurs during the vibration"
    )
    parsed = parse_constraints(text)
    assert parsed.errors == []
    window, during = parsed.constraints
    assert (window.antecedent, window.relation, window.consequent, window.window_s) == (
        "abnormal_vibration", "within", "operator_report", 15.0,
    )
    assert (during.antecedent, during.relation, during.consequent) == (
        "acoustic_anomaly", "during", "abnormal_vibration",
    )
    ordered = parse_constraints("t_vibration < t_report")
    assert ordered.constraints[0].relation == "before"
    assert ordered.constraints[0].formula() == "t_abnormal_vibration < t_operator_report"


def test_fifteen_second_window_accepts_and_rejects():
    def event(type_id, start, end, channel):
        return Event(type_id, "clip", channel, start, end, 1.0, "evidence")

    figure = [
        event("abnormal_vibration", 10, 30, "visual"),
        event("acoustic_anomaly", 15, 30, "acoustic"),
        event("operator_report", 40, 50, "spoken"),
    ]
    constraints = parse_constraints(MANUFACTURING_QUERY).constraints
    solution = solve(figure, constraints)[0]
    assert solution.holds is False
    assert any("30.0s" in check.detail and not check.passed for check in solution.checks)
    assert any(check.passed and check.constraint.relation == "during" for check in solution.checks)

    tight = [
        event("abnormal_vibration", 28, 36, "visual"),
        event("acoustic_anomaly", 30, 34, "acoustic"),
        event("operator_report", 40, 48, "spoken"),
    ]
    passed = solve(tight, constraints)[0]
    assert passed.holds is True
    assert passed.order() == ["abnormal_vibration", "acoustic_anomaly", "operator_report"]


def test_short_spike_in_a_long_series_stays_tight():
    bin_s = 0.5
    n = int(300 / bin_s)
    times = np.arange(n) * bin_s
    energy = np.full(n, 0.2)
    energy[(times >= 200) & (times < 202.5)] = 8.0
    mask, _thresh = abnormal_mask(energy)
    assert mask.sum() == 5
    events = events_from_series(
        times, energy, source="long", bin_s=bin_s,
        high_type="abnormal_vibration", high_channel="visual", high_origin="motion",
    )
    assert len(events) == 1
    assert temporal_iou((events[0].start, events[0].end), (200.0, 202.5)) >= 0.95


def test_signal_bounds_win_over_a_wide_caption():
    wide = Event(
        "abnormal_vibration", "clip", "visual", 0, 30, 0.6,
        "Video frame showing vibration", origin="text",
    )
    tight = Event(
        "abnormal_vibration", "clip", "visual", 10, 12, 0.9,
        "motion peak 8.00", origin="motion",
    )
    merged = merge_events([wide, tight])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (10, 12)
    assert "motion peak" in merged[0].evidence
    assert "Video frame" in merged[0].evidence


def test_operator_speech_is_not_an_acoustic_anomaly():
    tone = Event("acoustic_anomaly", "clip", "acoustic", 15, 30, 1.0, "rms", origin="acoustic")
    voice = Event("acoustic_anomaly", "clip", "acoustic", 40, 50, 1.0, "rms", origin="acoustic")
    speech = [Chunk("operator report", "clip", "video", "speech", start=40, end=50)]
    kept = omit_voiced_anomalies([tone, voice], speech)
    assert [(e.start, e.end) for e in kept] == [(15, 30)]


def test_speech_does_not_become_a_visual_vibration():
    chunk = Chunk(
        "This is the operator report. Abnormal vibration was observed.",
        "clip", "video", "speech", start=40, end=50,
    )
    events = events_from_text([chunk])
    assert [e.type_id for e in events] == ["operator_report"]
    assert (events[0].start, events[0].end) == (40, 50)


def test_embedding_prototypes_respect_channel():
    class Stub:
        def embed_documents(self, texts, titles=None):
            rows = []
            for text in texts:
                low = text.lower()
                if "vibration" in low or "shaking" in low:
                    rows.append([1.0, 0.0, 0.0, 0.0])
                elif "operator" in low or "report" in low:
                    rows.append([0.0, 1.0, 0.0, 0.0])
                elif "sound" in low or "acoustic" in low:
                    rows.append([0.0, 0.0, 1.0, 0.0])
                else:
                    rows.append([0.0, 0.0, 0.0, 1.0])
            return np.asarray(rows, dtype=np.float32)

    visual = Chunk("the housing is shaking badly", "clip", "video", "visual", start=10, end=12)
    spoken = Chunk("the housing is shaking badly", "clip", "video", "speech", start=40, end=42)
    found = events_from_embeddings([visual, spoken], Stub())
    assert [(e.type_id, e.start) for e in found] == [("abnormal_vibration", 10)]


def test_benchmark_temporal_pipeline_beats_semantic_baseline():
    report = run_benchmark()
    by_name = {row.metric: row for row in report.rows}
    assert set(by_name) == {
        "Temporal IoU", "Recall@5", "Event-ordering accuracy", "Answer correctness",
    }
    for row in report.rows:
        assert row.temporal > row.semantic, row.metric
    assert by_name["Temporal IoU"].temporal >= 0.9
    assert by_name["Recall@5"].temporal == 1
    assert by_name["Recall@5"].semantic == 0
    assert by_name["Event-ordering accuracy"].temporal == 1
    assert by_name["Answer correctness"].temporal == 1
    assert report.temporal_answer == GOLD_HOLDS

    corpus = manufacturing_corpus()
    detected = {rec.source: detect_recording(rec) for rec in corpus}
    # Line A keeps the figure's order, and the unusual sound sits inside the vibration.
    line_a = {e.type_id: e for e in detected["line_a"]}
    assert line_a["abnormal_vibration"].start < line_a["operator_report"].start
    assert line_a["abnormal_vibration"].start <= line_a["acoustic_anomaly"].start
    assert line_a["acoustic_anomaly"].end <= line_a["abnormal_vibration"].end


def test_timeline_names_every_channel():
    events = detect_recording(manufacturing_corpus()[0])
    html = render_timeline(events, 50, "Line A")
    assert "Video events" in html
    assert "Audio events" in html
    assert "Spoken events" in html
    assert "Abnormal vibration" in html
    assert "Operator report" in html
