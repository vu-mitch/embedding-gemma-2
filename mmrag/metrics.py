"""Retrieval metrics for temporal multimodal evidence.

* Temporal IoU — how tightly a predicted span covers the true event
* Recall@k — fraction of true spans matched in the top k at a minimum IoU
* Event-ordering accuracy — pairwise order of event types
* Answer correctness — token F1 between the predicted and gold answer
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence


Span = tuple[float, float]


def temporal_iou(predicted: Span, gold: Span) -> float:
    start = max(predicted[0], gold[0])
    end = min(predicted[1], gold[1])
    intersection = max(0.0, end - start)
    union = max(predicted[1], gold[1]) - min(predicted[0], gold[0])
    if union <= 0:
        return 1.0 if intersection == 0 and predicted[0] == gold[0] else 0.0
    return intersection / union


def recall_at_k(
    predicted: Sequence[Span],
    gold: Sequence[Span],
    k: int,
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of gold spans matched by a distinct top-k prediction."""
    if not gold:
        return 1.0
    used: set[int] = set()
    matched = 0
    pool = list(predicted[:k])
    for truth in gold:
        for index, span in enumerate(pool):
            if index in used:
                continue
            if temporal_iou(span, truth) >= iou_threshold:
                used.add(index)
                matched += 1
                break
    return matched / len(gold)


def mean_best_iou(predicted: Sequence[Span], gold: Sequence[Span]) -> float:
    """Mean, over gold spans, of the best Temporal IoU against any prediction."""
    if not gold:
        return 1.0
    if not predicted:
        return 0.0
    total = 0.0
    for truth in gold:
        total += max(temporal_iou(span, truth) for span in predicted)
    return total / len(gold)


def event_ordering_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Pairwise accuracy of event-type order. Missing types count as wrong."""
    if len(gold) < 2:
        return 1.0 if list(predicted) == list(gold) else 0.0
    gold_pos: dict[str, int] = {}
    for index, type_id in enumerate(gold):
        gold_pos.setdefault(type_id, index)
    pred_pos: dict[str, int] = {}
    for index, type_id in enumerate(predicted):
        pred_pos.setdefault(type_id, index)
    types = list(gold_pos)
    pairs = 0
    correct = 0
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            left, right = types[i], types[j]
            pairs += 1
            if left not in pred_pos or right not in pred_pos:
                continue
            gold_before = gold_pos[left] < gold_pos[right]
            pred_before = pred_pos[left] < pred_pos[right]
            correct += int(gold_before == pred_before)
    return correct / pairs if pairs else 1.0


_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def answer_correctness(predicted: str, gold: str) -> float:
    """SQuAD-style token F1 between a predicted answer and the gold answer."""
    pred = tokenize(predicted)
    truth = tokenize(gold)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    overlap = sum((Counter(pred) & Counter(truth)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)
