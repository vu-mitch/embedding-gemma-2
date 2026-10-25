"""Explicit temporal constraints across visual, acoustic and spoken events.

Semantic similarity can surface the clip that mentions a vibration. It cannot
say whether that vibration begins before the operator report, or whether the
unusual sound sits inside the vibration. These constraints are the test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product

from .events import Event, fmt_clock, resolve_event_id


@dataclass(frozen=True)
class Constraint:
    antecedent: str
    relation: str  # "before" | "within" | "during"
    consequent: str
    window_s: float | None = None
    side: str | None = None  # "before" for a bounded window

    def formula(self) -> str:
        if self.relation == "before":
            return f"t_{self.antecedent} < t_{self.consequent}"
        if self.relation == "within":
            return (
                f"0 < t_{self.consequent} - t_{self.antecedent} "
                f"≤ {self.window_s:g}s"
            )
        return f"t_{self.antecedent} during t_{self.consequent}"


@dataclass
class Check:
    constraint: Constraint
    passed: bool
    detail: str


@dataclass
class Solution:
    source: str
    holds: bool
    bindings: dict[str, Event] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    score: float = 0.0

    def order(self) -> list[str]:
        return [e.type_id for e in sorted(self.bindings.values(), key=lambda e: (e.start, e.end))]

    def target_span(self, type_id: str = "abnormal_vibration") -> tuple[float, float] | None:
        event = self.bindings.get(type_id)
        if event is None:
            return None
        return (event.start, event.end)


@dataclass
class ParseResult:
    constraints: list[Constraint]
    errors: list[str]


_WITHIN = re.compile(
    r"^(?P<a>.+?)\s+within\s+(?P<n>\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\s+before\s+(?P<b>.+)$",
    re.IGNORECASE,
)
_DURING = re.compile(r"^(?P<a>.+?)\s+during\s+(?P<b>.+)$", re.IGNORECASE)
_BEFORE = re.compile(r"^(?P<a>.+?)\s+before\s+(?P<b>.+)$", re.IGNORECASE)
_T_ORDER = re.compile(r"^t_([a-z][a-z_]*)\s*<\s*t_([a-z][a-z_]*)$", re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:,\s*while|,|\bwhile\b|\band\b)\s*", re.IGNORECASE)


def parse_constraints(text: str) -> ParseResult:
    """Parse a constraint expression.

    Examples
    --------
    ``vibration within 15 seconds before the report, while an acoustic
    anomaly occurs during the vibration``

    ``t_vibration < t_report``
    """
    raw = (text or "").strip().strip(".")
    if not raw:
        return ParseResult([], [])
    constraints: list[Constraint] = []
    errors: list[str] = []
    for part in [p.strip(" .") for p in _SPLIT.split(raw) if p.strip(" .")]:
        parsed = _parse_one(part)
        if isinstance(parsed, Constraint):
            constraints.append(parsed)
        else:
            errors.append(parsed)
    return ParseResult(constraints, errors)


def _parse_one(part: str) -> Constraint | str:
    ordered = _T_ORDER.match(part.replace(" ", ""))
    if ordered:
        left = resolve_event_id(ordered.group(1).replace("_", " "))
        right = resolve_event_id(ordered.group(2).replace("_", " "))
        if left and right:
            return Constraint(left, "before", right)
        return f"Unknown event in '{part}'"
    within = _WITHIN.match(part)
    if within:
        return _binary(within.group("a"), within.group("b"), "within", float(within.group("n")))
    during = _DURING.match(part)
    if during:
        return _binary(during.group("a"), during.group("b"), "during")
    before = _BEFORE.match(part)
    if before:
        return _binary(before.group("a"), before.group("b"), "before")
    return f"Could not read a constraint from '{part}'"


def _binary(left: str, right: str, relation: str, window_s: float | None = None) -> Constraint | str:
    antecedent = resolve_event_id(left)
    consequent = resolve_event_id(right)
    if antecedent is None or consequent is None:
        missing = left if antecedent is None else right
        return f"Unknown event '{missing.strip()}'"
    return Constraint(
        antecedent,
        relation,
        consequent,
        window_s=window_s,
        side="before" if relation == "within" else None,
    )


def check_pair(antecedent: Event, consequent: Event, constraint: Constraint) -> Check:
    if constraint.relation == "before":
        delta = consequent.start - antecedent.start
        passed = antecedent.start < consequent.start
        detail = (
            f"{constraint.formula()}: {fmt_clock(antecedent.start)} < {fmt_clock(consequent.start)} "
            f"({delta:.1f}s)"
        )
        return Check(constraint, passed, detail + ("" if passed else " failed"))
    if constraint.relation == "within":
        delta = consequent.start - antecedent.start
        window = float(constraint.window_s or 0.0)
        passed = 0.0 < delta <= window
        relation = "inside" if passed else "outside"
        detail = (
            f"{antecedent.label} begins {delta:.1f}s before {consequent.label} "
            f"({relation} the {window:g}s window). {constraint.formula()}"
        )
        return Check(constraint, passed, detail)
    mid = (antecedent.start + antecedent.end) / 2.0
    overlap = _overlap(antecedent, consequent)
    passed = overlap > 0 and consequent.start <= mid <= consequent.end
    detail = (
        f"{antecedent.label} {antecedent.locator()} during {consequent.label} "
        f"{consequent.locator()} (overlap {overlap:.1f}s)"
    )
    return Check(constraint, passed, detail)


def _overlap(left: Event, right: Event) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def solve(events: list[Event], constraints: list[Constraint], max_combinations: int = 400) -> list[Solution]:
    """Find, per recording, the event tuples that meet every constraint.

    Recordings that lack a required event are returned as failed solutions so
    the miss is visible instead of silent.
    """
    if not constraints:
        return []
    needed = _needed_types(constraints)
    by_source: dict[str, list[Event]] = {}
    for event in events:
        by_source.setdefault(event.source, []).append(event)
    solutions: list[Solution] = []
    for source, group in by_source.items():
        pools = {type_id: [e for e in group if e.type_id == type_id] for type_id in needed}
        missing = [type_id for type_id in needed if not pools[type_id]]
        if missing:
            solutions.append(Solution(
                source=source,
                holds=False,
                checks=[Check(constraints[0], False, "missing " + ", ".join(missing))],
            ))
            continue
        combinations = 0
        for combo in product(*[pools[type_id] for type_id in needed]):
            combinations += 1
            if combinations > max_combinations:
                break
            bindings = dict(zip(needed, combo))
            checks = [
                check_pair(bindings[c.antecedent], bindings[c.consequent], c) for c in constraints
            ]
            score = sum(event.score for event in bindings.values())
            score += 0.25 * sum(1.0 for check in checks if check.passed)
            solutions.append(Solution(
                source=source,
                holds=all(check.passed for check in checks),
                bindings=bindings,
                checks=checks,
                score=score,
            ))
    return rank_solutions(solutions)


def rank_solutions(solutions: list[Solution]) -> list[Solution]:
    """Passing tuples first, then higher evidence scores, then source name."""
    return sorted(solutions, key=lambda s: (not s.holds, -s.score, s.source))


# Expressions the manufacturing example is built around.
PRESETS: dict[str, str] = {
    "t_vibration < t_report": "t_vibration < t_report",
    "Acoustic anomaly during vibration": "acoustic anomaly during the vibration",
    "Vibration within 15s before the report": "vibration within 15 seconds before the report",
    "Full manufacturing constraint": (
        "vibration within 15 seconds before the report, "
        "while an acoustic anomaly occurs during the vibration"
    ),
}


def _needed_types(constraints: list[Constraint]) -> list[str]:
    ordered: list[str] = []
    for constraint in constraints:
        for type_id in (constraint.antecedent, constraint.consequent):
            if type_id not in ordered:
                ordered.append(type_id)
    return ordered


def holds_answer(solutions: list[Solution]) -> str:
    """Deterministic yes-list used by answer-correctness scoring."""
    names = []
    for solution in solutions:
        if solution.holds and solution.source not in names:
            names.append(solution.source)
    if not names:
        return "holds on none"
    return "holds on " + " and ".join(names)


def best_order(solutions: list[Solution]) -> list[str]:
    for solution in solutions:
        if solution.holds:
            return solution.order()
    return []
