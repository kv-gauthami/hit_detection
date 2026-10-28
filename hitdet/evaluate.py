"""Event-level evaluation of predicted hit frames against labels.

A prediction is a true positive if it lies within ``tol`` frames of a
label not already matched. Matching is one-to-one and greedy by
distance, so one label can never absorb two predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    errors: list[int] = field(default_factory=list)      # pred - label, for matched pairs
    matched: list[tuple[int, int]] = field(default_factory=list)
    false_pos: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_abs_error(self) -> float:
        return sum(abs(e) for e in self.errors) / len(self.errors) if self.errors else 0.0

    def __add__(self, other: "MatchResult") -> "MatchResult":
        return MatchResult(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
            self.errors + other.errors, self.matched + other.matched,
            self.false_pos + other.false_pos, self.missed + other.missed,
        )


def match_events(pred: list[int], gt: list[int], tol: int) -> MatchResult:
    pairs = sorted(
        (abs(p - g), p, g) for p in pred for g in gt if abs(p - g) <= tol
    )
    used_p, used_g = set(), set()
    matched = []
    for _, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            matched.append((p, g))
    matched.sort(key=lambda m: m[1])
    return MatchResult(
        tp=len(matched),
        fp=len(pred) - len(matched),
        fn=len(gt) - len(matched),
        errors=[p - g for p, g in matched],
        matched=matched,
        false_pos=sorted(set(pred) - used_p),
        missed=sorted(set(gt) - used_g),
    )
