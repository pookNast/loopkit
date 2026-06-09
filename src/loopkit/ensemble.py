"""GEM 7: Condorcet ensemble evaluation.

For binary verdicts (PASS/FAIL), N cheap evaluators with majority vote
outperform a single expensive evaluator when individual accuracy > 50%.

    P(majority_correct | p) = sum(C(N,k) * p^k * (1-p)^(N-k)) for k >= ceil(N/2)

    At p=0.70, N=3: P(majority) = 0.784
    At p=0.80, N=3: P(majority) = 0.896

Cost: 3 x Haiku ~ 0.6 x Sonnet, with equal or better accuracy.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def condorcet_accuracy(p_individual: float, n_voters: int = 3) -> float:
    """Compute majority-vote accuracy given individual voter accuracy.

    Uses the Condorcet jury theorem formula.
    """
    if n_voters % 2 == 0:
        raise ValueError("n_voters must be odd for majority vote")
    majority = (n_voters + 1) // 2
    total = 0.0
    for k in range(majority, n_voters + 1):
        binom = math.comb(n_voters, k)
        total += binom * (p_individual**k) * ((1 - p_individual) ** (n_voters - k))
    return total


@dataclass
class EvaluationResult:
    """Result of an ensemble evaluation."""

    verdict: Any
    agreement_ratio: float
    votes: list[Any]
    used_tiebreak: bool = False
    cost_ratio: float = 1.0  # vs single expensive evaluator


class EnsembleEvaluator:
    """Majority-vote evaluator using N cheap voters + optional expensive tiebreak.

    Usage::

        evaluator = EnsembleEvaluator(
            cheap_fn=lambda task: run_haiku(task),
            expensive_fn=lambda task: run_opus(task),
            n_voters=3,
        )
        result = evaluator.evaluate(my_task)
    """

    def __init__(
        self,
        cheap_fn: Callable[[Any], T],
        expensive_fn: Optional[Callable[[Any], T]] = None,
        n_voters: int = 3,
        cheap_cost: float = 0.2,
        expensive_cost: float = 1.0,
    ) -> None:
        if n_voters % 2 == 0:
            raise ValueError("n_voters must be odd")
        self.cheap_fn = cheap_fn
        self.expensive_fn = expensive_fn
        self.n_voters = n_voters
        self.cheap_cost = cheap_cost
        self.expensive_cost = expensive_cost
        self._total_evals = 0
        self._tiebreak_count = 0
        self._total_cost = 0.0

    def evaluate(self, task: Any) -> EvaluationResult:
        """Run the ensemble. Majority vote if unanimous or clear majority;
        escalate to expensive_fn on disagreement if available."""
        votes = [self.cheap_fn(task) for _ in range(self.n_voters)]
        counter = Counter(votes)
        majority_verdict, majority_count = counter.most_common(1)[0]
        agreement = majority_count / self.n_voters
        cost = self.n_voters * self.cheap_cost

        used_tiebreak = False

        # On disagreement, escalate if expensive_fn is available
        if agreement < 1.0 and self.expensive_fn is not None:
            majority_verdict = self.expensive_fn(task)
            cost += self.expensive_cost
            used_tiebreak = True
            self._tiebreak_count += 1

        self._total_evals += 1
        self._total_cost += cost
        cost_ratio = cost / self.expensive_cost if self.expensive_cost > 0 else 1.0

        return EvaluationResult(
            verdict=majority_verdict,
            agreement_ratio=agreement,
            votes=votes,
            used_tiebreak=used_tiebreak,
            cost_ratio=round(cost_ratio, 3),
        )

    @property
    def tiebreak_rate(self) -> float:
        if self._total_evals == 0:
            return 0.0
        return self._tiebreak_count / self._total_evals

    @property
    def avg_cost_ratio(self) -> float:
        if self._total_evals == 0:
            return 0.0
        avg = self._total_cost / self._total_evals
        return round(avg / self.expensive_cost, 3) if self.expensive_cost else 0.0

    def to_dict(self) -> dict:
        return {
            "total_evals": self._total_evals,
            "tiebreak_rate": round(self.tiebreak_rate, 4),
            "avg_cost_ratio": self.avg_cost_ratio,
        }
