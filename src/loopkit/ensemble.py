"""GEM 7: Condorcet ensemble evaluation.

For binary verdicts (PASS/FAIL), N cheap evaluators with majority vote
outperform a single expensive evaluator when individual accuracy > 50%
**and their errors are independent**.  Independence is the whole game:
N copies of the same deterministic voter give zero lift.

    P(majority_correct | p) = sum(C(N,k) * p^k * (1-p)^(N-k)) for k >= ceil(N/2)

    At p=0.70, N=3: P(majority) = 0.784
    At p=0.80, N=3: P(majority) = 0.896

Cost: 3 x Haiku ~ 0.6 x Sonnet, with equal or better accuracy -- but only
when voters actually differ (different model, temperature, seed, or prompt).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

T = TypeVar("T")


def condorcet_accuracy(p_individual: float, n_voters: int = 3) -> float:
    """Compute majority-vote accuracy given individual voter accuracy.

    Uses the Condorcet jury theorem formula.  ``p_individual`` is the
    probability that a single voter returns the correct verdict; assumes
    voters are independent.

    Raises:
        ValueError: if ``p_individual`` is outside [0, 1] or ``n_voters`` < 1.
        ValueError: if ``n_voters`` is even (no strict majority possible).
    """
    if not 0.0 <= p_individual <= 1.0:
        raise ValueError(
            f"p_individual must be in [0.0, 1.0], got {p_individual!r}"
        )
    if n_voters < 1:
        raise ValueError(f"n_voters must be >= 1, got {n_voters}")
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

    Two ways to specify voters (mutually exclusive):

    - ``voters``: a list of distinct callables, one per voter.  **This is the
      correct way to get the Condorcet benefit** -- each callable should
      encapsulate a different error mode (different model / temperature /
      seed / prompt).  ``n_voters`` is then derived from the list length.
    - ``cheap_fn``: a single callable invoked ``n_voters`` times.  Only use
      this when the callable is genuinely stochastic (e.g. temperature > 0
      or a seeded RNG that varies per call); a deterministic ``cheap_fn``
      produces N identical votes and majority voting adds nothing.

    Escalation policy (``escalate_on``):

    - ``"no_majority"`` (default): the expensive tiebreaker fires only when no
      single verdict holds a strict majority.  For binary verdicts with an
      odd number of voters this means it effectively never fires -- which is
      the point: the majority vote *is* the answer, and the cost saving comes
      from not paying for the expensive evaluator on every disagreement.
    - ``"any_disagreement"``: legacy behaviour -- escalate whenever the vote
      is not unanimous.  Costly; use only when close splits must be re-checked.
    """

    def __init__(
        self,
        cheap_fn: Callable[[Any], T] | None = None,
        voters: list[Callable[[Any], T]] | None = None,
        expensive_fn: Callable[[Any], T] | None = None,
        n_voters: int = 3,
        cheap_cost: float = 0.2,
        expensive_cost: float = 1.0,
        escalate_on: Literal["no_majority", "any_disagreement"] = "no_majority",
    ) -> None:
        if voters is not None:
            if cheap_fn is not None:
                raise ValueError("Pass either cheap_fn or voters, not both")
            self.voters: list[Callable[[Any], T]] = list(voters)
            n_voters = len(self.voters)
        elif cheap_fn is not None:
            self.voters = [cheap_fn] * n_voters
        else:
            raise ValueError("Provide either cheap_fn or voters")

        if n_voters % 2 == 0:
            raise ValueError("n_voters must be odd for majority vote")
        if not self.voters:
            raise ValueError("voters must not be empty")

        self.expensive_fn = expensive_fn
        self.n_voters = n_voters
        self.cheap_cost = cheap_cost
        self.expensive_cost = expensive_cost
        self.escalate_on = escalate_on
        self._total_evals = 0
        self._tiebreak_count = 0
        self._total_cost = 0.0

    def evaluate(self, task: Any) -> EvaluationResult:
        """Run the ensemble.

        Each voter is invoked exactly once on ``task``.  If a strict majority
        exists it is returned; otherwise (or per ``escalate_on``) the
        expensive tiebreaker is consulted when available.
        """
        votes = [v(task) for v in self.voters]
        counter = Counter(votes)
        majority_verdict, majority_count = counter.most_common(1)[0]
        agreement = majority_count / self.n_voters
        has_strict_majority = majority_count > self.n_voters // 2
        cost = self.n_voters * self.cheap_cost

        used_tiebreak = False
        need_escalation = (
            (self.escalate_on == "no_majority" and not has_strict_majority)
            or (self.escalate_on == "any_disagreement" and agreement < 1.0)
        )
        if need_escalation and self.expensive_fn is not None:
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
            "escalate_on": self.escalate_on,
            "n_voters": self.n_voters,
        }
