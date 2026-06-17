"""GEM 1: Beta-conjugate belief tracking with Thompson Sampling.

Every entity (skill, model, host, job type) gets a Beta(alpha, beta) prior
that updates on every observed outcome.  Thompson Sampling selects the best
entity by sampling from each posterior -- no hardcoded routing tables.

Mathematical basis:
    Prior:      theta ~ Beta(alpha, beta)
    Update:     alpha += success,  beta += failure
    Point est:  E[theta] = alpha / (alpha + beta)
    Variance:   Var[theta] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    Thompson:   sample theta_hat ~ Beta(alpha, beta) via gamma trick
    Decay:      alpha *= lambda, beta *= lambda  (recency weighting)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BetaBelief:
    """A single Beta-distributed belief about an entity's success probability."""

    alpha: float = 1.0
    beta: float = 1.0
    total_obs: int = 0
    _decay: float = 1.0  # no decay by default

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Approximate CI using normal approximation (valid when alpha+beta > 10)."""
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        mu = self.mean
        s = self.std
        return (max(0.0, mu - z * s), min(1.0, mu + z * s))

    def sample(self) -> float:
        """Thompson Sampling via the gamma trick -- no scipy needed."""
        x = random.gammavariate(self.alpha, 1.0)
        y = random.gammavariate(self.beta, 1.0)
        if x + y == 0:
            return 0.5
        return x / (x + y)

    def update(self, success: bool) -> None:
        """Update posterior with a binary outcome."""
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.total_obs += 1

    def update_weighted(self, success: bool, weight: float = 1.0) -> None:
        """Update with a weighted outcome (e.g., partial success)."""
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.total_obs += 1

    def decay(self, factor: float = 0.99) -> None:
        """Apply recency decay -- older observations count less."""
        self.alpha *= factor
        self.beta *= factor
        # Floor at 1.0 to maintain a valid prior
        self.alpha = max(1.0, self.alpha)
        self.beta = max(1.0, self.beta)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "total_obs": self.total_obs,
            "mean": round(self.mean, 4),
            "ci_95": tuple(round(v, 4) for v in self.credible_interval()),
        }


class BeliefEngine:
    """Registry of Beta beliefs keyed by (entity_type, entity_id, context).

    Example entity types: 'model', 'skill', 'host', 'job_type'.
    Context allows per-task-type beliefs (e.g., 'debug', 'routine').
    """

    def __init__(self, decay_factor: float = 0.99) -> None:
        self._beliefs: dict[tuple[str, str, str], BetaBelief] = {}
        self.decay_factor = decay_factor

    def _key(
        self, entity_type: str, entity_id: str, context: str = "global"
    ) -> tuple[str, str, str]:
        return (entity_type, entity_id, context)

    def get(
        self,
        entity_type: str,
        entity_id: str,
        context: str = "global",
    ) -> BetaBelief:
        """Get or create a belief for an entity."""
        key = self._key(entity_type, entity_id, context)
        if key not in self._beliefs:
            self._beliefs[key] = BetaBelief()
        return self._beliefs[key]

    def update(
        self,
        entity_type: str,
        entity_id: str,
        success: bool,
        context: str = "global",
    ) -> BetaBelief:
        """Record an outcome and return the updated belief."""
        belief = self.get(entity_type, entity_id, context)
        belief.update(success)
        return belief

    def select_best(
        self,
        entity_type: str,
        candidates: list[str] | None = None,
        context: str = "global",
        min_obs: int = 0,
    ) -> str | None:
        """Thompson Sampling: sample from each candidate's posterior, return argmax.

        If candidates is None, uses all known entities of that type.

        If min_obs > 0, a forced-exploration warmup is enforced: as long as
        any candidate has fewer than min_obs observations, the least-tried
        under-sampled candidate is returned deterministically (rather than
        sampling).  Once every candidate has been tried at least min_obs
        times, pure Thompson Sampling takes over -- wide priors then keep
        exploring naturally.  This prevents a lucky early sample from
        starving competitors of data.
        """
        if candidates is None:
            candidates = [
                eid
                for (etype, eid, ctx) in self._beliefs
                if etype == entity_type and ctx == context
            ]
        if not candidates:
            return None

        # Forced-exploration warmup: ensure each candidate is tried enough.
        if min_obs > 0:
            undertried = [
                eid for eid in candidates
                if self.get(entity_type, eid, context).total_obs < min_obs
            ]
            if undertried:
                return min(
                    undertried,
                    key=lambda eid: self.get(entity_type, eid, context).total_obs,
                )

        best_id = None
        best_sample = -1.0
        for eid in candidates:
            belief = self.get(entity_type, eid, context)
            s = belief.sample()
            if s > best_sample:
                best_sample = s
                best_id = eid
        return best_id

    def decay_all(self) -> None:
        """Apply daily decay to all beliefs for recency weighting."""
        for belief in self._beliefs.values():
            belief.decay(self.decay_factor)

    def ranked(
        self,
        entity_type: str,
        context: str = "global",
        by: str = "mean",
    ) -> list[tuple[str, BetaBelief]]:
        """Return entities ranked by mean or variance (for active learning)."""
        items = [
            (eid, belief)
            for (etype, eid, ctx), belief in self._beliefs.items()
            if etype == entity_type and ctx == context
        ]
        key_fn = (
            (lambda x: x[1].mean)
            if by == "mean"
            else (lambda x: x[1].variance)
        )
        return sorted(items, key=key_fn, reverse=True)

    def to_dict(self) -> dict:
        return {
            f"{etype}:{eid}:{ctx}": belief.to_dict()
            for (etype, eid, ctx), belief in self._beliefs.items()
        }
