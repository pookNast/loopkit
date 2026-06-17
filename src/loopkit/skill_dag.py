"""GEM 3: Skill dependency DAG with chain reliability estimation.

Models skill chains as a directed graph.  When an upstream skill's
reliability drops, that uncertainty propagates downstream.

    Chain reliability: P(chain) = prod(E[theta_i])
    Bottleneck:        argmin_i E[theta_i] in the chain
    Info gain:         Skill with highest Var[theta_i] benefits most from testing

Uses BetaBelief from the beliefs module for per-skill probability estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loopkit.beliefs import BeliefEngine


@dataclass
class ChainReliability:
    """Result of computing reliability for a skill chain."""

    chain: list[str]
    point_estimate: float
    ci_lower: float
    ci_upper: float
    bottleneck: str
    bottleneck_mean: float
    per_skill: dict[str, float]


class SkillDAG:
    """Directed acyclic graph of skill dependencies.

    Edges represent data/control flow: if skill A feeds into skill B,
    add edge (A, B).  Chain reliability is the product of per-skill
    success probabilities from a BeliefEngine.
    """

    def __init__(self, beliefs: BeliefEngine | None = None) -> None:
        self.beliefs = beliefs or BeliefEngine()
        self._edges: dict[str, list[str]] = {}  # parent -> [children]
        self._reverse: dict[str, list[str]] = {}  # child -> [parents]
        self._all_nodes: set[str] = set()

    def add_edge(self, parent: str, child: str) -> None:
        self._edges.setdefault(parent, []).append(child)
        self._reverse.setdefault(child, []).append(parent)
        self._all_nodes.add(parent)
        self._all_nodes.add(child)

    def add_chain(self, skills: list[str]) -> None:
        """Add a linear chain: A -> B -> C -> ..."""
        for i in range(len(skills) - 1):
            self.add_edge(skills[i], skills[i + 1])

    def children(self, skill: str) -> list[str]:
        return self._edges.get(skill, [])

    def parents(self, skill: str) -> list[str]:
        return self._reverse.get(skill, [])

    def roots(self) -> list[str]:
        """Skills with no parents (chain entry points)."""
        return [n for n in self._all_nodes if n not in self._reverse]

    def walk_chain(self, root: str) -> list[str]:
        """Walk a linear chain from root to leaf. For DAGs, follows first child."""
        chain = [root]
        current = root
        seen = {root}
        while self._edges.get(current):
            next_node = self._edges[current][0]
            if next_node in seen:
                break
            chain.append(next_node)
            seen.add(next_node)
            current = next_node
        return chain

    def chain_reliability(
        self,
        root: str,
        context: str = "global",
    ) -> ChainReliability:
        """Compute reliability for a chain starting at root.

        P(chain) = product of E[theta_i] for each skill in the chain.
        CI approximated via log-normal product of Betas.
        """
        chain = self.walk_chain(root)
        per_skill: dict[str, float] = {}
        log_sum = 0.0
        log_var_sum = 0.0
        bottleneck = chain[0]
        bottleneck_mean = 1.0

        for skill in chain:
            belief = self.beliefs.get("skill", skill, context)
            mu = belief.mean
            var = belief.variance
            per_skill[skill] = round(mu, 4)

            if mu < bottleneck_mean:
                bottleneck_mean = mu
                bottleneck = skill

            # Log-normal approximation for product of Betas
            if mu > 0:
                log_sum += math.log(mu)
                if mu > 0 and mu < 1:
                    log_var_sum += var / (mu * mu)

        point = math.exp(log_sum)
        # Approximate CI using delta method on log scale
        log_std = math.sqrt(log_var_sum) if log_var_sum > 0 else 0.0
        ci_lower = math.exp(log_sum - 1.96 * log_std)
        ci_upper = min(1.0, math.exp(log_sum + 1.96 * log_std))

        return ChainReliability(
            chain=chain,
            point_estimate=round(point, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            bottleneck=bottleneck,
            bottleneck_mean=round(bottleneck_mean, 4),
            per_skill=per_skill,
        )

    def all_chain_reliabilities(
        self, context: str = "global"
    ) -> dict[str, ChainReliability]:
        """Compute reliability for all chains (one per root)."""
        return {
            root: self.chain_reliability(root, context) for root in self.roots()
        }

    def highest_uncertainty_skill(self, context: str = "global") -> str | None:
        """Return the skill with highest variance -- best target for active learning."""
        best = None
        best_var = -1.0
        for skill in self._all_nodes:
            belief = self.beliefs.get("skill", skill, context)
            if belief.variance > best_var:
                best_var = belief.variance
                best = skill
        return best

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self._all_nodes),
            "edges": {k: v for k, v in self._edges.items()},
            "roots": self.roots(),
        }
