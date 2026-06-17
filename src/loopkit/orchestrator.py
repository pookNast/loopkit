"""Orchestrator -- composes all seven gems into one high-level API.

The orchestrator implements the "probabilistic core, deterministic shell"
pattern: probabilistic components (beliefs, CUSUM, ensemble) influence
routing and alerting decisions, while deterministic guardrails (max
iterations, budget ceiling, circuit breakers) enforce hard stops.

Usage::

    ork = Orchestrator()

    # Register models, skills, hosts
    ork.beliefs.update("model", "sonnet", success=True)
    ork.beliefs.update("model", "opus", success=True)

    # Register CUSUM metrics
    ork.cusum.register("success_rate", baseline=0.85)

    # Register skill chains
    ork.dag.add_chain(["build", "test", "deploy", "verify"])

    # Dispatch loop
    decision = ork.should_dispatch(job_priority=20, host="sandbox")
    model = ork.select_model(["haiku", "sonnet", "opus"], job_type="debug")
    ork.record_outcome("skill", "build", success=True)

    # Health report
    report = ork.health_report()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loopkit.beliefs import BeliefEngine
from loopkit.circuit_breaker import CircuitBreakerRegistry
from loopkit.cusum import CUSUMBank
from loopkit.degradation import DegradationLadder, DegradationLevel, SystemMetrics
from loopkit.ncd import SpinDetector
from loopkit.skill_dag import SkillDAG
from loopkit.store import SQLiteStore


@dataclass
class DispatchDecision:
    """Result of should_dispatch()."""

    allowed: bool
    reason: str
    degradation_level: DegradationLevel
    model_override: str | None = None
    circuit_breaker_open: bool = False


class Orchestrator:
    """Composes all seven gems into a unified orchestration layer.

    All components are optional -- use only what you need.
    """

    def __init__(
        self,
        db_path: str | None = None,
        decay_factor: float = 0.99,
    ) -> None:
        self.store = SQLiteStore(db_path) if db_path else None

        # Load from store if available, otherwise fresh
        if self.store:
            self.beliefs = self.store.load_beliefs(decay_factor)
            self.cusum = self.store.load_cusum()
            self.breakers = self.store.load_circuit_breakers()
            self.dag = self.store.load_skill_graph(self.beliefs)
        else:
            self.beliefs = BeliefEngine(decay_factor)
            self.cusum = CUSUMBank()
            self.breakers = CircuitBreakerRegistry()
            self.dag = SkillDAG(self.beliefs)

        self.degradation = DegradationLadder()
        self._spin_detectors: dict[str, SpinDetector] = {}

    # ── Core dispatch decision ──────────────────────────────────────

    def should_dispatch(
        self,
        job_priority: int,
        host: str,
        metrics: SystemMetrics | None = None,
    ) -> DispatchDecision:
        """Unified dispatch decision combining all gems.

        Returns a DispatchDecision with allowed/denied, reason, and
        any model overrides from the degradation ladder.
        """
        # 1. Circuit breaker check
        if not self.breakers.should_attempt(host):
            cb = self.breakers.get(host)
            return DispatchDecision(
                allowed=False,
                reason=f"Circuit breaker OPEN for {host} (P(healthy)={cb.p_healthy:.2f}, cooldown={cb.current_cooldown:.0f}s)",
                degradation_level=self.degradation.current,
                circuit_breaker_open=True,
            )

        # 2. Degradation level check
        if metrics:
            self.degradation.compute(metrics)

        if not self.degradation.should_dispatch(job_priority):
            pol = self.degradation.policy()
            return DispatchDecision(
                allowed=False,
                reason=f"Degradation level {self.degradation.current.name}: priority {job_priority} > max {pol.max_priority}",
                degradation_level=self.degradation.current,
            )

        # 3. Allowed -- check for model override
        model_override = self.degradation.policy().model_override
        return DispatchDecision(
            allowed=True,
            reason="OK",
            degradation_level=self.degradation.current,
            model_override=model_override,
        )

    # ── Model selection ─────────────────────────────────────────────

    def select_model(
        self,
        candidates: list[str],
        job_type: str = "general",
        min_obs: int = 5,
        fallback: str | None = None,
    ) -> str:
        """Thompson Sampling model selection with cold-start fallback.

        - If *no* candidate has ``min_obs`` observations for this job_type
          (total cold start), returns ``fallback`` (or the first candidate).
        - Otherwise delegates to :meth:`BeliefEngine.select_best` with the
          same ``min_obs``, so the forced-exploration warmup applies
          consistently: under-tried candidates are explored before pure
          Thompson exploitation takes over.
        """
        # Total cold start: nothing is warm enough to trust a sample.
        has_data = any(
            self.beliefs.get("model", m, job_type).total_obs >= min_obs
            for m in candidates
        )
        if not has_data:
            return fallback or candidates[0]

        result = self.beliefs.select_best(
            "model", candidates, context=job_type, min_obs=min_obs
        )
        return result or fallback or candidates[0]

    # ── Outcome recording ───────────────────────────────────────────

    def record_outcome(
        self,
        entity_type: str,
        entity_id: str,
        success: bool,
        context: str = "global",
        host: str | None = None,
    ) -> None:
        """Record an outcome, updating beliefs and circuit breakers."""
        self.beliefs.update(entity_type, entity_id, success, context)
        if host:
            self.breakers.record(host, success)
        if self.store:
            self.store.log_event(
                source="orchestrator",
                event="outcome",
                data={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "success": success,
                    "context": context,
                    "host": host,
                },
            )

    # ── Spin detection ──────────────────────────────────────────────

    def check_spin(
        self,
        job_id: str,
        output: str,
        epsilon: float = 0.15,
        window: int = 3,
    ) -> bool:
        """Feed an iteration output for a job. Returns True if spinning."""
        if job_id not in self._spin_detectors:
            self._spin_detectors[job_id] = SpinDetector(epsilon=epsilon, window=window)
        return self._spin_detectors[job_id].feed(output)

    # ── Chain reliability ───────────────────────────────────────────

    def chain_ok(
        self,
        chain_root: str,
        min_reliability: float = 0.5,
        context: str = "global",
    ) -> bool:
        """Check if a chain's reliability is above the minimum threshold."""
        rel = self.dag.chain_reliability(chain_root, context)
        return rel.point_estimate >= min_reliability

    # ── Health report ───────────────────────────────────────────────

    def health_report(self) -> dict:
        """Generate a full health report from all gems."""
        report: dict[str, Any] = {
            "degradation": self.degradation.to_dict(),
            "circuit_breakers": self.breakers.to_dict(),
            "cusum": self.cusum.to_dict(),
            "cusum_warnings": self.cusum.alerts(),
            "open_breakers": self.breakers.open_services(),
        }

        # Chain reliabilities
        chains = self.dag.all_chain_reliabilities()
        if chains:
            report["chains"] = {
                root: {
                    "reliability": rel.point_estimate,
                    "ci": (rel.ci_lower, rel.ci_upper),
                    "bottleneck": rel.bottleneck,
                }
                for root, rel in chains.items()
            }

        # Top/bottom beliefs
        for etype in ("model", "skill", "host"):
            ranked = self.beliefs.ranked(etype, by="mean")
            if ranked:
                report[f"top_{etype}s"] = [
                    {"id": eid, "mean": round(b.mean, 3), "obs": b.total_obs}
                    for eid, b in ranked[:5]
                ]

        # Spin detectors
        spinning = {
            jid: sd.to_dict()
            for jid, sd in self._spin_detectors.items()
            if sd.mean_ncd is not None and sd.mean_ncd < sd.epsilon
        }
        if spinning:
            report["spinning_jobs"] = spinning

        return report

    # ── Persistence ─────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all state to SQLite (if store configured), atomically."""
        if not self.store:
            return
        self.store.save_all(
            engine=self.beliefs,
            cusum=self.cusum,
            breakers=self.breakers,
            dag=self.dag,
        )

    def decay_all(self) -> None:
        """Apply daily decay to all beliefs."""
        self.beliefs.decay_all()
