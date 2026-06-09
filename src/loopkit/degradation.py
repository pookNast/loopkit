"""GEM 6: Graceful degradation ladder.

Five-level hierarchy that tells the system what to sacrifice when
resources are constrained.  Each level filters allowed job priorities
and can force model downgrades.

    Level 0 (NORMAL):      All systems go.
    Level 1 (CONSTRAINED): Pause BATCH jobs.
    Level 2 (STRESSED):    Reduce concurrency, downgrade models.
    Level 3 (CRITICAL):    Essential work only, all models cheapest.
    Level 4 (EMERGENCY):   Halt dispatch, alert human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional


class DegradationLevel(IntEnum):
    NORMAL = 0
    CONSTRAINED = 1
    STRESSED = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class LevelPolicy:
    """What actions to take at a given degradation level."""

    max_priority: int = 100  # Only allow jobs with priority <= this
    model_override: Optional[str] = None  # Force all jobs to this model
    max_concurrency: Optional[int] = None  # Override max concurrent
    alert: bool = False
    halt: bool = False


# Sensible defaults -- users can override
DEFAULT_POLICIES: dict[DegradationLevel, LevelPolicy] = {
    DegradationLevel.NORMAL: LevelPolicy(),
    DegradationLevel.CONSTRAINED: LevelPolicy(max_priority=35),
    DegradationLevel.STRESSED: LevelPolicy(
        max_priority=25, model_override="haiku", max_concurrency=2
    ),
    DegradationLevel.CRITICAL: LevelPolicy(
        max_priority=10, model_override="haiku", max_concurrency=1, alert=True
    ),
    DegradationLevel.EMERGENCY: LevelPolicy(halt=True, alert=True),
}


@dataclass
class SystemMetrics:
    """Current system state used to compute degradation level."""

    budget_remaining_pct: float = 100.0
    host_load: float = 0.0
    failed_24h: int = 0
    running: int = 0
    max_concurrent: int = 6
    p_system_healthy: float = 1.0  # From belief engine


class DegradationLadder:
    """Computes the current degradation level from system metrics.

    Pressure signals contribute points; total pressure maps to a level.
    """

    def __init__(
        self,
        policies: Optional[dict[DegradationLevel, LevelPolicy]] = None,
        thresholds: Optional[dict[str, tuple[float, int]]] = None,
    ) -> None:
        self.policies = policies or dict(DEFAULT_POLICIES)
        # (threshold, pressure_points) for each signal
        self._thresholds = thresholds or {
            "budget_low": (20.0, 2),     # budget < 20% -> +2 pressure
            "host_overloaded": (15.0, 1),  # load > 15 -> +1
            "failure_spike": (10, 1),      # >10 failures/24h -> +1
            "near_capacity": (0.85, 1),    # running/max > 85% -> +1
            "unhealthy": (0.5, 2),         # P(healthy) < 0.5 -> +2
        }
        self._current = DegradationLevel.NORMAL
        self._history: list[tuple[DegradationLevel, str]] = []

    def compute(self, metrics: SystemMetrics) -> DegradationLevel:
        """Compute degradation level from current metrics."""
        pressure = 0

        t = self._thresholds
        if metrics.budget_remaining_pct < t["budget_low"][0]:
            pressure += t["budget_low"][1]
        if metrics.host_load > t["host_overloaded"][0]:
            pressure += t["host_overloaded"][1]
        if metrics.failed_24h > t["failure_spike"][0]:
            pressure += t["failure_spike"][1]
        if metrics.max_concurrent > 0:
            utilization = metrics.running / metrics.max_concurrent
            if utilization > t["near_capacity"][0]:
                pressure += t["near_capacity"][1]
        if metrics.p_system_healthy < t["unhealthy"][0]:
            pressure += t["unhealthy"][1]

        level = DegradationLevel(min(pressure, 4))

        if level != self._current:
            self._history.append((level, f"pressure={pressure}"))
            self._current = level

        return level

    @property
    def current(self) -> DegradationLevel:
        return self._current

    def policy(self, level: Optional[DegradationLevel] = None) -> LevelPolicy:
        """Get the policy for a level (default: current)."""
        return self.policies[level or self._current]

    def should_dispatch(self, job_priority: int) -> bool:
        """Should a job with this priority be dispatched at the current level?"""
        pol = self.policy()
        if pol.halt:
            return False
        return job_priority <= pol.max_priority

    def effective_model(self, requested_model: str) -> str:
        """Apply model override if degradation level requires it."""
        pol = self.policy()
        if pol.model_override:
            return pol.model_override
        return requested_model

    def to_dict(self) -> dict:
        return {
            "level": self._current.value,
            "name": self._current.name,
            "policy": {
                "max_priority": self.policy().max_priority,
                "model_override": self.policy().model_override,
                "halt": self.policy().halt,
                "alert": self.policy().alert,
            },
            "transitions": len(self._history),
        }
