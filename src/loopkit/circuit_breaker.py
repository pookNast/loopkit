"""GEM 5: Circuit Breaker with Bayesian cooldown.

Three-state FSM:
    CLOSED    -- Normal operation, tracking consecutive failures.
    OPEN      -- Fast-fail all requests; wait for cooldown.
    HALF_OPEN -- Send one probe request to test recovery.

The cooldown duration is Bayesian: longer for historically unreliable
services (low P(healthy)), shorter for usually-reliable ones.

    cooldown = base_cooldown / max(P(healthy), 0.01)
    P(healthy) = alpha / (alpha + beta)   [Beta posterior]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreaker:
    """Three-state circuit breaker with Bayesian adaptive cooldown."""

    threshold: int = 3
    base_cooldown: float = 60.0
    max_cooldown: float = 3600.0
    state: State = State.CLOSED
    failures: int = 0
    successes: int = 0
    alpha: float = 1.0    # Beta prior successes
    beta_param: float = 1.0     # Beta prior failures
    _last_fail_time: float = 0.0
    _cooldown: float = 60.0
    _total_transitions: int = 0

    @property
    def p_healthy(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def current_cooldown(self) -> float:
        return self._cooldown

    def should_attempt(self) -> bool:
        """Should we attempt a request to this service?"""
        if self.state == State.CLOSED:
            return True
        if self.state == State.OPEN:
            elapsed = time.monotonic() - self._last_fail_time
            # ponytail: monotonic clock regression guard only — upgrade: persist wall-clock + boot_id if cross-host failover needed
            # _last_fail_time==0.0 ⇒ never set (hand-constructed/corrupt state); treat as stale and probe.
            if self._last_fail_time == 0.0 or elapsed < 0 or elapsed >= self._cooldown:
                # elapsed < 0 ⇒ monotonic clock regressed (process/host
                # reboot since _last_fail_time was captured). Allow a probe
                # rather than deadlocking OPEN forever on stale timing.
                self.state = State.HALF_OPEN
                self._total_transitions += 1
                return True
            return False
        # HALF_OPEN: allow one probe
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self.alpha += 1.0
        self.successes += 1
        self.failures = 0
        if self.state != State.CLOSED:
            self._total_transitions += 1
        self.state = State.CLOSED
        self._cooldown = self.base_cooldown

    def record_failure(self) -> None:
        """Record a failed request."""
        self.beta_param += 1.0
        self.failures += 1
        self._last_fail_time = time.monotonic()

        if self.state == State.HALF_OPEN or self.failures >= self.threshold:
            self.state = State.OPEN
            self._total_transitions += 1
            # Bayesian cooldown: unreliable services wait longer
            p = max(self.p_healthy, 0.01)
            self._cooldown = min(self.max_cooldown, self.base_cooldown / p)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self.failures,
            "p_healthy": round(self.p_healthy, 4),
            "cooldown_s": round(self._cooldown, 1),
            "total_transitions": self._total_transitions,
        }


class CircuitBreakerRegistry:
    """Collection of circuit breakers keyed by service name."""

    def __init__(
        self,
        default_threshold: int = 3,
        default_base_cooldown: float = 60.0,
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_threshold = default_threshold
        self._default_base_cooldown = default_base_cooldown

    def get(self, service: str) -> CircuitBreaker:
        if service not in self._breakers:
            self._breakers[service] = CircuitBreaker(
                threshold=self._default_threshold,
                base_cooldown=self._default_base_cooldown,
            )
        return self._breakers[service]

    def should_attempt(self, service: str) -> bool:
        return self.get(service).should_attempt()

    def record(self, service: str, success: bool) -> None:
        cb = self.get(service)
        if success:
            cb.record_success()
        else:
            cb.record_failure()

    def open_services(self) -> list[str]:
        """Return names of services currently in OPEN state."""
        return [
            name for name, cb in self._breakers.items() if cb.state == State.OPEN
        ]

    def to_dict(self) -> dict:
        return {name: cb.to_dict() for name, cb in self._breakers.items()}
