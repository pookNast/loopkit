"""GEM 2: CUSUM (Cumulative Sum) anomaly detection.

Detects sustained drift in system metrics that threshold-based alerts miss.
A success rate dropping from 85% to 70% over 3 days triggers CUSUM long
before any single-point threshold fires.

Mathematical basis:
    Target:       mu_0 = baseline metric
    Allowance:    K = acceptable_shift / 2
    Upper CUSUM:  S_n+ = max(0, S_(n-1)+ + (x_n - mu_0 - K))
    Lower CUSUM:  S_n- = max(0, S_(n-1)- - (x_n - mu_0 + K))
    Alert:        if S_n+ > H or S_n- > H
    Reset:        S+ = S- = 0 after alert
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUSUM:
    """Single-metric CUSUM detector."""

    baseline: float
    allowance_k: float = 0.05
    threshold_h: float = 4.0
    upper: float = 0.0
    lower: float = 0.0
    observation_count: int = 0
    alert_count: int = 0
    _calibrating: bool = True
    _cal_sum: float = 0.0
    _cal_n: int = 0
    _cal_target: int = 30

    def update(self, observation: float) -> bool:
        """Feed an observation. Returns True if an alert fires."""
        # Auto-calibrate baseline from first N observations
        if self._calibrating and self._cal_n < self._cal_target:
            self._cal_sum += observation
            self._cal_n += 1
            if self._cal_n >= self._cal_target:
                self.baseline = self._cal_sum / self._cal_n
                self._calibrating = False
            return False

        self.observation_count += 1
        self.upper = max(0.0, self.upper + (observation - self.baseline - self.allowance_k))
        self.lower = max(0.0, self.lower - (observation - self.baseline + self.allowance_k))

        alert = self.upper > self.threshold_h or self.lower > self.threshold_h
        if alert:
            self.upper = 0.0
            self.lower = 0.0
            self.alert_count += 1
        return alert

    @property
    def distance_to_threshold(self) -> float:
        """How close the worst-side CUSUM is to the alert threshold (0.0 = alert, 1.0 = baseline)."""
        worst = max(self.upper, self.lower)
        if self.threshold_h == 0:
            return 0.0
        return max(0.0, 1.0 - worst / self.threshold_h)

    def reset(self) -> None:
        self.upper = 0.0
        self.lower = 0.0

    def to_dict(self) -> dict:
        return {
            "baseline": round(self.baseline, 4),
            "upper": round(self.upper, 4),
            "lower": round(self.lower, 4),
            "distance_to_threshold": round(self.distance_to_threshold, 4),
            "observation_count": self.observation_count,
            "alert_count": self.alert_count,
        }


class CUSUMBank:
    """Collection of named CUSUM detectors for multiple metrics."""

    def __init__(self) -> None:
        self._detectors: dict[str, CUSUM] = {}

    def register(
        self,
        name: str,
        baseline: float,
        allowance_k: float = 0.05,
        threshold_h: float = 4.0,
        auto_calibrate: bool = True,
        calibration_window: int = 30,
    ) -> CUSUM:
        detector = CUSUM(
            baseline=baseline,
            allowance_k=allowance_k,
            threshold_h=threshold_h,
            _calibrating=auto_calibrate,
            _cal_target=calibration_window,
        )
        self._detectors[name] = detector
        return detector

    def update(self, name: str, observation: float) -> bool:
        """Update a named detector. Returns True on alert."""
        if name not in self._detectors:
            raise KeyError(f"Unknown metric: {name}")
        return self._detectors[name].update(observation)

    def update_many(self, observations: dict[str, float]) -> dict[str, bool]:
        """Update multiple detectors at once. Returns {name: alerted}."""
        return {name: self.update(name, val) for name, val in observations.items()}

    def get(self, name: str) -> CUSUM:
        return self._detectors[name]

    def alerts(self) -> list[str]:
        """Return names of detectors that are >80% toward threshold."""
        return [
            name
            for name, d in self._detectors.items()
            if d.distance_to_threshold < 0.2
        ]

    def to_dict(self) -> dict:
        return {name: d.to_dict() for name, d in self._detectors.items()}
