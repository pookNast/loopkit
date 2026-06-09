"""GEM 4: Normalized Compression Distance for loop spin detection.

Detects when an agent loop is producing repetitive output by measuring
information-theoretic similarity between consecutive iterations using gzip.

NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

where C(.) = len(gzip.compress(.))

NCD ~ 0.0 => outputs are nearly identical (spinning)
NCD ~ 1.0 => outputs are maximally different (novel)
"""

from __future__ import annotations

import gzip
from collections import deque
from dataclasses import dataclass, field


def ncd(x: str, y: str) -> float:
    """Compute Normalized Compression Distance between two strings.

    Returns a value in [0, 1] where 0 means identical and 1 means
    maximally different.  Uses gzip from stdlib -- no external deps.
    """
    if not x and not y:
        return 0.0
    bx = x.encode("utf-8", errors="replace")
    by = y.encode("utf-8", errors="replace")
    cx = len(gzip.compress(bx, compresslevel=6))
    cy = len(gzip.compress(by, compresslevel=6))
    cxy = len(gzip.compress(bx + by, compresslevel=6))
    denominator = max(cx, cy)
    if denominator == 0:
        return 0.0
    return (cxy - min(cx, cy)) / denominator


@dataclass
class SpinDetector:
    """Detects when an agent loop is spinning (producing near-identical output).

    Maintains a sliding window of recent outputs and computes mean NCD.
    When mean NCD drops below epsilon, the loop is spinning.

    Args:
        epsilon: NCD threshold below which the loop is considered spinning.
        window: Number of consecutive NCD values to average.
    """

    epsilon: float = 0.15
    window: int = 3
    _outputs: deque = field(default_factory=lambda: deque(maxlen=50))
    _ncds: deque = field(default_factory=lambda: deque(maxlen=50))
    spin_count: int = 0

    def feed(self, output: str) -> bool:
        """Feed a new iteration output. Returns True if spinning detected."""
        if self._outputs:
            d = ncd(self._outputs[-1], output)
            self._ncds.append(d)
        self._outputs.append(output)

        if len(self._ncds) < self.window:
            return False

        recent = list(self._ncds)[-self.window :]
        mean_ncd = sum(recent) / len(recent)
        is_spinning = mean_ncd < self.epsilon

        if is_spinning:
            self.spin_count += 1

        return is_spinning

    @property
    def last_ncd(self) -> float | None:
        return self._ncds[-1] if self._ncds else None

    @property
    def mean_ncd(self) -> float | None:
        if len(self._ncds) < self.window:
            return None
        recent = list(self._ncds)[-self.window :]
        return sum(recent) / len(recent)

    def to_dict(self) -> dict:
        return {
            "last_ncd": round(self.last_ncd, 4) if self.last_ncd is not None else None,
            "mean_ncd": round(self.mean_ncd, 4) if self.mean_ncd is not None else None,
            "spin_count": self.spin_count,
            "outputs_seen": len(self._outputs),
            "is_spinning": (
                self.mean_ncd is not None and self.mean_ncd < self.epsilon
            ),
        }
