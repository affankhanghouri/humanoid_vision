"""Bounded latency statistics; each instance belongs to its measuring thread."""
from collections import deque
import numpy as np


class RollingLatency:
    def __init__(self, maxlen: int = 100):
        if maxlen < 1:
            raise ValueError("metrics window must be positive")
        self._values = deque(maxlen=maxlen)

    def add(self, value_ms: float) -> None:
        self._values.append(value_ms)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def last(self) -> float:
        return self._values[-1] if self._values else 0.0

    @property
    def average(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self._values)) if self._values else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._values, 95)) if self._values else 0.0
