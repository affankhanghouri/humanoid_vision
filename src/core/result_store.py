"""Atomic publication of immutable tracking snapshots."""
from threading import Lock
from core.types import TrackingResult


class ResultStore:
    def __init__(self):
        self._lock = Lock()
        self._result: TrackingResult | None = None

    def publish(self, result: TrackingResult) -> None:
        with self._lock:
            self._result = result

    def get_latest(self) -> TrackingResult | None:
        with self._lock:
            return self._result
