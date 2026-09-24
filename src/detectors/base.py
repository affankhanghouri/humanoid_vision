"""Minimal synchronous interface, scheduled independently by a worker."""
from abc import ABC, abstractmethod
import numpy as np
from core.types import Detection


class Detector(ABC):
    @abstractmethod
    def warmup(self) -> None:
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Read a shared frame without modifying it."""
        pass
