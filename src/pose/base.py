"""Synchronous pose interface for the shared compute worker."""
from abc import ABC, abstractmethod
import numpy as np
from core.pose_types import PosePerson


class PoseEstimator(ABC):
    @abstractmethod
    def warmup(self) -> None:
        pass

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> tuple[PosePerson, ...]:
        """Read the shared frame without modifying it."""
        pass
