"""Definitions for heavy compute tasks and their scheduling policy."""
from dataclasses import dataclass

from core.types import FramePacket


@dataclass(frozen=True)
class TaskPolicy:
    task_type: str
    # Larger number = more important.
    priority: int
    # Minimum time between starts; zero runs whenever compute is available.
    min_interval: float
    # Drop input older than this many seconds.
    max_input_age: float


@dataclass(frozen=True)
class ComputeTask:
    task_type: str
    source_frame_id: int
    source_timestamp: float
    # Shared reference to the source frame; do not modify packet.frame.
    packet: FramePacket
