"""One shared frame reference, replaced atomically; no queue or polling copies."""
from threading import Lock
from core.types import FramePacket


class LatestFrameBuffer:
    def __init__(self):
        self._lock = Lock()
        self._packet: FramePacket | None = None
        self._finished = False

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            self._packet = packet

    def get_latest(self) -> FramePacket | None:
        with self._lock:
            return self._packet

    def mark_finished(self) -> None:
        with self._lock:
            self._finished = True

    def is_finished(self) -> bool:
        with self._lock:
            return self._finished
