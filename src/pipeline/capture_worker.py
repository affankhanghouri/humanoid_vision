"""Read and pace the source, replacing the latest frame at source FPS."""
import math
import time
from threading import Event
import cv2
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.types import FramePacket


def capture_worker(config: VisionConfig, frame_buffer: LatestFrameBuffer,
                   stop_event: Event) -> None:
    cap = cv2.VideoCapture(config.video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = config.fallback_source_fps
        print(f"Source FPS: {source_fps:.2f}")
        frame_period = 1.0 / source_fps
        frame_id = 0
        next_frame_time = time.perf_counter()
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                break
            frame_id += 1
            frame_buffer.publish(FramePacket(frame_id, time.perf_counter(), frame))
            next_frame_time += frame_period
            remaining = next_frame_time - time.perf_counter()
            if remaining > 0:
                stop_event.wait(remaining)
            else:
                next_frame_time = time.perf_counter()
    finally:
        cap.release()
        frame_buffer.mark_finished()
