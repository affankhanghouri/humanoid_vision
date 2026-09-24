"""Own worker lifetimes; keep GUI calls on the main thread."""
import threading
import time
import traceback
import cv2
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.result_store import ResultStore
from detectors.factory import create_detector
from pipeline.capture_worker import capture_worker
from pipeline.detector_worker import detector_worker
from rendering.renderer import Renderer
from tracking.tracker import MultiObjectTracker


class VisionPipeline:
    def __init__(self, config: VisionConfig):
        self.config = config

    def run(self) -> None:
        cv2.setNumThreads(self.config.opencv_threads)
        print("Loading detector...")
        detector = create_detector(self.config)
        # Backend imports can change OpenCV globals; reapply our runtime setting.
        cv2.setNumThreads(self.config.opencv_threads)
        print("Warming up detector...")
        detector.warmup()
        print("Warm-up complete.")
        frame_buffer = LatestFrameBuffer()
        result_store = ResultStore()
        stop_event = threading.Event()
        tracker = MultiObjectTracker(self.config)
        renderer = Renderer(self.config)
        failures = []
        failure_lock = threading.Lock()

        def guarded(target, *args):
            try:
                target(*args)
            except Exception as exc:
                traceback.print_exc()
                with failure_lock:
                    failures.append(exc)
                stop_event.set()

        threads = [
            threading.Thread(name="capture", target=guarded,
                             args=(capture_worker, self.config, frame_buffer, stop_event)),
            threading.Thread(name="detector", target=guarded,
                             args=(detector_worker, self.config, detector, tracker,
                                   frame_buffer, result_store, stop_event)),
        ]
        started_threads = []
        last_frame = -1
        display_frames = 0
        display_fps = 0.0
        fps_timer = time.perf_counter()
        try:
            for thread in threads:
                thread.start()
                started_threads.append(thread)
            while not stop_event.is_set():
                packet = frame_buffer.get_latest()
                if packet is None or packet.frame_id == last_frame:
                    if frame_buffer.is_finished():
                        break
                    # Pump keyboard events even when capture has no new frame.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    stop_event.wait(self.config.empty_poll_seconds if packet is None
                                    else self.config.repeat_poll_seconds)
                    continue
                last_frame = packet.frame_id
                display_frames += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    display_fps = display_frames / (now - fps_timer)
                    display_frames = 0
                    fps_timer = now
                frame = renderer.render(packet, result_store.get_latest(), display_fps)
                cv2.imshow(self.config.window_title, frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            stop_event.set()
            for thread in started_threads:
                thread.join()
            cv2.destroyAllWindows()
        if failures:
            raise RuntimeError("Vision worker failed; pipeline stopped") from failures[0]
