"""Native-resolution road-only Nano inference, opt-in shared-worker use."""
import time
import cv2
import numpy as np
from core.perception_state import RoadSegObservation


class RoadSegmenter:
    def __init__(self, model_path, threads=2):
        import openvino as ov
        self.model = ov.Core().compile_model(str(model_path), 'CPU', {
            'INFERENCE_NUM_THREADS': threads, 'NUM_STREAMS': 1,
            'INFERENCE_PRECISION_HINT': 'f32', 'PERFORMANCE_HINT': 'LATENCY',
        })
        if list(self.model.input(0).shape) != [1, 3, 384, 640]:
            raise ValueError('Road model must have input [1, 3, 384, 640]')
        if len(self.model.outputs) != 1 or list(self.model.output(0).shape) != [1, 2, 384, 640]:
            raise ValueError('Expected road-only output [1, 2, 384, 640]')

    def observe(self, frame, frame_id, timestamp):
        preprocessing_start = time.perf_counter()
        h, w = frame.shape[:2]
        scale = min(640 / w, 384 / h)
        rw, rh = round(w * scale), round(h * scale)
        left, top = (640-rw)//2, (384-rh)//2
        padded = np.full((384, 640, 3), 114, np.uint8)
        padded[top:top+rh, left:left+rw] = cv2.resize(frame, (rw, rh))
        tensor = cv2.dnn.blobFromImage(padded, 1/255., swapRB=True)
        preprocessing_ms = (time.perf_counter()-preprocessing_start)*1000
        start = time.perf_counter()
        output = self.model([tensor])[self.model.output(0)]
        inference_ms = (time.perf_counter()-start)*1000
        postprocessing_start = time.perf_counter()
        if not np.isfinite(output).all():
            raise ValueError('Non-finite road prediction')
        mask = (output[0, 1] > output[0, 0]).astype(np.uint8)
        postprocessing_ms = (time.perf_counter()-postprocessing_start)*1000
        return RoadSegObservation(
            frame_id, timestamp, time.perf_counter(), inference_ms, mask,
            (left, top, rw, rh), (w, h), preprocessing_ms, postprocessing_ms,
        )

    def warmup(self, runs=5):
        durations = []
        for _ in range(runs):
            start = time.perf_counter()
            self.observe(np.zeros((720, 1152, 3), np.uint8), 0, start)
            durations.append(time.perf_counter()-start)
        return float(np.percentile(durations, 95))
