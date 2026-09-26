"""Own the complete real-time perception pipeline."""

import threading
import time
import traceback

import cv2


from config import VisionConfig


from core.latest_frame import (
    LatestFrameBuffer,
)


from core.perception_store import (
    PerceptionStore,
)


from detectors.factory import (
    create_detector,
)


from pipeline.capture_worker import (
    capture_worker,
)


from pipeline.compute_worker import (
    compute_worker,
)


from pose.ultralytics_pose import (
    UltralyticsPoseEstimator,
)


from rendering.renderer import (
    Renderer,
)


from rendering.demo_renderer import (
    DemoRenderer,
)


from scheduling.scheduler import (
    ComputeScheduler,
    scheduler_worker,
)


from scheduling.task import (
    TaskPolicy,
)


from tracking.tracker import (
    MultiObjectTracker,
)


class VisionPipeline:

    def __init__(
        self,
        config: VisionConfig,
    ):

        self.config = config

    def run(
        self,
    ) -> None:

        # ==================================================
        # OpenCV runtime
        # ==================================================

        cv2.setNumThreads(
            self.config.opencv_threads
        )

        # ==================================================
        # Detection model
        # ==================================================

        print(
            "Loading detector..."
        )

        detector = create_detector(
            self.config
        )

        cv2.setNumThreads(
            self.config.opencv_threads
        )

        print(
            "Warming up detector..."
        )

        detector.warmup()

        print(
            "Detector warm-up complete."
        )

        # ==================================================
        # Pose model
        # ==================================================

        print(
            "Loading pose model..."
        )

        pose_estimator = (
            UltralyticsPoseEstimator(

                model_path=(
                    self.config
                    .pose_model_path
                ),

                image_size=(
                    self.config
                    .pose_image_size
                ),

                confidence=(
                    self.config
                    .pose_confidence
                ),

                warmup_runs=(
                    self.config
                    .pose_warmup_runs
                ),
                onnx_threads=self.config.onnx_intra_op_threads,
            )
        )

        cv2.setNumThreads(
            self.config.opencv_threads
        )

        print(
            "Warming up pose..."
        )

        pose_estimator.warmup()

        print(
            "Pose warm-up complete."
        )

        print(
            "Warm-up complete."
        )

        # ==================================================
        # Shared stores
        # ==================================================

        frame_buffer = (
            LatestFrameBuffer()
        )

        perception_store = (
            PerceptionStore()
        )

        stop_event = (
            threading.Event()
        )

        tracker = (
            MultiObjectTracker(
                self.config
            )
        )

        # ==================================================
        # Scheduler
        # ==================================================

        scheduler = ComputeScheduler(
            max_tracking_age=self.config.max_render_age,
            policies=(

                TaskPolicy(

                    task_type=(
                        "detection"
                    ),

                    priority=100,

                    min_interval=(
                        self.config
                        .detection_min_interval
                    ),

                    max_input_age=(
                        self.config
                        .detection_max_input_age
                    ),
                ),

                TaskPolicy(

                    task_type="pose",

                    priority=70,

                    min_interval=(
                        self.config
                        .pose_min_interval
                    ),

                    max_input_age=(
                        self.config
                        .pose_max_input_age
                    ),
                ),
            )
        )

        # ==================================================
        # Renderer
        # ==================================================

        if (
            self.config.render_mode
            == "demo"
        ):

            renderer = DemoRenderer(
                self.config
            )

        else:

            renderer = Renderer(
                self.config
            )

        print(
            f"Renderer active: "
            f"{type(renderer).__name__}"
        )

        # ==================================================
        # Failure handling
        # ==================================================

        failures = []

        failure_lock = (
            threading.Lock()
        )

        def guarded(
            target,
            *args,
        ):

            try:

                target(
                    *args
                )

            except Exception as exc:

                traceback.print_exc()

                with failure_lock:

                    failures.append(
                        exc
                    )

                stop_event.set()

                scheduler.wake()

        # ==================================================
        # Threads
        # ==================================================

        capture_thread = (
            threading.Thread(

                name="capture",

                target=guarded,

                args=(
                    capture_worker,
                    self.config,
                    frame_buffer,
                    stop_event,
                ),
            )
        )

        scheduler_thread = (
            threading.Thread(

                name="scheduler",

                target=guarded,

                args=(
                    scheduler_worker,
                    self.config,
                    frame_buffer,
                    perception_store,
                    scheduler,
                    stop_event,
                ),
            )
        )

        compute_thread = (
            threading.Thread(

                name="compute",

                target=guarded,

                args=(
                    compute_worker,
                    self.config,
                    scheduler,
                    detector,
                    pose_estimator,
                    tracker,
                    perception_store,
                    stop_event,
                ),
            )
        )

        threads = (
            capture_thread,
            scheduler_thread,
            compute_thread,
        )

        started_threads = []

        # ==================================================
        # Display FPS
        # ==================================================

        last_frame_id = -1

        display_frames = 0

        display_fps = 0.0

        fps_start = (
            time.perf_counter()
        )

        # ==================================================
        # Start
        # ==================================================

        try:
            cv2.namedWindow(
                self.config.window_title,
                cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL,
            )
            cv2.setWindowProperty(
                self.config.window_title,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

            for thread in threads:

                thread.start()

                started_threads.append(
                    thread
                )

            # ==============================================
            # MAIN DISPLAY LOOP
            # ==============================================

            while not stop_event.is_set():

                packet = (
                    frame_buffer
                    .get_latest()
                )

                # ------------------------------------------
                # no frame
                # ------------------------------------------

                if packet is None:

                    if (
                        frame_buffer
                        .is_finished()
                    ):
                        break

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key in (
                        ord("q"),
                        27,
                    ):
                        break

                    stop_event.wait(
                        self.config
                        .empty_poll_seconds
                    )

                    continue

                # ------------------------------------------
                # same frame
                # ------------------------------------------

                if (
                    packet.frame_id
                    == last_frame_id
                ):

                    if (
                        frame_buffer
                        .is_finished()
                    ):
                        break

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key in (
                        ord("q"),
                        27,
                    ):
                        break

                    stop_event.wait(
                        self.config
                        .repeat_poll_seconds
                    )

                    continue

                last_frame_id = (
                    packet.frame_id
                )

                # ------------------------------------------
                # FPS
                # ------------------------------------------

                display_frames += 1

                now = (
                    time.perf_counter()
                )

                elapsed = (
                    now
                    - fps_start
                )

                if elapsed >= 1.0:

                    display_fps = (
                        display_frames
                        / elapsed
                    )

                    display_frames = 0

                    fps_start = now

                # ------------------------------------------
                # STATE
                # ------------------------------------------

                state = (
                    perception_store
                    .get_latest()
                )

                # ------------------------------------------
                # RENDER
                # ------------------------------------------

                frame = renderer.render(
                    packet,
                    state,
                    display_fps,
                )

                cv2.imshow(
                    self.config
                    .window_title,
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    break

        # ==================================================
        # Shutdown
        # ==================================================

        finally:

            stop_event.set()

            scheduler.wake()

            for thread in (
                started_threads
            ):

                thread.join()

            cv2.destroyAllWindows()

        # ==================================================
        # Worker failure
        # ==================================================

        if failures:

            raise RuntimeError(
                "Vision worker failed; "
                "pipeline stopped"
            ) from failures[0]