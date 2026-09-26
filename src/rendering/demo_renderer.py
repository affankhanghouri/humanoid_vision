"""LinkedIn/demo renderer.

This does not change perception.
It only changes how perception is shown.
"""

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


from config import VisionConfig
from tracking.motion import BoxMotionHistory


from core.perception_state import (
    PerceptionEntity,
    PerceptionState,
)


from core.types import (
    FramePacket,
)


from rendering.color import (
    WHITE,
    LIGHT_TEXT,
    CYAN,
    GREEN,
    color_for_entity,
)


from rendering.entity_history import (
    EntityHistoryStore,
)


from rendering.overlays import (
    dark_panel,
    draw_corner_box,
    draw_entity_label,
    draw_motion_trail,
    draw_pose,
    draw_velocity_arrow,
)


from tracking.smoothing import (
    DisplayBoxSmoother,
    DisplayPoseSmoother,
)


@dataclass
class EntityView:

    entity: PerceptionEntity

    box: tuple[
        int,
        int,
        int,
        int,
    ]

    center: tuple[
        int,
        int,
    ]

    color: tuple[
        int,
        int,
        int,
    ]

    pose_ready: bool

    speed: float

    projection_box: tuple[float, float, float, float]


def predicted_box(
    entity,
    age,
    config,
):

    prediction_age = min(
        max(
            age,
            0.0,
        ),
        config.max_prediction_age,
    )

    center_x = (
        entity.center_x
        + entity.velocity_x
        * prediction_age
    )

    center_y = (
        entity.center_y
        + entity.velocity_y
        * prediction_age
    )

    return (
        center_x
        - entity.width / 2,

        center_y
        - entity.height / 2,

        center_x
        + entity.width / 2,

        center_y
        + entity.height / 2,
    )


def clamp_box(
    frame,
    box,
):

    height, width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = box

    return (
        max(
            0,
            min(
                int(x1),
                width - 1,
            ),
        ),

        max(
            0,
            min(
                int(y1),
                height - 1,
            ),
        ),

        max(
            0,
            min(
                int(x2),
                width - 1,
            ),
        ),

        max(
            0,
            min(
                int(y2),
                height - 1,
            ),
        ),
    )


def movement_text(
    entity,
):

    vx = entity.velocity_x
    vy = entity.velocity_y

    speed = math.hypot(
        vx,
        vy,
    )

    if speed < 12.0:

        return "STABLE"

    if abs(vx) > abs(vy):

        if vx > 0:
            return "RIGHT"

        return "LEFT"

    if vy > 0:
        return "DOWN"

    return "UP"


class DemoRenderer:

    def __init__(
        self,
        config: VisionConfig,
    ):

        print(
            ">>> DEMO RENDERER INITIALIZED <<<"
        )

        self.config = config
        self.motion = BoxMotionHistory(config)

        self.box_smoother = (
            DisplayBoxSmoother(
                config
            )
        )

        self.pose_smoother = (
            DisplayPoseSmoother(
                config
            )
        )

        self.history = (
            EntityHistoryStore(
                max_points=24
            )
        )

        self.detector_history = deque(
            maxlen=45
        )

        self.pose_history = deque(
            maxlen=45
        )

    # ==================================================
    # LEFT METRIC PANEL
    # ==================================================

    def draw_metrics(
        self,
        frame,
        state,
        display_fps,
        state_age,
    ):

        x = 14
        y = 14

        panel_width = 280
        panel_height = 255

        dark_panel(
            frame,
            x,
            y,
            panel_width,
            panel_height,
            alpha=0.84,
            border_color=CYAN,
        )

        cv2.putText(
            frame,
            "SYSTEM METRICS",
            (
                x + 14,
                y + 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            CYAN,
            2,
            cv2.LINE_AA,
        )

        detector_hz = 0.0
        pose_hz = 0.0

        detector_ms = 0.0
        pose_ms = 0.0

        latency = 0.0

        entity_count = 0

        if state is not None:

            detector_hz = (
                state.detector_hz
            )

            pose_hz = (
                state.pose_hz
            )

            detector_ms = (
                state.inference_ms
            )

            pose_ms = (
                state.pose_inference_ms
            )

            latency = (
                state.processing_latency_ms
            )

            entity_count = len(
                state.entities
            )

            self.detector_history.append(
                detector_ms
            )

            self.pose_history.append(
                pose_ms
            )

        age_ms = 0.0

        if state_age is not None:

            age_ms = (
                state_age
                * 1000.0
            )

        rows = (
            (
                "DISPLAY FPS",
                f"{display_fps:.1f}",
            ),
            (
                "DETECTION",
                f"{detector_hz:.2f} Hz",
            ),
            (
                "POSE",
                f"{pose_hz:.2f} Hz",
            ),
            (
                "YOLO AI",
                f"{detector_ms:.0f} ms",
            ),
            (
                "POSE AI",
                f"{pose_ms:.0f} ms",
            ),
            (
                "STATE AGE",
                f"{age_ms:.0f} ms",
            ),
            (
                "ENTITIES",
                f"{entity_count}",
            ),
            (
                "LATENCY",
                f"{latency:.0f} ms",
            ),
        )

        for index, (
            name,
            value,
        ) in enumerate(rows):

            yy = (
                y
                + 52
                + index * 21
            )

            cv2.putText(
                frame,
                name,
                (
                    x + 14,
                    yy,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                LIGHT_TEXT,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                value,
                (
                    x + 155,
                    yy,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                GREEN,
                1,
                cv2.LINE_AA,
            )

        self.draw_graph(
            frame,
            x + 12,
            y + 220,
            panel_width - 24,
            25,
        )

    # ==================================================
    # SMALL GRAPH
    # ==================================================

    def draw_graph(
        self,
        frame,
        x,
        y,
        width,
        height,
    ):

        cv2.rectangle(
            frame,
            (
                x,
                y,
            ),
            (
                x + width,
                y + height,
            ),
            (
                30,
                45,
                55,
            ),
            1,
        )

        self.draw_series(
            frame,
            self.detector_history,
            x,
            y,
            width,
            height,
            CYAN,
        )

        self.draw_series(
            frame,
            self.pose_history,
            x,
            y,
            width,
            height,
            (
                0,
                150,
                255,
            ),
        )

    def draw_series(
        self,
        frame,
        values,
        x,
        y,
        width,
        height,
        color,
    ):

        if len(values) < 2:
            return

        maximum = max(
            max(values),
            1.0,
        )

        points = []

        count = len(values)

        for index, value in enumerate(
            values
        ):

            px = int(
                x
                + (
                    index
                    / max(
                        count - 1,
                        1,
                    )
                )
                * width
            )

            py = int(
                y
                + height
                - (
                    value
                    / maximum
                )
                * height
            )

            points.append(
                (
                    px,
                    py,
                )
            )

        if len(points) > 1:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False,
                          color, 1, cv2.LINE_AA)

    # ==================================================
    # BUILD ENTITY VIEWS
    # ==================================================

    def build_views(self, frame, packet, state, state_age):
        self.motion.update(packet)
        views = []
        for entity in state.entities:
            if not entity.visible_at(packet.timestamp, self.config.max_coast_age):
                continue
            source_box = predicted_box(entity, 0, self.config)
            raw_box = self.motion.project(entity.entity_id, source_box, state.source_timestamp)
            if raw_box is None:
                raw_box = predicted_box(entity, state_age, self.config)
            smooth_box = self.box_smoother.smooth(
                entity.entity_id, raw_box, timestamp=packet.timestamp)
            box = clamp_box(frame, smooth_box)
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            pose_ready = (
                entity.class_name == "person" and entity.pose is not None
                and entity.pose.meta.is_valid_for(
                    packet.frame_id, packet.timestamp, self.config.pose_max_age))
            views.append(EntityView(
                entity=entity, box=box, center=((x1+x2)//2, (y1+y2)//2),
                color=color_for_entity(entity.entity_id), pose_ready=pose_ready,
                speed=math.hypot(entity.velocity_x, entity.velocity_y),
                projection_box=tuple(smooth_box),
            ))
        return views

    # ==================================================
    # DRAW ENTITIES
    # ==================================================

    def draw_entities(
        self,
        frame,
        packet,
        views,
    ):

        active_ids = []
        pose_ids = []

        for view in views:

            entity = view.entity

            active_ids.append(
                entity.entity_id
            )

            self.history.update(
                entity_id=(
                    entity.entity_id
                ),
                center=view.center,
                timestamp=(
                    packet.timestamp
                ),
            )

            trail = (
                self.history.trail(
                    entity.entity_id
                )
            )

            draw_motion_trail(
                frame,
                trail,
                view.color,
            )

            draw_velocity_arrow(
                frame,
                view.center,
                entity.velocity_x,
                entity.velocity_y,
                view.color,
            )

            pose_visible = False
            if view.pose_ready:
                pose_ids.append(entity.entity_id)
                pose_visible = draw_pose(
                    frame, entity, view.projection_box, packet.timestamp,
                    self.pose_smoother, self.config, view.color, packet.frame_id)

            draw_corner_box(
                frame,
                view.box,
                view.color,
                thickness=2,
            )

            draw_entity_label(
                frame,
                view.box,
                entity,
                view.color,
                pose_visible,
            )

        self.box_smoother.keep_only(
            active_ids
        )

        self.pose_smoother.keep_only(pose_ids)
        self.history.keep_only(active_ids)

        self.history.prune(
            timestamp=(
                packet.timestamp
            ),
            max_missing_age=2.0,
        )

    # ==================================================
    # SELECT MAIN TARGET
    # ==================================================

    def select_target(
        self,
        views,
    ):

        if not views:
            return None

        # Person with pose first
        choices = [
            view
            for view in views
            if (
                view.entity.class_name
                == "person"
                and view.pose_ready
            )
        ]

        if choices:

            return max(
                choices,
                key=self.box_area,
            )

        # Any person
        choices = [
            view
            for view in views
            if (
                view.entity.class_name
                == "person"
            )
        ]

        if choices:

            return max(
                choices,
                key=self.box_area,
            )

        # Otherwise biggest object
        return max(
            views,
            key=self.box_area,
        )

    @staticmethod
    def box_area(
        view,
    ):

        x1, y1, x2, y2 = (
            view.box
        )

        return (
            max(
                0,
                x2 - x1,
            )
            *
            max(
                0,
                y2 - y1,
            )
        )

    # ==================================================
    # RIGHT TARGET CARD
    # ==================================================

    def draw_target_card(
        self,
        frame,
        packet,
        target,
    ):

        if target is None:
            return

        frame_height, frame_width = (
            frame.shape[:2]
        )

        width = 340
        height = 205
        if frame_width < width + 24 or frame_height < height + 14:
            return

        x = max(
            10,
            frame_width
            - width
            - 14,
        )

        y = 14

        dark_panel(
            frame,
            x,
            y,
            width,
            height,
            alpha=0.86,
            border_color=(
                target.color
            ),
        )

        entity = target.entity

        title = (
            f"TRACKED TARGET: "
            f"{entity.class_name.upper()} "
            f"#{entity.entity_id}"
        )

        cv2.putText(
            frame,
            title,
            (
                x + 12,
                y + 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            target.color,
            2,
            cv2.LINE_AA,
        )

        # ------------------------------
        # thumbnail
        # ------------------------------

        bx1, by1, bx2, by2 = (
            target.box
        )

        bx1 = max(
            0,
            bx1,
        )

        by1 = max(
            0,
            by1,
        )

        bx2 = min(
            frame_width,
            bx2,
        )

        by2 = min(
            frame_height,
            by2,
        )

        crop = packet.frame[
            by1:by2,
            bx1:bx2,
        ]

        thumb_x = x + 12
        thumb_y = y + 40

        thumb_width = 105
        thumb_height = 145

        if crop.size > 0:

            resized = cv2.resize(
                crop,
                (
                    thumb_width,
                    thumb_height,
                ),
            )

            frame[
                thumb_y:
                thumb_y + thumb_height,

                thumb_x:
                thumb_x + thumb_width
            ] = resized

        cv2.rectangle(
            frame,
            (
                thumb_x,
                thumb_y,
            ),
            (
                thumb_x
                + thumb_width,

                thumb_y
                + thumb_height,
            ),
            target.color,
            2,
            cv2.LINE_AA,
        )

        # ------------------------------
        # data
        # ------------------------------

        track_age = (
            self.history.track_age(
                entity.entity_id,
                packet.timestamp,
            )
        )

        if target.pose_ready:

            pose_text = "READY"

        else:

            pose_text = "WAIT"

        rows = (
            (
                "ID",
                str(
                    entity.entity_id
                ),
            ),
            (
                "CONF",
                f"{entity.confidence:.2f}",
            ),
            (
                "POSE",
                pose_text,
            ),
            (
                "TRACK AGE",
                f"{track_age:.1f} s",
            ),
        )

        text_x = (
            thumb_x
            + thumb_width
            + 14
        )

        for index, (
            label,
            value,
        ) in enumerate(rows):

            yy = (
                y
                + 58
                + index * 23
            )

            cv2.putText(
                frame,
                (
                    f"{label}:"
                ),
                (
                    text_x,
                    yy,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (
                    180,
                    195,
                    210,
                ),
                1,
                cv2.LINE_AA,
            )

            value_color = LIGHT_TEXT

            if (
                label == "POSE"
                and target.pose_ready
            ):

                value_color = (
                    target.color
                )

            cv2.putText(
                frame,
                value,
                (
                    text_x + 78,
                    yy,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                value_color,
                1,
                cv2.LINE_AA,
            )

        # ------------------------------
        # connector
        # ------------------------------

        cv2.line(
            frame,
            (
                x,
                y + height,
            ),
            target.center,
            target.color,
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            target.center,
            4,
            target.color,
            -1,
            cv2.LINE_AA,
        )

    # ==================================================
    # MOTION MAP
    # ==================================================

    def draw_motion_map(
        self,
        frame,
        views,
    ):

        frame_height, frame_width = (
            frame.shape[:2]
        )

        width = 300
        height = 145

        x = max(
            10,
            frame_width
            - width
            - 14,
        )

        y = max(
            340,
            frame_height
            - height
            - 14,
        )

        dark_panel(
            frame,
            x,
            y,
            width,
            height,
            alpha=0.84,
            border_color=CYAN,
        )

        cv2.putText(
            frame,
            "IMAGE-SPACE TRACKING MAP",
            (
                x + 11,
                y + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            CYAN,
            1,
            cv2.LINE_AA,
        )

        map_x1 = x + 12
        map_y1 = y + 30

        map_x2 = (
            x
            + width
            - 12
        )

        map_y2 = (
            y
            + height
            - 12
        )

        # grid

        for index in range(
            1,
            6,
        ):

            gx = int(
                map_x1
                + (
                    map_x2
                    - map_x1
                )
                * index
                / 6
            )

            cv2.line(
                frame,
                (
                    gx,
                    map_y1,
                ),
                (
                    gx,
                    map_y2,
                ),
                (
                    30,
                    45,
                    55,
                ),
                1,
            )

        for index in range(
            1,
            4,
        ):

            gy = int(
                map_y1
                + (
                    map_y2
                    - map_y1
                )
                * index
                / 4
            )

            cv2.line(
                frame,
                (
                    map_x1,
                    gy,
                ),
                (
                    map_x2,
                    gy,
                ),
                (
                    30,
                    45,
                    55,
                ),
                1,
            )

        # entities

        for view in views:

            trail = (
                self.history.trail(
                    view.entity.entity_id
                )
            )

            points = []

            for px, py in trail:

                mapped_x = int(
                    map_x1
                    + (
                        px
                        / max(
                            frame_width,
                            1,
                        )
                    )
                    * (
                        map_x2
                        - map_x1
                    )
                )

                mapped_y = int(
                    map_y1
                    + (
                        py
                        / max(
                            frame_height,
                            1,
                        )
                    )
                    * (
                        map_y2
                        - map_y1
                    )
                )

                points.append(
                    (
                        mapped_x,
                        mapped_y,
                    )
                )

            if len(points) > 1:
                cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False,
                              view.color, 1, cv2.LINE_AA)

            if points:

                cv2.circle(
                    frame,
                    points[-1],
                    4,
                    view.color,
                    -1,
                    cv2.LINE_AA,
                )

    # ==================================================
    # BOTTOM STATUS BAR
    # ==================================================

    def draw_status_bar(
        self,
        frame,
        state,
    ):

        height, width = (
            frame.shape[:2]
        )

        bar_height = 31

        dark_panel(frame, 0, height-bar_height, width, bar_height,
                   alpha=0.85, background=(5, 11, 17))

        cv2.line(
            frame,
            (
                0,
                height
                - bar_height,
            ),
            (
                width,
                height
                - bar_height,
            ),
            CYAN,
            1,
        )

        entity_count = 0

        if state is not None:

            entity_count = len(
                state.entities
            )

        text = (f"TRACKING FRESH    POSE SCHEDULED    ACTIVE ENTITIES: {entity_count}"
                if state is not None else "WAITING FOR FRESH TRACKING    ACTIVE ENTITIES: 0")

        cv2.putText(
            frame,
            text,
            (
                18,
                height - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (
                180,
                230,
                210,
            ),
            1,
            cv2.LINE_AA,
        )

    # ==================================================
    # MAIN RENDER
    # ==================================================

    def render(
        self,
        packet: FramePacket,
        state: PerceptionState | None,
        display_fps: float,
    ):

        frame = (
            packet.frame.copy()
        )

        self.motion.update(packet)
        state_age = state.age_at(packet.timestamp) if state is not None else None
        valid = state is not None and state.is_valid_for(
            packet.frame_id, packet.timestamp, self.config.max_render_age)
        views = []
        if valid:
            views = self.build_views(frame, packet, state, state_age)
            self.draw_entities(frame, packet, views)
        else:
            self.box_smoother.keep_only(())
            self.pose_smoother.keep_only(())
            self.history.keep_only(())
        display_state = replace(state, entities=tuple(v.entity for v in views)) if valid else None

        # ----------------------------------------------
        # UI AFTER scene overlays
        # ----------------------------------------------

        self.draw_metrics(
            frame,
            display_state,
            display_fps,
            state_age,
        )

        target = (
            self.select_target(
                views
            )
        )

        self.draw_target_card(
            frame,
            packet,
            target,
        )

        self.draw_motion_map(
            frame,
            views,
        )

        self.draw_status_bar(
            frame,
            display_state,
        )

        return frame
