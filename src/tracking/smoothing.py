"""Display-only smoothing."""

import numpy as np

from config import VisionConfig


class DisplayBoxSmoother:

    def __init__(
        self,
        config: VisionConfig = VisionConfig(),
    ):

        self.config = config

        self.boxes = {}
        self.timestamps = {}

    def smooth(
        self,
        track_id,
        target_box,
        timestamp=None,
    ):

        x1, y1, x2, y2 = (
            target_box
        )

        target = np.array(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=np.float32,
        )

        # Alphas are calibrated at 24 FPS. Elapsed time keeps behavior stable
        # when the display skips frames; duplicate packets must not smooth twice.
        previous_time = self.timestamps.get(track_id)
        steps = 1.0
        if timestamp is not None:
            if previous_time is not None:
                if timestamp == previous_time:
                    return self.boxes[track_id]
                if timestamp > previous_time:
                    steps = (timestamp - previous_time) * 24.0
                else:
                    self.boxes.pop(track_id, None)
            self.timestamps[track_id] = timestamp

        if track_id not in self.boxes:

            self.boxes[
                track_id
            ] = target

            return target

        old = self.boxes[
            track_id
        ]

        old_center_x = (
            old[0] + old[2]
        ) / 2

        old_center_y = (
            old[1] + old[3]
        ) / 2

        new_center_x = (
            target[0]
            + target[2]
        ) / 2

        new_center_y = (
            target[1]
            + target[3]
        ) / 2

        movement = np.hypot(
            new_center_x
            - old_center_x,

            new_center_y
            - old_center_y,
        )

        width = max(
            1.0,
            target[2]
            - target[0],
        )

        height = max(
            1.0,
            target[3]
            - target[1],
        )

        box_size = np.hypot(
            width,
            height,
        )

        movement_ratio = (
            movement
            / box_size
        )

        if (
            movement_ratio
            <
            self.config
            .smoothing_thresholds[0]
        ):

            alpha_position = (
                self.config
                .smoothing_alphas[0]
            )

        elif (
            movement_ratio
            <
            self.config
            .smoothing_thresholds[1]
        ):

            alpha_position = (
                self.config
                .smoothing_alphas[1]
            )

        elif (
            movement_ratio
            <
            self.config
            .smoothing_thresholds[2]
        ):

            alpha_position = (
                self.config
                .smoothing_alphas[2]
            )

        else:

            alpha_position = (
                self.config
                .smoothing_alphas[3]
            )

        alpha_size = (
            self.config
            .display_size_alpha
        )

        alpha_position = 1.0 - (1.0 - alpha_position) ** steps
        alpha_size = 1.0 - (1.0 - alpha_size) ** steps

        old_width = (
            old[2] - old[0]
        )

        old_height = (
            old[3] - old[1]
        )

        target_width = (
            target[2]
            - target[0]
        )

        target_height = (
            target[3]
            - target[1]
        )

        smooth_center_x = (
            old_center_x
            + alpha_position
            * (
                new_center_x
                - old_center_x
            )
        )

        smooth_center_y = (
            old_center_y
            + alpha_position
            * (
                new_center_y
                - old_center_y
            )
        )

        smooth_width = (
            old_width
            + alpha_size
            * (
                target_width
                - old_width
            )
        )

        smooth_height = (
            old_height
            + alpha_size
            * (
                target_height
                - old_height
            )
        )

        smoothed = np.array(
            [
                smooth_center_x
                - smooth_width / 2,

                smooth_center_y
                - smooth_height / 2,

                smooth_center_x
                + smooth_width / 2,

                smooth_center_y
                + smooth_height / 2,
            ],
            dtype=np.float32,
        )

        self.boxes[
            track_id
        ] = smoothed

        return smoothed

    def keep_only(
        self,
        active_ids,
    ):

        active_ids = set(
            active_ids
        )

        dead_ids = [
            track_id

            for track_id
            in self.boxes

            if track_id
            not in active_ids
        ]

        for track_id in dead_ids:

            del self.boxes[track_id]
            self.timestamps.pop(track_id, None)

class DisplayPoseSmoother:
    """Smooth skeleton joints for display only."""

    def __init__(
        self,
        config: VisionConfig = VisionConfig(),
    ):

        self.config = config
        self.points = {}

    def smooth(
        self,
        entity_id,
        keypoints,
    ):

        previous = self.points.get(
            entity_id,
            {},
        )

        current = {}
        output = {}

        alpha = (
            self.config
            .pose_smoothing_alpha
        )

        for index, keypoint in enumerate(
            keypoints
        ):

            if (
                keypoint.confidence
                <
                self.config
                .pose_min_keypoint_confidence
            ):
                continue

            # --------------------------------------
            # Support both naming styles:
            #
            # x_relative / y_relative
            #
            # OR
            #
            # x / y
            # --------------------------------------

            if hasattr(
                keypoint,
                "x_relative",
            ):

                x = keypoint.x_relative
                y = keypoint.y_relative

            else:

                x = keypoint.x
                y = keypoint.y

            if not np.isfinite((x, y, keypoint.confidence)).all():
                continue

            target = np.array(
                [
                    x,
                    y,
                ],
                dtype=np.float32,
            )

            old = previous.get(
                index
            )

            if old is None:

                smooth_point = target

            else:

                smooth_point = (
                    old
                    + alpha
                    * (
                        target - old
                    )
                )

            current[index] = (
                smooth_point
            )

            output[index] = (
                float(
                    smooth_point[0]
                ),
                float(
                    smooth_point[1]
                ),
            )

        self.points[
            entity_id
        ] = current

        return output

    def keep_only(
        self,
        active_ids,
    ):

        active_ids = set(
            active_ids
        )

        dead_ids = [
            entity_id
            for entity_id
            in self.points
            if entity_id
            not in active_ids
        ]

        for entity_id in dead_ids:

            del self.points[
                entity_id
            ]
