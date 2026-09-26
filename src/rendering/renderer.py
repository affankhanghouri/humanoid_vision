"""Render fresh tracking boxes, attached skeletons, and independent metrics."""
import math
import cv2
from config import VisionConfig
from tracking.motion import BoxMotionHistory
from core.types import FramePacket
from core.perception_state import PerceptionState
from tracking.smoothing import DisplayBoxSmoother, DisplayPoseSmoother

# COCO 17-keypoint topology: face, shoulders/arms, hips/legs.
SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_entity(frame, entity, tracking_age, timestamp, smoother, pose_smoother, config, has_pose=False, projected_box=None):
    height, width = frame.shape[:2]
    prediction_age = min(tracking_age, config.max_prediction_age)
    cx = entity.center_x + entity.velocity_x * prediction_age
    cy = entity.center_y + entity.velocity_y * prediction_age
    box = smoother.smooth(entity.entity_id, projected_box if projected_box is not None else (
        cx-entity.width/2, cy-entity.height/2,
        cx+entity.width/2, cy+entity.height/2,
    ), timestamp=timestamp)
    # Keep the unclipped box for projecting joints; clipping must not stretch pose.
    x1, y1, x2, y2 = box
    x1, x2 = [max(0, min(int(x), width-1)) for x in (x1, x2)]
    y1, y2 = [max(0, min(int(y), height-1)) for y in (y1, y2)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{entity.class_name.upper()} #{entity.entity_id} {entity.confidence:.2f}'
    if entity.misses:
        label += ' PREDICTED'
    if has_pose:
        label += ' + POSE'
    cv2.putText(frame, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (0, 255, 0), 2)
    return box


def draw_pose(frame, entity, box, timestamp, pose_smoother, config):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    pose = entity.pose

    relative_points = pose_smoother.smooth(entity.entity_id, pose.keypoints)

    points = {}
    for index, relative_point in relative_points.items():
        relative_x, relative_y = relative_point
        x = int(x1 + relative_x * box_width)
        y = int(y1 + relative_y * box_height)
        points[index] = (x, y)

    rendered = {}
    for index, point in points.items():
        x, y = point
        rendered[index] = (x, y) if 0 <= x < width and 0 <= y < height else None

    for a, b in SKELETON_EDGES:
        if max(a, b) in rendered and a in rendered and b in rendered \
                and rendered[a] is not None and rendered[b] is not None:
            cv2.line(frame, rendered[a], rendered[b], (255, 220, 0), 2, cv2.LINE_AA)
    for point in rendered.values():
        if point is not None:
            cv2.circle(frame, point, 3, (0, 220, 255), -1, cv2.LINE_AA)


class Renderer:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.motion = BoxMotionHistory(config)
        self.smoother = DisplayBoxSmoother(config)
        self.pose_smoother = DisplayPoseSmoother(config)

    def render(self, packet: FramePacket, state: PerceptionState | None, display_fps: float):
        self.motion.update(packet)
        frame = packet.frame.copy()
        valid = state is not None and state.is_valid_for(
            packet.frame_id, packet.timestamp, self.config.max_render_age)
        if valid:
            state_age = state.age_at(packet.timestamp)
            pose_ids = []
            active_ids = []
            for entity in state.entities:
                if not entity.visible_at(packet.timestamp, self.config.max_coast_age):
                    continue
                active_ids.append(entity.entity_id)
                projected = self.motion.project(entity.entity_id, (
                    entity.center_x-entity.width/2, entity.center_y-entity.height/2,
                    entity.center_x+entity.width/2, entity.center_y+entity.height/2,
                ), state.source_timestamp)
                pose = entity.pose
                pose_valid = (entity.class_name == 'person' and pose is not None
                              and pose.meta.is_valid_for(packet.frame_id, packet.timestamp,
                                                         self.config.pose_max_age))
                box = draw_entity(frame, entity, state_age, packet.timestamp,
                                  self.smoother, self.pose_smoother, self.config,
                                  has_pose=pose_valid, projected_box=projected)
                if pose_valid:
                    pose_ids.append(entity.entity_id)
                    draw_pose(frame, entity, box, packet.timestamp,
                             self.pose_smoother, self.config)
            self.smoother.keep_only(active_ids)
            self.pose_smoother.keep_only(pose_ids)
        else:
            self.smoother.keep_only(())
            self.pose_smoother.keep_only(())

        pose_fresh = (state is not None and state.pose_meta is not None
                      and state.pose_meta.is_valid_for(packet.frame_id, packet.timestamp,
                                                       self.config.pose_max_age))
        lines = [
            'VISION SYSTEM', f'Display: {display_fps:.1f} FPS',
            f'Detection: {state.detector_hz if state else 0:.1f} Hz',
            f'Pose: {state.pose_hz if pose_fresh else 0:.1f} Hz',
            f'YOLO: {state.inference_ms if state else 0:.1f} ms',
            f'Pose AI: {state.pose_inference_ms if state else 0:.1f} ms',
        ]
        if state is not None:
            lines += [f'State age: {state.age_at(packet.timestamp)*1000:.0f} ms',
                      f'Processing latency: {state.processing_latency_ms:.0f} ms',
                      'State: ' + ('FRESH' if valid else 'STALE')]
        # Dim only the HUD region on our private display copy.
        h, w = frame.shape[:2]
        roi = frame[:min(h, 24*len(lines)+16), :min(w, 325)]
        cv2.convertScaleAbs(roi, dst=roi, alpha=.35)
        for index, text in enumerate(lines):
            cv2.putText(frame, text, (12, 25+index*24), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255) if index == 0 else (0, 255, 255), 1, cv2.LINE_AA)
        return frame
