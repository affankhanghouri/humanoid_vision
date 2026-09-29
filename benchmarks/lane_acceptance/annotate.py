#!/usr/bin/env python3
"""Small OpenCV click UI for human lane-boundary annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    BOUNDARY_STATUSES,
    NO_VALID_EGO_LANE,
    SCENE_TYPES,
    atomic_write_json,
    load_json,
    validate_annotation,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks/lane_acceptance/dataset.json"
WINDOW = "Lane acceptance annotation"
LEFT_COLOR = (0, 255, 255)
RIGHT_COLOR = (255, 128, 0)
PANEL_WIDTH = 340


class Annotator:
    def __init__(self, dataset_path: Path, scale: float, start_frame_id: int | None):
        self.dataset_path = dataset_path.resolve()
        self.directory = self.dataset_path.parent
        self.dataset = load_json(self.dataset_path)
        self.annotations = self.dataset["annotations"]
        self.scale = scale
        self.active_side = "left"
        self.message = ""
        self.index = self._start_index(start_frame_id)
        self.image = None
        self.display_image_width = 0
        self._load_image()

    def _start_index(self, frame_id: int | None) -> int:
        if frame_id is not None:
            for index, annotation in enumerate(self.annotations):
                if annotation["frame_id"] == frame_id:
                    return index
            raise ValueError(f"frame {frame_id} is not in the dataset")
        progress_id = self.dataset.get("progress", {}).get("last_frame_id")
        if progress_id is not None:
            for index, annotation in enumerate(self.annotations):
                if annotation["frame_id"] == progress_id:
                    return index
        for index, annotation in enumerate(self.annotations):
            if annotation.get("review_status") != "complete":
                return index
        return 0

    @property
    def annotation(self):
        return self.annotations[self.index]

    def _load_image(self) -> None:
        self.image = cv2.imread(str(self.directory / self.annotation["image"]))
        if self.image is None:
            raise RuntimeError(f"Cannot load {self.annotation['image']}")

    def save(self) -> None:
        self.dataset["progress"] = {
            "last_frame_id": self.annotation["frame_id"],
            "completed": sum(
                item.get("review_status") == "complete" for item in self.annotations
            ),
        }
        atomic_write_json(self.dataset_path, self.dataset)

    def move(self, amount: int) -> None:
        self.save()
        self.index = max(0, min(len(self.annotations) - 1, self.index + amount))
        self.message = ""
        self._load_image()

    def boundary(self) -> dict:
        return self.annotation[f"{self.active_side}_boundary"]

    def set_boundary_status(self, status: str) -> None:
        assert status in BOUNDARY_STATUSES
        boundary = self.boundary()
        boundary["status"] = status
        if status == "not_visible":
            boundary["points"] = []
        self.annotation["review_status"] = "pending"
        self.message = f"{self.active_side} boundary: {status}"
        self.save()

    def set_ego_status(self, status: str) -> None:
        annotation = self.annotation
        annotation["ego_lane_status"] = status
        annotation["review_status"] = "pending"
        if status == "invalid":
            annotation["annotation_flag"] = NO_VALID_EGO_LANE
            for side in ("left_boundary", "right_boundary"):
                annotation[side] = {"status": "not_visible", "points": []}
        else:
            annotation["annotation_flag"] = None
        self.message = f"ego lane: {status}"
        self.save()

    def toggle_scene(self, index: int) -> None:
        tag = SCENE_TYPES[index]
        tags = self.annotation["scene_type"]
        if tag in tags:
            if len(tags) == 1:
                self.message = "At least one scene tag is required"
                return
            tags.remove(tag)
        else:
            tags.append(tag)
            tags.sort(key=SCENE_TYPES.index)
        self.annotation["review_status"] = "pending"
        self.message = f"toggled {tag}"
        self.save()

    def add_point(self, display_x: int, display_y: int) -> None:
        image_height, image_width = self.image.shape[:2]
        x = int(round(display_x / self.scale))
        y = int(round(display_y / self.scale))
        if not (0 <= x < image_width and 0 <= y < image_height):
            return
        boundary = self.boundary()
        boundary["points"].append([x, y])
        boundary["status"] = "visible"
        self.annotation["review_status"] = "pending"
        self.message = f"added {self.active_side} point ({x}, {y})"
        self.save()

    def undo_point(self) -> None:
        points = self.boundary()["points"]
        if points:
            points.pop()
            self.annotation["review_status"] = "pending"
            self.message = f"removed last {self.active_side} point"
            self.save()

    def clear_active(self) -> None:
        self.boundary()["points"] = []
        self.boundary()["status"] = None
        self.annotation["review_status"] = "pending"
        self.message = f"cleared {self.active_side} boundary"
        self.save()

    def mark_complete(self) -> None:
        self.annotation["review_status"] = "complete"
        errors = validate_annotation(
            self.annotation,
            self.dataset["video"]["width"],
            self.dataset["video"]["height"],
            require_complete=True,
        )
        if errors:
            self.annotation["review_status"] = "pending"
            self.message = errors[0]
        else:
            self.message = "review complete"
        self.save()

    def mark_pending(self) -> None:
        self.annotation["review_status"] = "pending"
        self.message = "reopened for editing"
        self.save()

    def mouse(self, event, x, y, _flags, _parameter) -> None:
        if x >= self.display_image_width:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo_point()

    def _draw_boundary(self, image: np.ndarray, side: str, color: tuple[int, int, int]) -> None:
        boundary = self.annotation[f"{side}_boundary"]
        points = np.asarray(boundary["points"], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
        for number, point in enumerate(points, 1):
            cv2.circle(image, tuple(point), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                str(number),
                (int(point[0]) + 7, int(point[1]) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def render(self) -> np.ndarray:
        image = self.image.copy()
        self._draw_boundary(image, "left", LEFT_COLOR)
        self._draw_boundary(image, "right", RIGHT_COLOR)
        scaled_width = int(round(image.shape[1] * self.scale))
        scaled_height = int(round(image.shape[0] * self.scale))
        image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        self.display_image_width = scaled_width
        panel = np.full((scaled_height, PANEL_WIDTH, 3), 28, np.uint8)
        annotation = self.annotation
        completed = sum(item.get("review_status") == "complete" for item in self.annotations)
        lines = [
            f"Frame {annotation['frame_id']}  {annotation['timestamp']:.3f}s",
            f"Item {self.index + 1}/100  complete {completed}/100",
            f"Primary: {annotation['selection_stratum']}",
            f"Review: {annotation['review_status']}",
            f"Ego: {annotation['ego_lane_status']}",
            f"Active: {self.active_side.upper()}",
            f"Left: {annotation['left_boundary']['status']} "
            f"({len(annotation['left_boundary']['points'])} pts)",
            f"Right: {annotation['right_boundary']['status']} "
            f"({len(annotation['right_boundary']['points'])} pts)",
            "",
            "Mouse: left add | right undo",
            "L/R: active boundary",
            "V/H/A: visible/hidden/ambiguous",
            "G/X/U: ego valid/invalid/ambiguous",
            "Z: undo | D: clear active",
            "Space: complete | E: reopen",
            ", previous | . next | Q quit",
            "",
            "Scene tags (number toggles):",
        ]
        for number, tag in enumerate(SCENE_TYPES, 1):
            mark = "[x]" if tag in annotation["scene_type"] else "[ ]"
            lines.append(f"{number}: {mark} {tag}")
        if self.message:
            lines.extend(("", "Message:", self.message))
        y = 20
        for line in lines:
            color = (110, 230, 255) if line.startswith("Message") else (235, 235, 235)
            font_scale = 0.40 if len(line) > 38 else 0.44
            cv2.putText(
                panel,
                line[:46],
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 17
            if y >= scaled_height - 5:
                break
        return np.hstack((image, panel))

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKeyEx(30)
                if key < 0:
                    continue
                key &= 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    self.active_side = "left"
                elif key == ord("r"):
                    self.active_side = "right"
                elif key == ord("v"):
                    self.set_boundary_status("visible")
                elif key == ord("h"):
                    self.set_boundary_status("not_visible")
                elif key == ord("a"):
                    self.set_boundary_status("ambiguous")
                elif key == ord("g"):
                    self.set_ego_status("valid")
                elif key == ord("x"):
                    self.set_ego_status("invalid")
                elif key == ord("u"):
                    self.set_ego_status("ambiguous")
                elif key == ord("z"):
                    self.undo_point()
                elif key == ord("d"):
                    self.clear_active()
                elif key == ord(" "):
                    self.mark_complete()
                elif key == ord("e"):
                    self.mark_pending()
                elif key == ord(","):
                    self.move(-1)
                elif key == ord("."):
                    self.move(1)
                elif ord("1") <= key <= ord("8"):
                    self.toggle_scene(key - ord("1"))
        finally:
            self.save()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scale", type=float, default=0.70)
    parser.add_argument("--frame-id", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.25 <= args.scale <= 1.5:
        raise ValueError("--scale must be between 0.25 and 1.5")
    Annotator(args.dataset, args.scale, args.frame_id).run()


if __name__ == "__main__":
    main()
