from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2


@dataclass(frozen=True)
class Detection:
    timestamp: float
    card: str
    confidence: float
    source_frame: str | None = None


@dataclass(frozen=True)
class FrameDetection:
    """Normalized output from a single frame inference pass."""

    timestamp: float
    label: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    source_frame: str | None = None
    class_id: int | None = None
    detection_id: str | None = None
    canonical_label: str | None = None


class DetectorBackend(Protocol):
    def detect_frame(
        self,
        frame: object,
        *,
        timestamp: float,
        source_frame: str | None,
    ) -> list[FrameDetection]:
        ...


class CardDetector:
    """Placeholder interface for a future trained card detector."""

    def detect_video(self, video_path: Path) -> list[Detection]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            frame_index = 0
            detections: list[Detection] = []

            while True:
                ok, _frame = capture.read()
                if not ok:
                    break

                timestamp = frame_index / fps
                frame_index += 1

                # A trained detector will inspect frames here and append detections.
                _ = timestamp

            return detections
        finally:
            capture.release()

