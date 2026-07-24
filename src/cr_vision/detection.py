from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class Detection:
    timestamp: float
    card: str
    confidence: float
    source_frame: str | None = None


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

