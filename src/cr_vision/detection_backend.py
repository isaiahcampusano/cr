from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cr_vision.detection import FrameDetection


class FakeDetectorBackend:
    """Simple backend used by tests and offline CLI smoke tests."""

    def __init__(self, detections: list[FrameDetection] | None = None) -> None:
        self._detections = detections or []

    def detect_frame(
        self,
        frame: object,
        *,
        timestamp: float,
        source_frame: str | None,
    ) -> list[FrameDetection]:
        return [
            detection
            if detection.source_frame is not None else FrameDetection(
                timestamp=detection.timestamp,
                label=detection.label,
                confidence=detection.confidence,
                x_center=detection.x_center,
                y_center=detection.y_center,
                width=detection.width,
                height=detection.height,
                source_frame=source_frame,
            )
            for detection in self._detections
            if detection.timestamp == timestamp
        ]


def parse_roboflow_response(
    payload: dict[str, Any],
    *,
    timestamp: float,
    source_frame: str | None,
) -> list[FrameDetection]:
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Roboflow payload must contain a list of predictions")

    detections: list[FrameDetection] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Each Roboflow prediction must be an object")

        try:
            label = str(prediction["class"])
            confidence = float(prediction["confidence"])
            x_center = float(prediction["x_center"])
            y_center = float(prediction["y_center"])
            width = float(prediction["width"])
            height = float(prediction["height"])
        except KeyError as exc:
            raise ValueError(f"Roboflow prediction missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("Roboflow prediction values must be numeric") from exc

        detections.append(
            FrameDetection(
                timestamp=timestamp,
                label=label,
                confidence=confidence,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                source_frame=source_frame,
            )
        )

    return detections


def write_frame_detections_jsonl(
    path: Path,
    detections: list[FrameDetection],
    *,
    model_id: str,
    model_version: str,
    dataset_version: str,
    source_video: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for detection in detections:
            payload = {
                "model_id": model_id,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "source_video": source_video,
                "timestamp": detection.timestamp,
                "raw_label": detection.label,
                "canonical_label": None,
                "confidence": detection.confidence,
                "bbox": {
                    "x_center": detection.x_center,
                    "y_center": detection.y_center,
                    "width": detection.width,
                    "height": detection.height,
                },
                "source_frame": detection.source_frame,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def load_frame_detections_jsonl(path: Path) -> list[FrameDetection]:
    detections: list[FrameDetection] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            detections.append(
                FrameDetection(
                    timestamp=float(payload["timestamp"]),
                    label=str(payload["raw_label"]),
                    confidence=float(payload["confidence"]),
                    x_center=float(payload["bbox"]["x_center"]),
                    y_center=float(payload["bbox"]["y_center"]),
                    width=float(payload["bbox"]["width"]),
                    height=float(payload["bbox"]["height"]),
                    source_frame=payload.get("source_frame"),
                )
            )
    return detections


def detect_video(
    video_path: Path,
    *,
    backend: object,
    output_path: Path,
    model_id: str,
    model_version: str,
    dataset_version: str,
    source_video: str,
) -> list[FrameDetection]:
    capture = __import__("cv2").VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(__import__("cv2").CAP_PROP_FPS) or 30.0
        detections: list[FrameDetection] = []
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps
            frame_index += 1
            frame_detections = backend.detect_frame(
                frame,
                timestamp=timestamp,
                source_frame=f"frame_{frame_index:06d}.jpg",
            )
            detections.extend(frame_detections)

        write_frame_detections_jsonl(
            output_path,
            detections,
            model_id=model_id,
            model_version=model_version,
            dataset_version=dataset_version,
            source_video=source_video,
        )
        return detections
    finally:
        capture.release()
