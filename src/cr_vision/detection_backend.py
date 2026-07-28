from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cr_vision.cards import card_cost
from cr_vision.detection import FrameDetection


ROBOFLOW_LABEL_MAP = {
    "Kanon In Hand": "cannon",
    "Skelet In hand": "skeletons",
    "Vuurbal In Hand": "fireball",
}


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
    source_image_width: int | None = None,
    source_image_height: int | None = None,
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
            x_px = float(prediction["x"])
            y_px = float(prediction["y"])
            width_px = float(prediction["width"])
            height_px = float(prediction["height"])
            class_id = prediction.get("class_id")
            detection_id = prediction.get("detection_id")
        except KeyError as exc:
            raise ValueError(f"Roboflow prediction missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("Roboflow prediction values must be numeric") from exc

        if source_image_width is not None and source_image_height is not None:
            x_center = x_px / float(source_image_width)
            y_center = y_px / float(source_image_height)
            width = width_px / float(source_image_width)
            height = height_px / float(source_image_height)
        else:
            x_center = x_px
            y_center = y_px
            width = width_px
            height = height_px

        canonical_label = ROBOFLOW_LABEL_MAP.get(label)
        if canonical_label is None:
            try:
                card_cost(label)
            except ValueError:
                canonical_label = None
            else:
                canonical_label = label

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
                class_id=int(class_id) if class_id is not None else None,
                detection_id=str(detection_id) if detection_id is not None else None,
                canonical_label=canonical_label,
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
                "canonical_label": detection.canonical_label,
                "confidence": detection.confidence,
                "bbox": {
                    "x_center": detection.x_center,
                    "y_center": detection.y_center,
                    "width": detection.width,
                    "height": detection.height,
                },
                "source_frame": detection.source_frame,
                "class_id": detection.class_id,
                "detection_id": detection.detection_id,
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
                    class_id=int(payload["class_id"]) if payload.get("class_id") is not None else None,
                    detection_id=str(payload["detection_id"]) if payload.get("detection_id") is not None else None,
                    canonical_label=payload.get("canonical_label"),
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
    sample_fps: float | None = 3.0,
) -> list[FrameDetection]:
    if sample_fps is not None and sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive when provided")

    capture = __import__("cv2").VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(__import__("cv2").CAP_PROP_FPS) or 30.0
        sample_interval = 1.0 / sample_fps if sample_fps is not None else None
        next_sample_time = 0.0
        detections: list[FrameDetection] = []
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps
            frame_index += 1
            if sample_interval is not None and timestamp + 1e-9 < next_sample_time:
                continue

            if sample_interval is not None:
                while next_sample_time <= timestamp + 1e-9:
                    next_sample_time += sample_interval

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
