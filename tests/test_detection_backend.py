from __future__ import annotations

from pathlib import Path

from cr_vision.detection import FrameDetection
from cr_vision.detection_backend import (
    FakeDetectorBackend,
    load_frame_detections_jsonl,
    parse_roboflow_response,
    write_frame_detections_jsonl,
)


def test_parse_roboflow_response_to_frame_detections() -> None:
    payload = {
        "predictions": [
            {
                "class": "Kanon In Hand",
                "confidence": 0.91,
                "x": 205,
                "y": 153,
                "width": 48,
                "height": 56,
                "class_id": 29,
                "detection_id": "det-001",
            }
        ]
    }

    detections = parse_roboflow_response(
        payload,
        timestamp=1.5,
        source_frame="frame_000001.jpg",
        source_image_width=512,
        source_image_height=512,
    )

    assert detections == [
        FrameDetection(
            timestamp=1.5,
            label="Kanon In Hand",
            confidence=0.91,
            x_center=0.400390625,
            y_center=0.298828125,
            width=0.09375,
            height=0.109375,
            source_frame="frame_000001.jpg",
            class_id=29,
            detection_id="det-001",
            canonical_label="cannon",
        )
    ]


def test_frame_detections_jsonl_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_detections.jsonl"
    detections = [
        FrameDetection(
            timestamp=0.0,
            label="hog_rider",
            confidence=0.93,
            x_center=0.2,
            y_center=0.3,
            width=0.08,
            height=0.12,
            source_frame="frame_000000.jpg",
        ),
        FrameDetection(
            timestamp=1.2,
            label="cannon",
            confidence=0.88,
            x_center=0.5,
            y_center=0.4,
            width=0.07,
            height=0.09,
            source_frame="frame_000060.jpg",
        ),
    ]

    write_frame_detections_jsonl(
        output_path,
        detections,
        model_id="demo-model",
        model_version="1",
        dataset_version="1",
        source_video="match_001.mp4",
    )

    loaded = load_frame_detections_jsonl(output_path)

    assert loaded == detections


def test_fake_backend_returns_preloaded_detections() -> None:
    backend = FakeDetectorBackend(
        [
            FrameDetection(
                timestamp=1.0,
                label="hog_rider",
                confidence=0.9,
                x_center=0.1,
                y_center=0.2,
                width=0.05,
                height=0.06,
                source_frame="frame_000001.jpg",
            )
        ]
    )

    detections = backend.detect_frame(frame=None, timestamp=1.0, source_frame="frame_000001.jpg")

    assert detections[0].label == "hog_rider"
