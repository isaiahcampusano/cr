from __future__ import annotations

from pathlib import Path

from cr_vision.analyzer import analyze_events
from cr_vision.detector_adapter import detections_to_events
from cr_vision.detection import Detection
from cr_vision.detection_labels import load_detections


def test_load_detections_from_json(tmp_path: Path) -> None:
    label_path = tmp_path / "sample_detections.json"
    label_path.write_text(
        """
        [
          {"timestamp": 1.0, "card": "hog_rider", "confidence": 0.93, "source_frame": "frame_000001.jpg"},
          {"timestamp": 2.8, "card": "cannon", "confidence": 0.88, "source_frame": "frame_000014.jpg"}
        ]
        """,
        encoding="utf-8",
    )

    detections = load_detections(label_path)

    assert detections == [
        Detection(
            timestamp=1.0,
            card="hog_rider",
            confidence=0.93,
            source_frame="frame_000001.jpg",
        ),
        Detection(
            timestamp=2.8,
            card="cannon",
            confidence=0.88,
            source_frame="frame_000014.jpg",
        ),
    ]
    assert [d.timestamp for d in detections] == [1.0, 2.8]
    assert [d.card for d in detections] == ["hog_rider", "cannon"]


def test_loaded_detections_convert_and_analyze() -> None:
    detections = [
        Detection(timestamp=3.2, card="ice_spirit", confidence=0.99),
        Detection(timestamp=0.5, card="musketeer", confidence=0.94),
    ]

    events = detections_to_events(detections, player="opponent", min_confidence=0.0)
    tracker, snapshots = analyze_events(events)

    assert [snapshot.time for snapshot in snapshots] == [0.5, 3.2]
    assert [snapshot.event.card for snapshot in snapshots] == ["musketeer", "ice_spirit"]
    assert tracker.known_deck == ("musketeer", "ice_spirit")
