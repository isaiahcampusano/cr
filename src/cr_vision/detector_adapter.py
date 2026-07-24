from __future__ import annotations

from cr_vision.detection import Detection
from cr_vision.state import CardEvent


def detections_to_events(
    detections: list[Detection],
    *,
    player: str = "opponent",
    min_confidence: float = 0.0,
) -> list[CardEvent]:
    """Convert detector outputs into CardEvent objects for the analyzer."""
    filtered = [
        detection
        for detection in detections
        if detection.confidence >= min_confidence
    ]
    filtered.sort(key=lambda detection: detection.timestamp)

    return [
        CardEvent(
            time=detection.timestamp,
            player=player,
            card=detection.card,
            confidence=detection.confidence,
        )
        for detection in filtered
    ]
