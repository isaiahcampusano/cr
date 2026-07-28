from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cr_vision.detection import FrameDetection
from cr_vision.state import CardEvent


@dataclass(frozen=True)
class HandObservation:
    timestamp: float
    slots: tuple[str | None, str | None, str | None, str | None]
    confidences: tuple[float | None, float | None, float | None, float | None]
    evidence_frames: tuple[str | None, str | None, str | None, str | None]


@dataclass(frozen=True)
class HandStateTransition:
    timestamp: float
    previous_state: HandObservation
    current_state: HandObservation
    removed_cards: tuple[str, ...]
    added_cards: tuple[str, ...]
    confidence: float
    evidence_frame: str | None


@dataclass
class EventizerConfig:
    confidence_threshold: float = 0.50
    sample_fps: float = 3.0
    stability_observations: int = 3
    player_perspective: str = "self"
    slot_centers: tuple[float, ...] = (0.18, 0.38, 0.58, 0.78)
    slot_tolerance: float = 0.10


class HandEventizer:
    def __init__(self, config: EventizerConfig | None = None) -> None:
        self.config = config or EventizerConfig()
        self._observations: list[HandObservation] = []
        self._stable_states: list[tuple[HandObservation, str | None]] = []
        self._last_stable_slots: list[str | None] = [None, None, None, None]
        self._last_stable_confidences: list[float | None] = [None, None, None, None]
        self._last_stable_frames: list[str | None] = [None, None, None, None]
        self.diagnostics: list[dict[str, object]] = []

    def observe_detections(self, detections: list[FrameDetection]) -> list[HandObservation]:
        if not detections:
            return []

        grouped: list[HandObservation] = []
        for detection in sorted(detections, key=lambda item: item.timestamp):
            if detection.confidence < self.config.confidence_threshold:
                continue
            if detection.canonical_label is None:
                continue

            slot_index = self._assign_slot(detection)
            if slot_index is None:
                continue

            grouped.append(
                HandObservation(
                    timestamp=detection.timestamp,
                    slots=(None, None, None, None),
                    confidences=(None, None, None, None),
                    evidence_frames=(None, None, None, None),
                )
            )

        return grouped

    def _assign_slot(self, detection: FrameDetection) -> int | None:
        for index, center in enumerate(self.config.slot_centers):
            if abs(detection.x_center - center) <= self.config.slot_tolerance:
                return index
        return None

    def _build_observation(self, detections: list[FrameDetection]) -> HandObservation:
        slots: list[str | None] = [None, None, None, None]
        confidences: list[float | None] = [None, None, None, None]
        evidence_frames: list[str | None] = [None, None, None, None]

        for detection in sorted(detections, key=lambda item: item.timestamp):
            if detection.confidence < self.config.confidence_threshold:
                continue
            if detection.canonical_label is None:
                continue
            slot_index = self._assign_slot(detection)
            if slot_index is None:
                continue
            slots[slot_index] = detection.canonical_label
            confidences[slot_index] = detection.confidence
            evidence_frames[slot_index] = detection.source_frame

        return HandObservation(
            timestamp=detections[-1].timestamp if detections else 0.0,
            slots=tuple(slots),
            confidences=tuple(confidences),
            evidence_frames=tuple(evidence_frames),
        )

    def stable_observation(self, observation: HandObservation) -> HandObservation:
        self._observations.append(observation)
        if len(self._observations) < self.config.stability_observations:
            return observation

        window = self._observations[-self.config.stability_observations :]
        stable_slots: list[str | None] = [None, None, None, None]
        stable_confidences: list[float | None] = [None, None, None, None]
        stable_frames: list[str | None] = [None, None, None, None]

        for slot_index in range(4):
            slot_values = [item.slots[slot_index] for item in window]
            slot_confidences = [item.confidences[slot_index] for item in window]
            slot_frames = [item.evidence_frames[slot_index] for item in window]
            latest_value = next(
                (value for value in reversed(slot_values) if value is not None),
                None,
            )
            if latest_value is not None:
                stable_slots[slot_index] = latest_value
                stable_confidences[slot_index] = next(
                    (value for value in reversed(slot_confidences) if value is not None),
                    None,
                )
                stable_frames[slot_index] = next(
                    (frame for frame in reversed(slot_frames) if frame is not None),
                    None,
                )
            elif self._last_stable_slots[slot_index] is not None:
                stable_slots[slot_index] = self._last_stable_slots[slot_index]
                stable_confidences[slot_index] = self._last_stable_confidences[slot_index]
                stable_frames[slot_index] = self._last_stable_frames[slot_index]

        self._last_stable_slots = stable_slots
        self._last_stable_confidences = stable_confidences
        self._last_stable_frames = stable_frames

        return HandObservation(
            timestamp=observation.timestamp,
            slots=tuple(stable_slots),
            confidences=tuple(stable_confidences),
            evidence_frames=tuple(stable_frames),
        )

    def eventize(self, detections: list[FrameDetection]) -> list[CardEvent]:
        if not detections:
            return []

        observations = [
            self._build_observation([item])
            for item in detections
            if item.confidence >= self.config.confidence_threshold
            and item.canonical_label is not None
        ]
        if not observations:
            return []

        stable_observations: list[HandObservation] = []
        events: list[CardEvent] = []
        previous_state: HandObservation | None = None

        for observation in observations:
            stable_observation = self.stable_observation(observation)
            stable_observations.append(stable_observation)
            if previous_state is None:
                previous_state = stable_observation
                continue

            if _state_signature(stable_observation) == _state_signature(previous_state):
                continue

            previous_slots = tuple(slot for slot in previous_state.slots if slot is not None)
            current_slots = tuple(slot for slot in stable_observation.slots if slot is not None)
            removed = tuple(card for card in previous_slots if card not in current_slots)
            added = tuple(card for card in current_slots if card not in previous_slots)
            if len(removed) == 1 and len(added) == 1:
                evidence_frame = next(
                    (
                        frame
                        for frame in stable_observation.evidence_frames
                        if frame is not None
                    ),
                    None,
                )
                events.append(
                    CardEvent(
                        time=stable_observation.timestamp,
                        player=self.config.player_perspective,
                        card=removed[0],
                        confidence=0.5,
                        source_frame=evidence_frame,
                    )
                )
            elif removed or added:
                self.diagnostics.append(
                    {
                        "timestamp": stable_observation.timestamp,
                        "reason": "ambiguous_transition",
                        "previous_slots": list(previous_slots),
                        "current_slots": list(current_slots),
                    }
                )

            previous_state = stable_observation

        return events


def _state_signature(state: HandObservation) -> tuple[tuple[str | None, ...], tuple[float | None, ...]]:
    return (state.slots, state.confidences)


def load_frame_detections(path: Path) -> list[FrameDetection]:
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


def write_events(path: Path, events: list[CardEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "time": event.time,
            "player": event.player,
            "card": event.card,
            "confidence": event.confidence,
            "source_frame": event.source_frame,
        }
        for event in events
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
