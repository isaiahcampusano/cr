from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cr_vision.detection import FrameDetection
from cr_vision.detection_backend import load_frame_detections_jsonl
from cr_vision.state import CardEvent


@dataclass(frozen=True)
class HandObservation:
    """One sampled frame's visible hand state.

    ``None`` means the slot is unknown because the detector did not provide
    enough evidence. It does not mean the game UI showed an empty slot.
    """

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
    stability_observations: int = 3
    stability_window: int = 5
    player_perspective: str = "self"
    slot_centers: tuple[float, float, float, float] = (0.18, 0.38, 0.58, 0.78)
    slot_tolerance: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.stability_observations < 1:
            raise ValueError("stability_observations must be positive")
        if self.stability_window < self.stability_observations:
            raise ValueError("stability_window must be at least stability_observations")
        if self.slot_tolerance <= 0.0:
            raise ValueError("slot_tolerance must be positive")
        if tuple(sorted(self.slot_centers)) != self.slot_centers:
            raise ValueError("slot_centers must be ordered from left to right")


class HandEventizer:
    """Turn stable, complete local-hand observations into card-play events."""

    def __init__(self, config: EventizerConfig | None = None) -> None:
        self.config = config or EventizerConfig()
        self._observations: list[HandObservation] = []
        self._last_complete_state: HandObservation | None = None
        self.diagnostics: list[dict[str, object]] = []

    def observe_detections(self, detections: list[FrameDetection]) -> list[HandObservation]:
        """Group all boxes from each frame before assigning their fixed slots."""
        grouped: dict[tuple[float, str | None], list[FrameDetection]] = {}
        for detection in detections:
            key = (detection.timestamp, detection.source_frame)
            grouped.setdefault(key, []).append(detection)

        return [
            self._build_observation(frame_detections)
            for _key, frame_detections in sorted(
                grouped.items(),
                key=lambda item: (item[1][0].timestamp, item[1][0].source_frame or ""),
            )
        ]

    def _assign_slot(self, detection: FrameDetection) -> int | None:
        distances = [abs(detection.x_center - center) for center in self.config.slot_centers]
        nearest_slot = min(range(4), key=distances.__getitem__)
        if distances[nearest_slot] > self.config.slot_tolerance:
            return None
        return nearest_slot

    def _build_observation(self, detections: list[FrameDetection]) -> HandObservation:
        if not detections:
            raise ValueError("A hand observation requires at least one frame detection")

        selected: list[FrameDetection | None] = [None, None, None, None]
        for detection in detections:
            if detection.confidence < self.config.confidence_threshold:
                continue
            if detection.canonical_label is None:
                continue

            slot_index = self._assign_slot(detection)
            if slot_index is None:
                continue

            current = selected[slot_index]
            if current is not None and current.canonical_label != detection.canonical_label:
                self.diagnostics.append(
                    {
                        "timestamp": detection.timestamp,
                        "reason": "slot_conflict",
                        "slot": slot_index,
                        "existing_label": current.canonical_label,
                        "competing_label": detection.canonical_label,
                    }
                )
            if current is None or detection.confidence > current.confidence:
                selected[slot_index] = detection

        return HandObservation(
            timestamp=detections[0].timestamp,
            slots=tuple(
                detection.canonical_label if detection is not None else None
                for detection in selected
            ),
            confidences=tuple(
                detection.confidence if detection is not None else None
                for detection in selected
            ),
            evidence_frames=tuple(
                detection.source_frame if detection is not None else None
                for detection in selected
            ),
        )

    def stable_observation(self, observation: HandObservation) -> HandObservation:
        """Return a per-slot majority vote across the most recent observations."""
        self._observations.append(observation)
        window = self._observations[-self.config.stability_window :]

        stable_slots: list[str | None] = [None, None, None, None]
        stable_confidences: list[float | None] = [None, None, None, None]
        stable_frames: list[str | None] = [None, None, None, None]

        for slot_index in range(4):
            values = [
                item.slots[slot_index]
                for item in window
                if item.slots[slot_index] is not None
            ]
            if not values:
                continue

            card, count = Counter(values).most_common(1)[0]
            if count < self.config.stability_observations:
                continue

            stable_slots[slot_index] = card
            for item in reversed(window):
                if item.slots[slot_index] == card:
                    stable_confidences[slot_index] = item.confidences[slot_index]
                    stable_frames[slot_index] = item.evidence_frames[slot_index]
                    break

        return HandObservation(
            timestamp=observation.timestamp,
            slots=tuple(stable_slots),
            confidences=tuple(stable_confidences),
            evidence_frames=tuple(stable_frames),
        )

    def eventize(self, detections: list[FrameDetection]) -> list[CardEvent]:
        events: list[CardEvent] = []

        for observation in self.observe_detections(detections):
            stable_state = self.stable_observation(observation)
            if not _is_complete(stable_state):
                continue

            if self._last_complete_state is None:
                self._last_complete_state = stable_state
                continue

            if _state_signature(stable_state) == _state_signature(self._last_complete_state):
                continue

            removed, added = _state_difference(self._last_complete_state, stable_state)
            if len(removed) == 1 and len(added) == 1:
                removed_card = removed[0]
                added_card = added[0]
                removed_confidence, _removed_frame = _evidence_for_card(
                    self._last_complete_state,
                    removed_card,
                )
                added_confidence, added_frame = _evidence_for_card(
                    stable_state,
                    added_card,
                )
                confidence_values = [
                    value
                    for value in (removed_confidence, added_confidence)
                    if value is not None
                ]
                confidence = min(confidence_values) if confidence_values else 0.0
                events.append(
                    CardEvent(
                        time=stable_state.timestamp,
                        player=self.config.player_perspective,
                        card=removed_card,
                        confidence=confidence,
                        source_frame=added_frame,
                    )
                )
            else:
                self.diagnostics.append(
                    {
                        "timestamp": stable_state.timestamp,
                        "reason": "ambiguous_transition",
                        "previous_slots": list(self._last_complete_state.slots),
                        "current_slots": list(stable_state.slots),
                        "removed_cards": list(removed),
                        "added_cards": list(added),
                    }
                )

            # Move the baseline forward even after an ambiguous change. This
            # avoids cascading one uncertain transition into later plays.
            self._last_complete_state = stable_state

        return events


def _is_complete(state: HandObservation) -> bool:
    return all(card is not None for card in state.slots)


def _state_signature(state: HandObservation) -> tuple[str | None, str | None, str | None, str | None]:
    return state.slots


def _state_difference(
    previous_state: HandObservation,
    current_state: HandObservation,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    previous_cards = Counter(card for card in previous_state.slots if card is not None)
    current_cards = Counter(card for card in current_state.slots if card is not None)
    removed = tuple((previous_cards - current_cards).elements())
    added = tuple((current_cards - previous_cards).elements())
    return removed, added


def _evidence_for_card(
    state: HandObservation,
    card: str,
) -> tuple[float | None, str | None]:
    for index, value in enumerate(state.slots):
        if value == card:
            return state.confidences[index], state.evidence_frames[index]
    return None, None


def load_frame_detections(path: Path) -> list[FrameDetection]:
    return load_frame_detections_jsonl(path)


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
