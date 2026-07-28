from __future__ import annotations

from cr_vision.detection import FrameDetection
from cr_vision.eventizer import EventizerConfig, HandEventizer
from cr_vision.state import CardEvent


def test_four_slots_are_assigned_without_renumbering() -> None:
    eventizer = HandEventizer(EventizerConfig(confidence_threshold=0.5))
    detections = [
        FrameDetection(timestamp=0.0, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon"),
        FrameDetection(timestamp=0.1, label="x", confidence=0.9, x_center=0.38, y_center=0.2, width=0.05, height=0.05, canonical_label="fireball"),
        FrameDetection(timestamp=0.2, label="x", confidence=0.9, x_center=0.58, y_center=0.2, width=0.05, height=0.05, canonical_label="skeletons"),
        FrameDetection(timestamp=0.3, label="x", confidence=0.9, x_center=0.78, y_center=0.2, width=0.05, height=0.05, canonical_label="ice_spirit"),
    ]

    observation = eventizer._build_observation(detections)

    assert observation.slots == ("cannon", "fireball", "skeletons", "ice_spirit")


def test_transient_flicker_produces_no_event() -> None:
    eventizer = HandEventizer(EventizerConfig(confidence_threshold=0.5, stability_observations=3))
    first = [FrameDetection(timestamp=0.0, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon")]
    second = [FrameDetection(timestamp=0.1, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon")]
    third = [FrameDetection(timestamp=0.2, label="x", confidence=0.9, x_center=0.38, y_center=0.2, width=0.05, height=0.05, canonical_label="fireball")]

    events = eventizer.eventize(first + second + third)

    assert events == []


def test_confirmed_transition_emits_one_card_event() -> None:
    eventizer = HandEventizer(EventizerConfig(confidence_threshold=0.5, stability_observations=3))
    detections = [
        FrameDetection(timestamp=0.0, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_a.jpg"),
        FrameDetection(timestamp=0.4, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_b.jpg"),
        FrameDetection(timestamp=0.8, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_c.jpg"),
        FrameDetection(timestamp=1.2, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="fireball", source_frame="frame_d.jpg"),
    ]

    events = eventizer.eventize(detections)

    assert len(events) == 1
    assert isinstance(events[0], CardEvent)
    assert events[0].card == "cannon"
    assert events[0].player == "self"
    assert events[0].source_frame == "frame_d.jpg"


def test_ambiguous_transition_is_logged_and_rejected() -> None:
    eventizer = HandEventizer(EventizerConfig(confidence_threshold=0.5, stability_observations=3))
    detections = [
        FrameDetection(timestamp=0.0, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_a.jpg"),
        FrameDetection(timestamp=0.4, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_b.jpg"),
        FrameDetection(timestamp=0.8, label="x", confidence=0.9, x_center=0.18, y_center=0.2, width=0.05, height=0.05, canonical_label="cannon", source_frame="frame_c.jpg"),
        FrameDetection(timestamp=1.2, label="x", confidence=0.9, x_center=0.38, y_center=0.2, width=0.05, height=0.05, canonical_label="fireball", source_frame="frame_d.jpg"),
        FrameDetection(timestamp=1.6, label="x", confidence=0.9, x_center=0.58, y_center=0.2, width=0.05, height=0.05, canonical_label="skeletons", source_frame="frame_e.jpg"),
    ]

    events = eventizer.eventize(detections)

    assert events == []
    assert any(entry["reason"] == "ambiguous_transition" for entry in eventizer.diagnostics)
