from __future__ import annotations

from cr_vision.detection import FrameDetection
from cr_vision.eventizer import EventizerConfig, HandEventizer
from cr_vision.state import CardEvent


SLOT_CENTERS = (0.18, 0.38, 0.58, 0.78)
INITIAL_HAND = ("cannon", "fireball", "skeletons", "ice_spirit")
ROTATED_HAND = ("fireball", "skeletons", "ice_spirit", "musketeer")


def _frame(
    timestamp: float,
    slots: tuple[str | None, str | None, str | None, str | None],
    *,
    source_frame: str,
    confidence: float = 0.9,
) -> list[FrameDetection]:
    return [
        FrameDetection(
            timestamp=timestamp,
            label=card,
            confidence=confidence,
            x_center=SLOT_CENTERS[index],
            y_center=0.9,
            width=0.08,
            height=0.1,
            source_frame=source_frame,
            canonical_label=card,
        )
        for index, card in enumerate(slots)
        if card is not None
    ]


def _eventizer() -> HandEventizer:
    return HandEventizer(
        EventizerConfig(
            confidence_threshold=0.5,
            stability_observations=3,
            stability_window=5,
        )
    )


def test_fixed_slots_do_not_renumber_after_a_missing_middle_detection() -> None:
    eventizer = _eventizer()
    detections = _frame(
        0.0,
        ("cannon", None, "skeletons", "ice_spirit"),
        source_frame="frame_000000.jpg",
    )

    observation = eventizer.observe_detections(detections)[0]

    assert observation.slots == ("cannon", None, "skeletons", "ice_spirit")


def test_conflicting_detections_in_one_slot_are_diagnosed() -> None:
    eventizer = _eventizer()
    detections = _frame(
        0.0,
        ("cannon", None, None, None),
        source_frame="frame_000000.jpg",
        confidence=0.8,
    ) + [
        FrameDetection(
            timestamp=0.0,
            label="fireball",
            confidence=0.9,
            x_center=SLOT_CENTERS[0],
            y_center=0.9,
            width=0.08,
            height=0.1,
            source_frame="frame_000000.jpg",
            canonical_label="fireball",
        )
    ]

    observation = eventizer.observe_detections(detections)[0]

    assert observation.slots[0] == "fireball"
    assert eventizer.diagnostics[0]["reason"] == "slot_conflict"


def test_repeated_stable_hand_with_varying_confidence_emits_no_event() -> None:
    eventizer = _eventizer()
    detections = (
        _frame(0.0, INITIAL_HAND, source_frame="frame_000000.jpg", confidence=0.91)
        + _frame(0.4, INITIAL_HAND, source_frame="frame_000001.jpg", confidence=0.84)
        + _frame(0.8, INITIAL_HAND, source_frame="frame_000002.jpg", confidence=0.96)
        + _frame(1.2, INITIAL_HAND, source_frame="frame_000003.jpg", confidence=0.88)
    )

    assert eventizer.eventize(detections) == []


def test_one_frame_flicker_does_not_emit_a_play() -> None:
    eventizer = _eventizer()
    detections = (
        _frame(0.0, INITIAL_HAND, source_frame="frame_000000.jpg")
        + _frame(0.4, INITIAL_HAND, source_frame="frame_000001.jpg")
        + _frame(0.8, INITIAL_HAND, source_frame="frame_000002.jpg")
        + _frame(1.2, ROTATED_HAND, source_frame="frame_000003.jpg")
        + _frame(1.6, INITIAL_HAND, source_frame="frame_000004.jpg")
        + _frame(2.0, INITIAL_HAND, source_frame="frame_000005.jpg")
    )

    assert eventizer.eventize(detections) == []


def test_confirmed_complete_hand_transition_emits_one_play() -> None:
    eventizer = _eventizer()
    detections = (
        _frame(0.0, INITIAL_HAND, source_frame="old_0.jpg")
        + _frame(0.4, INITIAL_HAND, source_frame="old_1.jpg")
        + _frame(0.8, INITIAL_HAND, source_frame="old_2.jpg")
        + _frame(1.2, ROTATED_HAND, source_frame="new_0.jpg")
        + _frame(1.6, ROTATED_HAND, source_frame="new_1.jpg")
        + _frame(2.0, ROTATED_HAND, source_frame="new_2.jpg")
        + _frame(2.4, ROTATED_HAND, source_frame="new_3.jpg")
    )

    events = eventizer.eventize(detections)

    assert events == [
        CardEvent(
            time=2.0,
            player="self",
            card="cannon",
            confidence=0.9,
            source_frame="new_2.jpg",
        )
    ]


def test_ambiguous_complete_hand_transition_is_logged_and_rejected() -> None:
    eventizer = _eventizer()
    ambiguous_hand = ("fireball", "skeletons", "musketeer", "knight")
    detections = (
        _frame(0.0, INITIAL_HAND, source_frame="old_0.jpg")
        + _frame(0.4, INITIAL_HAND, source_frame="old_1.jpg")
        + _frame(0.8, INITIAL_HAND, source_frame="old_2.jpg")
        + _frame(1.2, ambiguous_hand, source_frame="new_0.jpg")
        + _frame(1.6, ambiguous_hand, source_frame="new_1.jpg")
        + _frame(2.0, ambiguous_hand, source_frame="new_2.jpg")
    )

    assert eventizer.eventize(detections) == []
    assert eventizer.diagnostics == [
        {
            "timestamp": 2.0,
            "reason": "ambiguous_transition",
            "previous_slots": list(INITIAL_HAND),
            "current_slots": list(ambiguous_hand),
            "removed_cards": ["cannon", "ice_spirit"],
            "added_cards": ["musketeer", "knight"],
        }
    ]


def test_eventizer_config_rejects_an_invalid_stability_window() -> None:
    try:
        EventizerConfig(stability_observations=4, stability_window=3)
    except ValueError as exc:
        assert "stability_window" in str(exc)
    else:
        raise AssertionError("Expected invalid eventizer configuration to fail")
