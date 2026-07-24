from cr_vision.analyzer import analyze_events
from cr_vision.detection import Detection
from cr_vision.detector_adapter import detections_to_events


def test_detections_to_events_filters_sorts_and_maps() -> None:
    detections = [
        Detection(timestamp=4.0, card="cannon", confidence=0.95),
        Detection(timestamp=1.0, card="hog_rider", confidence=0.90),
        Detection(timestamp=2.5, card="fireball", confidence=0.20),
    ]

    events = detections_to_events(detections, player="opponent", min_confidence=0.5)

    assert [(event.time, event.card, event.player, event.confidence) for event in events] == [
        (1.0, "hog_rider", "opponent", 0.90),
        (4.0, "cannon", "opponent", 0.95),
    ]
    assert all(event.tile is None for event in events)
    assert all(event.x is None for event in events)
    assert all(event.y is None for event in events)
    assert all(event.source_frame is None for event in events)


def test_adapter_events_feed_into_analyzer_pipeline() -> None:
    detections = [
        Detection(timestamp=2.8, card="cannon", confidence=0.88),
        Detection(timestamp=1.0, card="hog_rider", confidence=0.93),
        Detection(timestamp=4.0, card="fireball", confidence=0.82),
    ]

    events = detections_to_events(detections, player="opponent", min_confidence=0.8)
    tracker, snapshots = analyze_events(events)

    assert [snapshot.time for snapshot in snapshots] == [1.0, 2.8, 4.0]
    assert [snapshot.event.card for snapshot in snapshots] == ["hog_rider", "cannon", "fireball"]
    assert tracker.known_deck == ("hog_rider", "cannon", "fireball")
    assert round(snapshots[0].opponent_elixir, 2) == 1.36
