from cr_vision.state import CardEvent, OpponentTracker, elixir_gained_between


def test_single_elixir_generation_rate() -> None:
    assert round(elixir_gained_between(0, 2.8), 2) == 1.0


def test_tracker_subtracts_visible_opponent_cards() -> None:
    tracker = OpponentTracker()
    tracker.observe(CardEvent(time=2.8, player="opponent", card="hog_rider"))

    assert round(tracker.elixir.current, 2) == 2.0
    assert tracker.known_deck == ["hog_rider"]


def test_likely_hand_uses_recent_four_visible_cards() -> None:
    tracker = OpponentTracker()
    cards = ["hog_rider", "cannon", "fireball", "skeletons", "musketeer"]

    for index, card in enumerate(cards):
        tracker.observe(CardEvent(time=10.0 + index * 5.0, player="opponent", card=card))

    assert tracker.likely_hand == ["cannon", "fireball", "skeletons", "musketeer"]

