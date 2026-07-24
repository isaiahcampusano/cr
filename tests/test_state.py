from cr_vision.analyzer import analyze_events, build_report
from cr_vision.cards import DEFAULT_EVALUATION_DECK, card_cost
from cr_vision.state import CardEvent, ElixirState, OpponentTracker, elixir_gained_between


def test_single_elixir_generation_rate() -> None:
    assert round(elixir_gained_between(0, 2.8), 2) == 1.0


def test_elixir_generation_crosses_double_and_triple_segments() -> None:
    assert round(elixir_gained_between(119.0, 121.0), 2) == 1.07
    assert round(elixir_gained_between(179.0, 181.0), 2) == 1.79


def test_elixir_caps_at_ten() -> None:
    elixir = ElixirState()
    elixir.advance_to(100.0)

    assert elixir.current == 10.0


def test_tracker_subtracts_visible_opponent_cards() -> None:
    tracker = OpponentTracker()
    tracker.observe(CardEvent(time=2.8, player="opponent", card="hog_rider"))

    assert round(tracker.elixir.current, 2) == 2.0
    assert tracker.known_deck == ("hog_rider",)


def test_self_events_do_not_change_opponent_public_state() -> None:
    tracker = OpponentTracker()
    snapshot = tracker.observe(CardEvent(time=10.0, player="self", card="hog_rider"))

    assert snapshot is None
    assert tracker.played_cards == []
    assert tracker.elixir.current == 5.0


def test_recent_visible_plays_are_marked_unavailable() -> None:
    tracker = OpponentTracker()
    cards = ["hog_rider", "cannon", "fireball", "skeletons"]

    for index, card in enumerate(cards):
        tracker.observe(CardEvent(time=10.0 + index * 5.0, player="opponent", card=card))

    assert tracker.unavailable_cards == ("hog_rider", "cannon", "fireball", "skeletons")
    assert tracker.available_known_cards == ()


def test_card_can_be_available_after_four_other_visible_plays() -> None:
    tracker = OpponentTracker()
    cards = ["hog_rider", "cannon", "fireball", "skeletons", "musketeer"]

    for index, card in enumerate(cards):
        tracker.observe(CardEvent(time=10.0 + index * 5.0, player="opponent", card=card))

    assert tracker.available_known_cards == ("hog_rider",)
    assert tracker.unavailable_cards == ("cannon", "fireball", "skeletons", "musketeer")


def test_analyzer_sorts_events_and_builds_report() -> None:
    events = [
        CardEvent(time=9.8, player="opponent", card="cannon"),
        CardEvent(time=4.2, player="opponent", card="hog_rider"),
    ]

    _tracker, snapshots = analyze_events(events)
    report = build_report(snapshots)

    assert [snapshot.card for snapshot in report.snapshots] == ["hog_rider", "cannon"]
    assert report.final is not None
    assert report.final.known_deck == ["hog_rider", "cannon"]
    assert report.final.cycle_statuses[0].card == "hog_rider"
    assert report.final.cycle_statuses[0].plays_until_available == 3


def test_positioned_deployment_events_update_board_state() -> None:
    tracker = OpponentTracker()
    snapshot = tracker.observe(
        CardEvent(
            time=12.4,
            player="opponent",
            card="hog_rider",
            tile="self:regular:10:6",
            confidence=0.95,
            source_frame="frame_000062.jpg",
        )
    )

    assert snapshot is not None
    assert tracker.board.occupied_tile_ids == ("self:regular:10:6",)
    assert snapshot.board.units[0].card == "hog_rider"
    assert snapshot.board.units[0].source_frame == "frame_000062.jpg"


def test_card_only_events_do_not_require_board_positions() -> None:
    tracker = OpponentTracker()
    snapshot = tracker.observe(CardEvent(time=12.4, player="opponent", card="hog_rider"))

    assert snapshot is not None
    assert snapshot.board.units == ()


def test_self_positioned_events_do_not_change_opponent_cycle() -> None:
    tracker = OpponentTracker()
    snapshot = tracker.observe(
        CardEvent(
            time=12.4,
            player="self",
            card="hog_rider",
            tile="self:regular:10:6",
        )
    )

    assert snapshot is None
    assert tracker.played_cards == []
    assert tracker.elixir.current == 5.0
    assert tracker.board.occupied_tile_ids == ("self:regular:10:6",)


def test_hog_cycle_deck_has_eight_known_cards_with_costs() -> None:
    assert DEFAULT_EVALUATION_DECK == (
        "hog_rider",
        "musketeer",
        "cannon",
        "fireball",
        "the_log",
        "ice_spirit",
        "skeletons",
        "ice_golem",
    )
    assert len(DEFAULT_EVALUATION_DECK) == 8
    assert all(card_cost(card) > 0 for card in DEFAULT_EVALUATION_DECK)
