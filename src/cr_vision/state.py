from __future__ import annotations

from dataclasses import dataclass, field

from cr_vision.cards import card_cost


DOUBLE_ELIXIR_START_SECONDS = 120.0
TRIPLE_ELIXIR_START_SECONDS = 180.0
MAX_ELIXIR = 10.0


@dataclass(frozen=True)
class CardEvent:
    time: float
    player: str
    card: str


@dataclass
class ElixirState:
    current: float = 5.0
    last_time: float = 0.0

    def advance_to(self, timestamp: float) -> None:
        if timestamp < self.last_time:
            raise ValueError("Events must be sorted by time")

        gained = elixir_gained_between(self.last_time, timestamp)
        self.current = min(MAX_ELIXIR, self.current + gained)
        self.last_time = timestamp

    def spend(self, card_name: str) -> None:
        self.current = max(0.0, self.current - card_cost(card_name))


@dataclass
class OpponentTracker:
    elixir: ElixirState = field(default_factory=ElixirState)
    played_cards: list[str] = field(default_factory=list)

    def observe(self, event: CardEvent) -> None:
        if event.player != "opponent":
            return

        self.elixir.advance_to(event.time)
        self.elixir.spend(event.card)
        self.played_cards.append(event.card)

    @property
    def known_deck(self) -> list[str]:
        return list(dict.fromkeys(self.played_cards))

    @property
    def likely_hand(self) -> list[str]:
        if len(self.played_cards) < 4:
            return self.played_cards[:]
        return self.played_cards[-4:]


def elixir_gained_between(start: float, end: float) -> float:
    if end < start:
        raise ValueError("end must be greater than or equal to start")

    return (
        _segment_gain(start, min(end, DOUBLE_ELIXIR_START_SECONDS), 1.0)
        + _segment_gain(
            max(start, DOUBLE_ELIXIR_START_SECONDS),
            min(end, TRIPLE_ELIXIR_START_SECONDS),
            2.0,
        )
        + _segment_gain(max(start, TRIPLE_ELIXIR_START_SECONDS), end, 3.0)
    )


def _segment_gain(start: float, end: float, multiplier: float) -> float:
    if end <= start:
        return 0.0

    seconds_per_elixir = 2.8 / multiplier
    return (end - start) / seconds_per_elixir

