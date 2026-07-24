from __future__ import annotations

from dataclasses import dataclass, field

from cr_vision.arena import tile_by_id
from cr_vision.cards import card_cost


CARDS_NEEDED_TO_CYCLE = 4


@dataclass(frozen=True)
class CardEvent:
    time: float
    player: str
    card: str
    tile: str | None = None
    x: float | None = None
    y: float | None = None
    confidence: float | None = None
    source_frame: str | None = None


@dataclass(frozen=True)
class UnitState:
    card: str
    owner: str
    tile: str
    deployed_at: float
    confidence: float | None = None
    source_frame: str | None = None


@dataclass(frozen=True)
class TowerState:
    tower_id: str
    owner: str
    tile: str
    hitpoints: int | None = None


@dataclass(frozen=True)
class BoardState:
    units: tuple[UnitState, ...] = ()
    towers: tuple[TowerState, ...] = ()

    def with_deployment(self, event: CardEvent) -> "BoardState":
        if event.tile is None:
            return self

        tile_by_id(event.tile)
        unit = UnitState(
            card=event.card,
            owner=event.player,
            tile=event.tile,
            deployed_at=event.time,
            confidence=event.confidence,
            source_frame=event.source_frame,
        )
        return BoardState(units=(*self.units, unit), towers=self.towers)

    @property
    def occupied_tile_ids(self) -> tuple[str, ...]:
        return tuple(unit.tile for unit in self.units)


@dataclass(frozen=True)
class ElixirRules:
    initial_elixir: float = 5.0
    max_elixir: float = 10.0
    seconds_per_elixir: float = 2.8
    double_elixir_start: float = 120.0
    triple_elixir_start: float = 180.0


@dataclass(frozen=True)
class CardCycleStatus:
    card: str
    last_played_at: float
    plays_since_last_seen: int
    plays_until_available: int

    @property
    def can_be_available(self) -> bool:
        return self.plays_until_available == 0


@dataclass(frozen=True)
class StateSnapshot:
    time: float
    event: CardEvent
    opponent_elixir: float
    known_deck: tuple[str, ...]
    unavailable_cards: tuple[str, ...]
    available_known_cards: tuple[str, ...]
    cycle_statuses: tuple[CardCycleStatus, ...]
    board: BoardState = field(default_factory=BoardState)


@dataclass(frozen=True)
class GameState:
    time: float
    opponent_elixir: float
    known_deck: tuple[str, ...]
    unavailable_cards: tuple[str, ...]
    available_known_cards: tuple[str, ...]
    board: BoardState


@dataclass
class ElixirState:
    rules: ElixirRules = field(default_factory=ElixirRules)
    current: float = field(init=False)
    last_time: float = 0.0

    def __post_init__(self) -> None:
        self.current = self.rules.initial_elixir

    def advance_to(self, timestamp: float) -> None:
        if timestamp < self.last_time:
            raise ValueError("Events must be sorted by time")

        gained = elixir_gained_between(self.last_time, timestamp, self.rules)
        self.current = min(self.rules.max_elixir, self.current + gained)
        self.last_time = timestamp

    def spend(self, card_name: str) -> None:
        self.current = max(0.0, self.current - card_cost(card_name))


@dataclass
class OpponentTracker:
    rules: ElixirRules = field(default_factory=ElixirRules)
    played_cards: list[str] = field(default_factory=list)
    played_events: list[CardEvent] = field(default_factory=list)
    board: BoardState = field(default_factory=BoardState)
    elixir: ElixirState = field(init=False)

    def __post_init__(self) -> None:
        self.elixir = ElixirState(rules=self.rules)

    def observe(self, event: CardEvent) -> StateSnapshot | None:
        self.board = self.board.with_deployment(event)

        if event.player != "opponent":
            return None

        self.elixir.advance_to(event.time)
        self.elixir.spend(event.card)
        self.played_cards.append(event.card)
        self.played_events.append(event)
        return self.snapshot(event)

    @property
    def known_deck(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.played_cards))

    @property
    def unavailable_cards(self) -> tuple[str, ...]:
        return tuple(
            status.card
            for status in self.cycle_statuses
            if not status.can_be_available
        )

    @property
    def available_known_cards(self) -> tuple[str, ...]:
        return tuple(
            status.card
            for status in self.cycle_statuses
            if status.can_be_available
        )

    @property
    def cycle_statuses(self) -> tuple[CardCycleStatus, ...]:
        statuses: list[CardCycleStatus] = []
        for card_name in self.known_deck:
            last_index = (
                len(self.played_cards) - 1 - self.played_cards[::-1].index(card_name)
            )
            plays_since_last_seen = len(self.played_cards) - last_index - 1
            plays_until_available = max(
                0,
                CARDS_NEEDED_TO_CYCLE - plays_since_last_seen,
            )
            statuses.append(
                CardCycleStatus(
                    card=card_name,
                    last_played_at=self.played_events[last_index].time,
                    plays_since_last_seen=plays_since_last_seen,
                    plays_until_available=plays_until_available,
                )
            )

        return tuple(
            sorted(
                statuses,
                key=lambda status: (
                    status.plays_until_available,
                    -status.last_played_at,
                ),
            )
        )

    def snapshot(self, event: CardEvent) -> StateSnapshot:
        return StateSnapshot(
            time=event.time,
            event=event,
            opponent_elixir=self.elixir.current,
            known_deck=self.known_deck,
            unavailable_cards=self.unavailable_cards,
            available_known_cards=self.available_known_cards,
            cycle_statuses=self.cycle_statuses,
            board=self.board,
        )

    def game_state(self, time: float) -> GameState:
        return GameState(
            time=time,
            opponent_elixir=self.elixir.current,
            known_deck=self.known_deck,
            unavailable_cards=self.unavailable_cards,
            available_known_cards=self.available_known_cards,
            board=self.board,
        )


def elixir_gained_between(
    start: float,
    end: float,
    rules: ElixirRules | None = None,
) -> float:
    if end < start:
        raise ValueError("end must be greater than or equal to start")

    active_rules = rules or ElixirRules()
    return (
        _segment_gain(
            start,
            min(end, active_rules.double_elixir_start),
            1.0,
            active_rules,
        )
        + _segment_gain(
            max(start, active_rules.double_elixir_start),
            min(end, active_rules.triple_elixir_start),
            2.0,
            active_rules,
        )
        + _segment_gain(
            max(start, active_rules.triple_elixir_start),
            end,
            3.0,
            active_rules,
        )
    )


def _segment_gain(
    start: float,
    end: float,
    multiplier: float,
    rules: ElixirRules,
) -> float:
    if end <= start:
        return 0.0

    seconds_per_elixir = rules.seconds_per_elixir / multiplier
    return (end - start) / seconds_per_elixir
