from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cr_vision.state import CardEvent, OpponentTracker, StateSnapshot


class EventRecord(BaseModel):
    time: float = Field(ge=0)
    player: Literal["self", "opponent"]
    card: str
    tile: str | None = None
    x: float | None = None
    y: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_frame: str | None = None


class CycleStatusRecord(BaseModel):
    card: str
    last_played_at: float
    plays_since_last_seen: int
    plays_until_available: int
    can_be_available: bool


class UnitRecord(BaseModel):
    card: str
    owner: str
    tile: str
    deployed_at: float
    confidence: float | None
    source_frame: str | None


class SnapshotRecord(BaseModel):
    time: float
    card: str
    opponent_elixir: float
    known_deck: list[str]
    unavailable_cards: list[str]
    available_known_cards: list[str]
    cycle_statuses: list[CycleStatusRecord]
    deployed_units: list[UnitRecord]


class MatchAnalysis(BaseModel):
    snapshots: list[SnapshotRecord]
    final: SnapshotRecord | None


def load_events(path: Path) -> list[CardEvent]:
    raw_events = json.loads(path.read_text(encoding="utf-8"))
    records = [EventRecord.model_validate(event) for event in raw_events]
    return [
        CardEvent(
            time=record.time,
            player=record.player,
            card=record.card,
            tile=record.tile,
            x=record.x,
            y=record.y,
            confidence=record.confidence,
            source_frame=record.source_frame,
        )
        for record in sorted(records, key=lambda item: item.time)
    ]


def analyze_events(events: list[CardEvent]) -> tuple[OpponentTracker, list[StateSnapshot]]:
    tracker = OpponentTracker()
    snapshots: list[StateSnapshot] = []
    for event in sorted(events, key=lambda item: item.time):
        snapshot = tracker.observe(event)
        if snapshot is not None:
            snapshots.append(snapshot)
    return tracker, snapshots


def build_report(snapshots: list[StateSnapshot]) -> MatchAnalysis:
    records = [_snapshot_record(snapshot) for snapshot in snapshots]
    return MatchAnalysis(snapshots=records, final=records[-1] if records else None)


def write_report(path: Path, snapshots: list[StateSnapshot]) -> None:
    report = build_report(snapshots)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _snapshot_record(snapshot: StateSnapshot) -> SnapshotRecord:
    return SnapshotRecord(
        time=snapshot.time,
        card=snapshot.event.card,
        opponent_elixir=round(snapshot.opponent_elixir, 2),
        known_deck=list(snapshot.known_deck),
        unavailable_cards=list(snapshot.unavailable_cards),
        available_known_cards=list(snapshot.available_known_cards),
        cycle_statuses=[
            CycleStatusRecord(
                card=status.card,
                last_played_at=status.last_played_at,
                plays_since_last_seen=status.plays_since_last_seen,
                plays_until_available=status.plays_until_available,
                can_be_available=status.can_be_available,
            )
            for status in snapshot.cycle_statuses
        ],
        deployed_units=[
            UnitRecord(
                card=unit.card,
                owner=unit.owner,
                tile=unit.tile,
                deployed_at=unit.deployed_at,
                confidence=unit.confidence,
                source_frame=unit.source_frame,
            )
            for unit in snapshot.board.units
        ],
    )
