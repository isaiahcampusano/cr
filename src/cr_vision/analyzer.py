from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from cr_vision.state import CardEvent, OpponentTracker


class EventRecord(BaseModel):
    time: float = Field(ge=0)
    player: str
    card: str


def load_events(path: Path) -> list[CardEvent]:
    raw_events = json.loads(path.read_text(encoding="utf-8"))
    records = [EventRecord.model_validate(event) for event in raw_events]
    return [
        CardEvent(time=record.time, player=record.player, card=record.card)
        for record in sorted(records, key=lambda item: item.time)
    ]


def analyze_events(events: list[CardEvent]) -> OpponentTracker:
    tracker = OpponentTracker()
    for event in events:
        tracker.observe(event)
    return tracker

