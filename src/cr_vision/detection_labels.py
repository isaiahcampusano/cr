from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from cr_vision.detection import Detection


class DetectionLabelRecord(BaseModel):
    timestamp: float = Field(ge=0)
    card: str
    confidence: float = Field(ge=0, le=1)
    source_frame: str | None = None


def load_detections(path: Path) -> list[Detection]:
    raw_labels = json.loads(path.read_text(encoding="utf-8"))
    records = [DetectionLabelRecord.model_validate(label) for label in raw_labels]
    sorted_records = sorted(records, key=lambda record: record.timestamp)
    return [
        Detection(
            timestamp=record.timestamp,
            card=record.card,
            confidence=record.confidence,
            source_frame=record.source_frame,
        )
        for record in sorted_records
    ]
