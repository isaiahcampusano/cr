from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cr_vision.arena import (
    BRIDGE_TILES_PER_LANE,
    REGULAR_COLUMNS,
    REGULAR_ROWS_PER_SIDE,
    RIVER_TILE_COUNT,
    bridge_tile_id,
    regular_tile_id,
    river_tile_id,
    tile_by_id,
)


@dataclass(frozen=True)
class GridCalibration:
    image_width: int
    image_height: int
    board_left: float
    board_top: float
    board_width: float
    board_height: float


def load_calibration(path: Path) -> GridCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    board = payload.get("board", payload)
    return GridCalibration(
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        board_left=float(board["left"]),
        board_top=float(board["top"]),
        board_width=float(board["width"]),
        board_height=float(board["height"]),
    )


def map_point_to_tile(calibration: GridCalibration, x: float, y: float) -> str | None:
    if not _point_inside_board(calibration, x, y):
        return None

    relative_x = (x - calibration.board_left) / calibration.board_width
    relative_y = (y - calibration.board_top) / calibration.board_height
    regular_column = min(REGULAR_COLUMNS - 1, int(relative_x * REGULAR_COLUMNS))
    logical_rows = (REGULAR_ROWS_PER_SIDE * 2) + 1
    row = min(logical_rows - 1, int(relative_y * logical_rows))

    if row < REGULAR_ROWS_PER_SIDE:
        tile_id = regular_tile_id("opponent", regular_column, row)
    elif row > REGULAR_ROWS_PER_SIDE:
        self_row = row - REGULAR_ROWS_PER_SIDE - 1
        tile_id = regular_tile_id("self", regular_column, self_row)
    else:
        tile_id = _river_or_bridge_tile(relative_x)

    return tile_by_id(tile_id).tile_id


def _point_inside_board(calibration: GridCalibration, x: float, y: float) -> bool:
    return (
        calibration.board_left <= x < calibration.board_left + calibration.board_width
        and calibration.board_top <= y < calibration.board_top + calibration.board_height
    )


def _river_or_bridge_tile(relative_x: float) -> str:
    if 0.18 <= relative_x < 0.32:
        lane_x = (relative_x - 0.18) / 0.14
        index = min(BRIDGE_TILES_PER_LANE - 1, int(lane_x * BRIDGE_TILES_PER_LANE))
        return bridge_tile_id("left", index)
    if 0.68 <= relative_x < 0.82:
        lane_x = (relative_x - 0.68) / 0.14
        index = min(BRIDGE_TILES_PER_LANE - 1, int(lane_x * BRIDGE_TILES_PER_LANE))
        return bridge_tile_id("right", index)

    river_x = min(RIVER_TILE_COUNT - 1, int(relative_x * RIVER_TILE_COUNT))
    return river_tile_id(river_x)
