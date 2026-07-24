from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


REGULAR_COLUMNS = 16
REGULAR_ROWS_PER_SIDE = 16
REGULAR_TILE_COUNT_PER_SIDE = REGULAR_COLUMNS * REGULAR_ROWS_PER_SIDE
RIVER_TILE_COUNT = 24
BRIDGE_TILE_COUNT = 8
BRIDGE_TILES_PER_LANE = 4
BOARD_TILE_COUNT = (REGULAR_TILE_COUNT_PER_SIDE * 2) + RIVER_TILE_COUNT + BRIDGE_TILE_COUNT

PlayerSide = Literal["self", "opponent"]
TileRegion = Literal["regular", "river", "bridge"]


@dataclass(frozen=True)
class BoardPosition:
    x: int
    y: int
    region: TileRegion
    side: PlayerSide | None = None
    lane: str | None = None


@dataclass(frozen=True)
class ArenaTile:
    tile_id: str
    position: BoardPosition
    playable: bool = True


def regular_tile_id(side: PlayerSide, x: int, y: int) -> str:
    if side not in ("self", "opponent"):
        raise ValueError("side must be 'self' or 'opponent'")
    if not is_regular_coordinate(x, y):
        raise ValueError("regular tile coordinates must be within 0..15")
    return f"{side}:regular:{x}:{y}"


def river_tile_id(x: int) -> str:
    if not 0 <= x < RIVER_TILE_COUNT:
        raise ValueError("river tile x must be within 0..23")
    return f"neutral:river:{x}:0"


def bridge_tile_id(lane: str, index: int) -> str:
    if lane not in ("left", "right"):
        raise ValueError("bridge lane must be 'left' or 'right'")
    if not 0 <= index < BRIDGE_TILES_PER_LANE:
        raise ValueError("bridge tile index must be within 0..3")
    return f"neutral:bridge:{lane}:{index}"


def is_regular_coordinate(x: int, y: int) -> bool:
    return 0 <= x < REGULAR_COLUMNS and 0 <= y < REGULAR_ROWS_PER_SIDE


def build_arena_tiles() -> tuple[ArenaTile, ...]:
    tiles: list[ArenaTile] = []
    for side in ("self", "opponent"):
        for y in range(REGULAR_ROWS_PER_SIDE):
            for x in range(REGULAR_COLUMNS):
                tiles.append(
                    ArenaTile(
                        tile_id=regular_tile_id(side, x, y),
                        position=BoardPosition(x=x, y=y, region="regular", side=side),
                    )
                )

    for x in range(RIVER_TILE_COUNT):
        tiles.append(
            ArenaTile(
                tile_id=river_tile_id(x),
                position=BoardPosition(x=x, y=0, region="river"),
            )
        )

    for lane in ("left", "right"):
        for index in range(BRIDGE_TILES_PER_LANE):
            tiles.append(
                ArenaTile(
                    tile_id=bridge_tile_id(lane, index),
                    position=BoardPosition(
                        x=index,
                        y=0,
                        region="bridge",
                        lane=lane,
                    ),
                )
            )

    return tuple(tiles)


ARENA_TILES: tuple[ArenaTile, ...] = build_arena_tiles()
TILES_BY_ID: dict[str, ArenaTile] = {tile.tile_id: tile for tile in ARENA_TILES}


def tile_by_id(tile_id: str) -> ArenaTile:
    try:
        return TILES_BY_ID[tile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown arena tile '{tile_id}'") from exc


def tile_counts_by_region() -> dict[str, int]:
    counts = {"regular_self": 0, "regular_opponent": 0, "river": 0, "bridge": 0}
    for tile in ARENA_TILES:
        if tile.position.region == "regular" and tile.position.side == "self":
            counts["regular_self"] += 1
        elif tile.position.region == "regular" and tile.position.side == "opponent":
            counts["regular_opponent"] += 1
        elif tile.position.region == "river":
            counts["river"] += 1
        elif tile.position.region == "bridge":
            counts["bridge"] += 1
    return counts


def validate_arena() -> None:
    if len(ARENA_TILES) != BOARD_TILE_COUNT:
        raise ValueError(f"Expected {BOARD_TILE_COUNT} tiles, found {len(ARENA_TILES)}")
    if len(TILES_BY_ID) != len(ARENA_TILES):
        raise ValueError("Arena tile IDs must be unique")

    counts = tile_counts_by_region()
    if counts["regular_self"] != REGULAR_TILE_COUNT_PER_SIDE:
        raise ValueError("Self regular half must contain 256 tiles")
    if counts["regular_opponent"] != REGULAR_TILE_COUNT_PER_SIDE:
        raise ValueError("Opponent regular half must contain 256 tiles")
    if counts["river"] != RIVER_TILE_COUNT:
        raise ValueError("River must contain 24 tiles")
    if counts["bridge"] != BRIDGE_TILE_COUNT:
        raise ValueError("Bridge must contain 8 tiles")
