import pytest

from cr_vision.arena import (
    ARENA_TILES,
    BOARD_TILE_COUNT,
    BRIDGE_TILE_COUNT,
    REGULAR_TILE_COUNT_PER_SIDE,
    RIVER_TILE_COUNT,
    regular_tile_id,
    tile_by_id,
    tile_counts_by_region,
    validate_arena,
)


def test_arena_has_expected_tile_counts() -> None:
    validate_arena()

    counts = tile_counts_by_region()
    assert len(ARENA_TILES) == BOARD_TILE_COUNT == 544
    assert counts["regular_self"] == REGULAR_TILE_COUNT_PER_SIDE == 256
    assert counts["regular_opponent"] == REGULAR_TILE_COUNT_PER_SIDE == 256
    assert counts["river"] == RIVER_TILE_COUNT == 24
    assert counts["bridge"] == BRIDGE_TILE_COUNT == 8


def test_regular_tile_ids_are_stable_and_parseable() -> None:
    tile = tile_by_id(regular_tile_id("self", 10, 6))

    assert tile.tile_id == "self:regular:10:6"
    assert tile.position.side == "self"
    assert tile.position.region == "regular"
    assert tile.position.x == 10
    assert tile.position.y == 6


def test_invalid_tile_ids_fail_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown arena tile"):
        tile_by_id("self:regular:99:99")


def test_out_of_bounds_regular_coordinates_fail_clearly() -> None:
    with pytest.raises(ValueError, match="regular tile coordinates"):
        regular_tile_id("self", 16, 0)
