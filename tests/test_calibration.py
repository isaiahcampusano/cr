from pathlib import Path

from cr_vision.calibration import GridCalibration, load_calibration, map_point_to_tile


def test_map_point_to_regular_tiles() -> None:
    calibration = GridCalibration(
        image_width=160,
        image_height=330,
        board_left=0,
        board_top=0,
        board_width=160,
        board_height=330,
    )

    assert map_point_to_tile(calibration, 105, 275) == "self:regular:10:10"
    assert map_point_to_tile(calibration, 105, 65) == "opponent:regular:10:6"


def test_map_point_to_river_and_bridge_tiles() -> None:
    calibration = GridCalibration(
        image_width=160,
        image_height=330,
        board_left=0,
        board_top=0,
        board_width=160,
        board_height=330,
    )

    assert map_point_to_tile(calibration, 80, 165) == "neutral:river:12:0"
    assert map_point_to_tile(calibration, 32, 165) == "neutral:bridge:left:0"
    assert map_point_to_tile(calibration, 123, 165) == "neutral:bridge:right:2"


def test_point_outside_board_returns_none() -> None:
    calibration = GridCalibration(
        image_width=160,
        image_height=330,
        board_left=10,
        board_top=10,
        board_width=100,
        board_height=200,
    )

    assert map_point_to_tile(calibration, 5, 20) is None


def test_load_calibration_from_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "image_width": 160,
          "image_height": 330,
          "board": { "left": 0, "top": 0, "width": 160, "height": 330 }
        }
        """,
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration.image_width == 160
    assert calibration.board_width == 160
