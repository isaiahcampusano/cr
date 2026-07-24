from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cr_vision.frames import extract_frames, load_manifest


def test_extract_frames_writes_expected_count_and_manifest(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "match.avi")
    output_dir = tmp_path / "frames"

    manifest = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=5.0,
        match_start_offset=0.4,
    )

    assert manifest.frame_count == 8
    assert manifest.frames[0].filename == "frame_000000.jpg"
    assert manifest.frames[0].video_time_seconds == 0.4
    assert manifest.frames[0].match_time_seconds == 0.0
    assert manifest.frames[-1].video_time_seconds == 1.8
    assert (output_dir / "manifest.json").exists()
    assert len(list(output_dir.glob("frame_*.jpg"))) == 8


def test_extract_frames_respects_max_seconds(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "match.avi")
    output_dir = tmp_path / "frames"

    manifest = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=5.0,
        match_start_offset=0.4,
        max_seconds=0.5,
    )

    assert manifest.frame_count == 3
    assert [record.match_time_seconds for record in manifest.frames] == [0.0, 0.2, 0.4]


def test_extract_frames_applies_crop_and_resize(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "match.avi")
    output_dir = tmp_path / "frames"

    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=1.0,
        match_start_offset=0.0,
        max_seconds=0.5,
        crop=(10, 8, 30, 20),
        resize=(40, 24),
    )

    image = cv2.imread(str(output_dir / "frame_000000.jpg"))
    assert image is not None
    assert image.shape[:2] == (24, 40)


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "match.avi")
    output_dir = tmp_path / "frames"

    manifest = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=5.0,
        match_start_offset=0.4,
        image_format="png",
    )

    loaded_manifest = load_manifest(output_dir / "manifest.json")
    assert loaded_manifest == manifest


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "match.avi")
    output_dir = tmp_path / "frames"

    manifest = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=5.0,
        match_start_offset=0.4,
        dry_run=True,
    )

    assert manifest.frame_count == 8
    assert not output_dir.exists()


def _create_synthetic_video(path: Path) -> Path:
    width = 64
    height = 48
    fps = 10.0
    frame_count = 20

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video at {path}")

    try:
        for index in range(frame_count):
            frame = np.full(
                (height, width, 3),
                ((index * 10) % 255, (index * 20) % 255, (index * 30) % 255),
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()

    return path
