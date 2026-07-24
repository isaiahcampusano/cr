from __future__ import annotations

"""Offline frame extraction for recorded friendly-match videos.

This module only reads video files the user already recorded from their own
friendly matches. It does not do live capture, hidden-state access, gameplay
automation, or any other unfair assist behavior.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


SUPPORTED_IMAGE_FORMATS = {"jpg", "png"}


@dataclass(frozen=True)
class FrameRecord:
    index: int
    filename: str
    video_time_seconds: float
    match_time_seconds: float


@dataclass(frozen=True)
class FrameManifest:
    video_path: str
    sample_rate_fps: float
    match_start_offset_seconds: float
    video_fps: float
    frame_count: int
    frames: list[FrameRecord]


@dataclass(frozen=True)
class ExtractionPlan:
    manifest: FrameManifest
    output_size: tuple[int, int]
    estimated_storage_bytes: int


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 5.0,
    match_start_offset: float = 0.0,
    max_seconds: float | None = None,
    crop: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
    image_format: str = "jpg",
    jpeg_quality: int = 90,
    dry_run: bool = False,
) -> FrameManifest:
    plan = plan_frame_extraction(
        video_path=video_path,
        fps=fps,
        match_start_offset=match_start_offset,
        max_seconds=max_seconds,
        crop=crop,
        resize=resize,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )

    if dry_run:
        return plan.manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(video_path)
    image_params = _image_write_params(image_format, jpeg_quality)

    try:
        for record in plan.manifest.frames:
            capture.set(cv2.CAP_PROP_POS_MSEC, record.video_time_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Could not read frame at {record.video_time_seconds:.3f}s "
                    f"from {video_path}"
                )

            processed_frame = _transform_frame(frame, crop=crop, resize=resize)
            frame_path = output_dir / record.filename
            if not cv2.imwrite(str(frame_path), processed_frame, image_params):
                raise ValueError(f"Could not write frame to {frame_path}")
    finally:
        capture.release()

    write_manifest(output_dir / "manifest.json", plan.manifest)
    return plan.manifest


def plan_frame_extraction(
    video_path: Path,
    fps: float = 5.0,
    match_start_offset: float = 0.0,
    max_seconds: float | None = None,
    crop: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
    image_format: str = "jpg",
    jpeg_quality: int = 90,
) -> ExtractionPlan:
    _validate_sampling_inputs(
        fps=fps,
        match_start_offset=match_start_offset,
        max_seconds=max_seconds,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )

    capture = _open_capture(video_path)
    try:
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if video_fps <= 0:
        raise ValueError(f"Could not determine source video FPS for {video_path}")
    if source_frame_count <= 0:
        raise ValueError(f"Could not determine source frame count for {video_path}")

    duration_seconds = source_frame_count / video_fps
    effective_end_seconds = min(
        duration_seconds,
        match_start_offset + max_seconds if max_seconds is not None else duration_seconds,
    )
    if match_start_offset > duration_seconds:
        raise ValueError(
            f"match_start_offset {match_start_offset} exceeds video duration "
            f"{duration_seconds:.3f}s"
        )

    output_size = _resolve_output_size(
        source_size=(frame_width, frame_height),
        crop=crop,
        resize=resize,
    )
    frames = _build_frame_records(
        fps=fps,
        match_start_offset=match_start_offset,
        end_seconds=effective_end_seconds,
        image_format=image_format,
    )
    manifest = FrameManifest(
        video_path=str(video_path),
        sample_rate_fps=fps,
        match_start_offset_seconds=match_start_offset,
        video_fps=round(video_fps, 6),
        frame_count=len(frames),
        frames=frames,
    )

    return ExtractionPlan(
        manifest=manifest,
        output_size=output_size,
        estimated_storage_bytes=_estimate_storage_bytes(
            frame_count=manifest.frame_count,
            output_size=output_size,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        ),
    )


def write_manifest(path: Path, manifest: FrameManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> FrameManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FrameManifest(
        video_path=payload["video_path"],
        sample_rate_fps=payload["sample_rate_fps"],
        match_start_offset_seconds=payload["match_start_offset_seconds"],
        video_fps=payload["video_fps"],
        frame_count=payload["frame_count"],
        frames=[
            FrameRecord(
                index=record["index"],
                filename=record["filename"],
                video_time_seconds=record["video_time_seconds"],
                match_time_seconds=record["match_time_seconds"],
            )
            for record in payload["frames"]
        ],
    )


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return capture


def _build_frame_records(
    fps: float,
    match_start_offset: float,
    end_seconds: float,
    image_format: str,
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    index = 0
    epsilon = 1e-9

    while True:
        video_time = match_start_offset + (index / fps)
        if video_time >= end_seconds - epsilon:
            break

        records.append(
            FrameRecord(
                index=index,
                filename=f"frame_{index:06d}.{image_format}",
                video_time_seconds=round(video_time, 6),
                match_time_seconds=round(video_time - match_start_offset, 6),
            )
        )
        index += 1

    return records


def _resolve_output_size(
    source_size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
) -> tuple[int, int]:
    width, height = source_size
    if crop is not None:
        x, y, crop_width, crop_height = crop
        if x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0:
            raise ValueError("crop must be x,y,w,h with positive width and height")
        if x + crop_width > width or y + crop_height > height:
            raise ValueError("crop must stay within the source video dimensions")
        width, height = crop_width, crop_height

    if resize is not None:
        resize_width, resize_height = resize
        if resize_width <= 0 or resize_height <= 0:
            raise ValueError("resize must contain positive width and height")
        width, height = resize_width, resize_height

    return width, height


def _transform_frame(
    frame,
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
):
    if crop is not None:
        x, y, width, height = crop
        frame = frame[y : y + height, x : x + width]

    if resize is not None:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

    return frame


def _validate_sampling_inputs(
    fps: float,
    match_start_offset: float,
    max_seconds: float | None,
    image_format: str,
    jpeg_quality: int,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if match_start_offset < 0:
        raise ValueError("match_start_offset must be greater than or equal to 0")
    if max_seconds is not None and max_seconds < 0:
        raise ValueError("max_seconds must be greater than or equal to 0")
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"image_format must be one of {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
        )
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100")


def _image_write_params(image_format: str, jpeg_quality: int) -> list[int]:
    if image_format == "jpg":
        return [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    return []


def _estimate_storage_bytes(
    frame_count: int,
    output_size: tuple[int, int],
    image_format: str,
    jpeg_quality: int,
) -> int:
    width, height = output_size
    raw_bytes_per_frame = width * height * 3
    if image_format == "png":
        compression_ratio = 0.55
    else:
        compression_ratio = max(0.08, 0.45 - (jpeg_quality / 250.0))
    return int(frame_count * raw_bytes_per_frame * compression_ratio)
