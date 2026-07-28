from __future__ import annotations

import argparse
from pathlib import Path

from cr_vision.analyzer import analyze_events, load_events, write_report
from cr_vision.arena import BOARD_TILE_COUNT, tile_by_id, tile_counts_by_region, validate_arena
from cr_vision.calibration import load_calibration, map_point_to_tile
from cr_vision.detection_backend import FakeDetectorBackend, detect_video
from cr_vision.eventizer import HandEventizer, EventizerConfig, load_frame_detections, write_events
from cr_vision.frames import extract_frames, plan_frame_extraction
from cr_vision.roboflow_backend import RoboflowDetectorBackend


def _parse_resize(raw_value: str) -> tuple[int, int]:
    try:
        width_text, height_text = raw_value.lower().split("x", maxsplit=1)
        return int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resize must look like WIDTHxHEIGHT") from exc


def _parse_crop(raw_value: str) -> tuple[int, int, int, int]:
    try:
        x_text, y_text, width_text, height_text = raw_value.split(",", maxsplit=3)
        return int(x_text), int(y_text), int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must look like x,y,w,h") from exc


def _parse_slot_centers(raw_value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in raw_value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "slot centers must look like LEFT,LEFT_CENTER,RIGHT_CENTER,RIGHT"
        ) from exc

    if len(values) != 4 or tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError(
            "slot centers must contain four left-to-right values"
        )
    return (values[0], values[1], values[2], values[3])


def _format_bytes(size_bytes: int) -> str:
    suffixes = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for suffix in suffixes:
        if size < 1024.0 or suffix == suffixes[-1]:
            return f"{size:.1f} {suffix}"
        size /= 1024.0

    return f"{size_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(prog="cr-vision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("events", type=Path)
    analyze_parser.add_argument("--output", type=Path)

    subparsers.add_parser("describe-grid")

    map_parser = subparsers.add_parser("map-point")
    map_parser.add_argument("calibration", type=Path)
    map_parser.add_argument("--x", type=float, required=True)
    map_parser.add_argument("--y", type=float, required=True)

    extract_parser = subparsers.add_parser("extract-frames")
    extract_parser.add_argument("video_path", type=Path)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--fps", type=float, default=5.0)
    extract_parser.add_argument("--match-start", type=float, default=0.0)
    extract_parser.add_argument("--max-seconds", type=float)
    extract_parser.add_argument("--crop", type=_parse_crop)
    extract_parser.add_argument("--resize", type=_parse_resize)
    extract_parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    extract_parser.add_argument("--jpeg-quality", type=int, default=90)
    extract_parser.add_argument("--dry-run", action="store_true")

    detect_parser = subparsers.add_parser("detect-video")
    detect_parser.add_argument("video_path", type=Path)
    detect_parser.add_argument("--output", type=Path, required=True)
    detect_parser.add_argument("--model-id")
    detect_parser.add_argument("--model-version", default="0")
    detect_parser.add_argument("--dataset-version", default="0")
    detect_parser.add_argument("--source-video")
    detect_parser.add_argument("--backend", choices=["fake", "roboflow"], default="fake")
    detect_parser.add_argument("--roboflow-endpoint")
    detect_parser.add_argument("--sample-fps", type=float, default=3.0)

    eventize_parser = subparsers.add_parser("eventize")
    eventize_parser.add_argument("raw_detections", type=Path)
    eventize_parser.add_argument("--output", type=Path, required=True)
    eventize_parser.add_argument("--confidence-threshold", type=float, default=0.50)
    eventize_parser.add_argument("--player-perspective", default="self")
    eventize_parser.add_argument("--stability-observations", type=int, default=3)
    eventize_parser.add_argument("--stability-window", type=int, default=5)
    eventize_parser.add_argument(
        "--slot-centers",
        type=_parse_slot_centers,
        default=(0.18, 0.38, 0.58, 0.78),
    )
    eventize_parser.add_argument("--slot-tolerance", type=float, default=0.10)

    args = parser.parse_args()

    try:
        if args.command == "analyze":
            events = load_events(args.events)
            tracker, snapshots = analyze_events(events)
            if args.output is not None:
                write_report(args.output, snapshots)

            for snapshot in snapshots:
                unavailable = ", ".join(snapshot.unavailable_cards) or "none"
                available = ", ".join(snapshot.available_known_cards) or "none"
                print(
                    f"{snapshot.time:6.1f}s  {snapshot.event.card:<18} "
                    f"elixir={snapshot.opponent_elixir:4.1f}  "
                    f"unavailable={unavailable}  available_seen={available}"
                )

            print(f"Estimated opponent elixir: {tracker.elixir.current:.1f}")
            print(f"Known opponent cards: {', '.join(tracker.known_deck) or 'none'}")
            print(
                "Definitely unavailable from visible cycle: "
                f"{', '.join(tracker.unavailable_cards) or 'none'}"
            )
            print(
                "Seen cards that may be available again: "
                f"{', '.join(tracker.available_known_cards) or 'none'}"
            )
            return

        if args.command == "describe-grid":
            validate_arena()
            counts = tile_counts_by_region()
            print(f"Total tiles: {BOARD_TILE_COUNT}")
            print(f"Self regular tiles: {counts['regular_self']}")
            print(f"Opponent regular tiles: {counts['regular_opponent']}")
            print(f"River tiles: {counts['river']}")
            print(f"Bridge tiles: {counts['bridge']}")
            return

        if args.command == "map-point":
            calibration = load_calibration(args.calibration)
            tile_id = map_point_to_tile(calibration, args.x, args.y)
            if tile_id is None:
                print("Point is not on the calibrated board")
                return

            tile = tile_by_id(tile_id)
            side = tile.position.side or "neutral"
            print(f"Tile: {tile.tile_id}")
            print(f"Region: {tile.position.region}")
            print(f"Side: {side}")
            return

        if args.command == "extract-frames":
            plan = plan_frame_extraction(
                video_path=args.video_path,
                fps=args.fps,
                match_start_offset=args.match_start,
                max_seconds=args.max_seconds,
                crop=args.crop,
                resize=args.resize,
                image_format=args.format,
                jpeg_quality=args.jpeg_quality,
            )
            if args.dry_run:
                width, height = plan.output_size
                print(f"Would extract {plan.manifest.frame_count} frames")
                print(f"Output size per frame: {width}x{height}")
                print(
                    "Estimated storage: "
                    f"{_format_bytes(plan.estimated_storage_bytes)}"
                )
                if plan.manifest.frames:
                    print(
                        "Match timeline covered: "
                        f"0.0s to {plan.manifest.frames[-1].match_time_seconds:.1f}s"
                    )
                else:
                    print("Match timeline covered: no frames")
                return

            manifest = extract_frames(
                video_path=args.video_path,
                output_dir=args.output,
                fps=args.fps,
                match_start_offset=args.match_start,
                max_seconds=args.max_seconds,
                crop=args.crop,
                resize=args.resize,
                image_format=args.format,
                jpeg_quality=args.jpeg_quality,
            )
            print(f"Extracted {manifest.frame_count} frames")
            print(f"Manifest: {args.output / 'manifest.json'}")
            return

        if args.command == "detect-video":
            if args.backend == "roboflow":
                backend = RoboflowDetectorBackend(
                    model_id=args.model_id,
                    endpoint=args.roboflow_endpoint,
                )
                resolved_model_id = backend.model_id
            else:
                backend = FakeDetectorBackend()
                resolved_model_id = args.model_id or "offline-fake-backend"

            detections = detect_video(
                args.video_path,
                backend=backend,
                output_path=args.output,
                model_id=resolved_model_id,
                model_version=args.model_version,
                dataset_version=args.dataset_version,
                source_video=args.source_video or str(args.video_path),
                sample_fps=args.sample_fps,
            )
            print(f"Wrote {len(detections)} raw detections to {args.output}")
            return

        if args.command == "eventize":
            detections = load_frame_detections(args.raw_detections)
            config = EventizerConfig(
                confidence_threshold=args.confidence_threshold,
                player_perspective=args.player_perspective,
                stability_observations=args.stability_observations,
                stability_window=args.stability_window,
                slot_centers=args.slot_centers,
                slot_tolerance=args.slot_tolerance,
            )
            eventizer = HandEventizer(config)
            events = eventizer.eventize(detections)
            write_events(args.output, events)
            print(f"Wrote {len(events)} events to {args.output}")
            return
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
