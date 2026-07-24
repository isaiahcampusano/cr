from __future__ import annotations

import argparse
from pathlib import Path

from cr_vision.analyzer import analyze_events, load_events, write_report
from cr_vision.arena import BOARD_TILE_COUNT, tile_by_id, tile_counts_by_region, validate_arena
from cr_vision.calibration import load_calibration, map_point_to_tile
from cr_vision.frames import extract_frames, plan_frame_extraction


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
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
