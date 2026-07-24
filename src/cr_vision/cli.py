from __future__ import annotations

import argparse
from pathlib import Path

from cr_vision.analyzer import analyze_events, load_events


def main() -> None:
    parser = argparse.ArgumentParser(prog="cr-vision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("events", type=Path)

    args = parser.parse_args()

    if args.command == "analyze":
        events = load_events(args.events)
        tracker = analyze_events(events)
        print(f"Estimated opponent elixir: {tracker.elixir.current:.1f}")
        print(f"Known opponent cards: {', '.join(tracker.known_deck) or 'none'}")
        print(f"Recent opponent cycle: {', '.join(tracker.likely_hand) or 'none'}")


if __name__ == "__main__":
    main()

