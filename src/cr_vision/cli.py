from __future__ import annotations

import argparse
from pathlib import Path

from cr_vision.analyzer import analyze_events, load_events, write_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="cr-vision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("events", type=Path)
    analyze_parser.add_argument("--output", type=Path)

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
