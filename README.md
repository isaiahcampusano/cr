# Clash Royale Vision Learning Project

This repo is a learning project for computer vision and game-state estimation using
recorded friendly Clash Royale matches.

The initial scope is intentionally fair and educational:

- Analyze recorded videos or manually labeled match events.
- Detect only visible opponent card plays.
- Estimate opponent elixir from public timing and card costs.
- Track opponent card cycle from cards that have already appeared.
- Avoid hidden-information hacks, memory inspection, network interception, or gameplay automation.

## Project Roadmap

1. Build the game-state logic from simple timestamped events.
2. Collect short friendly-match clips and label visible opponent card plays.
3. Train a card detector on screenshots or video frames.
4. Feed detector output into the elixir and cycle tracker.
5. Display or export the resulting public-information estimates.
6. Evaluate accuracy against hand-labeled test matches.

## Tech Stack

- Python for the main project.
- OpenCV for reading videos and frames.
- Ultralytics YOLO or another object detection model for card detection.
- Pytest for tests.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m cr_vision analyze examples/sample_events.json
python -m cr_vision analyze examples/sample_events.json --output data/processed/sample_report.json
pytest
```

## Event Format

Until the vision model is trained, the analyzer can run from labeled events:

```json
[
  { "time": 4.2, "player": "opponent", "card": "hog_rider" },
  { "time": 9.8, "player": "opponent", "card": "cannon" }
]
```

Times are seconds from the start of the match. The analyzer assumes the match
starts at 5 elixir, caps elixir at 10, applies normal/double/triple elixir
timing, and subtracts known card costs when opponent plays are observed.

The cycle tracker only uses public information. It reports:

- `known_deck`: opponent cards that have visibly appeared.
- `unavailable_cards`: seen cards that were played fewer than four opponent
  cards ago, so they cannot have cycled back yet.
- `available_known_cards`: seen cards that have had enough later plays to be
  possible again. This does not prove they are currently in hand.

## Game-State Stage Status

The current implementation can:

- Load manually labeled timestamped card events.
- Estimate opponent elixir over time.
- Track visible opponent cards without inventing hidden deck information.
- Print a timeline in the terminal.
- Export a JSON report with per-play public-state snapshots.

## GitHub

After reviewing the files, create a repository on GitHub and push this folder:

```powershell
git init
git add .
git commit -m "Create Clash Royale vision learning scaffold"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```
