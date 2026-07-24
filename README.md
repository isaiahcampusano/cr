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
python -m cr_vision extract-frames raw/match_001.mp4 --output data/frames/match_001 --fps 5.0 --match-start 8.5 --dry-run
python -m cr_vision describe-grid
python -m cr_vision map-point examples/sample_calibration.json --x 105 --y 275
python -m cr_vision analyze examples/sample_events.json
python -m cr_vision analyze examples/sample_events.json --output data/processed/sample_report.json
pytest
```

## Frame Extraction

The first computer-vision utility is frame extraction from recorded friendly
matches. It samples video frames at a fixed rate and writes a `manifest.json`
that maps each saved frame to both video time and match time.

```powershell
python -m cr_vision extract-frames raw/match_001.mp4 `
  --output data/frames/match_001 `
  --fps 5.0 `
  --match-start 8.5
```

Helpful options:

- `--dry-run`: estimate frame count and storage before writing files.
- `--max-seconds`: stop after a match-time window.
- `--crop x,y,w,h`: extract a region of interest before saving.
- `--resize WIDTHxHEIGHT`: resize each saved frame.
- `--format png`: switch from JPEG to PNG output.

## Event Format

Until the vision model is trained, the analyzer can run from labeled events:

```json
[
  { "time": 4.2, "player": "opponent", "card": "hog_rider" },
  { "time": 9.8, "player": "opponent", "card": "cannon" }
]
```

Positioned deployment labels can also include tile or screenshot metadata. Older
card-only events remain valid.

For a mocked local workflow, detector-style detections can be loaded from a
small JSON label file, converted into `CardEvent` objects with
`cr_vision.detector_adapter.detections_to_events`, and then passed to
`cr_vision.analyzer.analyze_events`.

```json
[
  {
    "timestamp": 1.0,
    "card": "hog_rider",
    "confidence": 0.93,
    "source_frame": "frame_000001.jpg"
  },
  {
    "timestamp": 2.8,
    "card": "cannon",
    "confidence": 0.88,
    "source_frame": "frame_000014.jpg"
  }
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
- Load optional positioned deployment labels.
- Estimate opponent elixir over time.
- Track visible opponent cards without inventing hidden deck information.
- Represent a 544-tile logical arena grid.
- Track visible card deployments on arena tiles.
- Map calibrated screenshot points to logical tiles.
- Print a timeline in the terminal.
- Export a JSON report with per-play public-state snapshots.

The default future evaluation deck is the classic 2.6 Hog Cycle list:
`hog_rider`, `musketeer`, `cannon`, `fireball`, `the_log`, `ice_spirit`,
`skeletons`, and `ice_golem`.

## Arena Grid

The logical board contains 544 tiles:

- 256 self-side regular tiles.
- 256 opponent-side regular tiles.
- 24 river tiles.
- 8 bridge tiles.

Tile IDs are stable strings such as `self:regular:10:6`,
`opponent:regular:3:12`, `neutral:river:12:0`, and
`neutral:bridge:left:0`.

Use a calibration file to map screenshot pixels into this logical grid:

```powershell
python -m cr_vision map-point examples/sample_calibration.json --x 105 --y 275
```

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
