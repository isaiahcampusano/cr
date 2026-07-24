# Copilot Handoff: Detector Adapter Milestone

## Paste This Into Copilot Chat

```text
You are helping on a local Python project called cr-vision. Work only within the existing offline, public-information scope of the repo. Do not add live gameplay automation, RL, memory inspection, hidden-information logic, or any paid API dependency.

Current goal: implement the smallest possible detector adapter milestone that converts detector output into the existing CardEvent pipeline.

Repo facts:
- Python package: src/cr_vision
- Current detector placeholder: src/cr_vision/detection.py
- State contract: src/cr_vision/state.py
- Analysis entrypoint: src/cr_vision/analyzer.py
- Tests currently pass with `python -m pytest` and the handoff baseline is 25 passing tests

What to build:
1. Add a small adapter module, preferably `src/cr_vision/detector_adapter.py`
2. Implement a function like:
   `detections_to_events(detections: list[Detection], *, player: str = "opponent", min_confidence: float = 0.0) -> list[CardEvent]`
3. Behavior:
   - filter out detections below min_confidence
   - sort by timestamp
   - map each Detection(timestamp, card, confidence) into CardEvent(time, player, card, confidence)
   - leave tile/x/y/source_frame unset for this first milestone
4. Add tests in `tests/test_detector_adapter.py`
5. Include at least one golden-path integration test that converts mocked detections into events, runs `analyze_events`, and verifies time ordering, known_deck, and elixir behavior

Constraints:
- Keep the change small and local
- Do not redesign the tracker
- Do not require a real detector model
- Use mocked detections only
- Preserve existing style and typing

Definition of done:
- adapter output is directly usable by `analyze_events`
- new tests pass
- full suite still passes

Before changing code, briefly summarize the files you plan to touch and why. After changes, summarize what you added and show the test result.
```

## Project Snapshot

As of Friday, July 24, 2026, this repo has a solid offline foundation. The main
gap is no longer basic state tracking. The missing piece is a small bridge from
future detector output into the existing `CardEvent` analysis pipeline.

This is a good local-development milestone because it does not require
DeepSeek, OpenAI, or any paid model access. The work can be completed entirely
with mocked detections and tests.

## Verified Current Status

Already implemented:

- Offline frame extraction from recorded videos with manifest output.
- Logical arena representation with 544 stable tile IDs.
- Screenshot-point to logical-tile calibration utilities.
- Public-state tracking for opponent elixir and visible cycle.
- Optional positioned deployment events on the board.
- JSON report export and CLI commands for the current offline workflow.

Key files:

- `src/cr_vision/frames.py`: frame extraction and manifest handling.
- `src/cr_vision/arena.py`: logical board grid and tile IDs.
- `src/cr_vision/calibration.py`: pixel-to-tile mapping.
- `src/cr_vision/state.py`: `CardEvent`, board state, elixir, and cycle logic.
- `src/cr_vision/analyzer.py`: event loading, analysis, and report writing.
- `src/cr_vision/detection.py`: placeholder detector interface only.

Verified on July 24, 2026:

```powershell
python -m pytest
```

Result at handoff time:

```text
25 passed
```

## Scope Guardrails

In scope:

- Recorded friendly-match videos.
- Visible opponent card-play inference.
- Public elixir and cycle estimation.
- Offline testing and evaluation.
- Mocked detector outputs for local development.

Out of scope:

- Hidden information access.
- Memory inspection or network interception.
- Live gameplay automation.
- Reinforcement learning or a play bot.
- Simulated touch input or action execution.
- Any dependency on paid API credits for this milestone.

## Why This Milestone Matters

The repo already has:

1. Offline frame extraction.
2. Logical board and calibration utilities.
3. Public-state tracking from `CardEvent`.

What it does not yet have is:

4. A clean adapter that turns detector output into `CardEvent`.

That adapter is the narrowest next step because it connects future CV work to
existing analysis code without forcing a redesign.

## Recommended First Detection Target

Start with opponent card-play popups, not deployed troop sprites.

Why:

- They map cleanly to `time` plus `card`.
- They do not require tile mapping for the first pass.
- They avoid movement tracking.
- They can feed directly into `analyze_events` immediately.

## Proposed Implementation

### Detector Contract

The current placeholder in `src/cr_vision/detection.py` already defines:

```python
@dataclass(frozen=True)
class Detection:
    timestamp: float
    card: str
    confidence: float
```

That is enough for the first adapter milestone.

### Adapter Module

Add a small module:

- `src/cr_vision/detector_adapter.py`

Suggested function:

```python
def detections_to_events(
    detections: list[Detection],
    *,
    player: str = "opponent",
    min_confidence: float = 0.0,
) -> list[CardEvent]:
    ...
```

Expected behavior:

- Filter out detections below `min_confidence`.
- Sort detections by timestamp.
- Convert detections into `CardEvent` instances.
- Set `confidence` on the resulting event.
- Leave `tile`, `x`, `y`, and `source_frame` as `None`.

Example mapping:

```python
CardEvent(
    time=detection.timestamp,
    player=player,
    card=detection.card,
    confidence=detection.confidence,
)
```

Do not add duplicate suppression yet unless mocked tests clearly require it.

## Test Plan

Create:

- `tests/test_detector_adapter.py`

Recommended tests:

1. Golden path:
   mocked detections like `hog_rider` then `cannon`, converted into events and
   passed into `analyze_events`.
2. Confidence filtering:
   detections below `min_confidence` are skipped.
3. Sorting:
   out-of-order detections become time-ordered events.

The key proof is that adapter output works with the existing tracker unchanged.

## Acceptance Criteria

This milestone is done when:

- Detector output converts directly into `list[CardEvent]`.
- `analyze_events` works on adapter output without tracker changes.
- A golden test proves the detector-output to state-analysis path.
- The full test suite still passes.

## Likely Files To Touch

- `src/cr_vision/detector_adapter.py`
- `tests/test_detector_adapter.py`
- `README.md` only if a short note is helpful

`src/cr_vision/detection.py` should probably remain minimal unless a tiny import
or typing cleanup is needed.

## What Comes After

After this adapter milestone:

1. Collect a tiny labeled dataset for popup detection.
2. Produce real detector output on extracted frames.
3. Measure how cleanly those detections become usable events.
4. Only then consider tile-aware deployment detections.

Core question for the next phase:

Can visible detections be turned into reliable public-state events?
