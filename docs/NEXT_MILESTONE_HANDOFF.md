# Copilot Handoff: First Labeled Detection Loop

## Why This Is The Next Step

The current repo already has:

- Offline frame extraction from match videos.
- A public-state tracker driven by `CardEvent`.
- A detector adapter that converts `Detection` objects into `CardEvent`.

The next missing link is not frontend work and not API-key access. It is the
first real offline vision loop:

1. extract frames,
2. attach simple labels to a tiny sample,
3. load those labels as `Detection` objects,
4. feed them through the adapter into `analyze_events`.

That keeps the project Python-only, local, and within the current fair-use
scope.

## Paste This Into Copilot Chat

```text
You are helping on a local Python project called cr-vision. Stay within the existing offline, public-information scope. Do not add live gameplay automation, hidden-information logic, paid API dependencies, or frontend/UI work.

Current goal: build the first labeled detection loop so a tiny local set of extracted frames can produce `Detection` objects and flow into the existing detector adapter and analyzer pipeline.

Repo facts:
- Frame extraction exists in `src/cr_vision/frames.py`
- Detector placeholder exists in `src/cr_vision/detection.py`
- Detector adapter exists in `src/cr_vision/detector_adapter.py`
- State/event contract exists in `src/cr_vision/state.py`
- Analysis entrypoint exists in `src/cr_vision/analyzer.py`
- Example data already exists in `examples/`

What to build:
1. Add a tiny label format for mocked/local detection records tied to extracted frames
2. Add loader code that reads those labels into `Detection` objects
3. Keep the format intentionally small, for example records with:
   - `timestamp`
   - `card`
   - `confidence`
   - optional `source_frame`
4. Add one or two example label files under `examples/`
5. Add tests proving:
   - label files load correctly into `Detection`
   - loaded detections can be converted with `detections_to_events`
   - resulting events still work with `analyze_events`

Recommended shape:
- Prefer a new module like `src/cr_vision/detection_labels.py`
- Keep parsing strict and typed
- Use JSON for the first pass

Constraints:
- Keep scope tight and local
- Do not train a model yet
- Do not add YOLO training code yet
- Do not redesign existing analyzer or tracker logic
- Do not build a frontend
- Preserve repo style and typing

Definition of done:
- a tiny label file can be loaded into `Detection` objects
- those detections can flow through the existing adapter
- tests cover the label loader and end-to-end local mocked pipeline
- full test suite still passes

Before changing code, summarize the exact files you plan to touch and why. After changes, summarize what you added and show the test result.
```

## What This Milestone Should Probably Add

Suggested file touches:

- `src/cr_vision/detection_labels.py`
- `tests/test_detection_labels.py`
- `examples/sample_detections.json`
- `README.md` only if a tiny usage note is helpful

Possible function surface:

```python
def load_detections(path: Path) -> list[Detection]:
    ...
```

Suggested JSON shape:

```json
[
  {
    "timestamp": 1.0,
    "card": "hog_rider",
    "confidence": 0.93,
    "source_frame": "frame_000005.jpg"
  },
  {
    "timestamp": 2.8,
    "card": "cannon",
    "confidence": 0.88,
    "source_frame": "frame_000014.jpg"
  }
]
```

## Why Not Frontend Yet

A frontend would be early right now because the project still needs the first
real offline CV data path. The most valuable next proof is:

Can labeled frame-derived detections move cleanly through the detector adapter
and into the public-state analyzer?

If that path works, a future CLI improvement or lightweight visualization will
be much easier to justify.

## What Comes After This

If this milestone lands cleanly, the follow-up sequence should be:

1. Create a slightly larger labeled sample from extracted frames.
2. Add a lightweight evaluation script for precision and recall on the mocked
   or label-derived detections.
3. Only then consider a first real detector implementation or training stub.

That keeps the project grounded in measurable Python-side progress instead of
jumping ahead to UI or paid-model dependencies.
