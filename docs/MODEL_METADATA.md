# Model metadata contract

This project keeps model metadata alongside raw frame detections without storing secrets.

## Recorded from the handoff

- Workspace: zay-clio1
- Project: clash-royale-of3d3-mts11
- Dataset version: 1
- Dataset size: 972 images
- Split: 673 train / 186 validation / 113 test
- Preprocessing: stretch to 512 x 512
- RF-DETR Small model ID: zay-clio1/clash-royale-of3d3-mts11-1-rfdetr-small-t1
- RF-DETR Medium model ID: zay-clio1/clash-royale-of3d3-mts11-1-rfdetr-medium-t2

## Pending confirmation

- Final Medium metrics
- Small vs Medium performance on entirely unseen match footage
- Selected production model
- Recommended confidence threshold
- Exact raw class-name list
- Confirmation that every trained class corresponds to a visible hand-card icon
- Hosted Roboflow vs local/self-hosted inference choice

## Storage format

Raw detections are written as JSONL records with:

- model_id
- model_version
- dataset_version
- source_video
- timestamp
- raw_label
- canonical_label
- confidence
- bbox
- source_frame
