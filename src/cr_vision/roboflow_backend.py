from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

import cv2

from cr_vision.detection import FrameDetection
from cr_vision.detection_backend import parse_roboflow_response


class RoboflowDetectorBackend:
    """Hosted Roboflow inference backend for offline frame detection."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID")
        self.endpoint = (endpoint or os.getenv("ROBOFLOW_HOST") or "https://detect.roboflow.com").rstrip("/")

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required for Roboflow inference")
        if not self.model_id:
            raise ValueError("ROBOFLOW_MODEL_ID environment variable is required for Roboflow inference")

    def detect_frame(
        self,
        frame: object,
        *,
        timestamp: float,
        source_frame: str | None,
    ) -> list[FrameDetection]:
        if frame is None:
            raise ValueError("Roboflow backend requires an image frame")

        if not hasattr(frame, "shape"):
            raise ValueError("Roboflow backend requires an image frame with array dimensions")

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode frame for Roboflow inference")

        image_bytes = encoded.tobytes()
        url = f"{self.endpoint}/{self.model_id}?api_key={self.api_key}"
        req = request.Request(
            url,
            data=image_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                payload = json.load(response)
        except error.HTTPError as exc:
            raise ValueError(f"Roboflow inference request failed ({exc.code}): {exc.reason}") from exc
        except error.URLError as exc:
            raise ValueError(f"Roboflow inference request failed: {exc.reason}") from exc

        height, width = frame.shape[:2]
        return parse_roboflow_response(
            payload,
            timestamp=timestamp,
            source_frame=source_frame,
            source_image_width=width,
            source_image_height=height,
        )
