"""The canonical frame -> model-input transform.

Every consumer — the cache builder, the local demo, the ONNX exporter, the
browser parity fixtures — must go through this one function so there is a
single definition of what the model sees.

Pipeline: keep the bottom ``CROP_BOTTOM`` rows (road, not sky), resize to
``INPUT_WIDTH`` x ``INPUT_HEIGHT`` with INTER_AREA, convert BGR -> RGB, and
return uint8 HWC. Scaling to [0, 1] float happens inside the model graph so
that exported ONNX carries its own normalization and the browser feeds raw
RGB bytes.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["preprocess_frame", "INPUT_HEIGHT", "INPUT_WIDTH", "CROP_BOTTOM"]

INPUT_HEIGHT = 66
INPUT_WIDTH = 200
CROP_BOTTOM = 150


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Transform one decoded BGR frame into the uint8 RGB model input."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got shape {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got {frame_bgr.dtype}")
    if frame_bgr.shape[0] < CROP_BOTTOM:
        raise ValueError(f"frame height {frame_bgr.shape[0]} < crop {CROP_BOTTOM}")
    cropped = frame_bgr[-CROP_BOTTOM:]
    resized = cv2.resize(cropped, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
