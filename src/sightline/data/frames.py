"""Multi-frame model input: the current frame plus earlier frames from the same camera.

Consecutive frames reveal the car's own rotation as horizontal image motion,
a cue that transfers across roads where a single frame's scenery does not.
History indices are clamped at frame 0; they never reach forward in time.

Frames are stacked along the channel axis, current frame first:
``[frame t RGB, frame t - lags[0] RGB, ...]``. With no lags the stack is the
plain single-frame input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["history_indices", "stack_frames"]


def history_indices(indices: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """(N, 1 + len(lags)) frame indices: each frame, then ``max(index - lag, 0)`` per lag."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    if any(lag <= 0 for lag in lags):
        raise ValueError(f"lags must be positive frame offsets into the past: {lags}")
    cols = [idx] + [np.maximum(idx - lag, 0) for lag in lags]
    return np.concatenate(cols, axis=1)


def stack_frames(inputs: np.ndarray, indices: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """uint8 (N, H, W, 3 * (1 + len(lags))) model input for ``indices``."""
    hist = history_indices(indices, lags)
    if not lags:
        return np.ascontiguousarray(inputs[hist[:, 0]])
    return np.ascontiguousarray(
        np.concatenate([inputs[hist[:, k]] for k in range(hist.shape[1])], axis=3)
    )
