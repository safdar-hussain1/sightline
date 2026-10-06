"""Synthetic drives: tiny, deterministic stand-ins for the real dataset.

Each frame is a 455x256 BGR image with a bright vertical bar whose horizontal
position follows the steering angle, so a model can learn the mapping in one
epoch and a broken pipeline is visible. Labels use the timestamped format.

With ``scenery=True`` the bar stands in front of a grey background that drifts
slowly from frame to frame, as roadside scenery does: neighbouring frames look
almost alike and frames far apart do not. Without it, every frame showing the
same bar position is pixel-identical, however far apart in time.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

W, H = 455, 256


def _scenery(seed: int, grid: tuple[int, int] = (6, 16), rho: float = 0.99):
    """Grey backgrounds, one per frame: a coarse random field that keeps a correlation
    of ``rho`` from one frame to the next, upsampled smoothly to the frame size."""
    rng = np.random.default_rng(seed)
    field = rng.normal(0, 1, grid)
    while True:
        smooth = cv2.resize(field.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        grey = np.clip(110.0 + 50.0 * smooth, 0, 255).astype(np.uint8)
        yield np.repeat(grey[:, :, None], 3, axis=2)
        field = rho * field + np.sqrt(1 - rho * rho) * rng.normal(0, 1, grid)


def make_drive(root: Path, name: str, n: int, timestamps: bool = True,
               dropouts: tuple[int, ...] = (), scenery: bool = False,
               intervals_ms: tuple[int, ...] = (57,)) -> Path:
    """``intervals_ms`` are the gaps between consecutive timestamps, repeated in turn."""
    drive = root / name
    drive.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name) * 1000 + n)
    t = np.arange(n)
    angles = 40.0 * np.sin(t / 25.0) + rng.normal(0, 1.0, n)
    for i in dropouts:
        angles[i] = 0.0
    backgrounds = _scenery(len(name) * 1000 + n + 1) if scenery else None
    offsets_ms = np.concatenate([[0], np.cumsum(np.resize(np.asarray(intervals_ms), max(n - 1, 0)))])
    start = datetime(2018, 7, 1, 17, 0, 0)
    lines = []
    for i in range(n):
        img = np.full((H, W, 3), 60, np.uint8) if backgrounds is None else next(backgrounds)
        x = int(W / 2 + angles[i] * 2.0)
        cv2.rectangle(img, (max(0, x - 6), 120), (min(W - 1, x + 6), H - 1), (220, 220, 220), -1)
        cv2.imwrite(str(drive / f"{i}.jpg"), img)
        if timestamps:
            ts = start + timedelta(milliseconds=int(offsets_ms[i]))
            lines.append(f"{i}.jpg {angles[i]:.6f},{ts:%Y-%m-%d %H:%M:%S}:{ts.microsecond // 1000}")
        else:
            lines.append(f"{i}.jpg {angles[i]:.6f}")
    (drive / "data.txt").write_text("\n".join(lines) + "\n")
    return drive


@pytest.fixture
def tiny_drive(tmp_path: Path) -> Path:
    return make_drive(tmp_path / "data", "drivetest", n=900)
