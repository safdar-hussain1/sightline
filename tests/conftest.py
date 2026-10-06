"""Synthetic drives: tiny, deterministic stand-ins for the real dataset.

Each frame is a 455x256 BGR image with a bright vertical bar whose horizontal
position follows the steering angle, so a model can learn the mapping in one
epoch and a broken pipeline is visible. Labels use the timestamped format.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

W, H = 455, 256


def make_drive(root: Path, name: str, n: int, timestamps: bool = True,
               dropouts: tuple[int, ...] = ()) -> Path:
    drive = root / name
    drive.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name) * 1000 + n)
    t = np.arange(n)
    angles = 40.0 * np.sin(t / 25.0) + rng.normal(0, 1.0, n)
    for i in dropouts:
        angles[i] = 0.0
    start = datetime(2018, 7, 1, 17, 0, 0)
    lines = []
    for i in range(n):
        img = np.full((H, W, 3), 60, np.uint8)
        x = int(W / 2 + angles[i] * 2.0)
        cv2.rectangle(img, (max(0, x - 6), 120), (min(W - 1, x + 6), H - 1), (220, 220, 220), -1)
        cv2.imwrite(str(drive / f"{i}.jpg"), img)
        if timestamps:
            ts = start + timedelta(milliseconds=57 * i)
            lines.append(f"{i}.jpg {angles[i]:.6f},{ts:%Y-%m-%d %H:%M:%S}:{ts.microsecond // 1000}")
        else:
            lines.append(f"{i}.jpg {angles[i]:.6f}")
    (drive / "data.txt").write_text("\n".join(lines) + "\n")
    return drive


@pytest.fixture
def tiny_drive(tmp_path: Path) -> Path:
    return make_drive(tmp_path / "data", "drivetest", n=900)
