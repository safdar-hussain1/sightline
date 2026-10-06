"""Steering label parsing and sensor-dropout repair.

Label files follow the driving-datasets convention, one frame per line:

    ``filename.jpg angle``                          (2017 drive)
    ``filename.jpg angle,YYYY-MM-DD HH:MM:SS:fff``  (2018 drive)

Angles are steering-wheel degrees, positive clockwise. The 2018 format
carries a wall-clock timestamp with millisecond precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

__all__ = ["DriveLabels", "load_labels", "median_interval_s", "repair_dropouts"]


def _parse_timestamp(ts: str) -> float:
    """Parse ``YYYY-MM-DD HH:MM:SS:mmm`` where mmm is UNPADDED milliseconds.

    The recorder printed milliseconds as a bare integer, so ``:11`` means
    11 ms — ``strptime``'s ``%f`` would silently read it as 110 ms.
    """
    base, ms_str = ts.rsplit(":", 1)
    ms = int(ms_str)
    if not 0 <= ms <= 999:
        raise ValueError(f"millisecond field out of range in {ts!r}")
    return datetime.strptime(base, "%Y-%m-%d %H:%M:%S").timestamp() + ms / 1000.0


def median_interval_s(timestamps_s: np.ndarray) -> float:
    """Median of the positive intervals between consecutive timestamps, in seconds."""
    deltas = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise ValueError("timestamps are not increasing anywhere")
    return float(np.median(deltas))


@dataclass
class DriveLabels:
    """Labels for one contiguous dashcam drive."""

    root: Path
    filenames: list[str]
    angles_deg: np.ndarray
    timestamps_s: np.ndarray | None = None
    repaired_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        if len(self.filenames) != len(self.angles_deg):
            raise ValueError(
                f"{len(self.filenames)} filenames but {len(self.angles_deg)} angles"
            )
        if self.timestamps_s is not None and len(self.timestamps_s) != len(self.angles_deg):
            raise ValueError("timestamps length does not match angles")

    def __len__(self) -> int:
        return len(self.filenames)

    @property
    def has_timestamps(self) -> bool:
        return self.timestamps_s is not None

    def frame_path(self, index: int) -> Path:
        return self.root / self.filenames[index]

    def median_dt_s(self) -> float:
        """Median inter-frame interval in seconds (timestamped drives only)."""
        if self.timestamps_s is None:
            raise ValueError("drive has no timestamps")
        return median_interval_s(self.timestamps_s)


def load_labels(root: str | Path, filename: str = "data.txt") -> DriveLabels:
    """Parse a drive's label file. Fails fast on malformed lines."""
    root = Path(root)
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")

    filenames: list[str] = []
    angles: list[float] = []
    stamps: list[float] = []
    saw_timestamp = False

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            name, rest = line.split(maxsplit=1)
            if "," in rest:
                angle_str, ts_str = rest.split(",", 1)
                stamps.append(_parse_timestamp(ts_str.strip()))
                saw_timestamp = True
            else:
                angle_str = rest
            angles.append(float(angle_str))
            filenames.append(name)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed label line {line!r}") from exc

    if not filenames:
        raise ValueError(f"{path}: no labels parsed")
    if saw_timestamp and len(stamps) != len(filenames):
        raise ValueError(f"{path}: timestamps present on some lines but not all")

    return DriveLabels(
        root=root,
        filenames=filenames,
        angles_deg=np.asarray(angles, dtype=np.float64),
        timestamps_s=np.asarray(stamps, dtype=np.float64) if saw_timestamp else None,
    )


def repair_dropouts(
    labels: DriveLabels,
    min_neighbor_deg: float = 15.0,
) -> DriveLabels:
    """Repair single-frame steering-sensor dropouts by linear interpolation.

    The drives contain frames where the logged angle is exactly 0.0 while both
    neighbors hold large same-sign angles (e.g. 501.78 -> 0.0 -> 498.55): the
    sensor dropped for one frame mid-turn. A frame is repaired only when

    * its angle is exactly 0.0, and
    * both neighbors exceed ``min_neighbor_deg`` in magnitude, and
    * both neighbors share the same sign.

    Genuine straight-driving zeros never satisfy the neighbor conditions.
    Returns a new :class:`DriveLabels` with ``repaired_indices`` recording
    every touched frame, so the count is publishable and testable.
    """
    a = labels.angles_deg
    is_zero = a == 0.0
    prev_ok = np.abs(np.roll(a, 1)) >= min_neighbor_deg
    next_ok = np.abs(np.roll(a, -1)) >= min_neighbor_deg
    same_sign = np.sign(np.roll(a, 1)) == np.sign(np.roll(a, -1))
    mask = is_zero & prev_ok & next_ok & same_sign
    mask[0] = mask[-1] = False  # edges have no two neighbors

    repaired = a.copy()
    idx = np.flatnonzero(mask)
    repaired[idx] = (a[idx - 1] + a[idx + 1]) / 2.0

    return DriveLabels(
        root=labels.root,
        filenames=list(labels.filenames),
        angles_deg=repaired,
        timestamps_s=None if labels.timestamps_s is None else labels.timestamps_s.copy(),
        repaired_indices=idx,
    )
